import os
import sys
import math

import numpy as np
import pandas as pd
import time
import param

from functools import lru_cache

import scipy.special
import scipy.stats as st

from numba import jit

from bokeh.layouts import row, column
from bokeh.models import CustomJS, Slider, Band, Spinner, RangeSlider
from bokeh.plotting import figure, curdoc, ColumnDataSource
from bokeh.models.widgets import AutocompleteInput, Div, Toggle
from bokeh.models.widgets import DataTable, TableColumn
from bokeh.models import Range1d

from figures import *

path = os.path.abspath(os.path.dirname(__file__))
if not path in sys.path:
    sys.path.append(path)

from get_station_data import get_daily_UR, get_annual_inst_peaks

from skew_calc import calculate_skew

from stations import IDS_AND_DAS, STATIONS_DF, IDS_TO_NAMES, NAMES_TO_IDS


def calculate_sample_statistics(x):
    return (np.mean(x), np.var(x), np.std(x), st.skew(x))


def update_UI_text_output(n_years):
    ffa_info.text = """
    Simulated measurement error is assumed to be a linear function of flow. 
    Coloured bands represent the 67 and 95 % confidence intervals of the 
    curve fit MCMC simulation.  The LP3 shape parameter is the generalized skew,
    note how poor the fit when the skew is negative, common for small samples.
    For information on this application,\
    see <a href="https://www.dkhydrotech.com/entry/5/" target="_blank">this 
    writeup.</a>
    """.format()
    error_info.text = ""


def randomize_msmt_err(val, msmt_err_params):
    msmt_error = val * msmt_err_params[0] + msmt_err_params[1]
    return val * np.random.uniform(low=1. - msmt_error, 
                             high=1. + msmt_error)


def LP3_calc(data, exceedance):
    # calculate the log-pearson III distribution
    mean, variance, stdev, skew = calculate_sample_statistics(np.log10(data))
    lp3_model = st.pearson3.ppf(exceedance, abs(skew), loc=mean, scale=stdev)
    return np.power(10, lp3_model)


def calculate_measurement_error_params(data):
    """
    Assume measurement error is a linear function
    of magnitude of flow.
    """
    min_e, max_e = np.divide(msmt_error_input.value, 100.)
    min_q, max_q = min(data), max(data)
    m = (max_e - min_e) / (max_q - min_q)
    b = min_e - m * min_q
    return (m, b)


def set_up_model(df):
    mean, variance, stdev, skew = calculate_sample_statistics(np.log10(df['PEAK']))
    model = pd.DataFrame()
    model['Tr'] = np.linspace(1.01, 200, 500)
    model['theoretical_cdf'] = 1 / model['Tr']
    log_q = st.pearson3.ppf(1 - model['theoretical_cdf'], abs(skew),
                            loc=mean, scale=stdev)
    model['theoretical_quantiles'] = np.power(10, log_q)

    mean, variance, stdev, skew = calculate_sample_statistics(np.log10(df['PEAK_SIM']))
    log_q_sim = st.pearson3.ppf(1 - model['theoretical_cdf'], abs(skew),
                        loc=mean, scale=stdev)
    model['theoretical_quantiles_sim'] = np.power(10, log_q_sim)
    return model


def run_ffa_simulation(data, n_simulations):
    """
    Monte Carlo simulation of measurement error.
    Reference:
    https://nbviewer.jupyter.org/github/demotu/BMC/blob/master/notebooks/CurveFitting.ipynb
    """

    peak_values = data[['PEAK']].to_numpy().flatten()
    years = data[['YEAR']].to_numpy().flatten
    flags = data[['SYMBOL']].to_numpy().flatten

    data = calculate_distributions(peak_values, years, flags)
    model = set_up_model(data)

    simulation_matrix = np.tile(peak_values, (n_simulations, 1))

    msmt_error_params = calculate_measurement_error_params(peak_values)

    simulated_error = np.apply_along_axis(randomize_msmt_err, 1,
                                          simulation_matrix,
                                          msmt_err_params=msmt_error_params)

    exceedance = 1 - model['theoretical_cdf'].values.flatten()

    simulation = np.apply_along_axis(LP3_calc, 1,
                                     simulated_error,
                                     exceedance=exceedance)
  
    model['lower_1s_bound'] = np.apply_along_axis(np.percentile, 0, simulation, q=33)
    model['upper_1s_bound'] = np.apply_along_axis(np.percentile, 0, simulation, q=67)
    model['lower_2s_bound'] = np.apply_along_axis(np.percentile, 0, simulation, q=5)
    model['upper_2s_bound'] = np.apply_along_axis(np.percentile, 0, simulation, q=95)
    model['expected_value'] = np.apply_along_axis(np.percentile, 0, simulation, q=50.)
    model['mean'] = np.apply_along_axis(np.mean, 0, simulation)

    return model

def calc_theoretical_quantiles(data):
    return 


def calc_simulated_msmt_error(data):
    msmt_error_params = calculate_measurement_error_params(data['PEAK'])
    return [randomize_msmt_err(v, msmt_error_params) for v in data['PEAK']]


def calculate_distributions(peak_values, years, flags, correction_factor=None):
    """
    Calculate return period, and empirical and theoretical CDF series.
    """   
    n_sample = len(peak_values)
    data = pd.DataFrame()
    data['PEAK'] = peak_values
    data['PEAK_SIM'] = calc_simulated_msmt_error(data)
    data['YEAR'] = years
    data['SYMBOL'] = flags

    mean, variance, stdev, skew = calculate_sample_statistics(np.log10(data['PEAK']))
    data['Tr'] = (n_sample + 1) / data['PEAK'].rank(ascending=False)
    data['empirical_cdf'] = st.pearson3.cdf(np.log10(data['PEAK']), abs(skew), loc=mean, scale=stdev)
    data['theoretical_cdf'] = 1 - 1 / data['Tr']
    data['theoretical_quantiles'] = np.power(10, st.pearson3.ppf(data['theoretical_cdf'], 
                                            abs(skew), loc=mean, scale=stdev))
    data = data.sort_values('Tr', ascending=False)

    mean, variance, stdev, skew = calculate_sample_statistics(np.log10(data['PEAK_SIM'])) 
    data['Tr_sim'] = (n_sample + 1) / data['PEAK_SIM'].rank(ascending=False)
    data['empirical_cdf_sim'] = st.pearson3.cdf(np.log10(data['PEAK_SIM']), 
                                                abs(skew), loc=mean, scale=stdev) 
    data['theoretical_cdf_sim'] = 1 - 1 / data['Tr_sim']    
    data['theoretical_quantiles_sim'] = np.power(10, st.pearson3.ppf(data['theoretical_cdf_sim'], 
                                                 abs(skew), loc=mean, scale=stdev))   
    return data


def get_data_and_initialize_dataframe():
    station_name = station_name_input.value.split(':')[-1].strip()
   
    df = get_annual_inst_peaks(
        NAMES_TO_IDS[station_name])

    if len(df) < 2:
        error_info.text = "Error, insufficient data in record (n = {}).  Resetting to default.".format(
            len(df))
        station_name_input.value = IDS_TO_NAMES['08MH016']
        return get_data_and_initialize_dataframe()

    df = calculate_distributions(df['PEAK'].values.flatten(),
                                 df['YEAR'].values.flatten(),
                                 df['SYMBOL'].values.flatten())
    return df

def update():
    
    df = get_data_and_initialize_dataframe()

    n_years = len(df)

    # Run the FFA fit simulation on a sample of specified size
    # number of times to run the simulation
    time0 = time.time()
    model = run_ffa_simulation(df, simulation_number_input.value)
    time_end = time.time()

    # print(model[['Tr', 'theoretical_quantiles', 'theoretical_cdf']].head())
    # print(model.columns)

    print("Time for {:.0f} simulations = {:0.2f} s".format(
        simulation_number_input.value, time_end - time0))

    # update the data sources  
    peak_source.data = peak_source.from_df(df)
    # peak_sim_source.data = peak_sim_source.from_df(df)
    data_flag_filter = df[~df['SYMBOL'].isin([None, ' '])]
    peak_flagged_source.data = peak_flagged_source.from_df(data_flag_filter)

    distribution_source.data = model
    update_UI_text_output(n_years)
    update_data_table()


def update_station(attr, old, new):
    peak_source.selected.indices = []
    refresh_histogram()
    update_data_table()
    update()


def update_n_simulations(attr, old, new):
    if new > simulation_number_input.high:
        simulation_number_input.value = 500
        error_info.text = "Max simulation size is 500"
    update()


def update_msmt_error(attr, old, new):
    update()


def update_simulation_sample_size(attr, old, new):
    update()

def refresh_histogram():
    df = get_data_and_initialize_dataframe()
    selection = df['PEAK']

    pv.y_range = Range1d(selection.min(), selection.max())

    vhist, vedges = np.histogram(selection, bins=10)
    vzeros = np.zeros(len(vedges)-1)
    vmax = max(vhist)*1.1

    hist_source.data["bottom"] = vedges[:-1]
    hist_source.data["top"] = vedges[1:]
    hist_source.data["right"] = vhist
    hist_source.data["left"] = np.zeros_like(vhist)
    update_pv_plot()

def update_pv_plot():
    indices = peak_source.selected.indices

    if len(indices) > 0:
        data = peak_source.data['PEAK']
        years = peak_source.data['YEAR'][indices]
        selection = data[indices]
    else:
        df = get_data_and_initialize_dataframe()
        selection = df['PEAK'].values.flatten()

    vhist, vedges = np.histogram(selection, bins=10)
    vzeros = np.zeros(len(vedges)-1)
    vmax = max(vhist)*1.1

    if len(indices) == 0 or len(indices) == len(data):
        vhist1, vhist2 = vzeros, vzeros
    else:
        neg_inds = np.ones_like(data, dtype=np.bool)
        neg_inds[indices] = False
        vhist1, _ = np.histogram(data[indices], bins=vedges)
        # vhist2, _ = np.histogram(data[neg_inds], bins=vedges)

    vh1.data_source.data["right"] = vhist1
    # vh2.data_source.data["right"] = -vhist2
    vh1.data_source.data["bottom"] = vedges[:-1]
    # vh2.data_source.data["bottom"] = vedges[:-1]
    vh1.data_source.data["top"] = vedges[1:]
    # vh2.data_source.data["top"] = vedges[1:]
    vh1.data_source.data["left"] = np.zeros_like(vhist1)
    # vh2.data_source.data["left"] = np.zeros_like(vhist2)


def update_UI(attr, old, new):
    inds = new
    # update the datatable only if at least three points are selected
    if len(inds) > 2:
        # retrieve the entire dataset and filter for 
        # the selected points
        years = peak_source.data['YEAR'][inds]
        df = get_data_and_initialize_dataframe()
        selected = df[df['YEAR'].isin(years)]

        selected = calculate_distributions(selected['PEAK'].values.flatten(),
                        selected['YEAR'].values.flatten(),
                        selected['SYMBOL'].values.flatten())

        model = run_ffa_simulation(selected, simulation_number_input.value)

        data = selected['PEAK'].values.flatten()

        update_pv_plot()
        stats = [round(e, 2) for e in calculate_sample_statistics(data)]
        datatable_source.data['value_selection'] = [stats[0], stats[2], stats[3], len(data)]
        distribution_source.data = distribution_source.from_df(model)
        update_data_table()


def update_data_table():
    """
    order of stats is mean, var, stdev, skew
    """
    indices = peak_source.selected.indices
    data = peak_source.data['PEAK']
    years = peak_source.data['YEAR'][indices]
    selection = data[indices]

    df = pd.DataFrame()
    stats_all = calculate_sample_statistics(np.log(data))

    if len(selection) > 1:
        selected_stats = calculate_sample_statistics(np.log(selection))
    else:
        selected_stats = stats_all

    df['parameter'] = ['Mean', 'Standard Deviation', 'Skewness', 'Sample Size']
    df['value_all'] = np.round([stats_all[0], stats_all[2], 
                                stats_all[3], len(data)], 2)
    df['value_selection'] = np.round([selected_stats[0], selected_stats[2], 
                                     selected_stats[3], len(selection)], 2)
    datatable_source.data = dict(df)


def update_simulated_msmt_error(val):
    update()

# configure Bokeh Inputs, data sources, and plots
autocomplete_station_names = list(STATIONS_DF['Station Name'])
peak_source = ColumnDataSource(data=dict())
peak_flagged_source = ColumnDataSource(data=dict())
distribution_source = ColumnDataSource(data=dict())
qq_source = ColumnDataSource(data=dict())
datatable_source = ColumnDataSource(data=dict())
hist_source = ColumnDataSource(data=dict())

station_name_input = AutocompleteInput(
    completions=autocomplete_station_names, 
    title='Enter Water Survey of Canada STATION NAME (USE ALL CAPS)',
    value=IDS_TO_NAMES['08MH016'], min_characters=3)

simulation_number_input = Spinner(
    high=5000, low=100, step=1, value=500, title="Number of Simulations",
)

sample_size_input = Spinner(
    high=200, low=2, step=1, value=10, title="Sample Size for Simulations"
)

msmt_error_input = RangeSlider(
    start=0, end=100., value=(10, 35), 
    step=2, title="Measurement Uncertainty [%]",
)

toggle_button = Toggle(label="Simulate Measurement Error", button_type="success")

ffa_info = Div(width=550,
    text="Mean of {} simulations for a sample size of {}.".format('x', 'y'))

error_info = Div(text="", style={'color': 'red'})


# Set up data table for summary statistics
datatable_columns = [
    TableColumn(field="parameter", title="Parameter"),
    TableColumn(field="value_all", title="All Data"),
    TableColumn(field="value_selection", title="Selected Data"),
]

data_table = DataTable(source=datatable_source, columns=datatable_columns,
                       width=450, height=125, index_position=None)

# callback for updating the plot based on a changes to inputs
station_name_input.on_change('value', update_station)
simulation_number_input.on_change('value', update_n_simulations)
msmt_error_input.on_change('value', update_msmt_error)
sample_size_input.on_change(
    'value', update_simulation_sample_size)
toggle_button.on_click(update_simulated_msmt_error)

# see documentation for threading information
# https://docs.bokeh.org/en/latest/docs/user_guide/server.html

update()

# widgets
ts_plot = create_ts_plot(peak_source, peak_flagged_source)

peak_source.selected.on_change('indices', update_UI)

vh1, pv, hist_source = create_vhist(peak_source, ts_plot)

ffa_plot = create_ffa_plot(peak_source, peak_flagged_source,
                           distribution_source)

qq_plot = create_qq_plot(peak_source)

pp_plot = create_pp_plot(peak_source)

# create page layout
info_input_block = column(simulation_number_input,
                          msmt_error_input,
                          ffa_info,
                          toggle_button)

input_layout = row(info_input_block,
                   column(station_name_input,
                          data_table),
                   sizing_mode='scale_both')

layout = column(input_layout,
                error_info,
                row(ts_plot, pv),
                row(ffa_plot, column(pp_plot, qq_plot))
                )

curdoc().add_root(layout)
