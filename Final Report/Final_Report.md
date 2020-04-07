---
date: 2020-04-06
author: Dan Kovacek (35402767)
school: University of British Columbia
subtitle: EOSC 510 - Data Analysis in Earth and Ocean Sciences
summary: Final Project Proposal
template: 'formats/class'
title: "EOSC 510 - Term Project: Assessing Similarity of Western Canadian River Basins Using Runoff and Spatial Distribution of Precipitation"
bibliography: bib/term_project.bib
geometry: "left=3cm,right=3cm,top=2cm,bottom=2cm"
numbersections: true
---

\maketitle
\thispagestyle{empty}
\clearpage
\tableofcontents
\pagenumbering{roman}
\clearpage
\pagenumbering{arabic}
\setcounter{page}{1}

# Abstract

Hydrological analysis is often undertaken on the basis that historical records from one basin can be used as a proxy for the characterization of long-term flow characteristics in an ungauged watershed of reasonably similar physiography and location.  Across British Columbia and Alberta, mountainous river basins exhibit a large amount of localized variability, confounding efforts to develop relationships that apply accurately at the regional level.  In this study, historical weather radar imagery is gathered for 144 basins in British Columbia and Alberta.  Spatial distribution of precipitation is estimated using a sample of precipitation events identified in summer and fall between 2007 and 2018.  Self organizing maps (SOM) are used to identify similarity in spatial distribution of precipitation.  SOMs are also applied to measured runoff at the same catchments to identify dominant runoff patterns.  The precipitation and runoff based SOMs are compared. The results show...

# Introduction

The characterization of water resources is a critical step in support of natural resource project proposals in Canada.  Hydrological analysis is critical to the planning, design, permitting, and operation of mines and hydropower facilities in particular.  The standard of resource engineering practice in assessing precipitation and runoff is often limited to quantification at seasonal or annual levels due to the limitations in the spatiotemporal resolution of available data.  Single point-in-space measurements are routinely applied to represent average precipitation across large areas, yet spatial distribution of precipitation is known to be highly variable.  As a result, there is considerable uncertainty associated with applying precipitation measurements to an ungauged basin of interest for analytical purposes, and derivative metrics such as runoff ratio and evapotranspiration are accordingly highly uncertain.  Similarly, regional relationships are applied in practice to adjust long-term estimates from one location to other ungauged locations some distance away.

Weather radar approximates the density of precipitation by measuring the reflection of transmitted microwave pulses off of water droplets in the air, enabling the estimation of spatial distribution of precipitation with reasonable resolution across large areas.  [@Urban_radar_2017] details the advantages of precipitation radar over traditional measurement methods, and as well discusses the difficulties in collecting, processing, and interpreting radar data and the limitations and uncertainty inherent in its use.  The goal of this research is to investigate the capacity of weather radar data for estimating runoff at ungauged locations.  Historical weather radar data from five stations in British Columbia (BC) and Alberta (AB) are combined with concurrent streamflow data from 144 hydrometric stations within sensing range of the radar stations in order to compare precipitation with resultant runoff.

# Data

## Historical Weather Radar

Environment Canada (EC) provides free access to historical weather radar data as far back as 2007 for some radar stations, though programmatic access is not provided at the time of writing.  Historical weather radar was as a result obtained by a web-crawling script targeting time periods associated with summer and fall runoff events.  Radar images from five radar stations in BC and AB were collected based on their coverage of mountainous basins.  The measurement range of radar is described by a circle with a radius of 250km centred at the radar station.  An example radar image is shown in [Figure 1](#Fig1)

![Example Radar Image (Prince George, BC.  Source: Environment Canada)](img/CASPG_ex1.gif){#Fig1 width=450}

The resolution of radar imagery corresponds to 1 pixel for every $1 km^2$, and the centre pixel represents the radar location.  It is these two pieces of information that allow a reasonable projection of a coordinate system onto the radar image which is unreferenced in the form it is acquired.  The processing of radar data is discussed further in Section X.  Note the information layers shown in [Figure 1](#Fig1), such as place names and concentric circles, are embedded in the image, causing issues for some catchments.

## Daily Average Streamflow (Runoff)

The Water Survey of Canada (WSC) provides open, programmatic access to historical daily average streamflow records for over 8000 active and inactive hydrometric stations across Canada.  WSC provides open access to a database file (HYDAT) containing all historical WSC streamflow data, and the latest available database file is used in this study.  An example streamflow time series is shown in [Figure 2](#Fig2).  The Bokeh data visualization library [@bokeh] for the Python programming language is used for plotting [Figure 2](#Fig2) and subsequent figures.

![Example Daily Average Flow Timeseries (WSC 08HB048: Carnation Creek at the Mouth)](img/ex_flow_timeseries.png){#Fig2 width=450}

The HYDAT database is filtered to include stations in BC and Alberta that fall within the sensing range of a radar station, and to include stations with historical record concurrent with the weather radar stations.  Since the radius of radar measurement is limited to 250 km, and because information layers embedded in the radar images obstruct some areas of the images, the WSC stations were also filtered to include stations capturing a drainage area of less than $1000 km^2$.  A total of 141 stations were found to fit the filtering criteria.  The smallest WSC catchments are in the order of $10 km^2$ which correspond to a mere ~10 pixels of the radar image.  As a result, the accuracy of the smallest catchments is expected to be poor.  The WSC stations satisfying the above criteria are plotted on [Figure 3.](#Fig3)

![Radar Stations and Range (Approximate), and WSC Stations within Radar Coverage](img/wsc_radar_stn_map.png){#Fig3 width=480}

## Catchment Boundaries

Geographic polygons corresponding to most of the WSC hydrometric stations are available from the Government of Canada's [Open Data Platform](http://donnees.ec.gc.ca/data/water/products/national-hydrometric-network-basin-polygons/?lang=en).  Given the low resolution of the radar images, the shape files are believed to be suitable for the intended purpose of extracting from the radar images the pixels corresponding to each catchment of interest.  An example catchment boundary with its corresponding station location is shown in [Figure 4](#Fig4)

![Catchment boundary for WSC 08HB048: Carnation Creek at the Mouth](img/wsc_example_catchment.png){#Fig4 width=480}

# Methodology

The primary research question involves the comparison of weather radar data and streamflow data to ultimately find clustering patterns of watersheds based on spatial distribution of rainfall, physiographic characteristics, and geographic location.  The data acquisition process necessitates an additional analytical step which will first be addressed to set the context for the subsequent analysis.  The remainder of data pre-processing is then described, including extraction of precipitation data from radar images corresponding to WSC station catchments, and the reconstruction of runoff hydrographs.  Finally, the methods used to evaluate the predictive power of radar data are presented.

## Data Acquisition and Preprocessing

### Radar Image Acquisition

A limiting step in the data acquisition process is radar imagery retrieval. The number of server requests required to capture 12 years of radar images at 10-minute intervals at five different stations is in the order of $3\times10^6$, which is an excessive imposition on a free service, and it invites a ban from use.  Focusing the study to summer and fall months simplifies the interpretation of radar data by avoiding precipitation as snow, but it does not alone reduce the number of server requests to a viable level.  An anomaly detection (AD) algorithm is used to identify isolated runoff events between June and October (inclusive) in order to reduce the total number of radar image requests to a reasonable number.  However, the execution time of a sufficiently complex AD algorithm could negate the (time) cost savings from a reduced number of image requests, and a sufficiently sensitive AD algorithm could label every oscillation in the input signal, no matter how minute, as a 'runoff event', resulting in more image requests and longer AD algorithm execution time.  A tradeoff then exists between running the anomaly detection algorithm to reduce the total radar image server calls to a viable number, the time required for the AD algorithm execution, and the number of runoff events identified.  The sample size for meaningful analysis adds a minimum constraint on the number of events identified.  Details of the AD algorithm are discussed in the subsequent section before returning to the question of radar image preprocessing.

### Anomaly Detection for Identifying Runoff Events

The identification of individual runoff events in streamflow signals is a difficult problem, and generalizing the problem across catchments of highly variable characteristics adds to the complexity.  For the specific use case of this study, it is important to identify a sufficient number of samples (true positives) in order to support meaningful analysis in the subsequent steps addressing the primary research question, which requires representative estimates of spatial distribution of precipitation.  False negatives (missing events) are considered less important than false positives (identifying a runoff event where there isn't one) in reducing the quality of the dataset, as false positives tend to result in biased outcomes corresponding to either 0 or infinite runoff ratio.

An initial approach to identifying runoff events used a simple anomaly detection (AD) algorithm based on moving windows and a percentile-based outlier threshold on the differenced flow series.  This moving-window approach proved ineffective at generalizing across all watersheds, and the results were difficult to validate and interpret.  A modified approach using PCA was successful at generalizing across watersheds and at capturing data that were easier to validate.  [Figure 5](#Fig5) shows an example of the AD algorithm identifying three events in the period of interest (June to October, inclusive).  A sample of the total events identified is plotted in [Figure 6](#Fig6) in a small-multiples format to facilitate the validation the runoff events identified by the AD algorithm for each station.

![Example AD Results (Timeseries) for WSC 08HB048: Carnation Creek at the Mouth](img/AD_results_ts.png){#Fig5}

![Example AD Results (Event Hydrographs) for WSC 08HB048: Carnation Creek at the Mouth ](img/AD_example_results.png){#Fig6}

The principal components (PCs) comprising a minimum 90% of the variance in lagged data are then used to create a time series corresponding to the Mahalanobis distance (MD) -- the Euclidean distance from each row of data points (each detrended observation and its n lags) to the corresponding PCs. A threshold Mahalanobis distance then represents some magnitude of deviation from the PC within a timeframe correspondent with the number of lags (components).  The runoff events identified by the AD algorithm correspond to the timestamps where the MD crosses the threshold (in either direction).

The PCA-based AD method takes a daily average runoff time series, and builds a matrix of some number of lag periods proportional to the size of the catchment.  Using up to 15 lag periods, or 15 days, is expected to be suitable for the time of concentration and hydrograph response of basins up to 1000 $km^2$.  Principal Component Analysis (PCA) is then applied to reduce the number of lag series to those components describing a minimum of 90% of the variance in the data.  [Figure 7](#Fig7) shows that most of the time just 2 components are required to meet the 90% variance target, however the expected AD performance is a maximum for between 4 and 5 components, and the confidence interval highlights the large amount of variance in the data for between 2 and 6 PCA components.  Much of this variance is expected to result from the runoff signals themselves, as for example few runoff events will be identified from June to October (inclusive) in the semi-arid climate of the BC interior.

![PCA Components and AD Performance](img/pca_component_numbers.png){#Fig7 width=500}

### Sensitivity to Training Period

The AD algorithm is supervised in the sense that a training period is provided as an input.  Initial testing of the AD algorithm demonstrated a high level of sensitivity to the training period selected.  Combining the random selection of a single year (2007-2018) for input training with a random selection of 1-12 months (inclusive) yields a total search space of roughly 50 thousand alternatives.  The execution time of the AD algorithm is such that evaluating the entire search space is intractable for practical purposes.  Better efficiency in code may be possible, however the main function of the AD algorithm uses the well-optimized Tensorflow Python library [@tensorflow2015-whitepaper].  

Monte Carlo (MC) simulation is used to illustrate the variability in AD performance as a function of training period selection.  [Figure 8](#Fig8) shows the variability in number of events identified by the AD algorithm across all WSC stations based on random selection.  Note that the results of each station are each comprised of 50 trials of randomly selected training parameters, and the MC simulation represents 1000 random selections from the aggregate results.  Since the training inputs are not continuous variables, a gradient-based search method cannot be applied directly, however the long tail of the distribution highlights the opportunity for an improved search method, which remains to be addressed in future work. 

![MC simulation: 1000 random selections of training period (KDE probability density function fit)](img/MC_training_sim.png){#Fig8 width=500}

### Radar Image Preprocessing

With a set of runoff events identified by the AD algorithm for each WSC station, start and end times of the runoff events are are used to form a series of queries to retrieve concurrent radar images from the nearest radar station.  Once the radar images are retrieved, a matrix is constructed with the same shape as the image (in pixels), and populated with values corresponding to the azimuthal equidistant projection of the radar image.  The known coordinates of the radar station correspond to the centre pixel, which is used to reproject the entire matrix to a projection consistent with that of the catchment basin geometry.  Each basin geometry is retrieved and used to create a boolean 'mask' such that the radar image pixels representing each basin can be captured in a batch process.  The projection error associated with conversion from the radar projection to the basin geometry projection is well within the image resolution of $1km^2$ per pixel, so the error is neglected. An example of a radar image mask representing a basin is shown in [Figure 9.](#Fig9)

![Sample Basin Captured from Radar Image (WSC 05BL014)](img/clipped_radar_example.png){#Fig9}

As shown in [Figure 1](#Fig1), rainfall intensity is mapped to an array of 14 colours unique to representing precipitation.  The final step in the radar image processing is to map the colour values in the masked radar images to their corresponding precipitation intensity.  The result is a series of matrices representing the volume of water that fell on each pixel or cell comprising the basin at each time step.  Reducing the time dimension by summing and normalizing yields the spatial distribution of precipitation across all events captured.  The results are discussed in the subsequent section.

# Results and Discussion

-having all the data would eliminate the costly AD step, negate the issues with false negatives, and give a more complete picture of total precip.

## Spatial Distribution of Precipitation

Summing the precipitation time series and applying a colour map to the normalized output volume yields a representation of the spatial distribution of precipitation for the sample events captured.  [Figure 9](#Fig9) shows a grid plot of a subsample of 100 WSC basins scaled to similar size to emphasize the differences in colour patterns, where yellow represents less precipitation and blue represents more precipitation.  

Vestiges of the information layer are apparent in some of the basins in [Figure 10](#Fig10) where parts of place names and the concentric rings from the image block the radar information.  In other basins, non-uniformities are evident (08MG001, 08HA003) suggesting orographic effects.  Spotted patterns (08NL071) suggest precipitation falling in convective cells in one or few events in the summer season, though this could also be attributable to noise or other interference in the radar data.

![Qualitative Representation of Spatial Distributions of Precipitation in Basins of Western Canada](img/spatial_precip_art1.png){#Fig10}

The [Pysheds](https://mattbartos.com/pysheds/) Python package [@pysheds] is used to facilitate the batch processing of radar image masking.

[@Loukas_1994] estimated precipitation lapse rate in the Seymour and Capilano basins, and found no functional relationship between *annual* precipitation and the distance from the beginning of the slope.  

[@precip_elevation_1973] investigated the effect of elevation on summer precipitation and found a rate of 7% increase per 100m elevation increase in the Santa Catalina Mountains of Arizona.

## Hydrograph Reconstruction

## Clustering Spatial Precipitation Distribution

Often basins of interest to a researcher are ungauged.  In the case of estimating water resources for ungauged basins, information from gauged basins is projected to the location of interest based on similarities in physiography.  [@obedkoff_2003] divides the hydrology of British Columbia into many subregions on the basis of there existing some level of homogeneity in runoff characteristics at a local level.  The hydrologic zones in BC are typically aligned with the coast as the source of air moisture is predominantly the Pacific Ocean.  Measured runoff statistics such as long-term averages and extremes then form the basis for the region for the purpose of estimating runoff at ungauged locations.  


# Conclusions

## Future Work

# References