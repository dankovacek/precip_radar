---
Date: 2020-03-03
Author: Dan Kovacek
School: University of British Columbia
Subtitle: EOSC 510 - Data Analysis in Earth and Ocean Sciences
Summary: Final Project Proposal
Template: 'formats/class'
Title: Exploring Structure in Correlation Between Precipitation Radar Imagery and Runoff
bibliography: bib/term_project.bib
geometry: "left=3cm,right=3cm,top=2cm,bottom=2cm"
---

# Term Project - Proposal

| **Date** |  **Prepared by** | **Student No.** | **Course** |
|---|---|---|---|
| 3 March 2020 | Dan Kovacek | 35402767 | EOSC 510 |

## Research Question

>What variables account for the strength of correlation between precipitation radar imagery and measured runoff in BC watersheds?

## Data

Input data for this investigation will include:

* **HYDAT Database**: daily average flow series will be extracted for a number of stations from the Water Survey of Canada (WSC) HYDAT database.  The final number of stations selected will depend upon the complexity of analysis undertaken, but I'm setting a goal of 50 stations.  Since the question focuses on larger magnitude runoff events, runoff events will be selected based on a target minimum of 50 distinct events, depending upon how easily they can be identified and extracted from the historical record, noting that the concurrent record for radar imagery goes back only as far as june 2007.  Filtering for stations smaller than some threshold will also be required to make it easier to acquire and process the radar images.  Given the targets described above, the total target number of runoff observations will be roughly 2500 events (each of which is comprised of multiple days of daily average flow observations), in addition to watershed characteristics such as drainage area, continentality/oceanity, location, and elevation, depending upon availability of information.

* **PRECIP-ET**: [historical weather radar images](https://www.canada.ca/en/environment-climate-change/services/weather-general-tools-resources/radar-overview/about.html) for precipitation will be retrieved from Environment Canada for some time window encompassing each flow event at 20 minute to 2 hour intervals based on the basin size and the event duration.  The resolution of precipitation imagery will be either 8 or 14 category intensity (mm/hr), depending upon availability.  Based on the target of 2500 precipitation events, the number of radar images could range from 10K to 1M+ observations, however there will likely be practical limitations to accessing such a large number of radar images.

* **WSC Hydrometric Station Catchments**: hydrometric watershed boundaries for WSC stations will be used to clip radar imagery to the catchments of interest in order to quantify the rainfall projected by the radar imagery.

## Methods

For many practical applications in water resource planning, the utility of point precipitation measurements is severely limited.  Radar imagery can provide more detail at the catchment level, which is especially important for convective precipitation events in larger catchments.  Radar imagery clipped to hydrometric survey station catchment boundaries will be used to approximately reconstruct a runoff hydrograph which will be compared to the measured runoff at the corresponding hydrometric station.  Given a large sample of both stations and runoff events, clustering will be used to try and identify features of catchments and precipitation patterns yielding strong correlations between the radar-precipitation based runoff hydrograph and the measured runoff hydrograph.  Of equal or greater importance might be the ability to distinguish what characteristics result in the poorest correlations, i.e. to identify and classify outliers in the input data, or identify geographic regions particularly ill-suited to using radar to predict rainfall-runoff response.