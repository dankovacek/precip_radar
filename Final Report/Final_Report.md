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

# EOSC 510 Term Project: Final Report

| **Date** |  **Prepared by** | **Student No.** | **Course** |
|---|---|---|---|
| 6 April 2020 | Dan Kovacek | 35402767 | EOSC 510 |

## Introduction

## Data

### Historical Weather Radar

Environment Canada (EC) provides free access to historical weather radar data as far back as 2007 for some radar stations, though programmatic access is not provided at the time of writing.  Historical weather radar as a result was obtained by a web-crawling script on specific time periods corresponding to summer and fall runoff events.  Five stations in BC and Alberta were used based on coverage of mountainous basins.  Weather radar coverage encompasses a circular area described by a radius of 250km from the radar station.  The resolution of radar imagery corresponds to 1 pixel for every 1kmx1km, and the image is centred on the station.  It is these two pieces of information that allow a reasonable projection of a coordinate system onto an unreferenced image.  The algorithm behind the extraction of radar data is discussed in greater detail in Section X.

### Daily Average Streamflow (Runoff)

The Water Survey of Canada (WSC) provides open, programmatic access to historical daily average streamflow records for over 8000 active and inactive hydrometric stations across Canada.  A database file (HYDAT) containing all historical WSC streamflow data is maintained and updated quartely, and the October 2019 HYDAT database file is used in this study.  The HYDAT database is filtered to include stations in BC and Alberta, to include stations with historical record concurrent with the weather radar stations, to include stations falling within a 200km radius of the nearest radar station.  Since the radar coverage has a radius of 250 km, and also because of the obstruction of some pixels due to overlaid information layers, the WSC stations were also filtered to include stations capturing a drainage area of less than $1000 km^2$.  The smallest WSC catchments in the order of $10 km^2$ which corresponds to a low resolution containing only 10 pixels in the radar image.  As a result, the smallest station considered in the study is $30 km^2$.

### Catchment Boundaries

Geographic polygons corresponding to most of the WSC hydrometric stations are available from the Government of Canada's [Open Data Platform](http://donnees.ec.gc.ca/data/water/products/national-hydrometric-network-basin-polygons/?lang=en).  Given the low resolution of the radar images, the polygons are believed to be suitable for the intended purpose of extracting from the radar images the pixels corresponding to each catchment of interest.

## Methodology

The primary line of investigation concerns the comparison of weather radar data concurrent to measured runoff in catchments monitored by WSC.  The data acquisition process necessitated an additional analytical step which will be addressed first to set the context for the subsequent analysis.  

### Anomaly Detection

A limiting step in the data aquisition process is radar imagery retrieval. The number of server requests required to capture 12 years of radar images at 10-minute intervals at 5 stations in BC and Alberta is in the order of 3E6, which is excessive for a free service and invites a ban from use.  Focusing the study to summer months to simplify the interpretation of radar data (by avoiding precipitation as snow) does not reduce the number of requests to a viable level.  As a result, an anomaly detection (AD) algorithm is used to identify isolated runoff events, which tend to be rare in summer but increasing in fall in many parts of BC and Alberta, to reduce the total number of radar image requests.  However, the execution time of a sufficiently complex AD algorithm could negate the (time) cost savings of a reduced number of image requests, and a sufficiently sensitive AD algorithm could label every oscillation, no matter how minute, as a 'runoff event', resulting in more image requests and longer AD algorithm execution time.  A tradeoff then exists between running the anomaly detection algorithm to reduce the total radar image server calls to a viable number, the time required for the AD algorithm execution, and the number of runoff events identified (true positives).  

The AD algorithm is supervised, in the sense that a training period must be provided.  Initial testing of the AD algorithm demonstrated a high level of sensitivity to the training period selected.  Combining the random selection of a single year for input training with a random selection of 1-12 months yields a search space of roughly 50K alternatives.  The execution time of the AD algorithm is roughly proportional to the length of training data, and the full search is considered intractable for practical purposes, noting that the main function of the AD algorithm is based on the well-optimized Tensorflow Python library [@tensorflow2015-whitepaper].  To illustrate the time cost of a random search for input parameters, a random sample of 30 input parameters applied to the AD algorithm took 70 minutes for 144 stations on a six-core gpu-enabled (CUDA) machine.  As shown in [Figure 1](#Fig_1) below, the number of runoff events detected by the AD algorithm using the random sample of 30 training parameter combinations per station are exponentially distributed, highlighting the opportunity for an improved search method.  

![KDE Distribution of AD performance based on random training input](img/AD_performance_distribution.png){#Fig_1}
