# EOSC 510 - Term Project

## Introduction

Input data for this project includes daily streamflow data from the HYDAT database made available by the Water Survey of Canada, and hourly weather radar images made available by Environment Canada.  The two datasets are obviously related, with one describing precipitation as rain and the other describing the resulting runoff.  Here, the two are connected with a third dataset of shape files representing the catchment boundaries of each WSC station where runoff is measured.  

In terms of the research question: I took this project as an opportunity to try letting the question evolve with data exploration, rather than set some distant target at the outset and pursue it unwavering, which is maybe more scientifically rigorous and honest.  In my limited experience, taking time to explore never fails to turn up interesting questions, and having the time to try is a luxury I'm grateful to have.   

Here's an example radar image showing Vancouver Island and the Lower Mainland, here's a Water Survey of Canada hydrometric Station, and here's the station's catchment boundary.  Now we add a bunch of sequential radar images showing a rainfall event occurring over a couple of days.  Here's a reconstructed rainfall hydrograph, and here's the concurrent runoff measured independently at the corresponding hydrometric station.  So that's a high-level description of the data I used, and the basic idea of how I put together my dataset, which ended up being about 95% of the work.  

(Show hydrograph and inset detail about single runoff event.)
So the samples in my dataset are single runoff events, and I needed to figure out a reliable way to get a large number of these so that I know what time range to query to get the radar images corresponding to the runoff event.

(Show gridplot of single events)
So how do you isolate hundreds of runoff events without manually combing through millions of data points?  It turns out PCA and its derivatives, including SSA have been applied to this question in the literature.  But I'll come back to this, because I want to briefly describe the part that gave me the most trouble, but yielded really satisfying results.  

I spent an obscene amount of time trying to figure out how to project coordinates onto an unreferenced radar image, but once I figured it out I was able to determine which pixels from the radar image fall within the catchment.  Now that I can automatically figure out which pixels I want to keep, I can start gathering all of the radar imagery.  Satellite radar is unfortunately not available on the Canada Open Data Portal, so I had to hit their webform a few hundred thousand times to retrieve the images individually, so my apologies to their IT team.

Also unfortunately, the radar images come with a few problems, like missing data, and a bunch of different things that add noise to the signal I'm trying to isolate, which is summer precipitation.  The effect of bad data goes to both extremes:  
  -(show 08NH132 which didn't record any rain in the radar, snoopy under umbrella cartoon) 
  -(show FitzSimmons Creek which has the noise in it saying it's always a downpour, show charlie brown cartoon)  

(Show the 2D histogram grid plot showing the spatio-temporal rainfall distribution)
The result of this effort is probably more satisfying for me given the lows I experienced getting here.  The shapes in this figure represent distinct watersheds in BC and Alberta.  They are roughly ordered geographically, but the size is normalized, or scaled to be roughly equal, and the colour represents where most of the rain fell, or how it's distributed spatially.  

I focused on rainfall events in typically drier periods like summer and early fall, mostly because I thought it would be a clearer signal of rainfall driving runoff in summer.  Whether or not there's glacier in the catchment, the runoff data is daily average, so any spikes in runoff should come mostly from rainfall, as the temporal scales of glacier melt are diurnal, and also seasonal due to temperature and solar gain.

Because the radar coverage has a radius of only 240 km, I eliminated stations falling outside 200 km from the radar, and those larger than 500 km^2 to actually fall within the radar area.  The result is I was able to collect about 500 samples of runoff events from about 50 different stations in BC and Alberta.  

In the process of collecting and processing the data, a number of interesting questions arose.  Returning to the question of anomaly detection:

(Show the first attempt figure with a million hits)
My first few attempts using simple z-scores were pathetic, and I realized I had to make a performance tradeoff.  In this case, if I miss a few events it doesn't matter, but if I capture a bunch of non-events, it's bad.  In other words, I only care about minimizing false positives because they'll add a lot of uncertainty to my results, and I don't care about false negatives unless I'm not finding any events at all.  PCA turned out to be a useful transformation for part of (a method of anomaly detection), and I was able to very effectively avoid capturing false positives, but the false negative performance varied by station and about half the stations captured few events or none at all.  So then I tried CCA to see if it was any better.  I also tried playing with training parameters to see the effect, but this kind of guess and check is like playing lawn darts: the fun quickly wears and it can be dangerous.

BC is big and diverse, so surely different model parameters will be better suited to different areas?   Recognizing any reasoned approach I could come up with to setting training parameters is probably biased, I decided to run a Monte Carlo simulation on the training parameters themselves.  By randomly selecting training parameters for each station, I found those that yielded the most samples (note that I didn't go as far as checking false positive performance).  I looked at the resulting distributions and then used k-means clustering to see if any physical characteristics of the catchments would turn up in the clusters.  Then I did the same using CCA and compared the two methods.  

(show expected value curves based on length of training period, broken down by different categories)

The next question we go back to the radar data.  Precipitation gauges are useful for a lot of things, but not for approximating rainfall over an area.  Because often it's the best available information, precipitation gauge data gets abused like this in practice.  This has caused me a lot of grief, so I wanted a way to look at spatial distribution of precipitation in catchments and relate it somehow to runoff.  

(show the spatial distribution plots again)
I used SOM to cluster (spatially and/or otherwise) basins whose summer runoff tends to fall more spatially uniformly in summer months.

Finally, using the radar data, I can also reconstruct a hydrograph to see how it relates to the runoff event.  In practice, runoff coefficients might be done only on a seasonal level, whereas in reality it depends on a number of things like soil moisture, vegetation, etc..

-start with 2d histogram of volume of water over the catchment (total volume)

-then try a 2d histogram to show where the water fell most often (frequency)

-then try incorporating the spatial intensity

-create a slide of the 2d histograms of precipitation intensity (and/or duration), or 3d of precipitation frequency (z) and average intensity (colour mapped)