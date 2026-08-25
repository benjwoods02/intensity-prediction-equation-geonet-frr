# Intensity Prediction Equation from GeoNet Felt RAPID Reports

Predicting Modified Mercalli Intensity (MMI) from crowdsourced earthquake shaking reports, using GeoNet's Felt RAPID Report (FRR) data for New Zealand.

Full write-up: [DATA601-FRR2-Report.pdf](DATA601-FRR2-Report.pdf)

## The problem

Intensity prediction equations estimate how strongly an earthquake is felt at a given location, which supports rapid post-event assessment and earthquake risk modelling. New Zealand's official equations rely on rupture plane data that is not available immediately after an event.

GeoNet's Felt RAPID Report survey asks the public which of six cartoons best matches what they experienced, mapping to MMI 3 through 8. That produces a large crowdsourced dataset covering the whole country. This project asks whether that data alone can produce a usable and physically sensible intensity prediction equation.

## Approach

- Around 300,000 felt reports across 103 earthquakes (magnitude 4.0 to 8.0), stratified across magnitude bins to avoid a handful of large events dominating.
- Reports aggregated to a 1 km grid, with a minimum of 5 reports per cell. Mode was used as the central tendency measure rather than mean or median, since MMI is ordinal and the distributions are skewed.
- Features: log hypocentral distance, magnitude, depth, and Vs30 (a site condition proxy for how soft the ground is).
- 39 candidate models evaluated, 13 classification and 26 regression, spanning linear models, tree ensembles, generalised additive models, and boosted methods.

## The interesting part

Models were screened against physical plausibility, not just predictive metrics. Shaking intensity must decrease as distance from the epicentre increases. Several models with strong R2 and MSE scores produced shake maps that violated this, showing patchy or non-monotonic intensity across space.

Those models were rejected despite scoring well. The final choice was a degree-2 polynomial Ridge regression, which was more interpretable and produced physically sensible attenuation.

Metrics alone would have selected a different and worse model.

## Results

Final model, degree-2 polynomial Ridge regression:

- R2 = 0.317
- MSE = 0.574
- 85.2% of predictions within 1 MMI unit
- 99.5% of predictions within 2 MMI units

A secondary finding: the model learned the opposite sign to the expected relationship for Vs30. Softer ground should amplify shaking, but the data suggested otherwise. The likely explanation is urban reporting bias, since cities cluster on soft sedimentary ground and also generate far more felt reports.

## Status

A rebuilt implementation is in progress as solo work. The `src/`, `notebooks/` and `tests/` directories are that rebuild; the report above documents the original project. The rebuild covers:

- A reproducible ingestion step pulling directly from the public GeoNet API
- The cleaning and grid aggregation pipeline
- Exploratory analysis, including a comparison of mean, median, and mode as the central tendency measure
- The model comparison across all candidates
- The spatial validation step that drove model rejection

## Context

Completed as a DATA601 research project at the University of Canterbury, in a team of two (Benjamin Woods and Zheng Chao), supervised by Hazel Fraser and Goldie Leung.

My contribution covered the data cleaning and feature pipeline, the central tendency analysis, sourcing and integrating the Vs30 data, the model sweep and metric collection, and the shake map validation function used to screen models for physical plausibility.

## Data source

[GeoNet Felt RAPID Reports](https://api.geonet.org.nz/) and the [GeoNet quake search API](https://quakesearch.geonet.org.nz/), both publicly available. Vs30 site condition data sourced separately and not redistributed here.
