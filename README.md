# Intensity Prediction Equation from GeoNet Felt RAPID Reports

Predicting Modified Mercalli Intensity (MMI) from crowdsourced earthquake shaking reports, using GeoNet's Felt RAPID Report data for New Zealand.

### Full analysis: [notebooks/](notebooks/)

The four notebooks run end to end from the public GeoNet APIs. Nothing is read from a pre-prepared file.

## The problem

Intensity prediction equations estimate how strongly an earthquake is felt at a given location, which supports rapid post-event assessment and earthquake risk modelling. New Zealand's official equations rely on rupture plane geometry that is not available immediately after an event.

GeoNet's Felt RAPID Report survey asks the public which of six cartoons best matches what they experienced, mapping to MMI 3 through 8. This project asks whether that crowdsourced data alone can produce a usable and physically sensible intensity prediction equation.

## Approach

- 99 earthquakes and 609,476 felt reports pulled from the public GeoNet APIs, magnitude 4.0 to 8.0, October 2016 to December 2025.
- Reports aggregated to a 1 km grid built in NZTM2000 rather than in degrees, so cells are genuinely 1 km on a side. 24,241 cells survive a minimum of 5 reports each.
- Rather than collapsing each cell to a single intensity, every reported level is kept as a weighted training row. This avoids choosing between mean, median and mode, avoids tie-breaking, and keeps every label a real integer report.
- Features span three groups: physical (magnitude, depth, log hypocentral distance), directional (azimuth), and reporting behaviour (local time of day, weekend). Vs30 site conditions are optional.
- Splits are by earthquake rather than by row, and stratified by magnitude. Most features are properties of an event, so 24,241 cells carry only 95 independent observations of them.
- 44 models compared on identical terms, then screened for physical plausibility.

## The interesting part

A model can score well on held out data and still be wrong in a way no metric catches. Shaking must weaken with distance from an earthquake, but nothing in a loss function says so.

Holding an earthquake fixed and sweeping distance outwards shows that only 2 of the 14 most accurate models produce physically possible attenuation. The most accurate model of all is among the failures: its predictions do not fall reliably with distance, so its shake maps would mislead.

Selecting on physics first costs about 10% of accuracy and buys a model whose output can be acted on.

![Shake maps for all fourteen candidates](maps/contact-sheet.png)

Green passed, red was rejected. Two of these fourteen are usable.

## Results

Selected model: Gradient Boosting Regressor.

| | Selected | Most accurate | Attenuation equation | Constant |
|---|---:|---:|---:|---:|
| Cell MAE | 0.369 | 0.337 | 0.545 | 0.590 |
| Worst Spearman | -0.980 | -0.899 | | |
| Physically plausible | yes | no | yes | no |

Three findings beyond the ranking.

### The numeric check had a hole in it, and the maps found it

The screening described above replaced the eyeball check the original version of this project used, because looking at maps does not scale past a handful of candidates and cannot be repeated by anyone else. That was the right move, and it was not the end of it.

The check swept distance from 5 km to 400 km. New Zealand is 900 km long. Drawing the maps for every candidate showed a model that had passed cleanly, at a rank correlation of -0.987, predicting the top of the intensity scale across Northland for a Kaikoura earthquake 780 km away, in the part of the map the check never looked at. It had been cited as evidence that constraining tree depth improves physical behaviour. It was scoring well on the half of the range that was being examined.

Two changes followed. The sweep now covers the full length of the country, and a second check looks at each compass bearing separately rather than averaging them, because averaging hides a model that behaves in twenty-three directions and predicts MMI 8 in the twenty-fourth. The pass count fell from 4 to 2. The selected model did not change.

The numbers are the test. The pictures are how you find out the test has a hole in it.

### The usual headline metric is broken

Intensity work normally quotes the percentage of predictions within one MMI unit. Against continuous predictions it measures the wrong thing.

MMI 3, 4 and 5 are 90.6% of all reports, so predicting exactly 4.0 sits within one unit of almost everything. Moving that prediction to 4.2 costs 9% in mean absolute error and 34% in within-1, because 4.2 now falls more than a unit from MMI 3. The metric collapses the moment a prediction stops being an integer, so it measures integer-ness rather than closeness.

That is enough to reverse the ranking between a real model and none. Scored the way the literature normally scores it, the selected model reaches 0.739 against 0.876 for a constant 4.0, even though its error is genuinely lower (report MAE 0.719 against 0.800). Mean absolute error and the Ranked Probability Score are used instead, with within-1 kept on rounded predictions where it is at least comparable.

### No model can name damaging shaking, but the leaders can rank it

MMI 7 and above is 1.3% of reports, so the most likely single level is essentially never 7 or 8 and per-class recall reads as zero for every candidate. Measured as a ranking problem instead, the leading models reach ROC AUC around 0.79 to 0.80 for identifying MMI 6 and above. They rank damaging cells higher; they just cannot name them.

The contact sheet above shows the same thing from another angle: most panels stay pale even at the epicentre of a magnitude 7.8 earthquake, because most of these models never predict strong shaking anywhere. Precision is capped by the base rate rather than by ranking quality, which is the clearest argument in this project for bringing instrumental measurements alongside felt reports.

## Notebooks

| # | Notebook | What it covers |
|---|---|---|
| 1 | [01_data_acquisition.ipynb](notebooks/01_data_acquisition.ipynb) | Rebuilding the dataset from the GeoNet APIs, and the data quality problems found |
| 2 | [02_exploratory_analysis.ipynb](notebooks/02_exploratory_analysis.ipynb) | Gridding, the target decision, and what the data does and does not contain |
| 3 | [03_model_comparison.ipynb](notebooks/03_model_comparison.ipynb) | 44 models on identical terms, and why the standard metric had to be replaced |
| 4 | [04_physical_validation.ipynb](notebooks/04_physical_validation.ipynb) | Screening for physically possible attenuation, and the final selection |

Start with [04](notebooks/04_physical_validation.ipynb) if you only read one.

## Maps

`python src/maps.py` regenerates every shake map into [maps/](maps/):

- [contact-sheet.png](maps/contact-sheet.png), all fourteen candidates on one page
- [selected-vs-rejected.png](maps/selected-vs-rejected.png), the two that matter side by side
- [maps/models/](maps/models/), one map per candidate in accuracy order

All are drawn for the same earthquake on the same fixed MMI scale, so the panels can be compared directly. Predictions are masked to land using a simplified Natural Earth coastline committed under [assets/](assets/), so the maps render offline like everything else here.

## Data quality problems worth noting

Five were found and handled, each documented in the notebooks:

- Teleseismic events. The GeoNet catalogue lists earthquakes New Zealand instruments detected, not New Zealand earthquakes. Filtering removes 72% of raw entries, including events in Chile and Japan.
- Duplicate magnitude solutions. Several catalogue entries for one physical earthquake.
- Null Island. 34 reporting locations at exactly longitude 0.005, latitude 0.003, carrying 469 reports. Failed geolocation defaulting to the coordinate system origin.
- Out-of-scale intensities. The survey produces MMI 3 to 8, but the archive holds a handful of MMI 1 and 2 reports.
- Event misattribution during aftershock sequences. Magnitude 4 to 5 events appear to produce rising intensity with distance, which is impossible. The far-field cells trace to two November 2016 Kaikoura aftershocks, where people could not tell which shake they were reporting.

## Repository layout

```
src/         pipeline modules: ingest, clean, features, models, bench, spatial, maps
notebooks/   the analysis, four notebooks in order
tests/       107 tests, entirely offline
maps/        rendered shake maps, one per candidate
assets/      simplified NZ coastline, for masking maps to land
```

## Running it

```bash
pip install -r requirements.txt
pytest tests/
jupyter notebook notebooks/
```

Run the notebooks in order. Notebook 01 fetches from the GeoNet APIs and caches responses under `data/raw`, so later runs do not hit the API again. No data is committed to this repository.

Vs30 site condition data is optional and not redistributed here; the pipeline runs without it.

## Tests

107 tests, run on Python 3.11, 3.12 and 3.13 by GitHub Actions, with a second pass promoting deprecation warnings to errors. The suite is entirely offline: API responses are stubbed and the geospatial and model checks use synthetic data, so a GeoNet outage cannot turn the build red for reasons unrelated to the code.

Six of those tests exist because of the far-field bug above. One of them builds a model that attenuates correctly to 400 km and then jumps to MMI 8, and asserts that the old truncated sweep would have accepted it.

## Context

This is a full rebuild, written independently, of a problem first worked on as a DATA601 research project at the University of Canterbury with Zheng Chao, supervised by Hazel Fraser and Goldie Leung. The pipeline, the analysis and the results here are my own, and differ substantially from that original work.

## Data source

[GeoNet Felt RAPID Reports](https://api.geonet.org.nz/) and the [GeoNet quake search API](https://quakesearch.geonet.org.nz/), both publicly available. Coastline from [Natural Earth](https://www.naturalearthdata.com/), public domain.
