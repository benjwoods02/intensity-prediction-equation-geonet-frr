# assets

`nz_coastline.json` is New Zealand's coastline as longitude/latitude rings,
used by `src/maps.py` to mask predictions to land.

It was derived once from Natural Earth's 1:10m physical land layer, clipped to
a box around New Zealand and simplified to a 0.01 degree tolerance. That
reduces it to 20 polygons and about 1,500 points, roughly 31 KB, which is small
enough to commit. The alternative was a runtime dependency on cartopy plus a
download on first use, in a project whose test suite and pipeline otherwise run
entirely offline.

The 1:10m layer is used rather than the coarser 1:50m one because at 1:50m the
Auckland isthmus is narrow enough to be simplified away, which puts the
country's largest city in the sea on every shake map. `tests/test_maps.py`
checks a list of cities against the mask so that cannot regress silently.

Natural Earth data is in the public domain. See naturalearthdata.com.

The Chatham Islands are excluded. They sit across the antimeridian and are
outside the NZTM2000 grid the models are mapped on.
