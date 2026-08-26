# assets

`nz_coastline.json` is New Zealand's coastline as longitude/latitude rings,
used by `src/maps.py` to mask predictions to land.

It was derived once from Natural Earth's 1:50m physical land layer, clipped to
a box around New Zealand and simplified to a 0.01 degree tolerance. That
reduces it to 7 polygons and 522 points, about 11 KB, which is small enough to
commit. The alternative was a runtime dependency on cartopy plus a download on
first use, in a project whose test suite and pipeline otherwise run offline.

Natural Earth data is in the public domain. See naturalearthdata.com.

The Chatham Islands are excluded. They sit across the antimeridian and are
outside the NZTM2000 grid the models are mapped on.
