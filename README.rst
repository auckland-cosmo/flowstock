flowstock
#########

.. image:: https://badge.fury.io/py/flowstock.svg
    :target: https://badge.fury.io/py/flowstock

Estimate flows of people from snapshots using the method of [Akagi2018]_.

There are two parts to this package

- a fake data generator
- an estimator

A couple IPython notebooks are included as demonstrations.
One of these generates fake data and attempts to recover the underlying parameters.
It is most useful for demonstration and testing.

Another is designed to run on real world data.
It requires the user to supply additional data: counts of people in New Zealand SA2 regions and properties of those regions.
The later information is available from StatsNZ:

- `SA2 locations`_
- `Centroid locations`_
- `Higher geographies`_

.. _`SA2 locations`: https://datafinder.stats.govt.nz/layer/92212-statistical-area-2-2018-generalised/data/
.. _`Centroid locations`: https://datafinder.stats.govt.nz/layer/93620-statistical-area-2-2018-centroid-true/
.. _`Higher geographies`: https://datafinder.stats.govt.nz/layer/95065-statistical-area-2-higher-geographies-2018-generalised/data/

We have assumed that the centroid-to-centroid distance separating regions is an appropriate metric.
In principle, the user can specify an arbitrary distance measure on the space, allowing distances to be set by actual driving time, non-local travel via airports, or even asymmetry during commutes.

There are still some rough edges and there may be more corner cases to be taken care of.

Installation
============

pip
---

The package is available on pypi.org_.

.. code-block:: bash

    pip install flowstock
    pytest

To run tests during the install process,

.. code-block:: bash

    pip install --install-option test flowstock


source
------

A conda environment specification is included.
This environment has been tested on Linux and OS X.

.. code-block:: bash

    conda env create -f environment.yml
    conda activate flowstock
    pip install -e .
    pytest

.. _pypi.org: https://pypi.org/project/layg/

.. [Akagi2018] Y. Akagi, T. Nishimura, T. Kurashima, and H. Toda, “A Fast and Accurate Method for Estimating People Flow from Spatiotemporal Population Data,” in Proceedings of the Twenty-Seventh International Joint Conference on Artificial Intelligence, Stockholm, Sweden, Jul. 2018, pp. 3293–3300, doi: 10.24963/ijcai.2018/457.
