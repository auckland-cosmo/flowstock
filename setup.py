# -*- coding: utf-8 -*-
"""Setup file for movement analysis package

Developer install
-----------------
Run following command in prompt/terminal:
    pip install -e .
"""
import setuptools  # type: ignore

with open("README.rst", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="movement",  # name of package on import
    version="0.0.1",
    description="Estimation of movement patterns from snapshots",
    long_description=long_description,
    long_description_content_type="text/x-rst",
    url="",
    author="Nathan Musoke, ",
    author_email="",
    # license="Apache Licence (2.0)",
    packages=setuptools.find_packages(),
    install_requires=[
        "matplotlib",
        "numpy",
        "scipy",
        "pandas",
        "geopandas",
        "numba",
    ],  # dependencies
    classifiers=[
        "Programming Language :: Python :: 3",
        # "License :: OSI Approved :: Apache Software License",
        "Intended Audience :: Science/Research",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.5",
)
