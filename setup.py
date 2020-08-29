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
    name="flowstock",  # name of package on import
    version="0.0.1-dev",
    description="Estimation of movements from snapshots",
    long_description=long_description,
    long_description_content_type="text/x-rst",
    url="",
    author="Mateja Gosenca, Lillian Guo, Emily Kendall,\
            Lerh Feng Low, Nathan Musoke, Richard Easther",
    author_email="n.musoke@auckland.ac.nz",
    # license="Apache Licence (2.0)",
    packages=setuptools.find_packages(),
    install_requires=[
        "matplotlib",
        "numpy",
        "scipy",
        "pandas",
        "pytest",
        "geopandas",
        "numba",
    ],  # dependencies
    setup_requires=["pytest-runner"],
    tests_require=["pytest"],
    classifiers=[
        "Programming Language :: Python :: 3",
        # "License :: OSI Approved :: Apache Software License",
        "Intended Audience :: Science/Research",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.5",
)
