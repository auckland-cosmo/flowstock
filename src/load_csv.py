import os

import geopandas  # type: ignore
import pandas as pd  # type: ignore
from shapely import wkt  # type: ignore


def load_telco_data(path: os.PathLike) -> pd.DataFrame:
    """
    Load telco data from a csv file to a dataframe

    Ignore some columns that don't look too useful and rename the others
    """

    dtype_dict = {
        # "WKT": str,
        # "region_2018_code": str,
        # "ta_2018_code": str,
        "sa2_2018_code": str,
        "time": str,
        "count": int,
    }

    df = pd.read_csv(
        path, usecols=dtype_dict.keys(), dtype=dtype_dict, parse_dates=["time"]
    )

    df = df.rename(columns={"sa2_2018_code": "region_code"})

    return df


def load_centroid_data_2018(path: os.PathLike) -> pd.DataFrame:
    """
    Load centroid data from a csv file to a GeoDataFrame

    Ignore some columns that don't look too useful and rename the others

    Assumes that the data's centroid coordinates are in the EPSG:2193 frame.
    """

    dtype_dict = {
        "WKT": str,
        "SA22018_V1_00": str,
        "SA22018_V1_NAME": str,
        # "LAND_AREA_SQ_KM": float,
        # "AREA_SQ_KM": float,
        # "EASTING": float,
        # "NORTHING": float,
        # "LATITUDE": float,
        # "LONGITUDE": float,
        # "Shape_X": float,
        # "Shape_Y": float,
    }

    df = pd.read_csv(path, usecols=dtype_dict.keys(), dtype=dtype_dict)

    df = df.rename(
        columns={"SA22018_V1_00": "region_code", "SA22018_V1_NAME": "region_name"}
    )

    df["centroid"] = df.WKT.apply(wkt.loads)
    df.drop("WKT", axis=1, inplace=True)

    df = geopandas.GeoDataFrame(data=df, geometry="centroid", crs="EPSG:2193")

    return df


def load_area_data_2018(path: os.PathLike) -> pd.DataFrame:
    """
    Load area data from a csv file to a dataframe

    Ignore some columns that don't look too useful and rename the others
    """

    dtype_dict = {
        "WKT": str,
        "SA22018_V1_00": str,
        "SA22018_V1_NAME": str,
        # "LAND_AREA_SQ_KM": float,
        "AREA_SQ_KM": float,
        # "Shape_Length": float,
    }

    df = pd.read_csv(path, usecols=dtype_dict.keys(), dtype=dtype_dict)

    df = df.rename(
        columns={
            "SA22018_V1_00": "region_code",
            "SA22018_V1_NAME": "region_name",
            "AREA_SQ_KM": "area",
        }
    )

    return df


def load_area_data_2019(path: os.PathLike) -> pd.DataFrame:
    """
    Load a csv file to a dataframe
    """

    dtype_dict = {
        # "WKT": str,
        "SA22018_V1_00": str,
        "SA22018_V1_00_NAME": str,
        "LAND_AREA_SQ_KM": float,
        "AREA_SQ_KM": float,
        "Shape_Length": float,
    }

    df = pd.read_csv(path, usecols=dtype_dict.keys(), dtype=dtype_dict)

    return df


def test_load_telco_data():

    load_telco_data("../data/telco/pop_data_2020-04-01.dat")


def test_load_area_data_2018():

    load_area_data_2018("../data/area/statistical-area-2-2018-generalised.csv")
