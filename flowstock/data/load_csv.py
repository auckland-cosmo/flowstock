import os

import geopandas  # type: ignore
import pandas as pd  # type: ignore
from shapely import wkt  # type: ignore


def load_telco_data(path: os.PathLike) -> pd.DataFrame:
    """
    Load telco data from a csv file to a dataframe

    Rename columns to be consistent with other data sources
    """

    dtype_dict = {
        "region_2018_code": str,
        "ta_2018_code": str,
        "sa2_2018_code": str,
        "time": str,
        "count": int,
    }

    df = pd.read_csv(
        path, usecols=dtype_dict.keys(), dtype=dtype_dict, parse_dates=["time"]
    )

    df = df.rename(
        columns={
            "region_2018_code": "rc_code",
            "ta_2018_code": "ta_code",
            "sa2_2018_code": "sa2_code",
        }
    )

    return df


def load_centroid_data_2018(path: os.PathLike) -> pd.DataFrame:
    """
    Load centroid data from a csv file to a GeoDataFrame

    Ignore some columns that don't look too useful and rename the others

    Assumes that the data's centroid coordinates are in the EPSG:2193 frame.
    """

    rename_dict = {"SA22018_V1_00": "sa2_code", "SA22018_V1_NAME": "sa2_name"}

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

    df = df.rename(columns=rename_dict)

    df["centroid"] = df.WKT.apply(wkt.loads)
    df.drop("WKT", axis=1, inplace=True)

    df = geopandas.GeoDataFrame(data=df, geometry="centroid", crs="EPSG:2193")

    return df


def load_hierarchy_data_2018(path: os.PathLike) -> pd.DataFrame:
    """
    Load hierarchical data from a csv file to a GeoDataFrame

    Rename columns to be consistent with other data.
    """

    rename_dict = {
        "SA22018_V1_00": "sa2_code",
        "SA22018_V1_00_NAME": "sa2_name",
        "REGC2018_V1_00": "rc_code",
        "REGC2018_V1_00_NAME": "rc_name",
        "TA2018_V1_00": "ta_code",
        "TA2018_V1_00_NAME": "ta_name",
    }

    dtype_dict = {
        "SA22018_V1_00": str,
        "SA22018_V1_00_NAME": str,
        "REGC2018_V1_00": str,
        "REGC2018_V1_00_NAME": str,
        "TA2018_V1_00": str,
        "TA2018_V1_00_NAME": str,
    }

    df = pd.read_csv(path, usecols=rename_dict.keys(), dtype=dtype_dict)

    df = df.rename(columns=rename_dict)

    return df


def load_area_data_2018(path: os.PathLike) -> pd.DataFrame:
    """
    Load area data from a csv file to a dataframe

    Ignore some columns that don't look too useful and rename the others

    https://datafinder.stats.govt.nz/layer/92212-statistical-area-2-2018-generalised/data/
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
            "SA22018_V1_00": "sa2_code",
            "SA22018_V1_NAME": "sa2_name",
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
