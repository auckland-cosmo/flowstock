from flowstock.data.load_csv import (
    load_area_data_2018,
    load_centroid_data_2018,
    load_hierarchy_data_2018,
    load_telco_data,
)


def test_load_telco_data():

    load_telco_data("data/telco/pop_data_2020-04-01.dat")


def test_load_centroid_data_2018():

    load_centroid_data_2018("data/area/statistical-area-2-2018-centroid-true.csv")


def test_load_hierarchy_data_2018():

    load_hierarchy_data_2018(
        "data/area/statistical-area-2-higher-geographies-2018-generalised.csv"
    )


def test_load_area_data_2018():

    load_area_data_2018("data/area/statistical-area-2-2018-generalised.csv")
