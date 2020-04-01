from datetime import datetime
import load_csv
import matplotlib.pyplot as plt  # type: ignore
import numpy as np  # type: ignore
import pandas as pd  # type: ignore
import seaborn as sns  # type: ignore


def main():
    area_data = load_csv.load_area_data_2018(
        "../data/area/statistical-area-2-2018-generalised.csv"
    )

    telco_data = load_csv.load_telco_data("../data/telco/pop_data_2020-04-01.dat")

    data = pd.merge(area_data, telco_data, on="region_code")

    data["density"] = data["count"] / data["area"]

    print(data)

    data_subset = data[
        np.logical_or.reduce(
            (
                data["region_name"] == "Auckland-University",
                # data["region_name"] == "Queen Street",
                data["region_name"] == "Balmoral",
                data["region_name"] == "Mount Maunganui North",
            )
        )
    ]

    sns_plot = sns.lineplot(data=data_subset, x="time", y="count", hue="region_name",)
    sns_plot.set_xlim(datetime(2020, 2, 1), datetime(2020, 2, 29))
    sns_plot.get_figure().autofmt_xdate()

    sns_plot.get_figure().savefig("cbd.png")


if __name__ == "__main__":
    main()
