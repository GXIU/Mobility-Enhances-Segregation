import numpy as np
import pandas as pd


def weighted_corr(x, y, w):
    """
    Weighted correlation coefficient
    """
    return np.cov(x, y, aweights=w)[0, 1] / np.sqrt(
        np.cov(x, aweights=w) * np.cov(y, aweights=w)
    )


def weighted_median(data, weights):
    """
    Compute the weighted median of data with weights.
    """
    # Combine the data and weights and sort by data
    combined = pd.DataFrame({"data": data, "weights": weights}).sort_values("data")

    # Calculate the cumulative sum of weights and the total sum
    combined["cumulative_weights"] = combined["weights"].cumsum()
    total_weight = combined["weights"].sum()

    # Find the data value where the cumulative weight is equal to or just exceeds half the total weight
    median_row = combined[combined["cumulative_weights"] >= total_weight / 2].iloc[0]
    return median_row["data"]


def weighted_quantile(data, weights, quantile):
    """
    Compute the weighted quantile of data with weights.
    """
    # Combine the data and weights and sort by data
    combined = pd.DataFrame({"data": data, "weights": weights}).sort_values("data")

    # Calculate the cumulative sum of weights and the total sum
    combined["cumulative_weights"] = combined["weights"].cumsum()
    total_weight = combined["weights"].sum()

    # Find the data value where the cumulative weight is equal to or just exceeds half the total weight
    median_row = combined[
        combined["cumulative_weights"] >= total_weight * quantile
    ].iloc[0]
    return median_row["data"]


# ----by mean----------------------------


def segregation(flow_matrix, poi_density, how="mean", quantile=0.8):
    if how == "mean":
        destination_poi_density_weighted_by_flow = flow_matrix.dot(
            poi_density
        ) / flow_matrix.sum(axis=1)
    elif how == "median":
        destination_poi_density_weighted_by_flow = flow_matrix.apply(
            lambda row: weighted_median(poi_density, row), axis=1
        )
    elif how == "quantile":
        destination_poi_density_weighted_by_flow = flow_matrix.apply(
            lambda row: weighted_quantile(poi_density, row, quantile), axis=1
        )
    segregation = pd.Series(
        [
            weighted_corr(poi_density, destination_poi_density_weighted_by_flow, w_i)
            for w_i in flow_matrix.values
        ],
        index=poi_density.index,
    )
    return segregation


# N = 100

# poi_density = pd.Series(np.random.random(N), index=range(N))
# flow_matrix = pd.DataFrame(np.random.random((N, N)), index=range(N), columns=range(N))
