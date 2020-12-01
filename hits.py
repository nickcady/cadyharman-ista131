import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import csv, os, sqlite3, math
import datetime
import math

def get_H_series():
    df = pd.read_csv("teamdata.csv", index_col=0)
    return df[["H", "R"]]

def make_image(df):
    x = df["H"]
    y = df["R"]
    X = sm.add_constant(x)
    model = sm.OLS(y.values, X)
    results = model.fit()
    Y = (results.params["H"] * x) + results.params["const"]
    plt.scatter(x,y)
    plt.plot(x, Y)
    plt.title("Hits")
    plt.xlabel("Hits")
    plt.ylabel("Runs")
    plt.show()

def main():
    df = get_H_series()
    make_image(df)

main()