import pandas as pd
import numpy as np


class Data:

    def __init__(self, config):
        self.config = config

    def loadfrom_csv(self, datafile):
        try:
            return pd.read_csv(datafile)
        except Exception as e:
            print("Error : " + str(e))

    def loadfrom_excel(self, datafile):
        try:
            return pd.read_excel(datafile)
        except Exception as e:
            print("Error : " + str(e))

    def loadfrom_df(self, df):
        print(df)

    def transform(self, data, trafo, power=1.):
        tf = trafo.split('-')
        if tf[0] == 'NA':  # only power transform
            return data
        elif tf[0] == 'pow':  # log_10 transform
            return data ** power
        elif tf[0] == 'log':  # log_10 transform
            return np.log10(data) ** power
        elif tf[0] == 'd1':  # first difference over period dx
            i = int(tf[1])
            return (data[i:] - data[:-i]) ** power
        elif tf[0] == 'pch':  # percentage change over period px
            i = int(tf[1])
            return (100. * (data[i:] - data[:-i]) / data[:-i]) ** power
        elif tf[0] == 'ld':  # log difference (approx pch for small changes)
            i = int(tf[1])
            return (100 * np.log(data[i:] / data[:-i])) ** power
        else:
            raise ValueError("Invalid transformation value.")
