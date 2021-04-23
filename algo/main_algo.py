
import numpy as np
import pandas as pd

from other import time2number

def split(F):

    F['time'] = [time2number(time) for time in F['Time ']]

    F.sort_values(by = ['Sour IP', 'time'], inplace = True)

    F.head()



    Ds = [F[F['Sour IP'] == sour] for sour in pd.unique(F['Sour IP'])]


    def T_setter(data):
        print(data.shape)
        T = np.empty(data.shape[0])
        for i in range(1, data.shape[0]):
            T[i] = data.loc[i, 'time'] - data.loc[i-1, 'time']

        T[0] = np.mean(T[1:])

        data['T2'] = T0 - T
        return data


    #F = pd.concat([T_setter(df) for df in Ds])

    toSee = [T_setter(df.reset_index()) for df in Ds if df.shape[0] > 1]

    return toSee

