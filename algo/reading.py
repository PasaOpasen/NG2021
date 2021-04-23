
import numpy as np
import pandas as pd

from filter import simple_filter
from other import time2number


def read_data(file):

    data = pd.read_csv(file, sep=';')

    return data


def AtoB(a_file = './A/A.csv', b_file = './B/B.csv', my_socket = None):
    
    A = read_data(a_file)
    A = simple_filter(A)
    if os.path.exists(b_file):
        B = read_data(b_file)
        A = pd.concat([A, B], ignore_index=True)

    # можно и из А убрать те 5 минут
    print(A.columns)
    times = np.array([time2number(time) for time in A['Time '].values])
    time_mask = times > (times.max() - 5*60)

    A.loc[time_mask,:].to_csv(b_file, index=False,  sep=';')

    return A



