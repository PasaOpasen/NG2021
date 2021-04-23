
import os
import socket

import numpy as np
import pandas as pd


def simple_filter(df, my_socket = None):
    
    my_socket = socket.gethostbyname(socket.gethostname()) if my_socket is None else my_socket

    data = df[df['Sour IP'] != socket]

    #data = data[data['Dest IP'] == socket] 

    return data

def second_filter(F,a_file, c_file):
    mask_Numb = F['Numb P'] < 5
    if np.logical_not(mask_Numb).sum() == F.shape[0]:
        F.to_csv(c_file, sep=';', index=False)
        os.remove(a_file)
    else:
        F = F[mask_Numb]

    F.head()


