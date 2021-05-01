
import os
import socket

import numpy as np
import pandas as pd


def simple_filter(df, my_socket = None):
    """
    базовая фильтрация
    """

    my_socket = socket.gethostbyname(socket.gethostname()) if my_socket is None else my_socket

    data = df[df['Src IP Addr'] != my_socket]

    data = data[data['Proto'].str.rstrip() == 'TCP']

    data = data[data['Dst IP Addr'] == my_socket] 

    return data

def second_filter(F, a_file, c_file):
    """
    дополнительная фильтрация
    """

    #mask_Numb = np.ones(F.shape[0], dtype = np.bool)
    mask_Numb = F['Packets'] <=3

    symb = {'P', 'F'}

    flags_mask = np.array([not symb.intersection(set(str(string))) for string in F['Flags']])

    both_masks = np.logical_and(mask_Numb.values, flags_mask)

    if np.logical_not(both_masks).sum() == F.shape[0]:
        F.to_csv(c_file, sep=',', index=False)
        os.remove(a_file)
    else:
        F = F.iloc[both_masks, :]

    return F


