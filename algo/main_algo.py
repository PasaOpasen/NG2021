

import math

import os

import numpy as np
import pandas as pd
import sklearn

import matplotlib.pyplot as plt

from .other import time2number, time2number_iso

from GNG import create_data_graph, convert_images_to_gif, GNG

from plttest import draw_image, create_gng, extract_subgraphs

from .plate import Plate

from .config import  T0, c_file, e_folder




def split(df):
    """
    разбивает таблицу на список таблиц по айпи, отсортированных по времени
    оставляет только таблицы с более чем 2 строками
    и добавляет переменную T2
    """

    F = df.copy()

    F['time'] = [time2number_iso(time) for time in F['Date first seen']]

    F.sort_values(by = ['Src IP Addr', 'time'], inplace = True)

    #F.head()

    Ds = [F[F['Src IP Addr'] == sour] for sour in pd.unique(F['Src IP Addr'])]


    def T_setter(data):
        print(data.shape)
        T = np.empty(data.shape[0])
        for i in range(1, data.shape[0]):
            T[i] = data.loc[i, 'time'] - data.loc[i-1, 'time']

        T[0] = np.mean(T[1:])

        data['T2'] = T0 - T
        return data[data['T2'] > 0]


    #F = pd.concat([T_setter(df) for df in Ds])

    toSee = [T_setter(df.reset_index()) for df in Ds if df.shape[0] > 2]  # здесь отбирает по числу соединений в одной IP

    toSee = [d for d in toSee if d.shape[0] > 2]

    return toSee


def df2XYZ(df):
    """
    достает из таблицы нужные для дальнейшего координаты
    """

    result = pd.DataFrame({
        'X': df['Bytes'],
        'Y': df['T2']
    })

    return result


def algo_work2(df, output_images_dir: str, sour: str, start_df):
    """
    основной алгоритм, запускает нейронный газ, рисует в случае подозрений и т п
    """
    
    data = df.values # sklearn.preprocessing.normalize(df.values, axis=1, norm='l1', copy=False)
    
    data[:,0] = data[:,0].astype(float)
    data[:,1] = data[:,1].astype(float)
    
    print(f"\n\n Source = {sour}  \n\n")


    # сам нейронный газ
    gng = create_gng(max_nodes  =   math.ceil(data.shape[0] / 3)) # здесь число узлов
    gng.train(data, epochs = 50) # число эпох


    xmin = 5*60-15
    ymin = 112


    #pl = Plate(xmin, ymin)

    clusters = extract_subgraphs(gng.graph)
    
    is_out_flag = False

    # проверяем, есть ли хотя бы один узел на неправильной стороне линии
    for cluster in clusters:    
        #f = any((pl.is_out(node.weight[0][1], node.weight[0][0]) for node in cluster))
        f = any((node.weight[0][1] <=xmin and node.weight[0][0]<=ymin for node in cluster))
        if f:
            is_out_flag = True
            break
        
    # если есть, рисуем
    if is_out_flag:
    #if True:

        plt.scatter(data[:,1], data[:,0], label = 'Out points')

        draw_image(gng.graph)
        
        clusters = len(clusters)

        #plt.plot([xmin, 0],[0, ymin], '--', c = 'red')
        plt.plot([xmin, xmin],[ 0, ymin], '--', c = 'red')
        plt.plot([xmin, 0],[ymin, ymin], '--', c='red', label = 'Line')


        plt.xlabel('time')
        plt.ylabel('bytes')

        plt.legend()
        plt.title(f"Source = {sour}, points = {data.shape[0]}, clusters = {clusters}")
        
        def center_range(arr):
            center = (arr.max()+arr.min())/2
            rng = (arr.max()-arr.min())*1.1
            return [center - rng, center + rng]
        
        plt.xlim(center_range(data[:,1]))
        plt.ylim(center_range(data[:,0]))

        plt.savefig(output_images_dir, dpi = 200)
        plt.show()




        start_df.to_csv(os.path.join(e_folder, f"{sour}.csv"), index = False)
    



























def algo_work(df, output_images_dir: str, output_gif: str):

    data = sklearn.preprocessing.normalize(df.values, axis=1, norm='l1', copy=False)
    G = create_data_graph(data)

    gng = GNG(data, surface_graph=G, output_images_dir=output_images_dir)
    gng.train(max_iterations=100, save_step=20)

    print('Saving GIF file...')
    convert_images_to_gif(output_images_dir, output_gif)

    #print('{}\n{}\n{}'.format(frame, 'Applying detector to the normal activity using the training set...', frame))
    
    gng.detect_anomalies(data)


    XYZ = sklearn.preprocessing.normalize(np.array([[5,T0-15]]), axis=1, norm='l1', copy=False)[0]

    pl = Plate(XYZ[0], XYZ[1]) # X Y Z bounds


    nodes = [d['pos'] for d in gng._graph.nodes._nodes.values()]

    nodes = set([i for i, node in enumerate(nodes) if pl.is_out(*node)])

    if len(nodes) > 0:

        cl = [gng._determine_closest_vertice(data[i]) for i in range(data.shape[0])]

        is_anomaly = np.array([c in nodes for c in cl])

        print(f"Found {len(nodes)} bad nodes with {np.sum(is_anomaly)} anomalies")

        return df.loc[is_anomaly,:]

    else:
        print('No anomalies')

        return None


