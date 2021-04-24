
import numpy as np
import pandas as pd

from other import time2number

from GNG import create_data_graph, convert_images_to_gif, GNG

from plate import Plate

from config import  T0, c_file

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


def df2XYZ(df):
    pass



def algo_work(df, output_images_dir: str, output_gif: str):

    data = preprocessing.normalize(df.values, axis=1, norm='l1', copy=False)
    G = create_data_graph(data)

    gng = alg(data, surface_graph=G, output_images_dir=output_images_dir)
    gng.train(max_iterations=max_iters, save_step=20)

    print('Saving GIF file...')
    convert_images_to_gif(output_images_dir, output_gif)

    print('{}\n{}\n{}'.format(frame, 'Applying detector to the normal activity using the training set...', frame))
    
    gng.detect_anomalies(data)


    pl = Plate(5,1,T0 - 0)

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


