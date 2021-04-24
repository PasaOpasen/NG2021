
import numpy as np
import pandas as pd
import sklearn

from .other import time2number, time2number_iso

from GNG import create_data_graph, convert_images_to_gif, GNG

from .plate import Plate

from .config import  T0, c_file




def split(df):

    F = df.copy()

    F['time'] = [time2number_iso(time) for time in F['Date first seen']]

    F.sort_values(by = ['Src IP Addr', 'time'], inplace = True)

    F.head()



    Ds = [F[F['Src IP Addr'] == sour] for sour in pd.unique(F['Src IP Addr'])]


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

    result = pd.DataFrame({
        'X': df['Packets'],
        'Y': df['Bytes'],
        'Z': df['T2']
    })

    return result



def algo_work(df, output_images_dir: str, output_gif: str):

    data = sklearn.preprocessing.normalize(df.values, axis=1, norm='l1', copy=False)
    G = create_data_graph(data)

    gng = GNG(data, surface_graph=G, output_images_dir=output_images_dir)
    gng.train(max_iterations=100, save_step=20)

    print('Saving GIF file...')
    convert_images_to_gif(output_images_dir, output_gif)

    #print('{}\n{}\n{}'.format(frame, 'Applying detector to the normal activity using the training set...', frame))
    
    gng.detect_anomalies(data)


    XYZ = sklearn.preprocessing.normalize(np.array([[5,5,T0-15]]), axis=1, norm='l1', copy=False)[0]

    pl = Plate(XYZ[0], XYZ[1], XYZ[2]) # X Y Z bounds


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


