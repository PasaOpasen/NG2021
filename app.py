

import os, sys

import numpy as np

from algo import AtoB, a_file, b_file, c_file, d_file, read_data
from algo import algo_work2, split, df2XYZ, second_filter


output_folder = "./output"

f_folder = "./folders/F"


if __name__ == '__main__':

    A_base = read_data(a_file)
    
    def check_shape(df):
        return df.shape[0] == 0
    
    def save(df):
        df.to_csv(os.path.join(f_folder, 'F.csv'), index = False)
    
    def close(df):
        if check_shape(df):
            save(A_base)
            sys.exit()
        
    
    close(A_base)
    #raise Exception()
    A = AtoB(A_base.copy(), a_file, b_file, my_socket = '192.168.200.9')
    
    close(A)
    
    F = second_filter(A, a_file, c_file)
    
    close(F)
    
    #if np.unique(F['Src IP Addr'].values.size) == F.shape[0]:
    #    save(A_base)
    
    
    dfs = split(F)
    
    if len(dfs) == 0:
        save(A_base)

    for df in dfs:
        #print(df)
        sour = df["Src IP Addr"].iloc[0]
        algo_work2(df2XYZ(df), os.path.join(output_folder, f'images_{sour}.png'), sour, start_df = df)





















