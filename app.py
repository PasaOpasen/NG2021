

import os, sys

import numpy as np


from algo import AtoB, a_file, b_file, c_file, d_file, read_data, a_name, e_folder
from algo import algo_work2, split, df2XYZ, second_filter


output_folder = e_folder # pictures

f_folder = "./folders/F"




if __name__ == '__main__':

    if a_file is None:
        sys.exit()

    A_base = read_data(a_file)
    
    def check_shape(df):
        return df.shape[0] == 0
    
    def save(df, df_base):
        df_base.to_csv(os.path.join(f_folder, f"{a_name}.csv"), index = False)
        df.to_csv(os.path.join(f_folder, f"{a_name}_filter.csv"), sep=',', index=False)
        os.remove(a_file)
    
    def close(df):
        if check_shape(df):
            save(df, A_base)
            sys.exit()
        
    
    close(A_base)

    A = AtoB(A_base.copy(), a_file, b_file, my_socket = '192.168.220.16')
    
    close(A)
    
    F = second_filter(A, a_file, c_file)
    
    close(F)
    
    
    dfs = split(F)
    
    save(F, A_base)

    for df in dfs:
        sour = df["Src IP Addr"].iloc[0]
        algo_work2(df2XYZ(df), os.path.join(output_folder, f'images_{sour}.png'), sour, start_df = df)
    





















