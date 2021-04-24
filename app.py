
import os

from algo import AtoB, a_file, b_file, c_file, d_file
from algo import algo_work, split, df2XYZ, second_filter


output_folder = "./output"



if __name__ == '__main__':

    A = AtoB(a_file, b_file)
    
    F = second_filter(A, a_file, c_file)
    
    dfs = split(F)

    for df in dfs:
        sour = df["Src IP Addr"][0]
        algo_work(df2XYZ(df), os.path.join(output_folder, f'images_{sour}'), os.path.join(output_folder,f'{sour}.gif'))





















