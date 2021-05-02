


#
# некоторые переменные конфигурации
#

import os
import glob


files = glob.glob('./folders/A/*.csv')

a_name = os.path.basename(files[0]).split('.')[0] if files else None


a_file = f'./folders/A/{a_name}.csv' if not (a_name is None) else None
b_file = f'./folders/B/B.csv'
c_file = './folders/F/F.csv'
d_file = './folders/D/D.csv'

e_folder = './folders/E'


T0 = 5*60


