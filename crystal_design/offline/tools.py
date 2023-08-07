import pandas as pd
import os
from tqdm import tqdm 
data = pd.read_csv('/home/mila/p/prashant.govindarajan/scratch/crystal_design_project/crystal-design/crystal_design/data/mp_20/train.csv')
files = os.listdir('new_calculations')
file_out = []
for file in files:
    if '.pwo' in file:
        file_out.append(file)
c = 0
metals = []
metals_yet = []
n = data.shape[0]
for i in tqdm(range(n)):
    bg = data.loc[i]['band_gap']
    if bg == 0.0:
        try:
            with open('new_calculations/espresso_'+str(i)+'.pwo', 'r') as f:
                if 'total energy' in f.read():
                    c+=1
                    metals.append(i)
                else:
                    metals_yet.append(i)
        except:
            metals_yet.append(i)


import pandas as pd
import os
from tqdm import tqdm 
import re
import shutil
data = pd.read_csv('/home/mila/p/prashant.govindarajan/scratch/crystal_design_project/crystal-design/crystal_design/data/mp_20/train.csv')
files = os.listdir('bands_tmp')
for file in files:
    if '.pwo' in tqdm(file):
        with open('bands_tmp/'+file, 'r') as f:
            i = int(re.findall(r'\d+', file)[0])
            if data.loc[i]['band_gap'] > 0.0:
                if 'highest' in f.read():
                    shutil.copy('bands_tmp/'+file, 'bands_nm_valid/'+file) 


import pandas as pd
import os
from tqdm import tqdm 
import re
import shutil
data = pd.read_csv('/home/mila/p/prashant.govindarajan/scratch/crystal_design_project/crystal-design/crystal_design/data/mp_20/train.csv')
files = os.listdir('new_calculations')
for file in files:
    if '.pwo' in tqdm(file):
        with open('new_calculations/'+file, 'r') as f:
            i = int(re.findall(r'\d+', file)[0])
            if data.loc[i]['band_gap'] == 0.0:
                if 'total energy' in f.read():
                    shutil.copy('new_calculations/'+file, 'energies_m_valid/'+file) 


