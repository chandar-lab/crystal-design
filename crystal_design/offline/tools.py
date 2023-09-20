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



import os 
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm 
import re
all_true = []
all_pred = []
data = pd.read_csv('/home/mila/p/prashant.govindarajan/scratch/crystal_design_project/crystal-design/crystal_design/data/mp_20/train.csv')
files = os.listdir('/home/mila/p/prashant.govindarajan/scratch/crystal_design_project/crystal-design/crystal_design/offline/matgl_pred_bg_mp_train')
for file in tqdm(files):
    try:
        ind = int(re.findall(r'\d+', file)[0])
        true_bg = data.loc[ind]['band_gap']
        with open('matgl_pred_bg_mp_train/'+file, 'r') as f:
            lines = f.read()
            pred_bg =  float(lines.split()[-2])
        all_pred.append(pred_bg)
        all_true.append(true_bg)
    except:
        pass

plt.figure(figsize=(10,8))
plt.rcParams['font.size'] = 14
plt.scatter(all_true, all_pred)
xpoints = ypoints = plt.xlim()
plt.plot(xpoints, ypoints, linestyle='--', color='k', lw=3, scalex=False, scaley=False, label = '45 degree line')
plt.legend(loc='best')
plt.xlabel('True Band Gap (MP)')
plt.ylabel('MEGNet predicted band gap')
plt.title('Scatter of plot of band gap prediction using MEGNet (ONLY NONMETALS)')
plt.savefig('plots/MEGNetbg.png', dpi = 400)

import pickle 
dft_bg = []
true_bg_2 = []
nm_ = pickle.load(open('nm_dict.pkl', 'rb'))
for i in tqdm(nm_):
    true_bg_2.append(data.loc[i]['band_gap'])
    dft_bg.append(nm_[i][0])


plt.figure(figsize=(10,8))
plt.rcParams['font.size'] = 14
plt.scatter(true_bg_2, dft_bg)
xpoints = ypoints = plt.xlim()
plt.plot(xpoints, ypoints, linestyle='--', color='k', lw=3, scalex=False, scaley=False, label = '45 degree line')
plt.legend(loc='best')
plt.xlabel('True Band Gap (MP)')
plt.ylabel('ESPRESSO predicted band gap')
plt.title('Scatter of plot of band gap prediction using ESPRESSO (ONLY NONMETALS)')
plt.savefig('plots/ESPRESSObg.png', dpi = 400)

