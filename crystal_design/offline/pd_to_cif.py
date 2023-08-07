import pandas as pd
from tqdm import tqdm

data = pd.read_csv('/home/mila/p/prashant.govindarajan/scratch/crystal_design_project/crystal-design/crystal_design/data/mp_20/train.csv')

for i in tqdm(range(len(data))):
    cif_string = data.loc[i]['cif']
    with open('mp_20_cifs/'+str(i)+'.cif', 'w') as f:
        f.write(cif_string)