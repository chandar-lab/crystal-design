import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pymatgen
from tqdm import tqdm
import pickle
from ase.io import read, write
from ase.calculators.espresso import Espresso
import mendeleev
from time import time
# from mendeleev.fetch import fetch_table
import os
import re
from p_tqdm import p_map
import argparse
pseudo_dir = '/home/pragov/projects/rrg-bengioy-ad/pragov/crystal_structure_design/SSSP'
# cifs = os.listdir('cifs_train')
outputs_dir = 'calculations_bands'
def calc_energy(file_path, pseudodict, i, start, pwo_path):
    # input_data = {'prefix':"myprefix",'electron_maxstep':100,'outdir':"calculations_bands/",'pseudo_dir': pseudo_dir, 'tstress':True,'tprnfor':True,'calculation':'bands', 
    #             'ecutrho':240,'verbosity':'high','ecutwfc':30, 'diagonalization': 'david', 'occupations':'smearing','fixed':'mp', 'mixing_mode':'plain', 
    #             'mixing_beta':0.7,'degauss':0.001, 'nspin':1, 'nstep': 1}
    input_data = {'prefix':"myprefix",'electron_maxstep':1000,'outdir':pwo_path,'pseudo_dir': pseudo_dir, 'tstress':False,'tprnfor':False,'calculation':'scf', 
                'ecutrho':240,'verbosity':'high','ecutwfc':30, 'diagonalization': 'david', 'occupations':'fixed','smearing':'gaussian', 'mixing_mode':'plain', 
                'mixing_beta':0.7,'degauss':0.001, 'nspin':1, 'nstep': 1, 'ntyp': 1, 'nbnd': 64 } ### Changed
    ase_obj = read(file_path)
    breakpoint()
    ase_obj.calc=Espresso(pseudopotentials=pseudodict,input_data=input_data, kpts=(3,3,3), label=pwo_path + '/espresso_'+str(i + start))
    # pseudo_dir = '/home/pragov/projects/rrg-bengioy-ad/pragov/crystal_structure_design/SSSP'
    # input_data = {'prefix':"myprefix",'electron_maxstep':200,'outdir':"./",'pseudo_dir': pseudo_dir, 'tstress':True,'tprnfor':True,'calculation':'scf', 
    #         'ecutrho':240,'verbosity':'high','ecutwfc':30, 'diagonalization': 'david', 'occupations':'smearing','smearing':'mp', 'mixing_mode':'plain', 
    #         'mixing_beta':0.7,'degauss':0.001, 'nspin':1, 'nstep': 0}
    # ase_obj = read(file_path)
    # ase_obj.calc=Espresso(pseudopotentials=pseudodict,input_data=input_data, kpts=(2,2,2))
    try: 
        energy = ase_obj.get_total_energy()
    except:
        return None
    return None 

if __name__ == '__main__':
    print('Starting')
    parser = argparse.ArgumentParser(description='Energy Calculation using Espresso')
    parser.add_argument('--save_path', default = '/home/pragov/projects/rrg-bengioy-ad/pragov/crystal_structure_design/crystal-design/crystal_design/data/energy_freq.pkl', type = str, help = 'save_path')
    parser.add_argument('--start', type = int, default = 0)
    parser.add_argument('--end', type = int, default = 1000)
    parser.add_argument('--datatype', type = str, default = 'train')
    parser.add_argument('--pwo_path', default = 'calculations_bands')

    args = parser.parse_args()
    save_path = args.save_path
    start = args.start
    end = args.end
    data_type = args.datatype
    pwo_path = args.pwo_path
    print(data_type)
    print('Loading Chunk ', start // 1000)
    # data = pd.read_pickle('/home/pragov/projects/rrg-bengioy-ad/pragov/crystal_structure_design/crystal-design/crystal_design/data/materials_project.pkl')
    data = pd.read_csv('/home/pragov/projects/rrg-bengioy-ad/pragov/crystal_structure_design/crystal-design/crystal_design/data/'+data_type+'.csv')
    data = data.loc[start:end].reset_index()
    # data = data.loc[start]#.reset_index()
    print('Loaded Data')
    n = len(data)
    tmp_dir = '/home/pragov/projects/rrg-bengioy-ad/pragov/crystal_structure_design/crystal-design/crystal_design/utils/tmp'
    # tmp_dir = '/home/pragov/projects/rrg-bengioy-ad/pragov/crystal_structure_design/crystal-design/crystal_design/utils/cifs_train'
    # n = 1000 #len(outputs)
    # pseudo_dir = '/home/pragov/projects/rrg-bengioy-ad/pragov/crystal_structure_design/SSSP'
    # elements_df = fetch_table('elements')
    # atom_list = [elements_df.loc[i]['symbol'] for i in range(len(elements_df))]
    # sssp = os.listdir(pseudo_dir)
    # pseudodict = {}
    # for ele in atom_list:
    #     for ps  in sssp:
    #         tmp = re.split('[\W^_]+', ps)[0]
    #         if ele.lower() == tmp.lower():
    #             pseudodict[ele] = ps
    #             break
    # print(pseudodict)
    # pickle.dump(pseudodict, open('pseudodict.pkl', 'wb'))
    # exit()

    pseudodict = pickle.load(open('pseudodict.pkl', 'rb'))
    energy_list = []
    file_paths = []
    energy_values = []
    for i in tqdm(range(n)):
        # if i%50 == 0:
        #     print('Length of energy list so far', i, len(energy_list))
        cif_string = data.loc[i]['cif']
        # try:
        #     with open(outputs_dir + '/espresso_' + str(i+start) + '.pwo', 'r') as f:
        #         lines = f.read()
        #         if 'convergence NOT achieved after' not in lines:
        #             continue
        #     # with open(outputs_dir + '/espresso_' + str(i+start) + '.pwo', 'r') as f:
        #     #     lines = f.read()
        #     #     if 'too few bands' not in lines:
        #     #         continue
        # except:
        #     continue
        with open(tmp_dir + '/tmp'+str(i)+'.cif', 'w') as f:
            f.write(cif_string)
        print(i,end = ',')
        file_path = tmp_dir + '/tmp'+str(i)+'.cif'
        # file_path = tmp_dir + '/'+str(i)+'.cif'
        file_paths.append(file_path)
        energy_values.append(calc_energy(file_path, pseudodict, i, start, pwo_path))
        # ase_obj = read(file_path)
        # input_ = 'calculations/espresso_'+str(i)+'.pwi'
        # output_ = 'calculations/espresso_'+str(i)+'.pwo'
        # os.environ['ASE_ESPRESSO_COMMAND'] = 'srun pw.x -in ' + input_ + ' > ' + output_
        # input_data = {'prefix':"myprefix",'electron_maxstep':100,'outdir':"calculations/",'pseudo_dir': pseudo_dir, 'tstress':True,'tprnfor':True,'calculation':'scf', 
        #             'ecutrho':240,'verbosity':'high','ecutwfc':30, 'diagonalization': 'david', 'occupations':'smearing','smearing':'mp', 'mixing_mode':'plain', 
        #             'mixing_beta':0.7,'degauss':0.001, 'nspin':1, 'nstep': 1}
        # ase_obj.calc=Espresso(pseudopotentials=pseudodict,input_data=input_data, kpts=None, label='calculations/espresso_'+str(i))
        # try:
        #     energy = ase_obj.get_potential_energy()
        #     print('Energy = ', energy)
        # except:
        #     energy = None
        # energy_list.append(energy)
    # assert len(file_paths) == n
    # energy_values =  p_map(calc_energy, file_paths, [pseudodict] * n, list(range(n)), [start] * n, num_cpus = 1)   
    print(energy_values)
    # pickle.dump(energy_values, open(save_path, 'wb'))

