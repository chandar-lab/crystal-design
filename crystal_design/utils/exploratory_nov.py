import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pymatgen.core.structure as S
from pymatgen.analysis import energy_models as em
import pymatgen.io.cif as cif
from tqdm import tqdm 
import pickle

if __name__ == '__main__':
    print('Starting')
    data = pd.read_pickle('../data/materials_project.pkl')
    print('Loaded Data')
    n = len(data)
    # _, freq = pickle.load(open('energy_freq.py', 'rb'))
    tol = 1e-6
################################################
    # elements_sorted = list(dict(sorted(freq.items(), key=lambda item: item[1], reverse = True)).keys())
    # for ele in elements_sorted:
    #     for i in range(n):
    #         cif_string = data.loc[i]['cif']
    #     # formula = data.loc[i]['full_formula']
    #         mat = cif.CifParser.from_string(cif_string).as_dict()
    #         mat = mat[list(mat.keys())[0]]
    #         lattice = cif.CifParser.from_string(cif_string).get_lattice(mat)
    #         ele = mat['_atom_site_type_symbol']                    
################################################
    freq_dict = {'cubic':0, 'hexagonal': 0, 'tetragonal':0, 'orthorhombic':0, 'trigonal': 0, 'monoclinic': 0, 'triclinic': 0}
    for i in tqdm(range(n)):
        cif_string = data.loc[i]['cif']
        mat = cif.CifParser.from_string(cif_string).as_dict()
        mat = mat[list(mat.keys())[0]]
        
        a, b, c = float(mat['_cell_length_a']), float(mat['_cell_length_b']), float(mat['_cell_length_c'])
        alpha, beta, gamma = float(mat['_cell_angle_alpha']), float(mat['_cell_angle_beta']), float(mat['_cell_angle_gamma'])
        
        if abs(a-b) < tol and abs(b-c) < tol:  ### a=b=c
            if abs(alpha - 90.) < tol and abs(beta - 90.) < tol and abs(gamma - 90.) < tol: 
                freq_dict['cubic'] += 1   #alpha = beta = gamma = 90
            elif (alpha < 120.) and (beta < 120.) and (gamma < 120.) and \
                abs(alpha-beta) < tol and abs(beta-gamma) < tol:
                freq_dict['trigonal'] += 1   #alpha = beta = gamma < 120
        elif (abs(a-b) < tol and abs(b-c) > tol) or (abs(b-c) < tol and abs(a-c) > tol) or \
            (abs(a-c) < tol and abs(b-c) > tol): # a=b!=c
            if abs(alpha - 90.) < tol and abs(beta - 90.) < tol and abs(gamma - 90.) < tol:
                freq_dict['tetragonal'] += 1 #alpha = beta = gamma = 90
            elif (abs(alpha - 120.) < tol and abs(beta - 90.) < tol and abs(gamma - 90.) < tol) or \
                (abs(beta - 120.) < tol and abs(alpha - 90.) < tol and abs(gamma - 90.) < tol) or \
                (abs(gamma - 120.) < tol and abs(alpha - 90.) < tol and abs(beta - 90.) < tol):
                freq_dict['hexagonal'] += 1 #a = 120, b=c=90
        else: #a!=b!=c
            if abs(alpha - 90.) < tol and abs(beta - 90.) < tol and abs(gamma - 90.) < tol:
                freq_dict['orthorhombic'] += 1 #alpha = beta = gamma = 90
            elif (abs(alpha - 90.)  > tol and abs(beta - 90.) < tol and abs(gamma - 90.) < tol) or \
                (abs(beta - 90.)  > tol and abs(alpha - 90.) < tol and abs(gamma - 90.) < tol) or \
                (abs(gamma - 90.) > tol and abs(alpha - 90.) < tol and abs(beta - 90.) < tol): 
                freq_dict['monoclinic'] += 1  #alpha = beta = 90, gamma != 90
            elif abs(alpha-beta) > tol and abs(beta-gamma) > tol:
                freq_dict['triclinic'] += 1  #alpha != beta != gamma
            else:
                print('Exceptions: ', a, b, c, alpha, beta, gamma)
    print(freq_dict)










