import os 
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from tqdm import tqdm 
import re
import numpy as np

# if __name__ == '__main__':
#     dict_ = {}
#     c = 0
#     files = os.listdir('bands_nm_valid')
#     for file in tqdm(files):
#         i = int(re.findall(r'\d+', file)[0])
#         try:
#             with open('bands_nm_valid/'+file, 'r') as f:
#                 lines = f.read()
#                 high, low = lines.split('highest occupied, lowest unoccupied level (ev):')[1].split('!')[0].split()
#                 bandgap = float(low)-float(high)
#                 if bandgap <0:
#                     bandgap = 0
#                     c+=1
#                 energy = float(lines.split('!    total energy')[1].split()[1])
#                 dict_[i] = (bandgap, energy)
#         except:
#             pass

# if __name__ == '__main__':
#     data = pd.read_csv('../data/mp_20/train.csv')
#     dict_ = {}
#     c = []
#     files = os.listdir('energies_m_valid')
#     for file in tqdm(files):
#         try:
#             i = int(re.findall(r'\d+', file)[0])
#             with open('energies_m_valid/'+file, 'r') as f:
#                 lines = f.read()
#                 try:
#                     energy = float(lines.split('!    total energy')[1].split()[1])
#                     dict_[i] = (0.0, energy)
#                 except:
#                     tmp = re.findall('total energy              =   .*', lines)[-1]
#                     energy = float(tmp.split()[-2])
#                     dict_[i] = (0.0, energy)
#         except:
#             c.append(file)
#             pass


if __name__ == '__main__':
    d1 = pickle.load(open('m_dict.pkl', 'rb'))
    d2 = pickle.load(open('nm_dict.pkl', 'rb'))
    ### alpha1 log(-E) 
    bgs = [d1[i][0] for i in d1]
    tmp = [d2[i][0] for i in d2]
    bgs += tmp
    beta = 5.
    # reward1 = [np.exp(-(d1[i][0]-1.12)**2 / beta) for i in d1]
    # tmp1 = [-np.exp((d2[i][0]-1.12)**2 / beta) for i in d2]
    # reward1 += tmp1
    reward2 = [np.log10(-d1[i][1]) + 5*np.exp(-(d1[i][0]-1.12)**2 / beta) for i in d1]
    tmp2 = [np.log10(-d2[i][1]) +5*np.exp(-(d2[i][0]-1.12)**2 / beta) for i in d2]
    reward2 += tmp2
    reward3 = [np.log10(-d1[i][1]) + 4*np.exp(-(d1[i][0]-1.12)**2 / beta) for i in d1]
    tmp3 = [np.log10(-d2[i][1]) +4*np.exp(-(d2[i][0]-1.12)**2 / beta) for i in d2]
    reward3 += tmp3
    reward4 = [np.log10(-d1[i][1]) + 3*np.exp(-(d1[i][0]-1.12)**2 / beta) for i in d1]
    tmp4 = [np.log10(-d2[i][1]) +3*np.exp(-(d2[i][0]-1.12)**2 / beta) for i in d2]
    reward4 += tmp4
    reward5 = [np.log10(-d1[i][1]) + 2*np.exp(-(d1[i][0]-1.12)**2 / beta) for i in d1]
    tmp5 = [np.log10(-d2[i][1]) +2*np.exp(-(d2[i][0]-1.12)**2 / beta) for i in d2]
    reward5 += tmp5
    reward6 = [1.*np.log10(-d1[i][1]) + 1*np.exp(-(d1[i][0]-1.12)**2 / beta) for i in d1]
    tmp6 = [1.*np.log10(-d2[i][1]) +1*np.exp(-(d2[i][0]-1.12)**2 / beta) for i in d2]
    reward6 += tmp6

    
    # reward7 = [beta*np.log10(-d1[i][1]) - np.exp((d1[i][0]-1.12)**2 / beta) for i in d1]
    # tmp7 = [beta*np.log10(-d2[i][1]) -np.exp((d2[i][0]-1.12)**2 / beta) for i in d2]
    # reward7 += tmp7

plt.figure(figsize=(9,7))
# plt.plot(bgs, reward1, 'o', label = '0.',markersize=3)
plt.plot(bgs, reward2, 'o', label = '5',markersize=3)
plt.plot(bgs, reward3, 'o', label = '4',markersize=3)
plt.plot(bgs, reward4, 'o', label = '3',markersize=3)
plt.plot(bgs, reward5, 'o', label = '2',markersize=3)
plt.plot(bgs, reward6, 'o', label = '1',markersize=3)
l = np.linspace(-20,20, 100)
plt.ylim((0,10))
plt.plot([1.12]*100, l, '-')
plt.legend(loc = 'best')
plt.savefig('rewardplot2.png', dpi = 400)

beta = 1.
plt.figure(figsize=(9,7))
reward2 = [np.log10(-d1[i][1]) + 5*np.exp(-(d1[i][0]-1.12)**2 / beta) for i in d1]
tmp2 = [np.log10(-d2[i][1]) +5*np.exp(-(d2[i][0]-1.12)**2 / beta) for i in d2]
reward2 += tmp2
reward3 = [np.log10(-d1[i][1]) + 4*np.exp(-(d1[i][0]-1.12)**2 / beta) for i in d1]
tmp3 = [np.log10(-d2[i][1]) +4*np.exp(-(d2[i][0]-1.12)**2 / beta) for i in d2]
reward3 += tmp3
reward4 = [np.log10(-d1[i][1]) + 3*np.exp(-(d1[i][0]-1.12)**2 / beta) for i in d1]
tmp4 = [np.log10(-d2[i][1]) +3*np.exp(-(d2[i][0]-1.12)**2 / beta) for i in d2]
reward4 += tmp4
reward5 = [np.log10(-d1[i][1]) + 2*np.exp(-(d1[i][0]-1.12)**2 / beta) for i in d1]
tmp5 = [np.log10(-d2[i][1]) +2*np.exp(-(d2[i][0]-1.12)**2 / beta) for i in d2]
reward5 += tmp5
reward6 = [1.*np.log10(-d1[i][1]) + 1*np.exp(-(d1[i][0]-1.12)**2 / beta) for i in d1]
tmp6 = [1.*np.log10(-d2[i][1]) +1*np.exp(-(d2[i][0]-1.12)**2 / beta) for i in d2]
reward6 += tmp6
plt.plot(bgs, reward2, 'o', label = '5',markersize=3)
plt.plot(bgs, reward3, 'o', label = '4',markersize=3)
plt.plot(bgs, reward4, 'o', label = '3',markersize=3)
plt.plot(bgs, reward5, 'o', label = '2',markersize=3)
plt.plot(bgs, reward6, 'o', label = '1',markersize=3)
# plt.plot(bgs, reward7, 'o', label = '5.')
plt.plot([1.12]*100, l, '-')
plt.legend(loc = 'best')
plt.xlabel('Band Gap')
plt.ylabel('Reward')
plt.ylim((0,10))
plt.savefig('rewardplot3.png', dpi = 400)



beta = 3.
plt.figure(figsize=(9,7))
reward2 = [np.log10(-d1[i][1]) + 5*np.exp(-(d1[i][0]-1.12)**2 / beta) for i in d1]
tmp2 = [np.log10(-d2[i][1]) +5*np.exp(-(d2[i][0]-1.12)**2 / beta) for i in d2]
reward2 += tmp2
reward3 = [np.log10(-d1[i][1]) + 4*np.exp(-(d1[i][0]-1.12)**2 / beta) for i in d1]
tmp3 = [np.log10(-d2[i][1]) +4*np.exp(-(d2[i][0]-1.12)**2 / beta) for i in d2]
reward3 += tmp3
reward4 = [np.log10(-d1[i][1]) + 3*np.exp(-(d1[i][0]-1.12)**2 / beta) for i in d1]
tmp4 = [np.log10(-d2[i][1]) +3*np.exp(-(d2[i][0]-1.12)**2 / beta) for i in d2]
reward4 += tmp4
reward5 = [np.log10(-d1[i][1]) + 2*np.exp(-(d1[i][0]-1.12)**2 / beta) for i in d1]
tmp5 = [np.log10(-d2[i][1]) +2*np.exp(-(d2[i][0]-1.12)**2 / beta) for i in d2]
reward5 += tmp5
reward6 = [1.*np.log10(-d1[i][1]) + 1*np.exp(-(d1[i][0]-1.12)**2 / beta) for i in d1]
tmp6 = [1.*np.log10(-d2[i][1]) +1*np.exp(-(d2[i][0]-1.12)**2 / beta) for i in d2]
reward6 += tmp6
plt.plot(bgs, reward2, 'o', label = '5',markersize=3)
plt.plot(bgs, reward3, 'o', label = '4',markersize=3)
plt.plot(bgs, reward4, 'o', label = '3',markersize=3)
plt.plot(bgs, reward5, 'o', label = '2',markersize=3)
plt.plot(bgs, reward6, 'o', label = '1',markersize=3)
# plt.plot(bgs, reward7, 'o', label = '5.')
plt.plot([1.12]*100, l, '-')
plt.legend(loc = 'best')
plt.xlabel('Band Gap')
plt.ylabel('Reward')
plt.ylim((0,10))
plt.savefig('rewardplot4.png', dpi = 400)
