import torch

data = torch.load('trajectories/train_mp_mg_24k.pt')
observations = data['observations']
n = len(observations)
print('initial length: ', len(observations))
new_observations = []
N  = observations[0]['atomic_number'].shape[0]
j = 0
while j<n:
    for i in range(j,N*2):
        if i % 2 ==0 or (i == 2*N -1):
            new_observations.append(new_observations[j])
        j += 1
    j = i+1
    N = observations[j]['atomic_number'].shape[0]

data['observations'] = new_observations