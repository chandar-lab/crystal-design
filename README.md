# Offline Reinforcement Learning for Crystal Design

## Introduction

In this work, we adopt conservative Q-learning, a popular offline reinforcement learning approach to learning a conditional policy that can design new and stable crystals having the desired band gap. Our targeted formulation of the reward function for offline RL is crafted from formation energy (per atom) and band gap values computed using first-principles DFT calculations, widely used in computational chemistry. Refer to our [paper](https://openreview.net/forum?id=VbjD8w2ctG) for more details about our methodology.


![alt text](images/workflow.png)

## Documentation
### Installation
To install, clone the repository  
```
git clone https://github.com/chandar-lab/crystal-design.git
cd crystal-design
pip install -r requirements.txt
pip install -e .
cd crystal_design
```
If `dgl` installation fails, please refer [https://www.dgl.ai/pages/start.html](https://www.dgl.ai/pages/start.html).
### Training 
We use the `crystal_cql.py` in the `runner` folder for training and generation. Use this command for the list of arguments. 
```
python crystal_cql.py -h
```
To train a conditional CQL model from scratch for 250000 steps, execute
```
python crystal_cql.py --data_path <path to trajectory data> --p_hat <target property> --max_timesteps 250000
```
To train an unconditional CQL model for 250000 steps, execute

```
python crystal_cql.py --data_path <path to trajectory data> --max_timesteps 250000 --nocondition True
```

To train a behavioral cloning model for 250000 steps, execute
```
python crystal_cql.py --data_path <path to trajectory data> --max_timesteps 250000 --bc True --nocondition True
```
### Generating Crystals

To generate crystal using a learned model, whose checkpoint (`.pt`) is stored in a directory, follow the command
```
python crystal_cql.py --mode eval --model_path <path to checkpoint> --p_hat <target property>
```

A recommended way to do the same is to directly provide the path to the wandb run, in which case it will run the checkpoint file named `checkpoint_250000.pt` (this can be modified)
```
python crystal_cql.py --mode eval --eval_wandb_path <path to wandb run> --no_condition <True/False>
```
