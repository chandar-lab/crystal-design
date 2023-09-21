# crystal-design
Reinforcement Learning for Crystal Structure Design

To install, clone the `megnet` repository, and do the following step

`git clone https://github.com/chandar-lab/crystal-design.git --branch megnet`

`cd crystal-design`

`pip install -e .`

To train model, follow these steps,

`cd crystal-design/crystal_design/runner`

`python cql_trainer.py [--args]`

For example, 

```python cql_trainer.py --alpha1 1 --alpha2 5 --beta 1 --cql_min_q_weight 1 --project 'CQL-NONMETALS-MOREDATA' --group 'Run1' --si_bg 1.12 --data_path traj.pt```
