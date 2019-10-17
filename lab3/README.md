## Memory Buffer and Target Network Explorations
This repository contains the code for the Lab3 assignment.

### Train a model on a certain environment
Ex. `python main.py --use_env='cartpole'`

| Parameters | Description | Default 
| --- | --- | --- |
| `--use_env` | OpenAI gym to use: 'cartpole', 'pong' or 'acrobat' | 'cartpole'
| `--batch_size` | Batch Size (set to 1 if using `--replay_size=1`) | 32
| `--target_update` | Target net update intervals | -1 (no updating)
| `--replay_size` | Memory Buffer size | 10000
| `--clip_rewards` | Use reward clipping | False
| `--gamma` | Reward Discount | 0.9 
| `--num_episodes` | How many episodes to train for | 1200
| `--seed` | Seed | 42


### Reproducing results (time consuming)

The easiest way to reproduce the results in the paper is by running the following:

#### Target Update Experiments:

`target_run.sh`

```python plot_training_results.py --load_path=checkpoints_tu/cartpole_s42_tu-1.pth+checkpoints_tu/cartpole_s43_tu-1.pth+checkpoints_tu/cartpole_s44_tu-1.pth,checkpoints_tu/cartpole_s42_tu20.pth+checkpoints_tu/cartpole_s43_tu20.pth+checkpoints_tu/cartpole_s44_tu20.pth,checkpoints_tu/cartpole_s42_tu50.pth+checkpoints_tu/cartpole_s43_tu50.pth+checkpoints_tu/cartpole_s44_tu50.pth,checkpoints_tu/cartpole_s42_tu100.pth+checkpoints_tu/cartpole_s43_tu100.pth+checkpoints_tu/cartpole_s44_tu100.pth,checkpoints_tu/cartpole_s42_tu200.pth+checkpoints_tu/cartpole_s43_tu200.pth+checkpoints_tu/cartpole_s44_tu200.pth --smoothing=20 --labels=target_update=-1,target_update=20,target_update=50,target_update=100,target_update=200```

#### Memory Buffer Experiments:

`mem_run.sh`

```python plot_training_results.py --load_path=saved_checkpoints/1_24.pth+saved_checkpoints/1_42.pth+saved_checkpoints/1_44.pth,saved_checkpoints/100_24.pth+saved_checkpoints/100_42.pth+saved_checkpoints/100_44.pth,saved_checkpoints/1000_24.pth+saved_checkpoints/1000_42.pth+saved_checkpoints/1000_44.pth,saved_checkpoints/5000_24.pth+saved_checkpoints/5000_42.pth+saved_checkpoints/5000_44.pth,saved_checkpoints/10000_24.pth+saved_checkpoints/10000_42.pth+saved_checkpoints/10000_44.pth --labels=1,100,1000,5000,10000 --smoothing=50```


### Recreate plots
This requires a checkpoint to exist created by training a model using the commands above:
`python plot_training_results.py --load_path=saved_checkpoints/train.pth --smoothing=10`

| Selected Parameters | Description | Default 
| ---- | --- | --- |
| `load_path` | list of checkpoint locations separated by commas. If using multiple checkpoints for the same configurations to get standard deviations, use +s between similar configs: <br/> `--load_path=1_v1.pth+1_v2.pth,2_v1.pth,2_v2.pth` | None
| `smoothing` | Do smoothing for the plot (in terms of episode) | 32
| `labels` | Which labels should appear in the plot? Length of the number of commas + 1 for the load path argument `--load_path` | inherit from checkpoint names





