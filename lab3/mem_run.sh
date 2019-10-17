#!/bin/bash
env="cartpole"
num_episodes=1200
FIX_BATCH_SIZE=32
target_updates=1
replay_sizes="1 100 1000 5000 10000"

seeds="42 43 44"

if [ "$(uname)" == "Linux" ]; then
    connect='/'
else
    connect='\'
fi

save_dir="checkpoints_tu"
for seed in $seeds; do
  for replay_size in $replay_sizes; do
    if [ $replay_size -eq 1 ]; then
      batch_size=1
    else 
      batch_size=$FIX_BATCH_SIZE
    fi
    save_path=$save_dir$connect$env"_s"$seed"_mem"$replay_size".pth"
    python main.py --seed=$seed --use_env=$env --num_episodes=$num_episodes --batch_size=$batch_size --replay_size=$replay_size --target_update=$target_updates --save_path=$save_path
  done
done
