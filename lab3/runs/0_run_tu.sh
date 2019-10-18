env="cartpole"
# num_episodes=1200
# batch_size=32
# replay_size=10000
# batch_size=1
# replay_size=1
num_episodes=400
batch_size=16
replay_size=1000

# seeds="42 43 44"
# target_updates="-1 20 50 100 200"
seeds="42 43"
target_updates="-1 2 5"

if [ "$(uname)" == "Linux" ]; then
    connect='/'
else
    connect='\'
fi

# save_dir="checkpoints_tu"
# save_dir="checkpoints_tu_raw"
save_dir="checkpoints_tu_small"
for seed in $seeds; do
  for target_update in $target_updates; do
    save_path=$save_dir$connect$env"_s"$seed"_tu"$target_update".pth"
    python main.py --seed=$seed --use_env=$env --num_episodes=$num_episodes --batch_size=$batch_size --replay_size=$replay_size --target_update=$target_update --save_path=$save_path
  done
done
