env="acrobot"
num_episodes=100
batch_size=16
replay_size=1000
save_dir="checkpoints_acrobot"

seeds="42"
target_updates="-1 20"

if [ "$(uname)" == "Linux" ]; then
    connect='/'
else
    connect='\'
fi

for seed in $seeds; do
  for target_update in $target_updates; do
    save_path=$save_dir$connect$env"_s"$seed"_tu"$target_update".pth"
    python main.py --seed=$seed --use_env=$env --num_episodes=$num_episodes --batch_size=$batch_size --replay_size=$replay_size --target_update=$target_update --save_path=$save_path
  done
done
