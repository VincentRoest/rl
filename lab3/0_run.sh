env="cartpole"
num_episodes=1200
batch_size=32
replay_size=10000

seeds="42 43 44"
target_updates="-1 20 50 100 200"

if [ "$(uname)" == "Linux" ]; then
    connect='/'
else
    connect='\'
fi

save_dir="checkpoints_tu"
for seed in $seeds; do
  for target_update in $target_updates; do
    save_path=$save_dir$connect$env"_s"$seed"_tu"$target_update".pth"
    python main.py --seed=$seed --use_env=$env --num_episodes=$num_episodes --batch_size=$batch_size --replay_size=$replay_size --target_update=$target_update --save_path=$save_path
  done
done
