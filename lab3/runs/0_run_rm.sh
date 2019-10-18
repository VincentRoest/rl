env="cartpole"
num_episodes=1200
target_update=-1

seeds="44"
replay_sizes="1 100 1000 5000 10000 500"

if [ "$(uname)" == "Linux" ]; then
    connect='/'
else
    connect='\'
fi

save_dir="checkpoints_rm"
for seed in $seeds; do
  for replay_size in $replay_sizes; do
    save_path=$save_dir$connect$env"_s"$seed"_rm"$replay_size".pth"
    if [ "$replay_size" == "1" ]; then
      batch_size=1
    else
      batch_size=32
    fi
    python main.py --seed=$seed --use_env=$env --num_episodes=$num_episodes --batch_size=$batch_size --replay_size=$replay_size --target_update=$target_update --save_path=$save_path
  done
done
