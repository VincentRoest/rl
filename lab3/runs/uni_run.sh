env="cartpole"
num_episodes=500
batch_size=32

seeds="42"
target_updates="-1 100"
replay_sizes="1 1000"

if [ "$(uname)" == "Linux" ]; then
    connect='/'
else
    connect='\'
fi

save_dir="checkpoints_uni"
for seed in $seeds; do
  for target_update in $target_updates; do
    for replay_size in $replay_sizes; do
      save_path=$save_dir$connect$env"_s"$seed"_tu"$target_update"_rm"$replay_size".pth"
      if [ "$replay_size" == "1" ]; then
        sample_repeat=--sample_repeat
      else
        sample_repeat=
      fi
      python main.py --seed=$seed --use_env=$env --num_episodes=$num_episodes --batch_size=$batch_size --replay_size=$replay_size --target_update=$target_update --save_path=$save_path $sample_repeat
    done
  done
done
