for tau in 0.1 0.9 0.7 0.3 0.4 0.6 0.8 0.2
do
  for seed in  0
  do
    python ppo_continuous_action.py  --seed $seed --tau $tau --device
  done
done