for tau in 0.99 0.95
do
  for seed in  0 1 2
  do
    python ppo_continuous_action.py  --seed $seed --tau $tau
  done
done