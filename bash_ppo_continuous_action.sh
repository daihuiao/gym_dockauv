for seed in  0 1 2
do
  for clip_coef in 0.1 0.05 0.25 0.15
  do
  for tau in  0.9
  do
#    python ppo_continuous_action.py  --seed $seed --tau $tau --w_velocity 0.1 --thruster_penalty 0.1
#    python ppo_continuous_action.py  --seed $seed --tau $tau --w_velocity 0.1 --thruster_penalty 0.5
    python ppo_continuous_action.py  --seed $seed --tau $tau --w_velocity 0.1 --thruster_penalty 1.0 --clip_coef $clip_coef
  done
  done
done






#for tau in 0.5 0.9 0.99
#do
#  for seed in  0
#  do
#    python ppo_continuous_action.py  --seed $seed --tau $tau --w_velocity 0. --thruster_penalty 0.
#    python ppo_continuous_action.py  --seed $seed --tau $tau --w_velocity 0.1 --thruster_penalty 0.
#    python ppo_continuous_action.py  --seed $seed --tau $tau --w_velocity 1. --thruster_penalty 0.
#    python ppo_continuous_action.py  --seed $seed --tau $tau --w_velocity 0.1 --thruster_penalty 0.1
#  done
#done


#for tau in 0.5 0.9 0.99
#do
#  for seed in  0
#  do
#    for w_velocity in 0.1 1.0
#    do
#      for thruster_penalty in 0.1 1.0
#      do
#        python ppo_continuous_action.py  --seed $seed --tau $tau --w_velocity $w_velocity --thruster_penalty $thruster_penalty
#      done
#    done
#  done
#done