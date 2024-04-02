for seed in  0 1 2
do
  for thruster in 400 500 600 700 800 900
  do
  for tau in  0.5
  do
    python timevaring_current_ppo_v1.py  --seed $seed --tau $tau --w_velocity 0.1 --thruster_penalty 1.0 --thruster $thruster
  done
  done
done
#todo :一个是加时间,一个是让位置随机.

