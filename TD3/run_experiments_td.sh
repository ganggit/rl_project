#!/bin/bash
# Script to reproduce results
for i in 4 5 #((i=0;i<2;i+=1))
do 
	nohup /usr/bin/python3 main.py --policy_name "TD3" --env_name "HalfCheetah-v2" --seed $i --start_timesteps 10000 &
	nohup /usr/bin/python3 main.py --policy_name "TD3" --env_name "Hopper-v2" --seed $i --start_timesteps 1000 &
	nohup /usr/bin/python3 main.py --policy_name "TD3" --env_name "Walker2d-v2" --seed $i --start_timesteps 1000 &
	nohup /usr/bin/python3 main.py --policy_name "TD3" --env_name "Ant-v2" --seed $i --start_timesteps 10000 &
	nohup /usr/bin/python3 main.py --policy_name "TD3" --env_name "InvertedPendulum-v2" --seed $i --start_timesteps 1000 &
	nohup /usr/bin/python3 main.py --policy_name "TD3" --env_name "InvertedDoublePendulum-v2" --seed $i --start_timesteps 1000 &
	nohup /usr/bin/python3 main.py --policy_name "TD3" --env_name "Reacher-v2" --seed $i --start_timesteps 1000 &
done