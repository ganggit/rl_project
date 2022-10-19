#!/bin/bash
# Script to reproduce results
for i in 0 #1 2 #((i=0;i<2;i+=1))
do 
	CUDA_VISIBLE_DEVICES=1 /usr/bin/python3 main_gaussian.py --policy_name "TD3" --env_name "HalfCheetah-v2" --seed $i --start_timesteps 10000 &
	CUDA_VISIBLE_DEVICES=1 /usr/bin/python3 main_gaussian.py --policy_name "TD3" --env_name "Hopper-v2" --seed $i --start_timesteps 1000 &
	CUDA_VISIBLE_DEVICES=1 /usr/bin/python3 main_gaussian.py --policy_name "TD3" --env_name "Walker2d-v2" --seed $i --start_timesteps 1000 &
	CUDA_VISIBLE_DEVICES=1 /usr/bin/python3 main_gaussian.py --policy_name "TD3" --env_name "Ant-v2" --seed $i --start_timesteps 10000 &
	CUDA_VISIBLE_DEVICES=1 /usr/bin/python3 main_gaussian.py --policy_name "TD3" --env_name "InvertedPendulum-v2" --seed $i --start_timesteps 1000 &
	CUDA_VISIBLE_DEVICES=1 /usr/bin/python3 main_gaussian.py --policy_name "TD3" --env_name "InvertedDoublePendulum-v2" --seed $i --start_timesteps 1000 &
	CUDA_VISIBLE_DEVICES=1 /usr/bin/python3 main_gaussian.py --policy_name "TD3" --env_name "Reacher-v2" --seed $i --start_timesteps 1000 &
done