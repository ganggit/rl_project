#!/bin/bash
# Script to reproduce results

nohup python3.6 main.py --num_episodes=20000 --policy_name "TD3" --env_name "Ant-v2" --gpu_id 0 --seed 2 >& ac_call_td3_Ant_pytorch_iter20000_v2.nonup &
nohup python3.6 main.py --num_episodes=20000 --policy_name "TD3" --env_name "HalfCheetah-v2" --gpu_id 1 --seed 2 >& ac_call_td3_HalfCheetah_pytorch_iter20000_v2.nohup &
nohup python3.6 main.py --num_episodes=20000 --policy_name "TD3" --env_name "Hopper-v2" --gpu_id 2 >& ac_call_td3_Hopper_pytorch_iter20000_v2.nohup &
nohup python3.6 main.py --num_episodes=20000 --policy_name "TD3" --env_name "Humanoid-v2" --gpu_id 3 >& ac_call_td3_Humanoid_pytorch_iter20000_v2.nohup &
nohup python3.6 main.py --num_episodes=20000 --policy_name "TD3" --env_name "Reacher-v2" --gpu_id 4 >& ac_call_td3_Reacher_pytorch_iter20000_v2.nohup &
nohup python3.6 main.py --num_episodes=20000 --policy_name "TD3" --env_name "Walker2d-v2" --gpu_id 5 >& ac_call_td3_Walker2d_pytorch_iter20000_v2.nohup &

for i in {1..10}
do
	python2 main.py \
	--policy_name "TD3" \
	--env_name "Ant-v2" \
	--seed $i \
	--start_timesteps 10000

	python2 main.py \
	--policy_name "TD3" \
	--env_name "HalfCheetah-v2" \
	--seed $i \
	--start_timesteps 10000

	python2 main.py \
	--policy_name "TD3" \
	--env_name "Hopper-v2" \
	--seed $i \
	--start_timesteps 1000

	python2 main.py \
	--policy_name "TD3" \
	--env_name "Humanoid-v2" \
	--seed $i \
	--start_timesteps 10000

	python2 main.py \
	--policy_name "TD3" \
	--env_name "Reacher-v2" \
	--seed $i \
	--start_timesteps 1000

	python2 main.py \
	--policy_name "TD3" \
	--env_name "Walker2d-v2" \
	--seed $i \
	--start_timesteps 1000

	python2 main.py \
	--policy_name "TD3" \
	--env_name "InvertedPendulum-v2" \
	--seed $i \
	--start_timesteps 1000

	python2 main.py \
	--policy_name "TD3" \
	--env_name "InvertedDoublePendulum-v2" \
	--seed $i \
	--start_timesteps 1000
done
