source /home/rsan/anaconda3/etc/profile.d/conda.sh
conda activate am

hyperfine --export-json benchmark_results.json --warmup 3 --runs 5 \
          -n 'No Agent Modelling' 'python main.py --config=mappo --env-config=gymma with env_args.time_limit=25 env_args.key="pz-mpe-simple-spread-v3" common_reward=False opponent_modelling=False t_max=100000' \
          -n 'Reward Reconstruction' 'python main.py --config=mappo --env-config=gymma with env_args.time_limit=25 env_args.key="pz-mpe-simple-spread-v3" common_reward=False opponent_modelling=True opponent_model_decode_observations=False opponent_model_decode_actions=False opponent_model_decode_rewards=True latent_dims=32 t_max=100000' \
          -n 'Action Reconstruction' 'python main.py --config=mappo --env-config=gymma with env_args.time_limit=25 env_args.key="pz-mpe-simple-spread-v3" common_reward=False opponent_modelling=True opponent_model_decode_observations=False opponent_model_decode_actions=True opponent_model_decode_rewards=False latent_dims=32 t_max=100000' \
          -n 'Observation Reconstruction' 'python main.py --config=mappo --env-config=gymma with env_args.time_limit=25 env_args.key="pz-mpe-simple-spread-v3" common_reward=False opponent_modelling=True opponent_model_decode_observations=True opponent_model_decode_actions=False opponent_model_decode_rewards=False latent_dims=32 t_max=100000' \
          -n 'Combined Reconstruction' 'python main.py --config=mappo --env-config=gymma with env_args.time_limit=25 env_args.key="pz-mpe-simple-spread-v3" common_reward=False opponent_modelling=True opponent_model_decode_observations=True opponent_model_decode_actions=True opponent_model_decode_rewards=True latent_dims=32 t_max=100000'
