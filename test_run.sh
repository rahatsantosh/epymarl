python3 src/main.py --config=mappo --env-config=gymma with env_args.time_limit=25 env_args.key="mpe:SimpleSpread-v0" opponent_modelling=True opponent_model_decode_observations=True opponent_model_decode_actions=True latent_dims=32

# Simple env runs for sanity checks etc.
python3 src/main.py --config=mappo --env-config=gymma with env_args.time_limit=25 env_args.key="lbforaging:Foraging-8x8-2p-3f-v2" common_reward=False opponent_modelling=True opponent_model_decode_observations=True opponent_model_decode_actions=True latent_dims=32

