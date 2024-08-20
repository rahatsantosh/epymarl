source /home/rsan/anaconda3/etc/profile.d/conda.sh
conda activate am

output_file="benchmark_results.txt"

echo "Starting resource usage benchmarking with perf..." > $output_file

programs=(
    "No Agent Modelling:python main.py --config=mappo --env-config=gymma with env_args.time_limit=25 env_args.key='pz-mpe-simple-spread-v3' common_reward=False opponent_modelling=False t_max=100000"
    "Reward Reconstruction:python main.py --config=mappo --env-config=gymma with env_args.time_limit=25 env_args.key='pz-mpe-simple-spread-v3' common_reward=False opponent_modelling=True opponent_model_decode_observations=False opponent_model_decode_actions=False opponent_model_decode_rewards=True latent_dims=32 t_max=100000"
    "Action Reconstruction:python main.py --config=mappo --env-config=gymma with env_args.time_limit=25 env_args.key='pz-mpe-simple-spread-v3' common_reward=False opponent_modelling=True opponent_model_decode_observations=False opponent_model_decode_actions=True opponent_model_decode_rewards=False latent_dims=32 t_max=100000"
    "Observation Reconstruction:python main.py --config=mappo --env-config=gymma with env_args.time_limit=25 env_args.key='pz-mpe-simple-spread-v3' common_reward=False opponent_modelling=True opponent_model_decode_observations=True opponent_model_decode_actions=False opponent_model_decode_rewards=False latent_dims=32 t_max=100000"
    "Combined Reconstruction:python main.py --config=mappo --env-config=gymma with env_args.time_limit=25 env_args.key='pz-mpe-simple-spread-v3' common_reward=False opponent_modelling=True opponent_model_decode_observations=True opponent_model_decode_actions=False opponent_model_decode_rewards=True latent_dims=32 t_max=100000"
)

for prog in "${programs[@]}"; do
    IFS=":" read -r label cmd <<< "$prog"
    echo "Running $label..." | tee -a $output_file
    
    # Benchmark runs with perf
    for i in {1..5}; do
        echo "Run $i for $label:" | tee -a $output_file
        perf stat -o perf_stat_tmp.log -d -d -d bash -c "$cmd"
        cat perf_stat_tmp.log >> $output_file
        echo "" >> $output_file
    done
done

echo "Benchmarking completed. Results saved to $output_file."
