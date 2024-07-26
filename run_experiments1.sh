#!/bin/bash

# Determine the total number of CPUs
total_cpus=$(nproc)
# Calculate the number of CPUs to use (half of the total CPUs)
half_cpus=$((total_cpus / 2))

# Define the CPU range to use (from 0 to half_cpus - 1)
cpu_range=0-$(($half_cpus - 1))

# Define environments and their specific configurations
declare -A env_configs=(
    ["pz-mpe-simple-spread-v3"]="gymma:25"
    ["lbforaging:Foraging-10x10-3p-3f-coop-v3"]="gymma:50"
    ["lbforaging:Foraging-10x10-3p-3f-v3"]="gymma:50"
    ["lbforaging:Foraging-2s-10x10-3p-3f-coop-v3"]="gymma:50"
    ["2s3z"]="smaclite:150"
)

# Function to run the command
run_experiment() {
    local config=$1
    local env_config=$2
    local env_key=$3
    local time_limit=$4
    local seed=$5
    taskset -c $cpu_range python src/main.py --config=$config --env-config=$env_config with env_args.time_limit=$time_limit env_args.key=$env_key opponent_modelling=True opponent_model_decode_observations=False opponent_model_decode_actions=True latent_dims=32 seed=$seed &
    echo "Running with $config and $env_key for seed=$seed on CPUs $cpu_range"
}

# Run commands for all environments and seeds
for env_key in "${!env_configs[@]}"
do
    IFS=":" read -r env_config time_limit <<< "${env_configs[$env_key]}"
    for seed in {0..3}
    do
        run_experiment "mappo" $env_config $env_key $time_limit $seed
    done
done

wait