#!/bin/bash
#SBATCH --job-name=env_tasks
#SBATCH --output=output_%j.txt
#SBATCH --error=error_%j.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --partition=standard

source /home/rsan/anaconda3/etc/profile.d/conda.sh
conda activate am

# Define environments and their specific configurations
declare -A env_configs=(
    ["pz-mpe-simple-spread-v3"]="gymma:25"
    ["lbforaging:Foraging-10x10-3p-3f-coop-v3"]="gymma:50"
    ["lbforaging:Foraging-10x10-3p-3f-v3"]="gymma:50"
    ["lbforaging:Foraging-2s-10x10-3p-3f-coop-v3"]="gymma:50"
    ["2s3z"]="smaclite:150"
)

# Function to run the command
run_command() {
    local config=$1
    local env_config=$2
    local env_key=$3
    local time_limit=$4
    local seed=$5
    srun --exclusive -c1 python src/main.py --config=$config --env-config=$env_config with env_args.time_limit=$time_limit env_args.key=$env_key opponent_modelling=True opponent_model_decode_observations=False opponent_model_decode_actions=True latent_dims=32 seed=$seed &
    echo "Running with $config and $env_key for seed=$seed"
}

# Export function to be used by parallel
export -f run_command

# Run commands for all environments and seeds
for env_key in "${!env_configs[@]}"
do
    IFS=":" read -r env_config time_limit <<< "${env_configs[$env_key]}"
    for seed in {0..3}
    do
        run_command "mappo" $env_config $env_key $time_limit $seed
    done
done

wait
