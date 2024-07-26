#!/bin/bash

envs=(3m 8m 2s3z 3s5z 1c3s5z)

for e in "${envs[@]}"
do
   for i in {0..9}
   do
      python src/main.py --config=mappo --env-config=sc2 with env_args.map_name=$e seed=$i &
      echo "Running with mappo and $e for seed=$i"
   done
done

wait
