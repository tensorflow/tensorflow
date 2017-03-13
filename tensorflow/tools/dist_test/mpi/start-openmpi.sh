#!/usr/bin/env bash

ps_hosts="localhost:12222"
worker_hosts="localhost:12223,localhost:12224,localhost:12225"
arguments=" ../python/mnist_replica.py --train_steps 10 --num_gpus 0  --num_parameter_servers=1 " 

#(Optional) Disable GPUs 
export CUDA_VISIBLE_DEVICES=""

mpirun \
      -np 1 python  ${arguments}  --task_index=0 --job_name=worker  --ps_hosts=$ps_hosts --worker_hosts=$worker_hosts  :\
      -np 1 python  ${arguments}  --task_index=1 --job_name=worker  --ps_hosts=$ps_hosts --worker_hosts=$worker_hosts  :\
      -np 1 python  ${arguments}  --task_index=2 --job_name=worker  --ps_hosts=$ps_hosts --worker_hosts=$worker_hosts  :\
      -np 1 python  ${arguments}  --task_index=0 --job_name=ps      --ps_hosts=$ps_hosts --worker_hosts=$worker_hosts \
                               2>&1 | tee log.txt 


