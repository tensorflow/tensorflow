# Multi-Host HLO Runner

[TOC]

This tool lets you run an HLO module on one or more GPUs. It also allows
compiling code targeting multiple GPUs without running it.

## Running multi-GPU (sharded) HLOs

We can identify these HLOs by seeing `sharding=` annotations. For example
`sharding={devices=[1,1,2,1]0,1}` means that the annotated tensor should be
sharded to 2 GPUs (GPU0 and GPU1) along the 3rd dimension.

The following instructions assume the working directory is the xla Git
repository and that `./configure.py` has been run.

If we have enough GPUs, we can replay these HLOs like this:

```
bazel run -c opt --config=cuda --dynamic_mode=off \
  //xla/tools/multihost_hlo_runner:hlo_runner_main -- my-hlo.txt
```

Tip: If the input generation takes too long or uses too much host memory,
consider using `--hlo_argument_mode=uninitialized`.

It is also possible to compile the same HLO without running it by setting
`--run=false`

```
bazel run -c opt --config=cuda --dynamic_mode=off \
  //xla/tools/multihost_hlo_runner:hlo_runner_main \
  -- --run=false my-hlo.txt
```

In that case, a single GPU is necessary, unless the
[autotuning cache](./persisted_autotuning) is used.

### Troubleshooting

-   Errors such as `Check failed: result.replicas >= 1 (0 vs. 1)`:
    -   We have to make sure that we have enough GPUs.
    -   `CUDA_VISIBLE_DEVICES` must be set correctly or not set at all.
-   Crashes:
    -   We may want to use `--dynamic_mode=off`.
    -   CUDA and cuDNN should be set up correctly.

# Examples

## Single process, multiple GPU example

### Setup and get the HLO

``` {note}
You can use a container with the following instructions:

  docker run -it --shm-size=1g --gpus all ghcr.io/nvidia/jax:pax-2024-06-03
  cd /opt/xla/

Note, those instructions can be outdated more quickly. Adjust as needed.
```

```
# The 8 below is the number of GPUs you have.
# test-pax.sh --help for more details on the parallelization options
(export XLA_FLAGS="--xla_dump_to=/tmp/dump --xla_dump_hlo_as_text"; test-pax.sh --fsdp 8 --batch-per-gpu 1)

ls -lSh /tmp/dump/*before_optimizations.txt
# The biggest file one is normally the one you care about.
# I picked one, for the rest of the scripts, but the name could change when you change the JAX or XLA version.
```

### Build XLA multinode runner

```
cd /opt/xla/
./configure.py --backend CUDA --nccl
bazel build -c opt --config=cuda --dynamic_mode=off //xla/tools/multihost_hlo_runner:hlo_runner_main
```

### Single process example: Before optimization graph replay

```
bazel run -c opt --config=cuda --dynamic_mode=off //xla/tools/multihost_hlo_runner:hlo_runner_main -- /tmp/dump/module_0023.pjit__wrapped_step_fn.before_optimizations.txt
```

### Single process example: After optimization graph replay

To replay an optimized HLO, you must use those two parameters
`--run_xla_backend_only=true --xla_disable_all_hlo_passes=true`. Otherwise, it
will try to recompile the HLO and this isn't supported. So it will give you many
strange errors.

Full command: `bazel run -c opt --config=cuda --dynamic_mode=off
//xla/tools/multihost_hlo_runner:hlo_runner_main -- --run_xla_backend_only=true
--xla_disable_all_hlo_passes=true
/tmp/dump/module_0023.pjit__wrapped_step_fn.sm_8.0_gpu_after_optimizations.txt`

## Multi-processes, single-node

### Launch container

Also install some missing librairies. (Note, that can be outdated more quickly.
Adjust as needed.)

```
docker run -it --shm-size=1g --gpus all ghcr.io/nvidia/jax:pax-2024-06-03
apt-get update && apt-get install -y openmpi-bin openmpi-common libopenmpi-dev
```

### Run original model and dump HLO.

For this example, we will use an 8-GPU PAXML model from `test-pax.sh`. (Note
this will be the same dump as the single process case. So you can do `cp -r
/tmp/dump /tmp/dump_multi_process` if you already have it. `export
XLA_FLAGS="--xla_dump_to=/tmp/dump_multi_process --xla_dump_hlo_as_text" mpirun
--allow-run-as-root -np 8 test-pax.sh --fsdp 8 --batch-per-gpu 1 -o
/tmp/checkpoint --multiprocess`

The HLO dump will be saved to `/tmp/dump_multi_process/`. For PAX specifically,
the main module will have "pjit__wrapped_step_fn" in the name. For this example
we will use
`/tmp/dump_multi_process/module_0023.pjit__wrapped_step_fn.before_optimizations.txt`.

### Run on a single node using MPI

Create a bash script called `run.sh`:

```
#!/bin/bash
export CUDA_VISIBLE_DEVICES=${OMPI_COMM_WORLD_LOCAL_RANK}
bazel run -c opt --config=cuda --dynamic_mode=off //xla/tools/multihost_hlo_runner:hlo_runner_main -- \
  --task_id=${OMPI_COMM_WORLD_RANK} \
  --num_nodes=${OMPI_COMM_WORLD_SIZE} \
  --address=127.0.0.1:12345 \
  /tmp/dump_multi_process/module_0023.pjit__wrapped_step_fn.before_optimizations.txt
```

Now, you can execute it using mpirun:

```
chmod a+x run.sh
mpirun --allow-run-as-root -np 8 run.sh
```

### Run on multiple nodes with SLURM

When running on multiple nodes using SLURM, you can forward the SLURM env
variables to the hlo runner like so in your slurm job:

```
bazel run -c opt --config=cuda --dynamic_mode=off //xla/tools/multihost_hlo_runner:hlo_runner_main -- \
  --task_id=${SLURM_PROCID} \
  --num_nodes=${SLURM_NTASKS} \
  --address="${SLURM_LAUNCH_NODE_IPADDR}:12345" \
  /tmp/dump_multi_process/module_0023.pjit__wrapped_step_fn.before_optimizations.txt
```
