# Multi-Host HLO Runner

[TOC]

This tool lets you run an HLO module on one or more GPUs.

## Running multi-GPU (sharded) HLOs

We can identify these HLOs by seeing `sharding=` annotations. For example
`sharding={devices=[1,1,2,1]0,1}` means that the annotated tensor should be
sharded to 2 GPUs (GPU0 and GPU1) along the 3rd dimension.

The following instructions assume the working directory is the Tensorflow Git
repository and that it had been ./configure'd.

If we have enough GPUs, we can replay these HLOs like this:

```
bazel run -c opt --config=cuda --dynamic_mode=off \
  //tensorflow/compiler/xla/tools/multihost_hlo_runner:hlo_runner_main \
  -- --device_type=gpu --use_spmd_partitioning=true \
  --num_partitions=2 --num_replicas=1 \
  --hlo_file=my-hlo.txt
```

Tip: If the input generation takes too long, we can use
--hlo_argument_mode=use_zeros_as_input.

### Troubleshooting
- Errors such as `Check failed: result.replicas >= 1 (0 vs. 1)`:
  -   We have to make sure that we have enough GPUs.
  -   `CUDA_VISIBLE_DEVICES` must be set correctly or not set at all.
-   Crashes:
    -   We may want to use `--dynamic_mode=off`.
    -   CUDA and Cudnn should be set up correctly.