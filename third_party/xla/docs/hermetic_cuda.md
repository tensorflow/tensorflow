# Hermetic CUDA, CUDNN, NCCL and NVSHMEM overview

The overview and usage examples are given in [rules_ml_toolchain project](https://github.com/google-ml-infra/rules_ml_toolchain/blob/main/third_party/gpus/hermetic_toolkits.md).

## DEPRECATED: Non-hermetic CUDA/CUDNN usage
Though non-hermetic CUDA/CUDNN usage is deprecated, it might be used for
some experiments currently unsupported officially (for example, building wheels
on Windows with CUDA).

Here are the steps to use non-hermetic CUDA installed locally in Google ML
projects:

1. Delete calls to hermetic CUDA repository rules from the `WORKSPACE`
   file of the project dependent on XLA.

2. Add the calls to non-hermetic CUDA repository rules to the bottom of the
   `WORKSPACE` file.

   For XLA and JAX:
   ```
   load("@local_xla//third_party/gpus:cuda_configure.bzl", "cuda_configure")
   cuda_configure(name = "local_config_cuda")
   load("@local_xla//third_party/nccl:nccl_configure.bzl", "nccl_configure")
   nccl_configure(name = "local_config_nccl")
   ```

   For Tensorflow:
   ```
   load("@local_xla//third_party/gpus:cuda_configure.bzl", "cuda_configure")
   cuda_configure(name = "local_config_cuda")
   load("@local_xla//third_party/nccl:nccl_configure.bzl", "nccl_configure")
   nccl_configure(name = "local_config_nccl")
   ```

3. Set the following environment variables directly in your shell or in
   `.bazelrc` file as shown below:
   ```
   build:cuda --action_env=TF_CUDA_VERSION=<locally installed cuda version>
   build:cuda --action_env=TF_CUDNN_VERSION=<locally installed cudnn version>
   build:cuda --action_env=TF_CUDA_COMPUTE_CAPABILITIES=<CUDA compute capabilities>
   build:cuda --action_env=LD_LIBRARY_PATH=<CUDA/CUDNN libraries folder locations divided by “:” sign>
   build:cuda --action_env=CUDA_TOOLKIT_PATH=<preinstalled CUDA folder location>
   build:cuda --action_env=TF_CUDA_PATHS=<preinstalled CUDA/CUDNN folder locations divided by “,” sign>
   build:cuda --action_env=NCCL_INSTALL_PATH=<preinstalled NCCL library folder location>
   ```

   Note that `TF_CUDA_VERSION` and `TF_CUDNN_VERSION` should consist of major and
   minor versions only (e.g. `12.3` for CUDA and `9.1` for CUDNN).

4. Now you can run `bazel` command to use locally installed CUDA and CUDNN.

   For XLA, no changes in the command options are needed.

   For JAX, use `--override_repository=tsl=<tsl_path>` flag in the Bazel command
   options.

   For Tensorflow, use `--override_repository=local_tsl=<tsl_path>` flag in the
   Bazel command options.
