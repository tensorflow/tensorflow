# Hermetic CUDA overview

Hermetic CUDA uses a specific downloadable version of CUDA instead of the user’s
locally installed CUDA. Bazel will download CUDA, CUDNN and NCCL distributions,
and then use CUDA libraries and tools as dependencies in various Bazel targets.
This enables more reproducible builds for Google ML projects and supported CUDA
versions.

## Supported hermetic CUDA, CUDNN versions

The supported CUDA versions are specified in `CUDA_REDIST_JSON_DICT`
dictionary,
[third_party/gpus/cuda/hermetic/cuda_redist_versions.bzl](https://github.com/openxla/xla/blob/main/third_party/gpus/cuda/hermetic/cuda_redist_versions.bzl).

The supported CUDNN versions are specified in `CUDNN_REDIST_JSON_DICT`
dictionary,
[third_party/gpus/cuda/hermetic/cuda_redist_versions.bzl](https://github.com/openxla/xla/blob/main/third_party/gpus/cuda/hermetic/cuda_redist_versions.bzl).

The `.bazelrc` files of individual projects have `HERMETIC_CUDA_VERSION`,
`HERMETIC_CUDNN_VERSION` environment variables set to the versions used by
default when `--config=cuda` is specified in Bazel command options.

## Environment variables controlling the hermetic CUDA/CUDNN versions

`HERMETIC_CUDA_VERSION` environment variable should consist of major, minor and
patch CUDA version, e.g. `12.6.3`.
`HERMETIC_CUDNN_VERSION` environment variable should consist of major, minor and
patch CUDNN version, e.g. `9.3.0`.

Three ways to set the environment variables for Bazel commands:

```
# Add an entry to your `.bazelrc` file
build:cuda --repo_env=HERMETIC_CUDA_VERSION="12.6.3"
build:cuda --repo_env=HERMETIC_CUDNN_VERSION="9.3.0"

# OR pass it directly to your specific build command
bazel build --config=cuda <target> \
--repo_env=HERMETIC_CUDA_VERSION="12.6.3" \
--repo_env=HERMETIC_CUDNN_VERSION="9.3.0"

# If .bazelrc doesn't have corresponding entries and the environment variables
# are not passed to bazel command, you can set them globally in your shell:
export HERMETIC_CUDA_VERSION="12.6.3"
export HERMETIC_CUDNN_VERSION="9.3.0"
```

If `HERMETIC_CUDA_VERSION` and `HERMETIC_CUDNN_VERSION` are not present, the
hermetic CUDA/CUDNN repository rules will look up `TF_CUDA_VERSION` and
`TF_CUDNN_VERSION` environment variables values. This is made for the backward
compatibility with non-hermetic CUDA/CUDNN repository rules.

The mapping between CUDA version and NCCL distribution version to be downloaded
is specified in [third_party/gpus/cuda/hermetic/cuda_redist_versions.bzl](https://github.com/openxla/xla/blob/main/third_party/gpus/cuda/hermetic/cuda_redist_versions.bzl)

## Configure hermetic CUDA

1. In the downstream project dependent on XLA, add the following lines to the
   bottom of the `WORKSPACE` file:

   Note: use @local_tsl instead of @tsl in Tensorflow project.

   ```
   load(
      "@local_xla//third_party/gpus/cuda/hermetic:cuda_json_init_repository.bzl",
      "cuda_json_init_repository",
   )

   cuda_json_init_repository()

   load(
      "@cuda_redist_json//:distributions.bzl",
      "CUDA_REDISTRIBUTIONS",
      "CUDNN_REDISTRIBUTIONS",
   )
   load(
      "@local_xla//third_party/gpus/cuda/hermetic:cuda_redist_init_repositories.bzl",
      "cuda_redist_init_repositories",
      "cudnn_redist_init_repository",
   )

   cuda_redist_init_repositories(
      cuda_redistributions = CUDA_REDISTRIBUTIONS,
   )

   cudnn_redist_init_repository(
      cudnn_redistributions = CUDNN_REDISTRIBUTIONS,
   )

   load(
      "@local_xla//third_party/gpus/cuda/hermetic:cuda_configure.bzl",
      "cuda_configure",
   )

   cuda_configure(name = "local_config_cuda")

   load(
      "@local_xla//third_party/nccl/hermetic:nccl_redist_init_repository.bzl",
      "nccl_redist_init_repository",
   )

   nccl_redist_init_repository()

   load(
      "@local_xla//third_party/nccl/hermetic:nccl_configure.bzl",
      "nccl_configure",
   )

   nccl_configure(name = "local_config_nccl")
   ```

2. To select specific versions of hermetic CUDA and CUDNN, set the
   `HERMETIC_CUDA_VERSION` and `HERMETIC_CUDNN_VERSION` environment variables
   respectively. Use only supported versions. You may set the environment
   variables directly in your shell or in `.bazelrc` file as shown below:
   ```
   build:cuda --repo_env=HERMETIC_CUDA_VERSION="12.6.3"
   build:cuda --repo_env=HERMETIC_CUDNN_VERSION="9.3.0"
   build:cuda --repo_env=HERMETIC_CUDA_COMPUTE_CAPABILITIES="sm_50,sm_60,sm_70,sm_80,compute_90"
   ```

3. To enable hermetic CUDA during test execution, or when running a binary via
   bazel, make sure to add `--@local_config_cuda//cuda:include_cuda_libs=true`
   flag to your bazel command. You can provide it either directly in a shell or
   in `.bazelrc`:
   ```
   build:cuda --@local_config_cuda//cuda:include_cuda_libs=true
   ```
   The flag is needed to make sure that CUDA dependencies are properly provided
   to test executables. The flag is false by default to avoid unwanted coupling
   of Google-released Python wheels to CUDA binaries.

4. To enforce CUDA forward compatibility mode, add
   `--@cuda_driver//:enable_forward_compatibility=true` flag to your bazel
   command. You can provide it either directly in a shell or in `.bazelrc`:
   ```
   test:cuda --@cuda_driver//:enable_forward_compatibility=true
   ```

   The default flag value is `false`.

   When CUDA forward compatibility mode is disabled, Bazel targets will use User
   Mode and Kernel Mode Drivers pre-installed on the system.

   When CUDA forward compatibility mode is enabled, Bazel targets will use User
   Mode Driver from CUDA driver redistribution downloaded into Bazel cache and
   Kernel Mode Driver pre-installed on the system. It allows enabling new CUDA
   Toolkit features while using older Kernel Mode Driver.

   Forward compatibility mode should be enforced only when it is appropriate -
   see [NVIDIA documentation](https://docs.nvidia.com/deploy/cuda-compatibility/index.html#forward-compatibility-support-across-major-toolkit-versions) for the
   details.

## Upgrade hermetic CUDA/CUDNN version

1.  Create and submit a pull request with updated `CUDA_REDIST_JSON_DICT`,
    `CUDNN_REDIST_JSON_DICT` dictionaries in
    [third_party/gpus/cuda/hermetic/cuda_redist_versions.bzl](https://github.com/openxla/xla/blob/main/third_party/gpus/cuda/hermetic/cuda_redist_versions.bzl).

    Update `CUDA_NCCL_WHEELS` in
    [third_party/gpus/cuda/hermetic/cuda_redist_versions.bzl](https://github.com/openxla/xla/blob/main/third_party/gpus/cuda/hermetic/cuda_redist_versions.bzl)
    if needed.

    Update `REDIST_VERSIONS_TO_BUILD_TEMPLATES` in
    [third_party/gpus/cuda/hermetic/cuda_redist_versions.bzl](https://github.com/openxla/xla/blob/main/third_party/gpus/cuda/hermetic/cuda_redist_versions.bzl)
    if needed.

    Update `PTX_VERSION_DICT` in
    [third_party/gpus/cuda/hermetic/cuda_redist_versions.bzl](https://github.com/openxla/xla/blob/main/third_party/gpus/cuda/hermetic/cuda_redist_versions.bzl)
    if needed.

2.  For each Google ML project create a separate pull request with updated
    `HERMETIC_CUDA_VERSION` and `HERMETIC_CUDNN_VERSION` in `.bazelrc` file.

    The PR presubmit job executions will launch bazel tests and download
    hermetic CUDA/CUDNN distributions. Verify that the presubmit jobs passed
    before submitting the PR.

3.  For the time optimization some build/test configurations utilize mirrored
    `.tar` redistributions. The `json` file with information about the mirrored
    `.tar` redistributions is uploaded some time later after
    `CUDA_REDIST_JSON_DICT` and `CUDNN_REDIST_JSON_DICT` are updated. One can
    download these files using
    `wget "https://storage.googleapis.com/mirror.tensorflow.org/developer.download.nvidia.com/compute/cuda/redist/redistrib_<cuda_version>_tar.json"`
    for `CUDA` and
    `wget "https://storage.googleapis.com/mirror.tensorflow.org/developer.download.nvidia.com/compute/cudnn/redist/redistrib_<cudnn_version>_tar.json"`
    for `CUDNN`.
    After that create and submit a pull request with updated
    `MIRRORED_TARS_CUDA_REDIST_JSON_DICT`,
    `MIRRORED_TARS_CUDNN_REDIST_JSON_DICT` dictionaries in
    [third_party/gpus/cuda/hermetic/cuda_redist_versions.bzl](https://github.com/openxla/xla/blob/main/third_party/gpus/cuda/hermetic/cuda_redist_versions.bzl).

## Pointing to CUDA/CUDNN/NCCL redistributions on local file system

You can use the local CUDA/CUDNN/NCCL dirs as a source of redistributions. The following additional environment variables are required:

```
LOCAL_CUDA_PATH
LOCAL_CUDNN_PATH
LOCAL_NCCL_PATH
```

Example:

```
# Add an entry to your `.bazelrc` file
build:cuda --repo_env=LOCAL_CUDA_PATH="/foo/bar/nvidia/cuda"
build:cuda --repo_env=LOCAL_CUDNN_PATH="/foo/bar/nvidia/cudnn"
build:cuda --repo_env=LOCAL_NCCL_PATH="/foo/bar/nvidia/nccl"

# OR pass it directly to your specific build command
bazel build --config=cuda <target> \
--repo_env=LOCAL_CUDA_PATH="/foo/bar/nvidia/cuda" \
--repo_env=LOCAL_CUDNN_PATH="/foo/bar/nvidia/cudnn" \
--repo_env=LOCAL_NCCL_PATH="/foo/bar/nvidia/nccl"

# If .bazelrc doesn't have corresponding entries and the environment variables
# are not passed to bazel command, you can set them globally in your shell:
export LOCAL_CUDA_PATH="/foo/bar/nvidia/cuda"
export LOCAL_CUDNN_PATH="/foo/bar/nvidia/cudnn"
export LOCAL_NCCL_PATH="/foo/bar/nvidia/nccl"
```

The structure of the folders inside CUDA dir should be the following (as if the archived redistributions were unpacked into one place):

```
<LOCAL_CUDA_PATH>/
    include/
    bin/
    lib/
    nvvm/
```

The structure of the folders inside CUDNN dir should be the following:

```
<LOCAL_CUDNN_PATH>
    include/
    lib/
```

The structure of the folders inside NCCL dir should be the following:

```
<LOCAL_NCCL_PATH>
    include/
    lib/
```

## Custom CUDA/CUDNN archives and NCCL wheels

There are three options that allow usage of custom CUDA/CUDNN distributions.

### Custom CUDA/CUDNN redistribution JSON files

This option allows to use custom distributions for all CUDA/CUDNN dependencies
in Google ML projects.

1. Create `cuda_redist.json` and/or `cudnn_redist.json` files.

   `cuda_redist.json` show follow the format below:

   ```
   {
      "cuda_cccl": {
         "linux-x86_64": {
            "relative_path": "cuda_cccl-linux-x86_64-12.4.99-archive.tar.xz",
         },
         "linux-sbsa": {
            "relative_path": "cuda_cccl-linux-sbsa-12.4.99-archive.tar.xz",
         }
      },
   }
   ```

   `cudnn_redist.json` show follow the format below:

   ```
   {
      "cudnn": {
         "linux-x86_64": {
            "cuda12": {
            "relative_path": "cudnn/linux-x86_64/cudnn-linux-x86_64-9.0.0.312_cuda12-archive.tar.xz",
            }
         },
         "linux-sbsa": {
            "cuda12": {
            "relative_path": "cudnn/linux-sbsa/cudnn-linux-sbsa-9.0.0.312_cuda12-archive.tar.xz",
            }
         }
      }
   }
   ```

   The `relative_path` field can be replaced with `full_path` for the full URLs
   and absolute local paths starting with `file:///`.

2. In the downstream project dependent on XLA, update the hermetic cuda JSON
   repository call in `WORKSPACE` file. Both web links and local file paths are
   allowed. Example:

   ```
   _CUDA_JSON_DICT = {
      "12.4.0": [
         "file:///home/user/Downloads/redistrib_12.4.0_updated.json",
      ],
   }

   _CUDNN_JSON_DICT = {
      "9.0.0": [
         "https://developer.download.nvidia.com/compute/cudnn/redist/redistrib_9.0.0.json",
      ],
   }

   cuda_json_init_repository(
      cuda_json_dict = _CUDA_JSON_DICT,
      cudnn_json_dict = _CUDNN_JSON_DICT,
   )
   ```

   If JSON files contain relative paths to distributions, the path prefix should
   be updated in `cuda_redist_init_repositories()` and
   `cudnn_redist_init_repository()` calls. Example

   ```
   cuda_redist_init_repositories(
      cuda_redistributions = CUDA_REDISTRIBUTIONS,
      cuda_redist_path_prefix = "file:///usr/Downloads/dists/",
   )
   ```

### Custom CUDA/CUDNN distributions

This option allows to use custom distributions for some CUDA/CUDNN dependencies
in Google ML projects.

1. In the downstream project dependent on XLA, remove the lines below:

   ```
   <...>
      "CUDA_REDIST_JSON_DICT",
   <...>
      "CUDNN_REDIST_JSON_DICT",
   <...>

   cuda_json_init_repository(
      cuda_json_dict = CUDA_REDIST_JSON_DICT,
      cudnn_json_dict = CUDNN_REDIST_JSON_DICT,
   )

   load(
      "@cuda_redist_json//:distributions.bzl",
      "CUDA_REDISTRIBUTIONS",
      "CUDNN_REDISTRIBUTIONS",
   )
   ```

2. In the same `WORKSPACE` file, create dictionaries with distribution paths.

   The dictionary with CUDA distributions show follow the format below:

   ```
   _CUSTOM_CUDA_REDISTRIBUTIONS = {
      "cuda_cccl": {
         "linux-x86_64": {
            "relative_path": "cuda_cccl-linux-x86_64-12.4.99-archive.tar.xz",
         },
         "linux-sbsa": {
            "relative_path": "cuda_cccl-linux-sbsa-12.4.99-archive.tar.xz",
         }
      },
   }
   ```

   The dictionary with CUDNN distributions show follow the format below:

   ```
   _CUSTOM_CUDNN_REDISTRIBUTIONS = {
      "cudnn": {
         "linux-x86_64": {
            "cuda12": {
            "relative_path": "cudnn/linux-x86_64/cudnn-linux-x86_64-9.0.0.312_cuda12-archive.tar.xz",
            }
         },
         "linux-sbsa": {
            "cuda12": {
            "relative_path": "cudnn/linux-sbsa/cudnn-linux-sbsa-9.0.0.312_cuda12-archive.tar.xz",
            }
         }
      }
   }
   ```

   The `relative_path` field can be replaced with `full_path` for the full URLs
   and absolute local paths starting with `file:///`.

2. In the same `WORKSPACE` file, pass the created dictionaries to the repository
   rule. If the dictionaries contain relative paths to distributions, the path
   prefix should be updated in `cuda_redist_init_repositories()` and
   `cudnn_redist_init_repository()` calls.

   ```
   cuda_redist_init_repositories(
      cuda_redistributions = _CUSTOM_CUDA_REDISTRIBUTIONS,
      cuda_redist_path_prefix = "file:///home/usr/Downloads/dists/",
   )

   cudnn_redist_init_repository(
      cudnn_redistributions = _CUSTOM_CUDNN_REDISTRIBUTIONS,
      cudnn_redist_path_prefix = "file:///home/usr/Downloads/dists/cudnn/"
   )
   ```
### Combination of the options above

In the example below, `CUDA_REDIST_JSON_DICT` is merged with custom JSON data in
`_CUDA_JSON_DICT`, and `CUDNN_REDIST_JSON_DICT` is merged with
`_CUDNN_JSON_DICT`.

The distributions data in `_CUDA_DIST_DICT` overrides the content of resulting
CUDA JSON file, and the distributions data in `_CUDNN_DIST_DICT` overrides the
content of resulting CUDNN JSON file. The NCCL wheels data is merged from
`CUDA_NCCL_WHEELS` and  `_NCCL_WHEEL_DICT`.

```
load(
    //third_party/gpus/cuda/hermetic:cuda_redist_versions.bzl",
    "CUDA_REDIST_PATH_PREFIX",
    "CUDA_NCCL_WHEELS",
    "CUDA_REDIST_JSON_DICT",
    "CUDNN_REDIST_PATH_PREFIX",
    "CUDNN_REDIST_JSON_DICT",
)

_CUDA_JSON_DICT = {
   "12.4.0": [
      "file:///usr/Downloads/redistrib_12.4.0_updated.json",
   ],
}

_CUDNN_JSON_DICT = {
   "9.0.0": [
      "https://developer.download.nvidia.com/compute/cudnn/redist/redistrib_9.0.0.json",
   ],
}

cuda_json_init_repository(
   cuda_json_dict = CUDA_REDIST_JSON_DICT | _CUDA_JSON_DICT,
   cudnn_json_dict = CUDNN_REDIST_JSON_DICT | _CUDNN_JSON_DICT,
)

load(
   "@cuda_redist_json//:distributions.bzl",
   "CUDA_REDISTRIBUTIONS",
   "CUDNN_REDISTRIBUTIONS",
)

load(
   "//third_party/gpus/cuda/hermetic:cuda_redist_init_repositories.bzl",
   "cuda_redist_init_repositories",
   "cudnn_redist_init_repository",
)

_CUDA_DIST_DICT = {
   "cuda_cccl": {
      "linux-x86_64": {
            "relative_path": "cuda_cccl-linux-x86_64-12.4.99-archive.tar.xz",
      },
      "linux-sbsa": {
            "relative_path": "cuda_cccl-linux-sbsa-12.4.99-archive.tar.xz",
      },
   },
   "libcusolver": {
      "linux-x86_64": {
            "full_path": "file:///usr/Downloads/dists/libcusolver-linux-x86_64-11.6.0.99-archive.tar.xz",
      },
      "linux-sbsa": {
         "relative_path": "libcusolver-linux-sbsa-11.6.0.99-archive.tar.xz",
      },
   },
}

_CUDNN_DIST_DICT = {
   "cudnn": {
      "linux-x86_64": {
            "cuda12": {
               "relative_path": "cudnn-linux-x86_64-9.0.0.312_cuda12-archive.tar.xz",
            },
      },
      "linux-sbsa": {
            "cuda12": {
               "relative_path": "cudnn-linux-sbsa-9.0.0.312_cuda12-archive.tar.xz",
            },
      },
   },
}

cudnn_redist_init_repositories(
   cuda_redistributions = CUDA_REDISTRIBUTIONS | _CUDA_DIST_DICT,
   cuda_redist_path_prefix = "file:///usr/Downloads/dists/",
)

cudnn_redist_init_repository(
   cudnn_redistributions = CUDNN_REDISTRIBUTIONS | _CUDNN_DIST_DICT,
   cudnn_redist_path_prefix = "file:///usr/Downloads/dists/cudnn/"
)

load(
    "//third_party/nccl/hermetic:nccl_redist_init_repository.bzl",
    "nccl_redist_init_repository",
)

_NCCL_WHEEL_DICT = {
   "12.4.0": {
      "x86_64-unknown-linux-gnu": {
            "url": "https://files.pythonhosted.org/packages/38/00/d0d4e48aef772ad5aebcf70b73028f88db6e5640b36c38e90445b7a57c45/nvidia_nccl_cu12-2.19.3-py3-none-manylinux1_x86_64.whl",
      },
   },
}

nccl_redist_init_repository(
   cuda_nccl_wheels = CUDA_NCCL_WHEELS | _NCCL_WHEEL_DICT,
)
```

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
