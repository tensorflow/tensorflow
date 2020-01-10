"""Repository rule for NCCL configuration.

`nccl_configure` depends on the following environment variables:

  * `TF_NCCL_VERSION`: Installed NCCL version or empty to build from source.
  * `NCCL_INSTALL_PATH` (deprecated): The installation path of the NCCL library.
  * `NCCL_HDR_PATH` (deprecated): The installation path of the NCCL header 
    files.
  * `TF_CUDA_PATHS`: The base paths to look for CUDA and cuDNN. Default is
    `/usr/local/cuda,usr/`.

"""

load(
    "//third_party/gpus:cuda_configure.bzl",
    "compute_capabilities",
    "enable_cuda",
    "find_cuda_config",
    "get_cpu_value",
)

_CUDA_TOOLKIT_PATH = "CUDA_TOOLKIT_PATH"
_NCCL_HDR_PATH = "NCCL_HDR_PATH"
_NCCL_INSTALL_PATH = "NCCL_INSTALL_PATH"
_TF_CUDA_COMPUTE_CAPABILITIES = "TF_CUDA_COMPUTE_CAPABILITIES"
_TF_NCCL_VERSION = "TF_NCCL_VERSION"
_TF_NEED_CUDA = "TF_NEED_CUDA"

_DEFINE_NCCL_MAJOR = "#define NCCL_MAJOR"
_DEFINE_NCCL_MINOR = "#define NCCL_MINOR"
_DEFINE_NCCL_PATCH = "#define NCCL_PATCH"

_NCCL_DUMMY_BUILD_CONTENT = """
filegroup(
  name = "LICENSE",
  visibility = ["//visibility:public"],
)

cc_library(
  name = "nccl",
  visibility = ["//visibility:public"],
)
"""

_NCCL_ARCHIVE_BUILD_CONTENT = """
filegroup(
  name = "LICENSE",
  data = ["@nccl_archive//:LICENSE.txt"],
  visibility = ["//visibility:public"],
)

alias(
  name = "nccl",
  actual = "@nccl_archive//:nccl",
  visibility = ["//visibility:public"],
)
"""

def _label(file):
    return Label("//third_party/nccl:{}".format(file))

def _nccl_configure_impl(repository_ctx):
    """Implementation of the nccl_configure repository rule."""
    if (not enable_cuda(repository_ctx) or
        get_cpu_value(repository_ctx) not in ("Linux", "FreeBSD")):
        # Add a dummy build file to make bazel query happy.
        repository_ctx.file("BUILD", _NCCL_DUMMY_BUILD_CONTENT)
        return

    nccl_version = ""
    if _TF_NCCL_VERSION in repository_ctx.os.environ:
        nccl_version = repository_ctx.os.environ[_TF_NCCL_VERSION].strip()
        nccl_version = nccl_version.split(".")[0]

    cuda_config = find_cuda_config(repository_ctx, ["cuda"])
    cuda_version = cuda_config["cuda_version"].split(".")
    cuda_major = cuda_version[0]
    cuda_minor = cuda_version[1]

    if nccl_version == "":
        # Alias to open source build from @nccl_archive.
        repository_ctx.file("BUILD", _NCCL_ARCHIVE_BUILD_CONTENT)

        # TODO(csigg): implement and reuse in cuda_configure.bzl.
        gpu_architectures = [
            "sm_" + capability.replace(".", "")
            for capability in compute_capabilities(repository_ctx)
        ]

        # Round-about way to make the list unique.
        gpu_architectures = dict(zip(gpu_architectures, gpu_architectures)).keys()
        config_wrap = {
            "%{gpu_architectures}": str(gpu_architectures),
            "%{use_bin2c_path}": "False",
        }
        if (int(cuda_major), int(cuda_minor)) <= (10, 1):
            config_wrap["%{use_bin2c_path}"] = "True"

        repository_ctx.template(
            "build_defs.bzl",
            _label("build_defs.bzl.tpl"),
            config_wrap,
        )
    else:
        # Create target for locally installed NCCL.
        config = find_cuda_config(repository_ctx, ["nccl"])
        config_wrap = {
            "%{nccl_version}": config["nccl_version"],
            "%{nccl_header_dir}": config["nccl_include_dir"],
            "%{nccl_library_dir}": config["nccl_library_dir"],
        }
        repository_ctx.template("BUILD", _label("system.BUILD.tpl"), config_wrap)

nccl_configure = repository_rule(
    implementation = _nccl_configure_impl,
    environ = [
        _CUDA_TOOLKIT_PATH,
        _NCCL_HDR_PATH,
        _NCCL_INSTALL_PATH,
        _TF_NCCL_VERSION,
        _TF_CUDA_COMPUTE_CAPABILITIES,
        _TF_NEED_CUDA,
        "TF_CUDA_PATHS",
    ],
)
"""Detects and configures the NCCL configuration.

Add the following to your WORKSPACE FILE:

```python
nccl_configure(name = "local_config_nccl")
```

Args:
  name: A unique name for this workspace rule.
"""
