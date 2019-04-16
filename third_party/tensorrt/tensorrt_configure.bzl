# -*- Python -*-
"""Repository rule for TensorRT configuration.

`tensorrt_configure` depends on the following environment variables:

  * `TF_TENSORRT_VERSION`: The TensorRT libnvinfer version.
  * `TENSORRT_INSTALL_PATH`: The installation path of the TensorRT library.
"""

load(
    "//third_party/gpus:cuda_configure.bzl",
    "find_cuda_config",
    "get_cpu_value",
    "lib_name",
    "make_copy_files_rule",
)

_TENSORRT_INSTALL_PATH = "TENSORRT_INSTALL_PATH"
_TF_TENSORRT_CONFIG_REPO = "TF_TENSORRT_CONFIG_REPO"
_TF_TENSORRT_VERSION = "TF_TENSORRT_VERSION"

_TF_TENSORRT_LIBS = ["nvinfer", "nvinfer_plugin"]
_TF_TENSORRT_HEADERS = ["NvInfer.h", "NvUtils.h", "NvInferPlugin.h"]

_DEFINE_TENSORRT_SONAME_MAJOR = "#define NV_TENSORRT_SONAME_MAJOR"
_DEFINE_TENSORRT_SONAME_MINOR = "#define NV_TENSORRT_SONAME_MINOR"
_DEFINE_TENSORRT_SONAME_PATCH = "#define NV_TENSORRT_SONAME_PATCH"

def _tpl(repository_ctx, tpl, substitutions):
    repository_ctx.template(
        tpl,
        Label("//third_party/tensorrt:%s.tpl" % tpl),
        substitutions,
    )

def _create_dummy_repository(repository_ctx):
    """Create a dummy TensorRT repository."""
    _tpl(repository_ctx, "build_defs.bzl", {"%{if_tensorrt}": "if_false"})

    _tpl(repository_ctx, "BUILD", {
        "%{tensorrt_genrules}": "",
        "%{tensorrt_headers}": "[]",
        "%{tensorrt_libs}": "[]",
    })

def _tensorrt_configure_impl(repository_ctx):
    """Implementation of the tensorrt_configure repository rule."""
    if _TF_TENSORRT_CONFIG_REPO in repository_ctx.os.environ:
        # Forward to the pre-configured remote repository.
        remote_config_repo = repository_ctx.os.environ[_TF_TENSORRT_CONFIG_REPO]
        repository_ctx.template("BUILD", Label(remote_config_repo + ":BUILD"), {})
        repository_ctx.template(
            "build_defs.bzl",
            Label(remote_config_repo + ":build_defs.bzl"),
            {},
        )
        return

    if _TF_TENSORRT_VERSION not in repository_ctx.os.environ:
        _create_dummy_repository(repository_ctx)
        return

    config = find_cuda_config(repository_ctx, ["tensorrt"])
    trt_version = config["tensorrt_version"]
    cpu_value = get_cpu_value(repository_ctx)

    # Copy the library and header files.
    libraries = [lib_name(lib, cpu_value, trt_version) for lib in _TF_TENSORRT_LIBS]
    library_dir = config["tensorrt_library_dir"] + "/"
    headers = _TF_TENSORRT_HEADERS
    include_dir = config["tensorrt_include_dir"] + "/"
    copy_rules = [
        make_copy_files_rule(
            repository_ctx,
            name = "tensorrt_lib",
            srcs = [library_dir + library for library in libraries],
            outs = ["tensorrt/lib/" + library for library in libraries],
        ),
        make_copy_files_rule(
            repository_ctx,
            name = "tensorrt_include",
            srcs = [include_dir + header for header in headers],
            outs = ["tensorrt/incude/" + header for header in headers],
        ),
    ]

    # Set up config file.
    _tpl(repository_ctx, "build_defs.bzl", {"%{if_tensorrt}": "if_true"})

    # Set up BUILD file.
    _tpl(repository_ctx, "BUILD", {
        "%{copy_rules}": "\n".join(copy_rules),
        "%{tensorrt_libs}": str(libraries),
    })

tensorrt_configure = repository_rule(
    implementation = _tensorrt_configure_impl,
    environ = [
        _TENSORRT_INSTALL_PATH,
        _TF_TENSORRT_VERSION,
        _TF_TENSORRT_CONFIG_REPO,
        "TF_CUDA_PATHS",
    ],
)
"""Detects and configures the local CUDA toolchain.

Add the following to your WORKSPACE FILE:

```python
tensorrt_configure(name = "local_config_tensorrt")
```

Args:
  name: A unique name for this workspace rule.
"""
