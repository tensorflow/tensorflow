"""Repository rule for TensorRT configuration.

`tensorrt_configure` depends on the following environment variables:

  * `TF_TENSORRT_VERSION`: The TensorRT libnvinfer version.
  * `TENSORRT_INSTALL_PATH`: The installation path of the TensorRT library.
"""

load(
    "//third_party/gpus:cuda_configure.bzl",
    "find_cuda_config",
    "lib_name",
    "make_copy_files_rule",
)
load(
    "//third_party/remote_config:common.bzl",
    "config_repo_label",
    "get_cpu_value",
    "get_host_environ",
)

_TENSORRT_INSTALL_PATH = "TENSORRT_INSTALL_PATH"
_TF_TENSORRT_CONFIG_REPO = "TF_TENSORRT_CONFIG_REPO"
_TF_TENSORRT_VERSION = "TF_TENSORRT_VERSION"
_TF_NEED_TENSORRT = "TF_NEED_TENSORRT"

_TF_TENSORRT_LIBS = ["nvinfer", "nvinfer_plugin"]
_TF_TENSORRT_HEADERS = ["NvInfer.h", "NvUtils.h", "NvInferPlugin.h"]
_TF_TENSORRT_HEADERS_V6 = [
    "NvInfer.h",
    "NvUtils.h",
    "NvInferPlugin.h",
    "NvInferVersion.h",
    "NvInferRuntime.h",
    "NvInferRuntimeCommon.h",
    "NvInferPluginUtils.h",
]

_DEFINE_TENSORRT_SONAME_MAJOR = "#define NV_TENSORRT_SONAME_MAJOR"
_DEFINE_TENSORRT_SONAME_MINOR = "#define NV_TENSORRT_SONAME_MINOR"
_DEFINE_TENSORRT_SONAME_PATCH = "#define NV_TENSORRT_SONAME_PATCH"

def _at_least_version(actual_version, required_version):
    actual = [int(v) for v in actual_version.split(".")]
    required = [int(v) for v in required_version.split(".")]
    return actual >= required

def _get_tensorrt_headers(tensorrt_version):
    if _at_least_version(tensorrt_version, "6"):
        return _TF_TENSORRT_HEADERS_V6
    return _TF_TENSORRT_HEADERS

def _tpl_path(repository_ctx, filename):
    return repository_ctx.path(Label("//third_party/tensorrt:%s.tpl" % filename))

def _tpl(repository_ctx, tpl, substitutions):
    repository_ctx.template(
        tpl,
        _tpl_path(repository_ctx, tpl),
        substitutions,
    )

def _create_dummy_repository(repository_ctx):
    """Create a dummy TensorRT repository."""
    _tpl(repository_ctx, "build_defs.bzl", {"%{if_tensorrt}": "if_false"})
    _tpl(repository_ctx, "BUILD", {
        "%{copy_rules}": "",
        "\":tensorrt_include\"": "",
        "\":tensorrt_lib\"": "",
    })
    _tpl(repository_ctx, "tensorrt/include/tensorrt_config.h", {
        "%{tensorrt_version}": "",
    })

    # Copy license file in non-remote build.
    repository_ctx.template(
        "LICENSE",
        Label("//third_party/tensorrt:LICENSE"),
        {},
    )

def enable_tensorrt(repository_ctx):
    """Returns whether to build with TensorRT support."""
    return int(get_host_environ(repository_ctx, _TF_NEED_TENSORRT, False))

def _create_local_tensorrt_repository(repository_ctx):
    # Resolve all labels before doing any real work. Resolving causes the
    # function to be restarted with all previous state being lost. This
    # can easily lead to a O(n^2) runtime in the number of labels.
    # See https://github.com/tensorflow/tensorflow/commit/62bd3534525a036f07d9851b3199d68212904778
    find_cuda_config_path = repository_ctx.path(Label("@org_tensorflow//third_party/gpus:find_cuda_config.py.gz.base64"))
    tpl_paths = {
        "build_defs.bzl": _tpl_path(repository_ctx, "build_defs.bzl"),
        "BUILD": _tpl_path(repository_ctx, "BUILD"),
        "tensorrt/include/tensorrt_config.h": _tpl_path(repository_ctx, "tensorrt/include/tensorrt_config.h"),
    }

    config = find_cuda_config(repository_ctx, find_cuda_config_path, ["tensorrt"])
    trt_version = config["tensorrt_version"]
    cpu_value = get_cpu_value(repository_ctx)

    # Copy the library and header files.
    libraries = [lib_name(lib, cpu_value, trt_version) for lib in _TF_TENSORRT_LIBS]
    library_dir = config["tensorrt_library_dir"] + "/"
    headers = _get_tensorrt_headers(trt_version)
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
            outs = ["tensorrt/include/" + header for header in headers],
        ),
    ]

    # Set up config file.
    repository_ctx.template(
        "build_defs.bzl",
        tpl_paths["build_defs.bzl"],
        {"%{if_tensorrt}": "if_true"},
    )

    # Set up BUILD file.
    repository_ctx.template(
        "BUILD",
        tpl_paths["BUILD"],
        {"%{copy_rules}": "\n".join(copy_rules)},
    )

    # Copy license file in non-remote build.
    repository_ctx.template(
        "LICENSE",
        Label("//third_party/tensorrt:LICENSE"),
        {},
    )

    # Set up tensorrt_config.h, which is used by
    # tensorflow/stream_executor/dso_loader.cc.
    repository_ctx.template(
        "tensorrt/include/tensorrt_config.h",
        tpl_paths["tensorrt/include/tensorrt_config.h"],
        {"%{tensorrt_version}": trt_version},
    )

def _tensorrt_configure_impl(repository_ctx):
    """Implementation of the tensorrt_configure repository rule."""

    if get_host_environ(repository_ctx, _TF_TENSORRT_CONFIG_REPO) != None:
        # Forward to the pre-configured remote repository.
        remote_config_repo = repository_ctx.os.environ[_TF_TENSORRT_CONFIG_REPO]
        repository_ctx.template("BUILD", config_repo_label(remote_config_repo, ":BUILD"), {})
        repository_ctx.template(
            "build_defs.bzl",
            config_repo_label(remote_config_repo, ":build_defs.bzl"),
            {},
        )
        repository_ctx.template(
            "tensorrt/include/tensorrt_config.h",
            config_repo_label(remote_config_repo, ":tensorrt/include/tensorrt_config.h"),
            {},
        )
        repository_ctx.template(
            "LICENSE",
            config_repo_label(remote_config_repo, ":LICENSE"),
            {},
        )
        return

    if not enable_tensorrt(repository_ctx):
        _create_dummy_repository(repository_ctx)
        return

    _create_local_tensorrt_repository(repository_ctx)

_ENVIRONS = [
    _TENSORRT_INSTALL_PATH,
    _TF_TENSORRT_VERSION,
    _TF_NEED_TENSORRT,
    "TF_CUDA_PATHS",
]

remote_tensorrt_configure = repository_rule(
    implementation = _create_local_tensorrt_repository,
    environ = _ENVIRONS,
    remotable = True,
    attrs = {
        "environ": attr.string_dict(),
    },
)

tensorrt_configure = repository_rule(
    implementation = _tensorrt_configure_impl,
    environ = _ENVIRONS + [_TF_TENSORRT_CONFIG_REPO],
)
"""Detects and configures the local CUDA toolchain.

Add the following to your WORKSPACE FILE:

```python
tensorrt_configure(name = "local_config_tensorrt")
```

Args:
  name: A unique name for this workspace rule.
"""
