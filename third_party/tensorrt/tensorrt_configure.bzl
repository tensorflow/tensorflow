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
_TF_TENSORRT_STATIC_PATH = "TF_TENSORRT_STATIC_PATH"
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
_TF_TENSORRT_HEADERS_V8 = [
    "NvInfer.h",
    "NvInferLegacyDims.h",
    "NvInferImpl.h",
    "NvUtils.h",
    "NvInferPlugin.h",
    "NvInferVersion.h",
    "NvInferRuntime.h",
    "NvInferRuntimeCommon.h",
    "NvInferPluginUtils.h",
]
_TF_TENSORRT_HEADERS_V8_6 = [
    "NvInfer.h",
    "NvInferConsistency.h",
    "NvInferConsistencyImpl.h",
    "NvInferImpl.h",
    "NvInferLegacyDims.h",
    "NvInferPlugin.h",
    "NvInferPluginUtils.h",
    "NvInferRuntime.h",
    "NvInferRuntimeBase.h",
    "NvInferRuntimeCommon.h",
    "NvInferRuntimePlugin.h",
    "NvInferSafeRuntime.h",
    "NvInferVersion.h",
    "NvUtils.h",
]
_TF_TENSORRT_HEADERS_V10 = [
    "NvInfer.h",
    "NvInferConsistency.h",
    "NvInferConsistencyImpl.h",
    "NvInferImpl.h",
    "NvInferLegacyDims.h",
    "NvInferPlugin.h",
    "NvInferPluginUtils.h",
    "NvInferRuntime.h",
    "NvInferRuntimeBase.h",
    "NvInferRuntimeCommon.h",
    "NvInferRuntimePlugin.h",
    "NvInferSafeRuntime.h",
    "NvInferVersion.h",
]

_DEFINE_TENSORRT_SONAME_MAJOR = "#define NV_TENSORRT_SONAME_MAJOR"
_DEFINE_TENSORRT_SONAME_MINOR = "#define NV_TENSORRT_SONAME_MINOR"
_DEFINE_TENSORRT_SONAME_PATCH = "#define NV_TENSORRT_SONAME_PATCH"

_TENSORRT_OSS_DUMMY_BUILD_CONTENT = """
cc_library(
  name = "nvinfer_plugin_nms",
  visibility = ["//visibility:public"],
)
"""

_TENSORRT_OSS_ARCHIVE_BUILD_CONTENT = """
alias(
  name = "nvinfer_plugin_nms",
  actual = "@tensorrt_oss_archive//:nvinfer_plugin_nms",
  visibility = ["//visibility:public"],
)
"""

def _at_least_version(actual_version, required_version):
    actual = [int(v) for v in actual_version.split(".")]
    required = [int(v) for v in required_version.split(".")]
    return actual >= required

def _get_tensorrt_headers(tensorrt_version):
    if _at_least_version(tensorrt_version, "10"):
        return _TF_TENSORRT_HEADERS_V10
    if _at_least_version(tensorrt_version, "8.6"):
        return _TF_TENSORRT_HEADERS_V8_6
    if _at_least_version(tensorrt_version, "8"):
        return _TF_TENSORRT_HEADERS_V8
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
        "%{oss_rules}": _TENSORRT_OSS_DUMMY_BUILD_CONTENT,
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

    # Set up tensorrt_config.py, which is used by gen_build_info to provide
    # build environment info to the API
    _tpl(
        repository_ctx,
        "tensorrt/tensorrt_config.py",
        _py_tmpl_dict({}),
    )

def enable_tensorrt(repository_ctx):
    """Returns whether to build with TensorRT support."""
    return int(get_host_environ(repository_ctx, _TF_NEED_TENSORRT, False))

def _get_tensorrt_static_path(repository_ctx):
    """Returns the path for TensorRT static libraries."""
    return get_host_environ(repository_ctx, _TF_TENSORRT_STATIC_PATH, None)

def _get_tensorrt_full_version(repository_ctx):
    """Returns the full version for TensorRT."""
    return get_host_environ(repository_ctx, _TF_TENSORRT_VERSION, None)

def _create_local_tensorrt_repository(repository_ctx):
    tpl_paths = {
        "build_defs.bzl": _tpl_path(repository_ctx, "build_defs.bzl"),
        "BUILD": _tpl_path(repository_ctx, "BUILD"),
        "tensorrt/include/tensorrt_config.h": _tpl_path(repository_ctx, "tensorrt/include/tensorrt_config.h"),
        "tensorrt/tensorrt_config.py": _tpl_path(repository_ctx, "tensorrt/tensorrt_config.py"),
        "plugin.BUILD": _tpl_path(repository_ctx, "plugin.BUILD"),
    }

    config = find_cuda_config(repository_ctx, ["cuda", "tensorrt"])
    cuda_version = config["cuda_version"]
    cuda_library_path = config["cuda_library_dir"] + "/"
    trt_version = config["tensorrt_version"]
    trt_full_version = _get_tensorrt_full_version(repository_ctx)
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

    tensorrt_static_path = _get_tensorrt_static_path(repository_ctx)
    if tensorrt_static_path:
        tensorrt_static_path = tensorrt_static_path + "/"
        if _at_least_version(trt_full_version, "8.4.1") and _at_least_version(cuda_version, "11.4"):
            raw_static_library_names = _TF_TENSORRT_LIBS
            nvrtc_ptxjit_static_raw_names = ["nvrtc", "nvrtc-builtins", "nvptxcompiler"]
            nvrtc_ptxjit_static_names = ["%s_static" % name for name in nvrtc_ptxjit_static_raw_names]
            nvrtc_ptxjit_static_libraries = [lib_name(lib, cpu_value, trt_version, static = True) for lib in nvrtc_ptxjit_static_names]
        elif _at_least_version(trt_version, "8"):
            raw_static_library_names = _TF_TENSORRT_LIBS
            nvrtc_ptxjit_static_libraries = []
        else:
            raw_static_library_names = _TF_TENSORRT_LIBS + ["nvrtc", "myelin_compiler", "myelin_executor", "myelin_pattern_library", "myelin_pattern_runtime"]
            nvrtc_ptxjit_static_libraries = []
        static_library_names = ["%s_static" % name for name in raw_static_library_names]
        static_libraries = [lib_name(lib, cpu_value, trt_version, static = True) for lib in static_library_names]
        copy_rules = copy_rules + [
            make_copy_files_rule(
                repository_ctx,
                name = "tensorrt_static_lib",
                srcs = [tensorrt_static_path + library for library in static_libraries] +
                       [cuda_library_path + library for library in nvrtc_ptxjit_static_libraries],
                outs = ["tensorrt/lib/" + library for library in static_libraries] +
                       ["tensorrt/lib/" + library for library in nvrtc_ptxjit_static_libraries],
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
        {
            "%{copy_rules}": "\n".join(copy_rules),
        },
    )

    # Set up the plugins folder BUILD file.
    repository_ctx.template(
        "plugin/BUILD",
        tpl_paths["plugin.BUILD"],
        {
            "%{oss_rules}": _TENSORRT_OSS_ARCHIVE_BUILD_CONTENT,
        },
    )

    # Copy license file in non-remote build.
    repository_ctx.template(
        "LICENSE",
        Label("//third_party/tensorrt:LICENSE"),
        {},
    )

    # Set up tensorrt_config.h, which is used by
    # tensorflow/compiler/xla/stream_executor/dso_loader.cc.
    repository_ctx.template(
        "tensorrt/include/tensorrt_config.h",
        tpl_paths["tensorrt/include/tensorrt_config.h"],
        {"%{tensorrt_version}": trt_version},
    )

    # Set up tensorrt_config.py, which is used by gen_build_info to provide
    # build environment info to the API
    repository_ctx.template(
        "tensorrt/tensorrt_config.py",
        tpl_paths["tensorrt/tensorrt_config.py"],
        _py_tmpl_dict({
            "tensorrt_version": trt_version,
        }),
    )

def _py_tmpl_dict(d):
    return {"%{tensorrt_config}": str(d)}

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
            "tensorrt/tensorrt_config.py",
            config_repo_label(remote_config_repo, ":tensorrt/tensorrt_config.py"),
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
    _TF_TENSORRT_STATIC_PATH,
    "TF_CUDA_PATHS",
]

remote_tensorrt_configure = repository_rule(
    implementation = _create_local_tensorrt_repository,
    environ = _ENVIRONS,
    remotable = True,
    attrs = {
        "environ": attr.string_dict(),
        "_find_cuda_config": attr.label(default = "@local_tsl//third_party/gpus:find_cuda_config.py"),
    },
)

tensorrt_configure = repository_rule(
    implementation = _tensorrt_configure_impl,
    environ = _ENVIRONS + [_TF_TENSORRT_CONFIG_REPO],
    attrs = {
        "_find_cuda_config": attr.label(default = "@local_tsl//third_party/gpus:find_cuda_config.py"),
    },
)
"""Detects and configures the local CUDA toolchain.

Add the following to your WORKSPACE FILE:

```python
tensorrt_configure(name = "local_config_tensorrt")
```

Args:
  name: A unique name for this workspace rule.
"""
