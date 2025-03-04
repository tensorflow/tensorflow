"""Repository rule for NCCL configuration.

NB: DEPRECATED! Use `hermetic/nccl_configure` rule instead.

`nccl_configure` depends on the following environment variables:

  * `TF_NCCL_VERSION`: Installed NCCL version or empty to build from source.
  * `NCCL_INSTALL_PATH` (deprecated): The installation path of the NCCL library.
  * `NCCL_HDR_PATH` (deprecated): The installation path of the NCCL header 
    files.
  * `TF_CUDA_PATHS`: The base paths to look for CUDA and cuDNN. Default is
    `/usr/local/cuda,usr/`.
  * `TF_NCCL_USE_STUB`: "1" if a NCCL stub that loads NCCL dynamically should
    be used, "0" if NCCL should be linked in statically.

"""

load(
    "//third_party/gpus:cuda_configure.bzl",
    "enable_cuda",
    "find_cuda_config",
)
load(
    "//third_party/remote_config:common.bzl",
    "config_repo_label",
    "get_cpu_value",
    "get_host_environ",
)

_CUDA_TOOLKIT_PATH = "CUDA_TOOLKIT_PATH"
_NCCL_HDR_PATH = "NCCL_HDR_PATH"
_NCCL_INSTALL_PATH = "NCCL_INSTALL_PATH"
_TF_CUDA_COMPUTE_CAPABILITIES = "TF_CUDA_COMPUTE_CAPABILITIES"
_TF_NCCL_VERSION = "TF_NCCL_VERSION"
_TF_NEED_CUDA = "TF_NEED_CUDA"
_TF_CUDA_PATHS = "TF_CUDA_PATHS"
_TF_NCCL_USE_STUB = "TF_NCCL_USE_STUB"

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

cc_library(
  name = "nccl_config",
  hdrs = ["nccl_config.h"],
  include_prefix = "third_party/nccl",
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

alias(
  name = "nccl_config",
  actual = "@nccl_archive//:nccl_config",
  visibility = ["//visibility:public"],
)
"""

_NCCL_ARCHIVE_STUB_BUILD_CONTENT = """
filegroup(
  name = "LICENSE",
  data = ["@nccl_archive//:LICENSE.txt"],
  visibility = ["//visibility:public"],
)

alias(
  name = "nccl",
  actual = "@nccl_archive//:nccl_via_stub",
  visibility = ["//visibility:public"],
)

alias(
  name = "nccl_headers",
  actual = "@nccl_archive//:nccl_headers",
  visibility = ["//visibility:public"],
)

alias(
  name = "nccl_config",
  actual = "@nccl_archive//:nccl_config",
  visibility = ["//visibility:public"],
)
"""

def _label(file):
    return Label("//third_party/nccl:{}".format(file))

def _create_local_nccl_repository(repository_ctx):
    nccl_version = get_host_environ(repository_ctx, _TF_NCCL_VERSION, "")
    if nccl_version:
        nccl_version = nccl_version.split(".")[0]

    cuda_config = find_cuda_config(repository_ctx, ["cuda"])
    cuda_version = cuda_config["cuda_version"].split(".")

    if nccl_version == "":
        # Alias to open source build from @nccl_archive.
        if get_host_environ(repository_ctx, _TF_NCCL_USE_STUB, "0") == "0":
            repository_ctx.file("BUILD", _NCCL_ARCHIVE_BUILD_CONTENT)
        else:
            repository_ctx.file("BUILD", _NCCL_ARCHIVE_STUB_BUILD_CONTENT)

        repository_ctx.template("generated_names.bzl", _label("generated_names.bzl.tpl"), {})
        repository_ctx.template(
            "build_defs.bzl",
            _label("build_defs.bzl.tpl"),
            {
                "%{cuda_version}": "(%s, %s)" % tuple(cuda_version),
                "%{nvlink_label}": "@local_config_cuda//cuda:cuda/bin/nvlink",
                "%{fatbinary_label}": "@local_config_cuda//cuda:cuda/bin/fatbinary",
                "%{bin2c_label}": "@local_config_cuda//cuda:cuda/bin/bin2c",
                "%{link_stub_label}": "@local_config_cuda//cuda:cuda/bin/crt/link.stub",
                "%{nvprune_label}": "@local_config_cuda//cuda:cuda/bin/nvprune",
            },
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
        repository_ctx.template("generated_names.bzl", _label("generated_names.bzl.tpl"), {})

def _create_remote_nccl_repository(repository_ctx, remote_config_repo):
    repository_ctx.template(
        "BUILD",
        config_repo_label(remote_config_repo, ":BUILD"),
        {},
    )
    nccl_version = get_host_environ(repository_ctx, _TF_NCCL_VERSION, "")
    if nccl_version == "":
        repository_ctx.template(
            "generated_names.bzl",
            config_repo_label(remote_config_repo, ":generated_names.bzl"),
            {},
        )
        repository_ctx.template(
            "build_defs.bzl",
            config_repo_label(remote_config_repo, ":build_defs.bzl"),
            {},
        )

def _nccl_autoconf_impl(repository_ctx):
    if (not enable_cuda(repository_ctx) or
        get_cpu_value(repository_ctx) not in ("Linux", "FreeBSD")):
        # Add a dummy build file to make bazel query happy.
        repository_ctx.file("BUILD", _NCCL_DUMMY_BUILD_CONTENT)
        repository_ctx.file("nccl_config.h", "#define TF_NCCL_VERSION \"\"")
    elif get_host_environ(repository_ctx, "TF_NCCL_CONFIG_REPO") != None:
        _create_remote_nccl_repository(repository_ctx, get_host_environ(repository_ctx, "TF_NCCL_CONFIG_REPO"))
    else:
        _create_local_nccl_repository(repository_ctx)

_ENVIRONS = [
    _CUDA_TOOLKIT_PATH,
    _NCCL_HDR_PATH,
    _NCCL_INSTALL_PATH,
    _TF_NCCL_VERSION,
    _TF_CUDA_COMPUTE_CAPABILITIES,
    _TF_NEED_CUDA,
    _TF_CUDA_PATHS,
]

remote_nccl_configure = repository_rule(
    implementation = _create_local_nccl_repository,
    environ = _ENVIRONS,
    remotable = True,
    attrs = {
        "environ": attr.string_dict(),
        "_find_cuda_config": attr.label(
            default = Label("@local_xla//third_party/gpus:find_cuda_config.py"),
        ),
    },
)

nccl_configure = repository_rule(
    implementation = _nccl_autoconf_impl,
    environ = _ENVIRONS,
    attrs = {
        "_find_cuda_config": attr.label(
            default = Label("@local_xla//third_party/gpus:find_cuda_config.py"),
        ),
    },
)
"""Detects and configures the NCCL configuration.

Add the following to your WORKSPACE FILE:

```python
nccl_configure(name = "local_config_nccl")
```

Args:
  name: A unique name for this workspace rule.
"""
