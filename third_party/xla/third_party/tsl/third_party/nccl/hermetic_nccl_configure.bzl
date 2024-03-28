"""Repository rule for hermetic NCCL configuration.

`hermetic_nccl_configure` depends on the following environment variables:

  * `TF_NCCL_USE_STUB`: "1" if a NCCL stub that loads NCCL dynamically should
    be used, "0" if NCCL should be linked in statically.

"""

load(
    "//third_party/gpus:hermetic_cuda_configure.bzl",
    "TF_NEED_CUDA",
    "enable_cuda",
    "get_cuda_version",
)
load(
    "//third_party/remote_config:common.bzl",
    "get_cpu_value",
    "get_host_environ",
)

_TF_NCCL_USE_STUB = "TF_NCCL_USE_STUB"

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
alias(
  name = "nccl_lib",
  actual = "@cuda_nccl//:nccl_lib",
)

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

def _create_local_nccl_repository(repository_ctx):
    cuda_version = get_cuda_version(repository_ctx).split(".")

    # Alias to open source build from @nccl_archive.
    if get_host_environ(repository_ctx, _TF_NCCL_USE_STUB, "0") == "0":
        repository_ctx.file("BUILD", _NCCL_ARCHIVE_BUILD_CONTENT)
    else:
        repository_ctx.file("BUILD", _NCCL_ARCHIVE_STUB_BUILD_CONTENT)

    repository_ctx.template("generated_names.bzl", repository_ctx.attr.generated_names_tpl, {})
    repository_ctx.template(
        "build_defs.bzl",
        repository_ctx.attr.build_defs_tpl,
        {
            "%{cuda_version}": "(%s, %s)" % tuple(cuda_version),
            "%{nvlink_label}": "@cuda_nvcc//:nvlink",
            "%{fatbinary_label}": "@cuda_nvcc//:fatbinary",
            "%{bin2c_label}": "@cuda_nvcc//:bin2c",
            "%{link_stub_label}": "@cuda_nvcc//:link_stub",
            "%{nvprune_label}": "@cuda_nvprune//:nvprune",
        },
    )

def _nccl_autoconf_impl(repository_ctx):
    if (not enable_cuda(repository_ctx) or
        get_cpu_value(repository_ctx) not in ("Linux", "FreeBSD")):
        # Add a dummy build file to make bazel query happy.
        repository_ctx.file("BUILD", _NCCL_DUMMY_BUILD_CONTENT)
        repository_ctx.file("nccl_config.h", "#define TF_NCCL_VERSION \"\"")
    else:
        _create_local_nccl_repository(repository_ctx)

_ENVIRONS = [
    TF_NEED_CUDA,
]

hermetic_nccl_configure = repository_rule(
    environ = _ENVIRONS,
    implementation = _nccl_autoconf_impl,
    attrs = {
        "environ": attr.string_dict(),
        "generated_names_tpl": attr.label(default = Label("//third_party/nccl:generated_names.bzl.tpl")),
        "build_defs_tpl": attr.label(default = Label("//third_party/nccl:build_defs.bzl.tpl")),
        "system_build_tpl": attr.label(default = Label("//third_party/nccl:system.BUILD.tpl")),
    },
)
"""Downloads and configures the hermetic NCCL configuration.

Add the following to your WORKSPACE FILE:

```python
hermetic_nccl_configure(name = "local_config_nccl")
```

Args:
  name: A unique name for this workspace rule.
"""
