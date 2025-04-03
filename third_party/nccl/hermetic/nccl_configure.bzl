"""Repository rule for hermetic NCCL configuration.

`nccl_configure` depends on the following environment variables:

  * `TF_NCCL_USE_STUB`: "1" if a NCCL stub that loads NCCL dynamically should
    be used, "0" if NCCL should be linked in statically.
  * `HERMETIC_CUDA_VERSION`: The version of the CUDA toolkit. If not specified,
  the version will be determined by the `TF_CUDA_VERSION`.

"""

load(
    "//third_party/gpus/cuda/hermetic:cuda_configure.bzl",
    "HERMETIC_CUDA_VERSION",
    "TF_CUDA_VERSION",
    "TF_NEED_CUDA",
    "USE_CUDA_REDISTRIBUTIONS",
    "enable_cuda",
    "get_cuda_version",
    "use_cuda_redistributions",
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
  actual = select({
      "@local_config_cuda//cuda:cuda_tools_and_libs": "@cuda_nccl//:nccl",
      "//conditions:default": "@nccl_archive//:nccl",
  }),
  visibility = ["//visibility:public"],
)

alias(
  name = "nccl_headers",
  actual = select({
      "@local_config_cuda//cuda:cuda_tools_and_libs": "@cuda_nccl//:headers",
      "//conditions:default": "@nccl_archive//:nccl_headers",
  }),
  visibility = ["//visibility:public"],
)

cc_library(
    name = "hermetic_nccl_config",
    hdrs = ["nccl_config.h"],
    include_prefix = "third_party/nccl",
    visibility = ["//visibility:public"],
)

alias(
  name = "nccl_config",
  actual = select({
      "@local_config_cuda//cuda:cuda_tools_and_libs": ":hermetic_nccl_config",
      "//conditions:default": "@nccl_archive//:nccl_config",
  }),
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
  actual = select({
      "@local_config_cuda//cuda:cuda_tools_and_libs": "@cuda_nccl//:nccl",
      "//conditions:default": "@nccl_archive//:nccl_via_stub",
  }),
  visibility = ["//visibility:public"],
)

alias(
  name = "nccl_headers",
  actual = select({
      "@local_config_cuda//cuda:cuda_tools_and_libs": "@cuda_nccl//:headers",
      "//conditions:default": "@nccl_archive//:nccl_headers",
  }),
  visibility = ["//visibility:public"],
)

cc_library(
    name = "hermetic_nccl_config",
    hdrs = ["nccl_config.h"],
    include_prefix = "third_party/nccl",
    visibility = ["//visibility:public"],
)

alias(
  name = "nccl_config",
  actual = select({
      "@local_config_cuda//cuda:cuda_tools_and_libs": ":hermetic_nccl_config",
      "//conditions:default": "@nccl_archive//:nccl_config",
  }),
  visibility = ["//visibility:public"],
)
"""

def _create_local_nccl_repository(repository_ctx):
    cuda_version = get_cuda_version(repository_ctx).split(".")[:2]
    nccl_version = repository_ctx.read(repository_ctx.attr.nccl_version)

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
    repository_ctx.file("nccl_config.h", "#define TF_NCCL_VERSION \"%s\"" % nccl_version)

def _nccl_autoconf_impl(repository_ctx):
    if (not enable_cuda(repository_ctx) or
        get_cpu_value(repository_ctx) != "Linux"):
        # Add a dummy build file to make bazel query happy.
        repository_ctx.file("BUILD", _NCCL_DUMMY_BUILD_CONTENT)
        if use_cuda_redistributions(repository_ctx):
            nccl_version = repository_ctx.read(repository_ctx.attr.nccl_version)
            repository_ctx.file(
                "nccl_config.h",
                "#define TF_NCCL_VERSION \"%s\"" % nccl_version,
            )
        else:
            repository_ctx.file("nccl_config.h", "#define TF_NCCL_VERSION \"\"")
    else:
        _create_local_nccl_repository(repository_ctx)

_ENVIRONS = [
    TF_NEED_CUDA,
    TF_CUDA_VERSION,
    _TF_NCCL_USE_STUB,
    HERMETIC_CUDA_VERSION,
    "LOCAL_NCCL_PATH",
    USE_CUDA_REDISTRIBUTIONS,
    "TF_NEED_ROCM",
]

nccl_configure = repository_rule(
    environ = _ENVIRONS,
    implementation = _nccl_autoconf_impl,
    attrs = {
        "environ": attr.string_dict(),
        "nccl_version": attr.label(default = Label("@cuda_nccl//:version.txt")),
        "generated_names_tpl": attr.label(default = Label("//third_party/nccl:generated_names.bzl.tpl")),
        "build_defs_tpl": attr.label(default = Label("//third_party/nccl:build_defs.bzl.tpl")),
    },
)
"""Downloads and configures the hermetic NCCL configuration.

Add the following to your WORKSPACE file:

```python
nccl_configure(name = "local_config_nccl")
```

Args:
  name: A unique name for this workspace rule.
"""  # buildifier: disable=no-effect
