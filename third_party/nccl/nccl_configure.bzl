# -*- Python -*-
"""Repository rule for NCCL configuration.

`nccl_configure` depends on the following environment variables:

  * `TF_NCCL_VERSION`: Installed NCCL version or empty to build from source.
  * `NCCL_INSTALL_PATH`: The installation path of the NCCL library.
  * `NCCL_HDR_PATH`: The installation path of the NCCL header files.
"""

load(
    "//third_party/gpus:cuda_configure.bzl",
    "auto_configure_fail",
    "compute_capabilities",
    "cuda_toolkit_path",
    "enable_cuda",
    "find_cuda_define",
    "get_cpu_value",
    "matches_version",
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

def _find_nccl_header(repository_ctx, nccl_install_path):
    """Finds the NCCL header on the system.

    Args:
      repository_ctx: The repository context.
      nccl_install_path: The NCCL library install directory.

    Returns:
      The path to the NCCL header.
    """
    header_path = repository_ctx.path("%s/include/nccl.h" % nccl_install_path)
    if not header_path.exists:
        auto_configure_fail("Cannot find %s" % str(header_path))
    return header_path

def _check_nccl_version(repository_ctx, nccl_install_path, nccl_hdr_path, nccl_version):
    """Checks whether the header file matches the specified version of NCCL.

    Args:
      repository_ctx: The repository context.
      nccl_install_path: The NCCL library install directory.
      nccl_hdr_path: The NCCL header path.
      nccl_version: The expected NCCL version.

    Returns:
      A string containing the library version of NCCL.
    """
    header_path = repository_ctx.path("%s/nccl.h" % nccl_hdr_path)
    if not header_path.exists:
        header_path = _find_nccl_header(repository_ctx, nccl_install_path)
    header_dir = str(header_path.realpath.dirname)
    major_version = find_cuda_define(
        repository_ctx,
        header_dir,
        "nccl.h",
        _DEFINE_NCCL_MAJOR,
    )
    minor_version = find_cuda_define(
        repository_ctx,
        header_dir,
        "nccl.h",
        _DEFINE_NCCL_MINOR,
    )
    patch_version = find_cuda_define(
        repository_ctx,
        header_dir,
        "nccl.h",
        _DEFINE_NCCL_PATCH,
    )
    header_version = "%s.%s.%s" % (major_version, minor_version, patch_version)
    if not matches_version(nccl_version, header_version):
        auto_configure_fail(
            ("NCCL library version detected from %s/nccl.h (%s) does not " +
             "match TF_NCCL_VERSION (%s). To fix this rerun configure again.") %
            (header_dir, header_version, nccl_version),
        )

def _nccl_configure_impl(repository_ctx):
    """Implementation of the nccl_configure repository rule."""
    if not enable_cuda(repository_ctx) or \
       get_cpu_value(repository_ctx) not in ("Linux", "FreeBSD"):
        # Add a dummy build file to make bazel query happy.
        repository_ctx.file("BUILD", _NCCL_DUMMY_BUILD_CONTENT)
        return

    nccl_version = ""
    if _TF_NCCL_VERSION in repository_ctx.os.environ:
        nccl_version = repository_ctx.os.environ[_TF_NCCL_VERSION].strip()
        nccl_version = nccl_version.split(".")[0]

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
        repository_ctx.template("build_defs.bzl", _label("build_defs.bzl.tpl"), {
            "%{gpu_architectures}": str(gpu_architectures),
        })
    else:
        # Create target for locally installed NCCL.
        nccl_install_path = repository_ctx.os.environ[_NCCL_INSTALL_PATH].strip()
        nccl_hdr_path = repository_ctx.os.environ[_NCCL_HDR_PATH].strip()
        _check_nccl_version(repository_ctx, nccl_install_path, nccl_hdr_path, nccl_version)
        repository_ctx.template("BUILD", _label("system.BUILD.tpl"), {
            "%{version}": nccl_version,
            "%{install_path}": nccl_install_path,
            "%{hdr_path}": nccl_hdr_path,
        })

nccl_configure = repository_rule(
    implementation = _nccl_configure_impl,
    environ = [
        _CUDA_TOOLKIT_PATH,
        _NCCL_HDR_PATH,
        _NCCL_INSTALL_PATH,
        _TF_NCCL_VERSION,
        _TF_CUDA_COMPUTE_CAPABILITIES,
        _TF_NEED_CUDA,
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
