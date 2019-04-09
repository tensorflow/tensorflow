# -*- Python -*-
"""Repository rule for RCCL configuration.

`rccl_configure` depends on the following environment variables:

  * `TF_RCCL_VERSION`: Installed RCCL version or empty to build from source.
  * `RCCL_INSTALL_PATH`: The installation path of the RCCL library.
  * `RCCL_HDR_PATH`: The installation path of the RCCL header files.
"""

load(
    "//third_party/gpus:cuda_configure.bzl",
    "auto_configure_fail",
    "find_cuda_define",
    "get_cpu_value",
    "matches_version",
)
load(
    "//third_party/gpus:rocm_configure.bzl",
    "enable_rocm",
)

_RCCL_HDR_PATH = "RCCL_HDR_PATH"
_RCCL_INSTALL_PATH = "RCCL_INSTALL_PATH"
_TF_RCCL_VERSION = "TF_RCCL_VERSION"
_TF_NEED_ROCM = "TF_NEED_ROCM"

_DEFINE_NCCL_MAJOR = "#define NCCL_MAJOR"
_DEFINE_NCCL_MINOR = "#define NCCL_MINOR"
_DEFINE_NCCL_PATCH = "#define NCCL_PATCH"

_RCCL_DUMMY_BUILD_CONTENT = """
filegroup(
  name = "LICENSE",
  visibility = ["//visibility:public"],
)

cc_library(
  name = "rccl",
  visibility = ["//visibility:public"],
)
"""

def _label(file):
    return Label("//third_party/rccl:{}".format(file))

def _find_rccl_header(repository_ctx, rccl_install_path):
    """Finds the RCCL header on the system.

    Args:
      repository_ctx: The repository context.
      rccl_install_path: The RCCL library install directory.

    Returns:
      The path to the RCCL header.
    """
    header_path = repository_ctx.path("%s/include/rccl.h" % rccl_install_path)
    if not header_path.exists:
        auto_configure_fail("Cannot find %s" % str(header_path))
    return header_path

def _check_rccl_version(repository_ctx, rccl_install_path, rccl_hdr_path, rccl_version):
    """Checks whether the header file matches the specified version of RCCL.

    Args:
      repository_ctx: The repository context.
      rccl_install_path: The RCCL library install directory.
      rccl_hdr_path: The RCCL header path.
      rccl_version: The expected RCCL version.

    Returns:
      A string containing the library version of RCCL.
    """
    header_path = repository_ctx.path("%s/rccl.h" % rccl_hdr_path)
    if not header_path.exists:
        header_path = _find_rccl_header(repository_ctx, rccl_install_path)
    header_dir = str(header_path.realpath.dirname)
    major_version = find_cuda_define(
        repository_ctx,
        header_dir,
        "rccl.h",
        _DEFINE_NCCL_MAJOR,
    )
    minor_version = find_cuda_define(
        repository_ctx,
        header_dir,
        "rccl.h",
        _DEFINE_NCCL_MINOR,
    )
    patch_version = find_cuda_define(
        repository_ctx,
        header_dir,
        "rccl.h",
        _DEFINE_NCCL_PATCH,
    )
    header_version = "%s.%s.%s" % (major_version, minor_version, patch_version)
    if not matches_version(rccl_version, header_version):
        auto_configure_fail(
            ("RCCL library version detected from %s/rccl.h (%s) does not " +
             "match TF_RCCL_VERSION (%s). To fix this rerun configure again.") %
            (header_dir, header_version, rccl_version),
        )

def _rccl_configure_impl(repository_ctx):
    """Implementation of the rccl_configure repository rule."""
    if (not enable_rocm(repository_ctx)
        or get_cpu_value(repository_ctx) not in ("Linux", "FreeBSD")):
        # Add a dummy build file to make bazel query happy.
        repository_ctx.file("BUILD", _RCCL_DUMMY_BUILD_CONTENT)
        return

    rccl_version = ""
    if _TF_RCCL_VERSION in repository_ctx.os.environ:
        rccl_version = repository_ctx.os.environ[_TF_RCCL_VERSION].strip()
        rccl_version = rccl_version.split(".")[0]

    if rccl_version == "":
        auto_configure_fail("Cannot find RCCL version")
    else:
        # Create target for locally installed RCCL.
        rccl_install_path = repository_ctx.os.environ[_RCCL_INSTALL_PATH].strip()
        rccl_hdr_path = repository_ctx.os.environ[_RCCL_HDR_PATH].strip()
        _check_rccl_version(repository_ctx, rccl_install_path, rccl_hdr_path, rccl_version)
        repository_ctx.template("BUILD", _label("system.BUILD.tpl"), {
            "%{version}": rccl_version,
            "%{install_path}": rccl_install_path,
            "%{hdr_path}": rccl_hdr_path,
        })

rccl_configure = repository_rule(
    implementation = _rccl_configure_impl,
    environ = [
        _RCCL_HDR_PATH,
        _RCCL_INSTALL_PATH,
        _TF_RCCL_VERSION,
        _TF_NEED_ROCM,
    ],
)
"""Detects and configures the RCCL configuration.

Add the following to your WORKSPACE FILE:

```python
rccl_configure(name = "local_config_rccl")
```

Args:
  name: A unique name for this workspace rule.
"""
