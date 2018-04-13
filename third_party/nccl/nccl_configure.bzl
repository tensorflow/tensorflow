# -*- Python -*-
"""Repository rule for NCCL configuration.

`nccl_configure` depends on the following environment variables:

  * `TF_NCCL_VERSION`: The NCCL version.
  * `NCCL_INSTALL_PATH`: The installation path of the NCCL library.
"""

load(
    "//third_party/gpus:cuda_configure.bzl",
    "auto_configure_fail",
    "find_cuda_define",
    "matches_version",
)

_NCCL_INSTALL_PATH = "NCCL_INSTALL_PATH"
_TF_NCCL_VERSION = "TF_NCCL_VERSION"

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

_NCCL_LOCAL_BUILD_TEMPLATE = """
filegroup(
  name = "LICENSE",
  data = ["nccl/NCCL-SLA.txt"],
  visibility = ["//visibility:public"],
)

cc_library(
  name = "nccl",
  srcs = ["nccl/lib/libnccl.so.%s"],
  hdrs = ["nccl/include/nccl.h"],
  include_prefix = "third_party/nccl",
  strip_include_prefix = "nccl/include",
  deps = [
      "@local_config_cuda//cuda:cuda_headers",
  ],
  visibility = ["//visibility:public"],
)
"""


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


def _check_nccl_version(repository_ctx, nccl_install_path, nccl_version):
  """Checks whether the header file matches the specified version of NCCL.

  Args:
    repository_ctx: The repository context.
    nccl_install_path: The NCCL library install directory.
    nccl_version: The expected NCCL version.

  Returns:
    A string containing the library version of NCCL.
  """
  header_path = _find_nccl_header(repository_ctx, nccl_install_path)
  header_dir = str(header_path.realpath.dirname)
  major_version = find_cuda_define(repository_ctx, header_dir, "nccl.h",
                                   _DEFINE_NCCL_MAJOR)
  minor_version = find_cuda_define(repository_ctx, header_dir, "nccl.h",
                                   _DEFINE_NCCL_MINOR)
  patch_version = find_cuda_define(repository_ctx, header_dir, "nccl.h",
                                   _DEFINE_NCCL_PATCH)
  header_version = "%s.%s.%s" % (major_version, minor_version, patch_version)
  if not matches_version(nccl_version, header_version):
    auto_configure_fail(
        ("NCCL library version detected from %s/nccl.h (%s) does not match " +
         "TF_NCCL_VERSION (%s). To fix this rerun configure again.") %
        (header_dir, header_version, nccl_version))


def _find_nccl_lib(repository_ctx, nccl_install_path, nccl_version):
  """Finds the given NCCL library on the system.

  Args:
    repository_ctx: The repository context.
    nccl_install_path: The NCCL library installation directory.
    nccl_version: The version of NCCL library files as returned
      by _nccl_version.

  Returns:
    The path to the NCCL library.
  """
  lib_path = repository_ctx.path("%s/lib/libnccl.so.%s" % (nccl_install_path,
                                                           nccl_version))
  if not lib_path.exists:
    auto_configure_fail("Cannot find NCCL library %s" % str(lib_path))
  return lib_path


def _nccl_configure_impl(repository_ctx):
  """Implementation of the nccl_configure repository rule."""
  if _TF_NCCL_VERSION not in repository_ctx.os.environ:
    # Add a dummy build file to make bazel query happy.
    repository_ctx.file("BUILD", _NCCL_DUMMY_BUILD_CONTENT)
    return

  nccl_version = repository_ctx.os.environ[_TF_NCCL_VERSION].strip()
  if matches_version("1", nccl_version):
    # Alias to GitHub target from @nccl_archive.
    if not matches_version(nccl_version, "1.3"):
      auto_configure_fail(
          "NCCL from GitHub must use version 1.3 (got %s)" % nccl_version)
    repository_ctx.file("BUILD", _NCCL_ARCHIVE_BUILD_CONTENT)
  else:
    # Create target for locally installed NCCL.
    nccl_install_path = repository_ctx.os.environ[_NCCL_INSTALL_PATH].strip()
    _check_nccl_version(repository_ctx, nccl_install_path, nccl_version)
    repository_ctx.symlink(nccl_install_path, "nccl")
    repository_ctx.file("BUILD", _NCCL_LOCAL_BUILD_TEMPLATE % nccl_version)


nccl_configure = repository_rule(
    implementation=_nccl_configure_impl,
    environ=[
        _NCCL_INSTALL_PATH,
        _TF_NCCL_VERSION,
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
