# -*- Python -*-
"""Repository rule for TensorRT configuration.

`tensorrt_configure` depends on the following environment variables:

  * `TF_TENSORRT_VERSION`: The TensorRT libnvinfer version.
  * `TENSORRT_INSTALL_PATH`: The installation path of the TensorRT library.
"""

load(
    "//third_party/gpus:cuda_configure.bzl",
    "auto_configure_fail",
    "get_cpu_value",
    "find_cuda_define",
    "find_lib",
    "lib_name",
    "matches_version",
    "make_copy_dir_rule",
    "make_copy_files_rule",
)

_TENSORRT_INSTALL_PATH = "TENSORRT_INSTALL_PATH"
_TF_TENSORRT_CONFIG_REPO = "TF_TENSORRT_CONFIG_REPO"
_TF_TENSORRT_VERSION = "TF_TENSORRT_VERSION"

_TF_TENSORRT_LIBS = ["nvinfer"]
_TF_TENSORRT_HEADERS = ["NvInfer.h", "NvUtils.h"]

_DEFINE_TENSORRT_SONAME_MAJOR = "#define NV_TENSORRT_SONAME_MAJOR"
_DEFINE_TENSORRT_SONAME_MINOR = "#define NV_TENSORRT_SONAME_MINOR"
_DEFINE_TENSORRT_SONAME_PATCH = "#define NV_TENSORRT_SONAME_PATCH"


def _headers_exist(repository_ctx, path):
  """Returns whether all TensorRT header files could be found in 'path'.

  Args:
    repository_ctx: The repository context.
    path: The TensorRT include path to check.

  Returns:
    True if all TensorRT header files can be found in the path.
  """
  for h in _TF_TENSORRT_HEADERS:
    if not repository_ctx.path("%s/%s" % (path, h)).exists:
      return False
  return True


def _find_trt_header_dir(repository_ctx, trt_install_path):
  """Returns the path to the directory containing headers of TensorRT.

  Args:
    repository_ctx: The repository context.
    trt_install_path: The TensorRT library install directory.

  Returns:
    The path of the directory containing the TensorRT header.
  """
  if trt_install_path == "/usr/lib/x86_64-linux-gnu":
    path = "/usr/include/x86_64-linux-gnu"
    if _headers_exist(repository_ctx, path):
      return path
  if trt_install_path == "/usr/lib/aarch64-linux-gnu":
    path = "/usr/include/aarch64-linux-gnu"
    if _headers_exist(repository_ctx, path):
      return path
  path = str(repository_ctx.path("%s/../include" % trt_install_path).realpath)
  if _headers_exist(repository_ctx, path):
    return path
  auto_configure_fail(
      "Cannot find NvInfer.h with TensorRT install path %s" % trt_install_path)


def _trt_lib_version(repository_ctx, trt_install_path):
  """Detects the library (e.g. libnvinfer) version of TensorRT.

  Args:
    repository_ctx: The repository context.
    trt_install_path: The TensorRT library install directory.

  Returns:
    A string containing the library version of TensorRT.
  """
  trt_header_dir = _find_trt_header_dir(repository_ctx, trt_install_path)
  major_version = find_cuda_define(repository_ctx, trt_header_dir, "NvInfer.h",
                                   _DEFINE_TENSORRT_SONAME_MAJOR)
  minor_version = find_cuda_define(repository_ctx, trt_header_dir, "NvInfer.h",
                                   _DEFINE_TENSORRT_SONAME_MINOR)
  patch_version = find_cuda_define(repository_ctx, trt_header_dir, "NvInfer.h",
                                   _DEFINE_TENSORRT_SONAME_PATCH)
  full_version = "%s.%s.%s" % (major_version, minor_version, patch_version)
  environ_version = repository_ctx.os.environ[_TF_TENSORRT_VERSION].strip()
  if not matches_version(environ_version, full_version):
    auto_configure_fail(
        ("TensorRT library version detected from %s/%s (%s) does not match " +
         "TF_TENSORRT_VERSION (%s). To fix this rerun configure again.") %
        (trt_header_dir, "NvInfer.h", full_version, environ_version))
  # Only use the major version to match the SONAME of the library.
  return major_version


def _find_trt_libs(repository_ctx, cpu_value, trt_install_path, trt_lib_version):
  """Finds the given TensorRT library on the system.

  Adapted from code contributed by Sami Kama (https://github.com/samikama).

  Args:
    repository_ctx: The repository context.
    trt_install_path: The TensorRT library installation directory.
    trt_lib_version: The version of TensorRT library files as returned
      by _trt_lib_version.

  Returns:
    The path to the library.
  """
  result = {}
  for lib in _TF_TENSORRT_LIBS:
    file_name = lib_name("nvinfer", cpu_value, trt_lib_version)
    path = find_lib(repository_ctx, ["%s/%s" % (trt_install_path, file_name)])
    result[file_name] = path
  return result


def _tpl(repository_ctx, tpl, substitutions):
  repository_ctx.template(tpl, Label("//third_party/tensorrt:%s.tpl" % tpl),
                          substitutions)


def _create_dummy_repository(repository_ctx):
  """Create a dummy TensorRT repository."""
  _tpl(repository_ctx, "build_defs.bzl", {"%{if_tensorrt}": "if_false"})

  _tpl(repository_ctx, "BUILD", {
      "%{tensorrt_genrules}": "",
      "%{tensorrt_headers}": "[]",
      "%{tensorrt_libs}": "[]"
  })

def _tensorrt_configure_impl(repository_ctx):
  """Implementation of the tensorrt_configure repository rule."""
  if _TF_TENSORRT_CONFIG_REPO in repository_ctx.os.environ:
    # Forward to the pre-configured remote repository.
    remote_config_repo = repository_ctx.os.environ[_TF_TENSORRT_CONFIG_REPO]
    repository_ctx.template("BUILD", Label(remote_config_repo + "/BUILD"), {})
    # Set up config file.
    _tpl(repository_ctx, "build_defs.bzl", {"%{if_tensorrt}": "if_true"})
    return

  if _TENSORRT_INSTALL_PATH not in repository_ctx.os.environ:
    _create_dummy_repository(repository_ctx)
    return

  cpu_value = get_cpu_value(repository_ctx)
  if (cpu_value != "Linux"):
    auto_configure_fail("TensorRT is supported only on Linux.")
  if _TF_TENSORRT_VERSION not in repository_ctx.os.environ:
    auto_configure_fail("TensorRT library (libnvinfer) version is not set.")
  trt_install_path = repository_ctx.os.environ[_TENSORRT_INSTALL_PATH].strip()
  if not repository_ctx.path(trt_install_path).exists:
    auto_configure_fail(
        "Cannot find TensorRT install path %s." % trt_install_path)

  # Copy the library files.
  trt_lib_version = _trt_lib_version(repository_ctx, trt_install_path)
  trt_libs = _find_trt_libs(repository_ctx, cpu_value, trt_install_path, trt_lib_version)
  trt_lib_srcs = []
  trt_lib_outs = []
  for path in trt_libs.values():
    trt_lib_srcs.append(str(path))
    trt_lib_outs.append("tensorrt/lib/" + path.basename)
  copy_rules = [make_copy_files_rule(
      repository_ctx,
      name = "tensorrt_lib",
      srcs = trt_lib_srcs,
      outs = trt_lib_outs,
  )]

  # Copy the header files header files.
  trt_header_dir = _find_trt_header_dir(repository_ctx, trt_install_path)
  trt_header_srcs = [
      "%s/%s" % (trt_header_dir, header) for header in _TF_TENSORRT_HEADERS
  ]
  trt_header_outs = [
      "tensorrt/include/" + header for header in _TF_TENSORRT_HEADERS
  ]
  copy_rules.append(
      make_copy_files_rule(
          repository_ctx,
          name = "tensorrt_include",
          srcs = trt_header_srcs,
          outs = trt_header_outs,
  ))

  # Set up config file.
  _tpl(repository_ctx, "build_defs.bzl", {"%{if_tensorrt}": "if_true"})

  # Set up BUILD file.
  _tpl(repository_ctx, "BUILD", {
      "%{copy_rules}": "\n".join(copy_rules),
      "%{tensorrt_headers}": '":tensorrt_include"',
      "%{tensorrt_libs}": str(trt_lib_outs),
  })


tensorrt_configure = repository_rule(
    implementation=_tensorrt_configure_impl,
    environ=[
        _TENSORRT_INSTALL_PATH,
        _TF_TENSORRT_VERSION,
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
