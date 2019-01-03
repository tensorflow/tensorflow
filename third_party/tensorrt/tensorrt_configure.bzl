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
    "matches_version",
    "symlink_genrule_for_dir",
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
  return environ_version


def _find_trt_libs(repository_ctx, trt_install_path, trt_lib_version):
  """Finds the given TensorRT library on the system.

  Adapted from code contributed by Sami Kama (https://github.com/samikama).

  Args:
    repository_ctx: The repository context.
    trt_install_path: The TensorRT library installation directory.
    trt_lib_version: The version of TensorRT library files as returned
      by _trt_lib_version.

  Returns:
    Map of library names to structs with the following fields:
      src_file_path: The full path to the library found on the system.
      dst_file_name: The basename of the target library.
  """
  objdump = repository_ctx.which("objdump")
  result = {}
  for lib in _TF_TENSORRT_LIBS:
    dst_file_name = "lib%s.so.%s" % (lib, trt_lib_version)
    src_file_path = repository_ctx.path("%s/%s" % (trt_install_path,
                                                   dst_file_name))
    if not src_file_path.exists:
      auto_configure_fail(
          "Cannot find TensorRT library %s" % str(src_file_path))
    if objdump != None:
      objdump_out = repository_ctx.execute([objdump, "-p", str(src_file_path)])
      for line in objdump_out.stdout.splitlines():
        if "SONAME" in line:
          dst_file_name = line.strip().split(" ")[-1]
    result.update({
        lib:
            struct(
                dst_file_name=dst_file_name,
                src_file_path=str(src_file_path.realpath))
    })
  return result


def _tpl(repository_ctx, tpl, substitutions):
  repository_ctx.template(tpl, Label("//third_party/tensorrt:%s.tpl" % tpl),
                          substitutions)


def _create_dummy_repository(repository_ctx):
  """Create a dummy TensorRT repository."""
  _tpl(repository_ctx, "build_defs.bzl", {"%{tensorrt_is_configured}": "False"})
  substitutions = {
      "%{tensorrt_genrules}": "",
      "%{tensorrt_headers}": "",
  }
  for lib in _TF_TENSORRT_LIBS:
    k = "%%{%s}" % lib.replace("nv", "nv_")
    substitutions.update({k: ""})
  _tpl(repository_ctx, "BUILD", substitutions)


def _tensorrt_configure_impl(repository_ctx):
  """Implementation of the tensorrt_configure repository rule."""
  if _TF_TENSORRT_CONFIG_REPO in repository_ctx.os.environ:
    # Forward to the pre-configured remote repository.
    repository_ctx.template("BUILD", Label("//third_party/tensorrt:remote.BUILD.tpl"), {
        "%{target}": repository_ctx.os.environ[_TF_TENSORRT_CONFIG_REPO],
    })
    # Set up config file.
    _tpl(repository_ctx, "build_defs.bzl", {"%{tensorrt_is_configured}": "True"})
    return

  if _TENSORRT_INSTALL_PATH not in repository_ctx.os.environ:
    _create_dummy_repository(repository_ctx)
    return

  if (get_cpu_value(repository_ctx) != "Linux"):
    auto_configure_fail("TensorRT is supported only on Linux.")
  if _TF_TENSORRT_VERSION not in repository_ctx.os.environ:
    auto_configure_fail("TensorRT library (libnvinfer) version is not set.")
  trt_install_path = repository_ctx.os.environ[_TENSORRT_INSTALL_PATH].strip()
  if not repository_ctx.path(trt_install_path).exists:
    auto_configure_fail(
        "Cannot find TensorRT install path %s." % trt_install_path)

  # Set up the symbolic links for the library files.
  trt_lib_version = _trt_lib_version(repository_ctx, trt_install_path)
  trt_libs = _find_trt_libs(repository_ctx, trt_install_path, trt_lib_version)
  trt_lib_src = []
  trt_lib_dest = []
  for lib in trt_libs.values():
    trt_lib_src.append(lib.src_file_path)
    trt_lib_dest.append(lib.dst_file_name)
  genrules = [
      symlink_genrule_for_dir(repository_ctx, None, "tensorrt/lib/",
                              "tensorrt_lib", trt_lib_src, trt_lib_dest)
  ]

  # Set up the symbolic links for the header files.
  trt_header_dir = _find_trt_header_dir(repository_ctx, trt_install_path)
  src_files = [
      "%s/%s" % (trt_header_dir, header) for header in _TF_TENSORRT_HEADERS
  ]
  dest_files = _TF_TENSORRT_HEADERS
  genrules.append(
      symlink_genrule_for_dir(repository_ctx, None, "tensorrt/include/",
                              "tensorrt_include", src_files, dest_files))

  # Set up config file.
  _tpl(repository_ctx, "build_defs.bzl", {"%{tensorrt_is_configured}": "True"})

  # Set up BUILD file.
  substitutions = {
      "%{tensorrt_genrules}": "\n".join(genrules),
      "%{tensorrt_headers}": '":tensorrt_include"',
  }
  for lib in _TF_TENSORRT_LIBS:
    k = "%%{%s}" % lib.replace("nv", "nv_")
    v = '"tensorrt/lib/%s"' % trt_libs[lib].dst_file_name
    substitutions.update({k: v})
  _tpl(repository_ctx, "BUILD", substitutions)


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
