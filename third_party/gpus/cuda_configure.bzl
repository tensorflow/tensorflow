# -*- Python -*-
"""Repository rule for CUDA autoconfiguration.

`cuda_configure` depends on the following environment variables:

  * `TF_NEED_CUDA`: Whether to enable building with CUDA.
  * `GCC_HOST_COMPILER_PATH`: The GCC host compiler path
  * `CUDA_TOOLKIT_PATH`: The path to the CUDA toolkit. Default is
    `/usr/local/cuda`.
  * `TF_CUDA_VERSION`: The version of the CUDA toolkit. If this is blank, then
    use the system default.
  * `TF_CUDNN_VERSION`: The version of the cuDNN library.
  * `CUDNN_INSTALL_PATH`: The path to the cuDNN library. Default is
    `/usr/local/cuda`.
  * `TF_CUDA_COMPUTE_CAPABILITIES`: The CUDA compute capabilities. Default is
    `3.5,5.2`.
"""

_GCC_HOST_COMPILER_PATH = "GCC_HOST_COMPILER_PATH"
_CUDA_TOOLKIT_PATH = "CUDA_TOOLKIT_PATH"
_TF_CUDA_VERSION = "TF_CUDA_VERSION"
_TF_CUDNN_VERSION = "TF_CUDNN_VERSION"
_CUDNN_INSTALL_PATH = "CUDNN_INSTALL_PATH"
_TF_CUDA_COMPUTE_CAPABILITIES = "TF_CUDA_COMPUTE_CAPABILITIES"

_DEFAULT_CUDA_VERSION = ""
_DEFAULT_CUDNN_VERSION = ""
_DEFAULT_CUDA_TOOLKIT_PATH = "/usr/local/cuda"
_DEFAULT_CUDNN_INSTALL_PATH = "/usr/local/cuda"
_DEFAULT_CUDA_COMPUTE_CAPABILITIES = ["3.5", "5.2"]


# TODO(dzc): Once these functions have been factored out of Bazel's
# cc_configure.bzl, load them from @bazel_tools instead.
# BEGIN cc_configure common functions.
def find_cc(repository_ctx):
  """Find the C++ compiler."""
  cc_name = "gcc"
  if _GCC_HOST_COMPILER_PATH in repository_ctx.os.environ:
    cc_name = repository_ctx.os.environ[_GCC_HOST_COMPILER_PATH].strip()
    if not cc_name:
      cc_name = "gcc"
  if cc_name.startswith("/"):
    # Absolute path, maybe we should make this suported by our which function.
    return cc_name
  cc = repository_ctx.which(cc_name)
  if cc == None:
    fail(
        "Cannot find gcc, either correct your path or set the CC" +
        " environment variable")
  return cc


_INC_DIR_MARKER_BEGIN = "#include <...>"


# OSX add " (framework directory)" at the end of line, strip it.
_OSX_FRAMEWORK_SUFFIX = " (framework directory)"
_OSX_FRAMEWORK_SUFFIX_LEN =  len(_OSX_FRAMEWORK_SUFFIX)
def _cxx_inc_convert(path):
  """Convert path returned by cc -E xc++ in a complete path."""
  path = path.strip()
  if path.endswith(_OSX_FRAMEWORK_SUFFIX):
    path = path[:-_OSX_FRAMEWORK_SUFFIX_LEN].strip()
  return path


def get_cxx_inc_directories(repository_ctx, cc):
  """Compute the list of default C++ include directories."""
  result = repository_ctx.execute([cc, "-E", "-xc++", "-", "-v"])
  index1 = result.stderr.find(_INC_DIR_MARKER_BEGIN)
  if index1 == -1:
    return []
  index1 = result.stderr.find("\n", index1)
  if index1 == -1:
    return []
  index2 = result.stderr.rfind("\n ")
  if index2 == -1 or index2 < index1:
    return []
  index2 = result.stderr.find("\n", index2 + 1)
  if index2 == -1:
    inc_dirs = result.stderr[index1 + 1:]
  else:
    inc_dirs = result.stderr[index1 + 1:index2].strip()

  return [repository_ctx.path(_cxx_inc_convert(p))
          for p in inc_dirs.split("\n")]

def auto_configure_fail(msg):
  """Output failure message when auto configuration fails."""
  red = "\033[0;31m"
  no_color = "\033[0m"
  fail("\n%sAuto-Configuration Error:%s %s\n" % (red, no_color, msg))
# END cc_configure common functions (see TODO above).


def _gcc_host_compiler_includes(repository_ctx, cc):
  """Generates the cxx_builtin_include_directory entries for gcc inc dirs.

  Args:
    repository_ctx: The repository context.
    cc: The path to the gcc host compiler.

  Returns:
    A string containing the cxx_builtin_include_directory for each of the gcc
    host compiler include directories, which can be added to the CROSSTOOL
    file.
  """
  inc_dirs = get_cxx_inc_directories(repository_ctx, cc)
  inc_entries = []
  for inc_dir in inc_dirs:
    inc_entries.append("  cxx_builtin_include_directory: \"%s\"" % inc_dir)
  return "\n".join(inc_entries)


def _enable_cuda(repository_ctx):
  if "TF_NEED_CUDA" in repository_ctx.os.environ:
    enable_cuda = repository_ctx.os.environ["TF_NEED_CUDA"].strip()
    return enable_cuda == "1"
  return False


def _cuda_toolkit_path(repository_ctx, cuda_version):
  """Finds the cuda toolkit directory.

  Args:
    repository_ctx: The repository context.
    cuda_version: The cuda toolkit version.

  Returns:
    A speculative real path of the cuda toolkit install directory.
  """
  cuda_toolkit_path = _DEFAULT_CUDA_TOOLKIT_PATH
  if _CUDA_TOOLKIT_PATH in repository_ctx.os.environ:
    cuda_toolkit_path = repository_ctx.os.environ[_CUDA_TOOLKIT_PATH].strip()
  if not repository_ctx.path(cuda_toolkit_path).exists:
    auto_configure_fail("Cannot find cuda toolkit path.")

  if cuda_version:
    # Handle typical configuration where the real path is
    # <basedir>/cuda-<version> and the provided path is <basedir>/cuda.
    version_suffixed = "%s-%s" % (cuda_toolkit_path, cuda_version)
    if repository_ctx.path(version_suffixed).exists:
      return version_suffixed
  # Returns the non-versioned path if cuda version is not provided or if the
  # installation does not use a cuda- directory, such as on ArchLinux where
  # CUDA installs directly to /opt/cuda.
  return cuda_toolkit_path


def _cudnn_install_basedir(repository_ctx):
  """Finds the cudnn install directory."""
  cudnn_install_path = _DEFAULT_CUDNN_INSTALL_PATH
  if _CUDNN_INSTALL_PATH in repository_ctx.os.environ:
    cudnn_install_path = repository_ctx.os.environ[_CUDNN_INSTALL_PATH].strip()
  if not repository_ctx.path(cudnn_install_path).exists:
    auto_configure_fail("Cannot find cudnn install path.")
  return cudnn_install_path


def _cuda_version(repository_ctx):
  """Detects the cuda version."""
  if _TF_CUDA_VERSION in repository_ctx.os.environ:
    return repository_ctx.os.environ[_TF_CUDA_VERSION].strip()
  else:
    return ""


def _cudnn_version(repository_ctx):
  """Detects the cudnn version."""
  if _TF_CUDNN_VERSION in repository_ctx.os.environ:
    return repository_ctx.os.environ[_TF_CUDNN_VERSION].strip()
  else:
    return ""


def _compute_capabilities(repository_ctx):
  """Returns a list of strings representing cuda compute capabilities."""
  if _TF_CUDA_COMPUTE_CAPABILITIES not in repository_ctx.os.environ:
    return _DEFAULT_CUDA_COMPUTE_CAPABILITIES
  capabilities_str = repository_ctx.os.environ[_TF_CUDA_COMPUTE_CAPABILITIES]
  capabilities = capabilities_str.split(",")
  for capability in capabilities:
    # Workaround for Skylark's lack of support for regex. This check should
    # be equivalent to checking:
    #     if re.match("[0-9]+.[0-9]+", capability) == None:
    parts = capability.split(".")
    if len(parts) != 2 or not parts[0].isdigit() or not parts[1].isdigit():
      auto_configure_fail("Invalid compute capability: %s" % capability)
  return capabilities


def _cpu_value(repository_ctx):
  os_name = repository_ctx.os.name.lower()
  if os_name.startswith("mac os"):
    return "Darwin"
  if os_name.find("windows") != -1:
    return "Windows"
  result = repository_ctx.execute(["uname", "-s"])
  return result.stdout.strip()


def _cuda_symlink_files(cpu_value, cuda_version, cudnn_version):
  """Returns a struct containing platform-specific paths.

  Args:
    cpu_value: The string representing the host OS.
    cuda_version: The cuda version as returned by _cuda_version
    cudnn_version: The cudnn version as returned by _cudnn_version
  """
  cuda_ext = ".%s" % cuda_version if cuda_version else ""
  cudnn_ext = ".%s" % cudnn_version if cudnn_version else ""
  if cpu_value == "Linux":
    return struct(
        cuda_lib_path = "lib64",
        cuda_rt_lib = "lib64/libcudart.so%s" % cuda_ext,
        cuda_rt_lib_static = "lib64/libcudart_static.a",
        cuda_blas_lib = "lib64/libcublas.so%s" % cuda_ext,
        cuda_dnn_lib = "lib64/libcudnn.so%s" % cudnn_ext,
        cuda_dnn_lib_alt = "libcudnn.so%s" % cudnn_ext,
        cuda_rand_lib = "lib64/libcurand.so%s" % cuda_ext,
        cuda_fft_lib = "lib64/libcufft.so%s" % cuda_ext,
        cuda_cupti_lib = "extras/CUPTI/lib64/libcupti.so%s" % cuda_ext)
  elif cpu_value == "Darwin":
    return struct(
        cuda_lib_path = "lib",
        cuda_rt_lib = "lib/libcudart%s.dylib" % cuda_ext,
        cuda_rt_lib_static = "lib/libcudart_static.a",
        cuda_blas_lib = "lib/libcublas%s.dylib" % cuda_ext,
        cuda_dnn_lib = "lib/libcudnn%s.dylib" % cudnn_ext,
        cuda_dnn_lib_alt = "libcudnn%s.dylib" % cudnn_ext,
        cuda_rand_lib = "lib/libcurand%s.dylib" % cuda_ext,
        cuda_fft_lib = "lib/libcufft%s.dylib" % cuda_ext,
        cuda_cupti_lib = "extras/CUPTI/lib/libcupti%s.dylib" % cuda_ext)
  elif cpu_value == "Windows":
    return struct(
        cuda_lib_path = "lib",
        cuda_rt_lib = "lib/cudart%s.dll" % cuda_ext,
        cuda_rt_lib_static = "lib/cudart_static.lib",
        cuda_blas_lib = "lib/cublas%s.dll" % cuda_ext,
        cuda_dnn_lib = "lib/cudnn%s.dll" % cudnn_ext,
        cuda_dnn_lib_alt = "cudnn%s.dll" % cudnn_ext,
        cuda_rand_lib = "lib/curand%s.dll" % cuda_ext,
        cuda_fft_lib = "lib/cufft%s.dll" % cuda_ext,
        cuda_cupti_lib = "extras/CUPTI/lib/cupti%s.dll" % cuda_ext)
  else:
    auto_configure_fail("Not supported CPU value %s" % cpu_value)


def _check_lib(repository_ctx, cuda_toolkit_path, cuda_lib):
  """Checks if cuda_lib exists under cuda_toolkit_path or fail if it doesn't.

  Args:
    repository_ctx: The repository context.
    cuda_toolkit_path: The cuda toolkit directory containing the cuda libraries.
    cuda_lib: The library to look for under cuda_toolkit_path.
  """
  lib_path = cuda_toolkit_path + "/" + cuda_lib
  if not repository_ctx.path(lib_path).exists:
    auto_configure_fail("Cannot find %s" % lib_path)


def _check_dir(repository_ctx, directory):
  """Checks whether the directory exists and fail if it does not.

  Args:
    repository_ctx: The repository context.
    directory: The directory to check the existence of.
  """
  if not repository_ctx.path(directory).exists:
    auto_configure_fail("Cannot find dir: %s" % directory)


def _find_cudnn_header_dir(repository_ctx, cudnn_install_basedir):
  """Returns the path to the directory containing cudnn.h

  Args:
    repository_ctx: The repository context.
    cudnn_install_basedir: The cudnn install directory as returned by
      _cudnn_install_basedir.

  Returns:
    The path of the directory containing the cudnn header.
  """
  if repository_ctx.path(cudnn_install_basedir + "/cudnn.h").exists:
    return cudnn_install_basedir
  if repository_ctx.path(cudnn_install_basedir + "/include/cudnn.h").exists:
    return cudnn_install_basedir + "/include"
  if repository_ctx.path("/usr/include/cudnn.h").exists:
    return "/usr/include"
  auto_configure_fail("Cannot find cudnn.h under %s" % cudnn_install_basedir)


def _find_cudnn_lib_path(repository_ctx, cudnn_install_basedir, symlink_files):
  """Returns the path to the directory containing libcudnn

  Args:
    repository_ctx: The repository context.
    cudnn_install_basedir: The cudnn install dir as returned by
      _cudnn_install_basedir.
    symlink_files: The symlink files as returned by _cuda_symlink_files.

  Returns:
    The path of the directory containing the cudnn libraries.
  """
  lib_dir = cudnn_install_basedir + "/" + symlink_files.cuda_dnn_lib
  if repository_ctx.path(lib_dir).exists:
    return lib_dir
  alt_lib_dir = cudnn_install_basedir + "/" + symlink_files.cuda_dnn_lib_alt
  if repository_ctx.path(alt_lib_dir).exists:
    return alt_lib_dir

  auto_configure_fail("Cannot find %s or %s under %s" %
       (symlink_files.cuda_dnn_lib, symlink_files.cuda_dnn_lib_alt,
        cudnn_install_basedir))


def _tpl(repository_ctx, tpl, substitutions={}, out=None):
  if not out:
    out = tpl.replace(":", "/")
  repository_ctx.template(
      out,
      Label("//third_party/gpus/%s.tpl" % tpl),
      substitutions)


def _file(repository_ctx, label):
  repository_ctx.template(
      label.replace(":", "/"),
      Label("//third_party/gpus/%s.tpl" % label),
      {})


_DUMMY_CROSSTOOL_BZL_FILE = """
def error_gpu_disabled():
  fail("ERROR: Building with --config=cuda but TensorFlow is not configured " +
       "to build with GPU support. Please re-run ./configure and enter 'Y' " +
       "at the prompt to build with GPU support.")

  native.genrule(
      name = "error_gen_crosstool",
      outs = ["CROSSTOOL"],
      cmd = "echo 'Should not be run.' && exit 1",
  )

  native.filegroup(
      name = "crosstool",
      srcs = [":CROSSTOOL"],
      output_licenses = ["unencumbered"],
  )
"""


_DUMMY_CROSSTOOL_BUILD_FILE = """
load("//crosstool:error_gpu_disabled.bzl", "error_gpu_disabled")

error_gpu_disabled()
"""


def _create_dummy_repository(repository_ctx):
  cpu_value = _cpu_value(repository_ctx)
  symlink_files = _cuda_symlink_files(cpu_value, _DEFAULT_CUDA_VERSION,
                                      _DEFAULT_CUDNN_VERSION)

  # Set up BUILD file for cuda/.
  _file(repository_ctx, "cuda:BUILD")
  _file(repository_ctx, "cuda:build_defs.bzl")
  _tpl(repository_ctx, "cuda:platform.bzl",
       {
           "%{cuda_version}": _DEFAULT_CUDA_VERSION,
           "%{cudnn_version}": _DEFAULT_CUDNN_VERSION,
           "%{platform}": cpu_value,
       })

  # Create dummy files for the CUDA toolkit since they are still required by
  # tensorflow/core/platform/default/build_config:cuda.
  repository_ctx.file("cuda/include/cuda.h", "")
  repository_ctx.file("cuda/include/cublas.h", "")
  repository_ctx.file("cuda/include/cudnn.h", "")
  repository_ctx.file("cuda/extras/CUPTI/include/cupti.h", "")
  repository_ctx.file("cuda/%s" % symlink_files.cuda_rt_lib, "")
  repository_ctx.file("cuda/%s" % symlink_files.cuda_rt_lib_static, "")
  repository_ctx.file("cuda/%s" % symlink_files.cuda_blas_lib, "")
  repository_ctx.file("cuda/%s" % symlink_files.cuda_dnn_lib, "")
  repository_ctx.file("cuda/%s" % symlink_files.cuda_rand_lib, "")
  repository_ctx.file("cuda/%s" % symlink_files.cuda_fft_lib, "")
  repository_ctx.file("cuda/%s" % symlink_files.cuda_cupti_lib, "")

  # Set up cuda_config.h, which is used by
  # tensorflow/stream_executor/dso_loader.cc.
  _tpl(repository_ctx, "cuda:cuda_config.h",
       {
           "%{cuda_version}": _DEFAULT_CUDA_VERSION,
           "%{cudnn_version}": _DEFAULT_CUDNN_VERSION,
           "%{cuda_compute_capabilities}": ",".join([
               "CudaVersion(\"%s\")" % c
               for c in _DEFAULT_CUDA_COMPUTE_CAPABILITIES]),
       })

  # If cuda_configure is not configured to build with GPU support, and the user
  # attempts to build with --config=cuda, add a dummy build rule to intercept
  # this and fail with an actionable error message.
  repository_ctx.file("crosstool/error_gpu_disabled.bzl",
                      _DUMMY_CROSSTOOL_BZL_FILE)
  repository_ctx.file("crosstool/BUILD", _DUMMY_CROSSTOOL_BUILD_FILE)

def _symlink_dir(repository_ctx, src_dir, dest_dir):
  """Symlinks all the files in a directory.

  Args:
    repository_ctx: The repository context.
    src_dir: The source directory.
    dest_dir: The destination directory to create the symlinks in.
  """
  files = repository_ctx.path(src_dir).readdir()
  for src_file in files:
    repository_ctx.symlink(src_file, dest_dir + "/" + src_file.basename)


def _create_cuda_repository(repository_ctx):
  """Creates the repository containing files set up to build with CUDA."""
  cuda_version = _cuda_version(repository_ctx)
  cuda_toolkit_path = _cuda_toolkit_path(repository_ctx, cuda_version)
  cudnn_install_basedir = _cudnn_install_basedir(repository_ctx)
  cudnn_version = _cudnn_version(repository_ctx)
  compute_capabilities = _compute_capabilities(repository_ctx)

  cpu_value = _cpu_value(repository_ctx)
  symlink_files = _cuda_symlink_files(cpu_value, cuda_version, cudnn_version)
  _check_lib(repository_ctx, cuda_toolkit_path, symlink_files.cuda_rt_lib)
  _check_lib(repository_ctx, cuda_toolkit_path, symlink_files.cuda_cupti_lib)
  _check_dir(repository_ctx, cudnn_install_basedir)

  cudnn_header_dir = _find_cudnn_header_dir(repository_ctx,
                                            cudnn_install_basedir)
  cudnn_lib_path = _find_cudnn_lib_path(repository_ctx, cudnn_install_basedir,
                                        symlink_files)

  # Set up symbolic links for the cuda toolkit. We link at the individual file
  # level not at the directory level. This is because the external library may
  # have a different file layout from our desired structure.
  _symlink_dir(repository_ctx, cuda_toolkit_path + "/include", "cuda/include")
  _symlink_dir(repository_ctx,
               cuda_toolkit_path + "/" + symlink_files.cuda_lib_path,
               "cuda/" + symlink_files.cuda_lib_path)
  _symlink_dir(repository_ctx, cuda_toolkit_path + "/bin", "cuda/bin")
  _symlink_dir(repository_ctx, cuda_toolkit_path + "/nvvm", "cuda/nvvm")
  _symlink_dir(repository_ctx, cuda_toolkit_path + "/extras/CUPTI/include",
               "cuda/extras/CUPTI/include")
  repository_ctx.symlink(cuda_toolkit_path + "/" + symlink_files.cuda_cupti_lib,
                         "cuda/" + symlink_files.cuda_cupti_lib)

  # Set up the symbolic links for cudnn if cudnn was was not installed to
  # CUDA_TOOLKIT_PATH.
  if not repository_ctx.path("cuda/include/cudnn.h").exists:
    repository_ctx.symlink(cudnn_header_dir + "/cudnn.h",
                           "cuda/include/cudnn.h")
  if not repository_ctx.path("cuda/" + symlink_files.cuda_dnn_lib).exists:
    repository_ctx.symlink(cudnn_lib_path, "cuda/" + symlink_files.cuda_dnn_lib)

  # Set up BUILD file for cuda/
  _file(repository_ctx, "cuda:BUILD")
  _file(repository_ctx, "cuda:build_defs.bzl")
  _tpl(repository_ctx, "cuda:platform.bzl",
       {
           "%{cuda_version}": cuda_version,
           "%{cudnn_version}": cudnn_version,
           "%{platform}": cpu_value,
       })

  # Set up crosstool/
  _file(repository_ctx, "crosstool:BUILD")
  cc = find_cc(repository_ctx)
  gcc_host_compiler_includes = _gcc_host_compiler_includes(repository_ctx, cc)
  _tpl(repository_ctx, "crosstool:CROSSTOOL",
       {
           "%{cuda_include_path}": cuda_toolkit_path + '/include',
           "%{gcc_host_compiler_includes}": gcc_host_compiler_includes,
       })
  _tpl(repository_ctx,
       "crosstool:clang/bin/crosstool_wrapper_driver_is_not_gcc",
       {
           "%{cpu_compiler}": str(cc),
           "%{gcc_host_compiler_path}": str(cc),
           "%{cuda_compute_capabilities}": ", ".join(
               ["\"%s\"" % c for c in compute_capabilities]),
       })

  # Set up cuda_config.h, which is used by
  # tensorflow/stream_executor/dso_loader.cc.
  _tpl(repository_ctx, "cuda:cuda_config.h",
       {
           "%{cuda_version}": cuda_version,
           "%{cudnn_version}": cudnn_version,
           "%{cuda_compute_capabilities}": ",".join(
               ["CudaVersion(\"%s\")" % c for c in compute_capabilities]),
       })


def _cuda_autoconf_impl(repository_ctx):
  """Implementation of the cuda_autoconf repository rule."""
  if not _enable_cuda(repository_ctx):
    _create_dummy_repository(repository_ctx)
  else:
    _create_cuda_repository(repository_ctx)


cuda_configure = repository_rule(
    implementation = _cuda_autoconf_impl,
    local = True,
)
"""Detects and configures the local CUDA toolchain.

Add the following to your WORKSPACE FILE:

```python
cuda_configure(name = "local_config_cuda")
```

Args:
  name: A unique name for this workspace rule.
"""
