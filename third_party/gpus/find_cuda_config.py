# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Prints CUDA library and header directories and versions found on the system.

The script searches for CUDA library and header files on the system, inspects
them to determine their version and prints the configuration to stdout.
The paths to inspect and the required versions are specified through environment
variables. If no valid configuration is found, the script prints to stderr and
returns an error code.

The list of libraries to find is specified as arguments. Supported libraries are
CUDA (includes cuBLAS), cuDNN, NCCL, and TensorRT.

The script takes a list of base directories specified by the TF_CUDA_PATHS
environment variable as comma-separated glob list. The script looks for headers
and library files in a hard-coded set of subdirectories from these base paths.
If TF_CUDA_PATHS is not specified, a OS specific default is used:

  Linux:   /usr/local/cuda, /usr, and paths from 'ldconfig -p'.
  Windows: CUDA_PATH environment variable, or
           C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\*

For backwards compatibility, some libraries also use alternative base
directories from other environment variables if they are specified. List of
library-specific environment variables:

  Library   Version env variable  Additional base directories
  ----------------------------------------------------------------
  CUDA      TF_CUDA_VERSION       CUDA_TOOLKIT_PATH
  cuBLAS    TF_CUBLAS_VERSION     CUDA_TOOLKIT_PATH
  cuDNN     TF_CUDNN_VERSION      CUDNN_INSTALL_PATH
  NCCL      TF_NCCL_VERSION       NCCL_INSTALL_PATH, NCCL_HDR_PATH
  TensorRT  TF_TENSORRT_VERSION   TENSORRT_INSTALL_PATH

Versions environment variables can be of the form 'x' or 'x.y' to request a
specific version, empty or unspecified to accept any version.

The output of a found library is of the form:
tf_<library>_version: x.y.z
tf_<library>_header_dir: ...
tf_<library>_library_dir: ...
"""

import io
import os
import glob
import platform
import re
import subprocess
import sys

# pylint: disable=g-import-not-at-top
try:
  from shutil import which
except ImportError:
  from distutils.spawn import find_executable as which
# pylint: enable=g-import-not-at-top


class ConfigError(Exception):
  pass


def _is_linux():
  return platform.system() == "Linux"


def _is_windows():
  return platform.system() == "Windows"


def _is_macos():
  return platform.system() == "Darwin"


def _matches_version(actual_version, required_version):
  """Checks whether some version meets the requirements.

      All elements of the required_version need to be present in the
      actual_version.

          required_version  actual_version  result
          -----------------------------------------
          1                 1.1             True
          1.2               1               False
          1.2               1.3             False
                            1               True

      Args:
        required_version: The version specified by the user.
        actual_version: The version detected from the CUDA installation.
      Returns: Whether the actual version matches the required one.
  """
  if actual_version is None:
    return False

  # Strip spaces from the versions.
  actual_version = actual_version.strip()
  required_version = required_version.strip()
  return actual_version.startswith(required_version)


def _at_least_version(actual_version, required_version):
  actual = [int(v) for v in actual_version.split(".")]
  required = [int(v) for v in required_version.split(".")]
  return actual >= required


def _get_header_version(path, name):
  """Returns preprocessor defines in C header file."""
  for line in io.open(path, "r", encoding="utf-8").readlines():
    match = re.match("#define %s +(\d+)" % name, line)
    if match:
      return match.group(1)
  return ""


def _cartesian_product(first, second):
  """Returns all path combinations of first and second."""
  return [os.path.join(f, s) for f in first for s in second]


def _get_ld_config_paths():
  """Returns all directories from 'ldconfig -p'."""
  if not _is_linux():
    return []
  ldconfig_path = which("ldconfig") or "/sbin/ldconfig"
  output = subprocess.check_output([ldconfig_path, "-p"])
  pattern = re.compile(".* => (.*)")
  result = set()
  for line in output.splitlines():
    try:
      match = pattern.match(line.decode("ascii"))
    except UnicodeDecodeError:
      match = False
    if match:
      result.add(os.path.dirname(match.group(1)))
  return sorted(list(result))


def _get_default_cuda_paths(cuda_version):
  if not cuda_version:
    cuda_version = "*"
  elif not "." in cuda_version:
    cuda_version = cuda_version + ".*"

  if _is_windows():
    return [
        os.environ.get(
            "CUDA_PATH",
            "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v%s\\" %
            cuda_version)
    ]
  return ["/usr/local/cuda-%s" % cuda_version, "/usr/local/cuda", "/usr",
         "/usr/local/cudnn"] + _get_ld_config_paths()


def _header_paths():
  """Returns hard-coded set of relative paths to look for header files."""
  return [
      "",
      "include",
      "include/cuda",
      "include/*-linux-gnu",
      "extras/CUPTI/include",
      "include/cuda/CUPTI",
      "local/cuda/extras/CUPTI/include",
  ]


def _library_paths():
  """Returns hard-coded set of relative paths to look for library files."""
  return [
      "",
      "lib64",
      "lib",
      "lib/*-linux-gnu",
      "lib/x64",
      "extras/CUPTI/*",
      "local/cuda/lib64",
      "local/cuda/extras/CUPTI/lib64",
  ]


def _not_found_error(base_paths, relative_paths, filepattern):
  base_paths = "".join(["\n        '%s'" % path for path in sorted(base_paths)])
  relative_paths = "".join(["\n        '%s'" % path for path in relative_paths])
  return ConfigError(
      "Could not find any %s in any subdirectory:%s\nof:%s\n" %
      (filepattern, relative_paths, base_paths))


def _find_file(base_paths, relative_paths, filepattern):
  for path in _cartesian_product(base_paths, relative_paths):
    for file in glob.glob(os.path.join(path, filepattern)):
      return file
  raise _not_found_error(base_paths, relative_paths, filepattern)


def _find_library(base_paths, library_name, required_version):
  """Returns first valid path to the requested library."""
  if _is_windows():
    filepattern = library_name + ".lib"
  elif _is_macos():
    filepattern = "%s*.dylib" % (".".join(["lib" + library_name] +
                                          required_version.split(".")[:1]))
  else:
    filepattern = ".".join(["lib" + library_name, "so"] +
                           required_version.split(".")[:1]) + "*"
  return _find_file(base_paths, _library_paths(), filepattern)


def _find_versioned_file(base_paths, relative_paths, filepatterns,
                         required_version, get_version):
  """Returns first valid path to a file that matches the requested version."""
  if type(filepatterns) not in [list, tuple]:
    filepatterns = [filepatterns]
  for path in _cartesian_product(base_paths, relative_paths):
    for filepattern in filepatterns:
      for file in glob.glob(os.path.join(path, filepattern)):
        actual_version = get_version(file)
        if _matches_version(actual_version, required_version):
          return file, actual_version
  raise _not_found_error(
      base_paths, relative_paths,
      ", ".join(filepatterns) + " matching version '%s'" % required_version)


def _find_header(base_paths, header_name, required_version, get_version):
  """Returns first valid path to a header that matches the requested version."""
  return _find_versioned_file(base_paths, _header_paths(), header_name,
                              required_version, get_version)


def _find_cuda_config(base_paths, required_version):

  def get_header_version(path):
    version = int(_get_header_version(path, "CUDA_VERSION"))
    if not version:
      return None
    return "%d.%d" % (version // 1000, version % 1000 // 10)

  cuda_header_path, header_version = _find_header(base_paths, "cuda.h",
                                                  required_version,
                                                  get_header_version)
  cuda_version = header_version  # x.y, see above.

  cuda_library_path = _find_library(base_paths, "cudart", cuda_version)

  def get_nvcc_version(path):
    pattern = "Cuda compilation tools, release \d+\.\d+, V(\d+\.\d+\.\d+)"
    for line in subprocess.check_output([path, "--version"]).splitlines():
      match = re.match(pattern, line.decode("ascii"))
      if match:
        return match.group(1)
    return None

  nvcc_name = "nvcc.exe" if _is_windows() else "nvcc"
  nvcc_path, nvcc_version = _find_versioned_file(base_paths, [
      "",
      "bin",
      "local/cuda/bin",
  ], nvcc_name, cuda_version, get_nvcc_version)

  nvvm_path = _find_file(base_paths, [
      "nvvm/libdevice",
      "share/cuda",
      "lib/nvidia-cuda-toolkit/libdevice",
      "local/cuda/nvvm/libdevice",
  ], "libdevice*.10.bc")

  cupti_header_path = _find_file(base_paths, _header_paths(), "cupti.h")
  cupti_library_path = _find_library(base_paths, "cupti", required_version)

  cuda_binary_dir = os.path.dirname(nvcc_path)
  nvvm_library_dir = os.path.dirname(nvvm_path)

  # XLA requires the toolkit path to find ptxas and libdevice.
  # TODO(csigg): pass in both directories instead.
  cuda_toolkit_paths = (
      os.path.normpath(os.path.join(cuda_binary_dir, "..")),
      os.path.normpath(os.path.join(nvvm_library_dir, "../..")),
  )
  if cuda_toolkit_paths[0] != cuda_toolkit_paths[1]:
    raise ConfigError("Inconsistent CUDA toolkit path: %s vs %s" %
                      cuda_toolkit_paths)

  return {
      "cuda_version": cuda_version,
      "cuda_include_dir": os.path.dirname(cuda_header_path),
      "cuda_library_dir": os.path.dirname(cuda_library_path),
      "cuda_binary_dir": cuda_binary_dir,
      "nvvm_library_dir": nvvm_library_dir,
      "cupti_include_dir": os.path.dirname(cupti_header_path),
      "cupti_library_dir": os.path.dirname(cupti_library_path),
      "cuda_toolkit_path": cuda_toolkit_paths[0],
  }


def _find_cublas_config(base_paths, required_version, cuda_version):

  if _at_least_version(cuda_version, "10.1"):

    def get_header_version(path):
      version = (
          _get_header_version(path, name)
          for name in ("CUBLAS_VER_MAJOR", "CUBLAS_VER_MINOR",
                       "CUBLAS_VER_PATCH"))
      return ".".join(version)

    header_path, header_version = _find_header(base_paths, "cublas_api.h",
                                               required_version,
                                               get_header_version)
    # cuBLAS uses the major version only.
    cublas_version = header_version.split(".")[0]

  else:
    # There is no version info available before CUDA 10.1, just find the file.
    header_version = cuda_version
    header_path = _find_file(base_paths, _header_paths(), "cublas_api.h")
    # cuBLAS version is the same as CUDA version (x.y).
    cublas_version = required_version

  library_path = _find_library(base_paths, "cublas", cublas_version)

  return {
      "cublas_version": header_version,
      "cublas_include_dir": os.path.dirname(header_path),
      "cublas_library_dir": os.path.dirname(library_path),
  }


def _find_cusolver_config(base_paths, required_version, cuda_version):

  if _at_least_version(cuda_version, "11.0"):

    def get_header_version(path):
      version = (
          _get_header_version(path, name)
          for name in ("CUSOLVER_VER_MAJOR", "CUSOLVER_VER_MINOR",
                       "CUSOLVER_VER_PATCH"))
      return ".".join(version)

    header_path, header_version = _find_header(base_paths, "cusolver_common.h",
                                               required_version,
                                               get_header_version)
    cusolver_version = header_version.split(".")[0]

  else:
    header_version = cuda_version
    header_path = _find_file(base_paths, _header_paths(), "cusolver_common.h")
    cusolver_version = required_version

  library_path = _find_library(base_paths, "cusolver", cusolver_version)

  return {
      "cusolver_version": header_version,
      "cusolver_include_dir": os.path.dirname(header_path),
      "cusolver_library_dir": os.path.dirname(library_path),
  }


def _find_curand_config(base_paths, required_version, cuda_version):

  if _at_least_version(cuda_version, "11.0"):

    def get_header_version(path):
      version = (
          _get_header_version(path, name)
          for name in ("CURAND_VER_MAJOR", "CURAND_VER_MINOR",
                       "CURAND_VER_PATCH"))
      return ".".join(version)

    header_path, header_version = _find_header(base_paths, "curand.h",
                                               required_version,
                                               get_header_version)
    curand_version = header_version.split(".")[0]

  else:
    header_version = cuda_version
    header_path = _find_file(base_paths, _header_paths(), "curand.h")
    curand_version = required_version

  library_path = _find_library(base_paths, "curand", curand_version)

  return {
      "curand_version": header_version,
      "curand_include_dir": os.path.dirname(header_path),
      "curand_library_dir": os.path.dirname(library_path),
  }


def _find_cufft_config(base_paths, required_version, cuda_version):

  if _at_least_version(cuda_version, "11.0"):

    def get_header_version(path):
      version = (
          _get_header_version(path, name)
          for name in ("CUFFT_VER_MAJOR", "CUFFT_VER_MINOR", "CUFFT_VER_PATCH"))
      return ".".join(version)

    header_path, header_version = _find_header(base_paths, "cufft.h",
                                               required_version,
                                               get_header_version)
    cufft_version = header_version.split(".")[0]

  else:
    header_version = cuda_version
    header_path = _find_file(base_paths, _header_paths(), "cufft.h")
    cufft_version = required_version

  library_path = _find_library(base_paths, "cufft", cufft_version)

  return {
      "cufft_version": header_version,
      "cufft_include_dir": os.path.dirname(header_path),
      "cufft_library_dir": os.path.dirname(library_path),
  }


def _find_cudnn_config(base_paths, required_version):

  def get_header_version(path):
    version = [
        _get_header_version(path, name)
        for name in ("CUDNN_MAJOR", "CUDNN_MINOR", "CUDNN_PATCHLEVEL")]
    return ".".join(version) if version[0] else None

  header_path, header_version = _find_header(base_paths,
                                             ("cudnn.h", "cudnn_version.h"),
                                             required_version,
                                             get_header_version)
  cudnn_version = header_version.split(".")[0]

  library_path = _find_library(base_paths, "cudnn", cudnn_version)

  return {
      "cudnn_version": cudnn_version,
      "cudnn_include_dir": os.path.dirname(header_path),
      "cudnn_library_dir": os.path.dirname(library_path),
  }


def _find_cusparse_config(base_paths, required_version, cuda_version):

  if _at_least_version(cuda_version, "11.0"):

    def get_header_version(path):
      version = (
          _get_header_version(path, name)
          for name in ("CUSPARSE_VER_MAJOR", "CUSPARSE_VER_MINOR",
                       "CUSPARSE_VER_PATCH"))
      return ".".join(version)

    header_path, header_version = _find_header(base_paths, "cusparse.h",
                                               required_version,
                                               get_header_version)
    cusparse_version = header_version.split(".")[0]

  else:
    header_version = cuda_version
    header_path = _find_file(base_paths, _header_paths(), "cusparse.h")
    cusparse_version = required_version

  library_path = _find_library(base_paths, "cusparse", cusparse_version)

  return {
      "cusparse_version": header_version,
      "cusparse_include_dir": os.path.dirname(header_path),
      "cusparse_library_dir": os.path.dirname(library_path),
  }


def _find_nccl_config(base_paths, required_version):

  def get_header_version(path):
    version = (
        _get_header_version(path, name)
        for name in ("NCCL_MAJOR", "NCCL_MINOR", "NCCL_PATCH"))
    return ".".join(version)

  header_path, header_version = _find_header(base_paths, "nccl.h",
                                             required_version,
                                             get_header_version)
  nccl_version = header_version.split(".")[0]

  library_path = _find_library(base_paths, "nccl", nccl_version)

  return {
      "nccl_version": nccl_version,
      "nccl_include_dir": os.path.dirname(header_path),
      "nccl_library_dir": os.path.dirname(library_path),
  }


def _find_tensorrt_config(base_paths, required_version):

  def get_header_version(path):
    version = (
        _get_header_version(path, name)
        for name in ("NV_TENSORRT_MAJOR", "NV_TENSORRT_MINOR",
                     "NV_TENSORRT_PATCH"))
    # `version` is a generator object, so we convert it to a list before using
    # it (muitiple times below).
    version = list(version)
    if not all(version):
      return None  # Versions not found, make _matches_version returns False.
    return ".".join(version)

  header_path, header_version = _find_header(base_paths, "NvInferVersion.h",
                                             required_version,
                                             get_header_version)

  tensorrt_version = header_version.split(".")[0]
  library_path = _find_library(base_paths, "nvinfer", tensorrt_version)

  return {
      "tensorrt_version": tensorrt_version,
      "tensorrt_include_dir": os.path.dirname(header_path),
      "tensorrt_library_dir": os.path.dirname(library_path),
  }


def _list_from_env(env_name, default=[]):
  """Returns comma-separated list from environment variable."""
  if env_name in os.environ:
    return os.environ[env_name].split(",")
  return default


def _get_legacy_path(env_name, default=[]):
  """Returns a path specified by a legacy environment variable.

  CUDNN_INSTALL_PATH, NCCL_INSTALL_PATH, TENSORRT_INSTALL_PATH set to
  '/usr/lib/x86_64-linux-gnu' would previously find both library and header
  paths. Detect those and return '/usr', otherwise forward to _list_from_env().
  """
  if env_name in os.environ:
    match = re.match("^(/[^/ ]*)+/lib/\w+-linux-gnu/?$", os.environ[env_name])
    if match:
      return [match.group(1)]
  return _list_from_env(env_name, default)


def _normalize_path(path):
  """Returns normalized path, with forward slashes on Windows."""
  path = os.path.realpath(path)
  if _is_windows():
    path = path.replace("\\", "/")
  return path


def find_cuda_config():
  """Returns a dictionary of CUDA library and header file paths."""
  libraries = [argv.lower() for argv in sys.argv[1:]]
  cuda_version = os.environ.get("TF_CUDA_VERSION", "")
  base_paths = _list_from_env("TF_CUDA_PATHS",
                              _get_default_cuda_paths(cuda_version))
  base_paths = [path for path in base_paths if os.path.exists(path)]

  result = {}
  if "cuda" in libraries:
    cuda_paths = _list_from_env("CUDA_TOOLKIT_PATH", base_paths)
    result.update(_find_cuda_config(cuda_paths, cuda_version))

    cuda_version = result["cuda_version"]
    cublas_paths = base_paths
    if tuple(int(v) for v in cuda_version.split(".")) < (10, 1):
      # Before CUDA 10.1, cuBLAS was in the same directory as the toolkit.
      cublas_paths = cuda_paths
    cublas_version = os.environ.get("TF_CUBLAS_VERSION", "")
    result.update(
        _find_cublas_config(cublas_paths, cublas_version, cuda_version))

    cusolver_paths = base_paths
    if tuple(int(v) for v in cuda_version.split(".")) < (11, 0):
      cusolver_paths = cuda_paths
    cusolver_version = os.environ.get("TF_CUSOLVER_VERSION", "")
    result.update(
        _find_cusolver_config(cusolver_paths, cusolver_version, cuda_version))

    curand_paths = base_paths
    if tuple(int(v) for v in cuda_version.split(".")) < (11, 0):
      curand_paths = cuda_paths
    curand_version = os.environ.get("TF_CURAND_VERSION", "")
    result.update(
        _find_curand_config(curand_paths, curand_version, cuda_version))

    cufft_paths = base_paths
    if tuple(int(v) for v in cuda_version.split(".")) < (11, 0):
      cufft_paths = cuda_paths
    cufft_version = os.environ.get("TF_CUFFT_VERSION", "")
    result.update(_find_cufft_config(cufft_paths, cufft_version, cuda_version))

    cusparse_paths = base_paths
    if tuple(int(v) for v in cuda_version.split(".")) < (11, 0):
      cusparse_paths = cuda_paths
    cusparse_version = os.environ.get("TF_CUSPARSE_VERSION", "")
    result.update(
        _find_cusparse_config(cusparse_paths, cusparse_version, cuda_version))

  if "cudnn" in libraries:
    cudnn_paths = _get_legacy_path("CUDNN_INSTALL_PATH", base_paths)
    cudnn_version = os.environ.get("TF_CUDNN_VERSION", "")
    result.update(_find_cudnn_config(cudnn_paths, cudnn_version))

  if "nccl" in libraries:
    nccl_paths = _get_legacy_path("NCCL_INSTALL_PATH", base_paths)
    nccl_version = os.environ.get("TF_NCCL_VERSION", "")
    result.update(_find_nccl_config(nccl_paths, nccl_version))

  if "tensorrt" in libraries:
    tensorrt_paths = _get_legacy_path("TENSORRT_INSTALL_PATH", base_paths)
    tensorrt_version = os.environ.get("TF_TENSORRT_VERSION", "")
    result.update(_find_tensorrt_config(tensorrt_paths, tensorrt_version))

  for k, v in result.items():
    if k.endswith("_dir") or k.endswith("_path"):
      result[k] = _normalize_path(v)

  return result


def main():
  try:
    for key, value in sorted(find_cuda_config().items()):
      print("%s: %s" % (key, value))
  except ConfigError as e:
    sys.stderr.write(str(e) + '\n')
    sys.exit(1)


if __name__ == "__main__":
  main()
