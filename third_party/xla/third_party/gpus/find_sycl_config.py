# Copyright 2024 The TensorFlow Authors. All Rights Reserved.
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
"""Prints SYCL library and header directories and versions found on the system.

The script searches for SYCL library and header files on the system, inspects
them to determine their version and prints the configuration to stdout. The path
to inspect is specified through an environment variable (SYCL_PATH). If no valid
configuration is found, the script prints to stderr and returns an error code.
The script takes the directory specified by the SYCL_PATH environment variable.
The script looks for headers and library files in a hard-coded set of
subdirectories from base path of the specified directory. If SYCL_PATH is not
specified, then "/opt/sycl" is used as it default value
"""

import io
import os
import glob
import re
import sys


class ConfigError(Exception):
  pass

def _get_default_sycl_toolkit_path():
  return "/opt/intel/oneapi/compiler/latest"

def _get_toolkit_path():
  """Determines and returns the SYCL installation path."""
  sycl_toolkit_path = None
  sycl_toolkit_path = _get_default_sycl_toolkit_path()
  if "SYCL_TOOLKIT_PATH" in os.environ:
    sycl_toolkit_path = os.environ["SYCL_TOOLKIT_PATH"]
  return os.path.realpath(sycl_toolkit_path)

def _get_basekit_path():
  return _get_toolkit_path().split("/compiler/")[0]

def _get_basekit_version():
  try:
    version = _get_toolkit_path().split("/compiler/")[1].split("/")[0]
  except:
    version = "unknown"
  return version

def _get_composite_version_number(major, minor, patch):
  return 10000 * major + 100 * minor + patch

def _list_from_env(env_name, default=[]):
  """Returns comma-separated list from environment variable."""
  if env_name in os.environ:
    return os.environ[env_name].split(",")
  return default

def _get_legacy_path(env_name, default=[]):
  if env_name in os.environ:
    match = re.match(r"^(/[^/ ]*)+/lib/\w+-linux-gnu/?$", os.environ[env_name])
    if match:
      return [match.group(1)]
  return _list_from_env(env_name, default)

def _get_header_version(path, name):
  """Returns preprocessor defines in C header file."""
  for line in io.open(path, "r", encoding="utf-8"):
    match = re.match(r"#define %s +(\d+)" % name, line)
    if match:
      value = match.group(1)
      return int(value)

  raise ConfigError(
      '#define "{}" is either\n'.format(name)
      + "  not present in file {} OR\n".format(path)
      + "  its value is not an integer literal"
  )


def _cartesian_product(first, second):
  """Returns all path combinations of first and second."""
  return [os.path.join(f, s) for f in first for s in second]

def _header_paths():
  """Returns hard-coded set of relative paths to look for header files."""
  return [
      "",
      "include",
      "include/level_zero",
      "include/oneapi",
  ]

def _library_paths():
  """Returns hard-coded set of relative paths to look for library files."""
  return [
      "",
      "lib64",
      "lib",
      "lib/*-linux-gnu",
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

def _find_header(base_paths, header_name):
  """Returns first valid path to a header."""
  return _find_file(base_paths, _header_paths(), header_name)

def _find_library(base_paths, library_name):
  """Returns first valid path to the requested library."""
  filepattern = ".".join(["lib" + library_name, "so"]) + "*"
  return _find_file(base_paths, _library_paths(), filepattern)

def _find_sycl_config(toolkit_path):

  def sycl_version_numbers(path):
    possible_version_files = [
        "linux/include/sycl/version.hpp",
        "include/sycl/version.hpp",
    ]
    version_file = None
    for f in possible_version_files:
      version_file_path = os.path.join(path, f)
      if os.path.exists(version_file_path):
        version_file = version_file_path
        break
    if not version_file:
      raise ConfigError(
          "SYCL version file not found in {}".format(possible_version_files)
      )

    major = _get_header_version(version_file, "__LIBSYCL_MAJOR_VERSION")
    minor = _get_header_version(version_file, "__LIBSYCL_MINOR_VERSION")
    patch = _get_header_version(version_file, "__LIBSYCL_PATCH_VERSION")
    return major, minor, patch

  major, minor, patch = sycl_version_numbers(toolkit_path)

  sycl_config = {
    "sycl_version_number": _get_composite_version_number(major, minor, patch),
    "sycl_basekit_version_number": _get_basekit_version(),
  }

  return sycl_config

def _find_mkl_config(base_paths):
  header_path = os.path.dirname(_find_header(base_paths, "mkl.hpp"))
  library_path = os.path.dirname(_find_library(base_paths, "mkl_sycl"))
  return {
    "mkl_include_dir": "/".join(header_path.split("/")[:-1]),
    "mkl_library_dir": library_path,
  }

def _find_l0_config(base_paths):
  header_path = os.path.dirname(_find_header(base_paths, "ze_api.h"))
  library_path = os.path.dirname(_find_library(base_paths, "ze_loader"))
  return {
    "l0_include_dir": header_path,
    "l0_library_dir": library_path,
  }

def find_sycl_config():
  """Returns a dictionary of SYCL components config info."""
  basekit_path = _get_basekit_path()
  toolkit_path = _get_toolkit_path()

  result = {}
  result["sycl_basekit_path"] = basekit_path
  result["sycl_toolkit_path"] = toolkit_path
  result.update(_find_sycl_config(toolkit_path))

  default_mkl_path = [basekit_path + "/mkl/" + _get_basekit_version()]
  mkl_paths = _get_legacy_path("MKL_INSTALL_PATH", default_mkl_path)
  result.update(_find_mkl_config(mkl_paths))

  default_l0_path = ["/usr"]
  l0_paths = _get_legacy_path("L0_INSTALL_PATH", default_l0_path)
  result.update(_find_l0_config(l0_paths))

  return result


def main():
  try:
    for key, value in sorted(find_sycl_config().items()):
      print("%s: %s" % (key, value))
  except ConfigError as e:
    sys.stderr.write("\nERROR: {}\n\n".format(str(e)))
    sys.exit(1)


if __name__ == "__main__":
  main()
