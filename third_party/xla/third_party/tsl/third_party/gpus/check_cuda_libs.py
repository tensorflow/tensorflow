# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""Verifies that a list of libraries is installed on the system.

NB: DEPRECATED! This script is a part of the deprecated `cuda_configure` rule.
Please use `hermetic/cuda_configure` instead.

Takes a list of arguments with every two subsequent arguments being a logical
tuple of (path, check_soname). The path to the library and either True or False
to indicate whether to check the soname field on the shared library.

Example Usage:
./check_cuda_libs.py /path/to/lib1.so True /path/to/lib2.so False
"""
import os
import os.path
import platform
import shutil
import subprocess
import sys


class ConfigError(Exception):
  pass


def _is_windows():
  return platform.system() == "Windows"


def check_cuda_lib(path, check_soname=True):
  """Tests if a library exists on disk and whether its soname matches the filename.

  Args:
    path: the path to the library.
    check_soname: whether to check the soname as well.

  Raises:
    ConfigError: If the library does not exist or if its soname does not match
    the filename.
  """
  if not os.path.isfile(path):
    raise ConfigError("No library found under: " + path)
  objdump = shutil.which("objdump")
  if check_soname and objdump is not None and not _is_windows():
    # Decode is necessary as in py3 the return type changed from str to bytes
    output = subprocess.check_output([objdump, "-p", path]).decode("utf-8")
    output = [line for line in output.splitlines() if "SONAME" in line]
    sonames = [line.strip().split(" ")[-1] for line in output]
    if not any(soname == os.path.basename(path) for soname in sonames):
      raise ConfigError("None of the libraries match their SONAME: " + path)


def main():
  try:
    args = [argv for argv in sys.argv[1:]]
    if len(args) % 2 == 1:
      raise ConfigError("Expected even number of arguments")
    checked_paths = []
    for i in range(0, len(args), 2):
      path = args[i]
      check_cuda_lib(path, check_soname=args[i + 1] == "True")
      checked_paths.append(path)
    # pylint: disable=superfluous-parens
    print(os.linesep.join(checked_paths))
    # pylint: enable=superfluous-parens
  except ConfigError as e:
    sys.stderr.write(str(e))
    sys.exit(1)


if __name__ == "__main__":
  main()
