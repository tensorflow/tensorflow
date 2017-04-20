#!/usr/bin/env bash
# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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

set -e -o errexit

if [ -d "../org_tensorflow" ]; then
  script_path="../org_tensorflow"
else
  # Prefix expected paths with ./ locally and external/reponame/ for remote repos.
  # TODO(kchodorow): remove once runfiles paths are fixed, see
  # https://github.com/bazelbuild/bazel/issues/848.
  script_path=$(dirname $(dirname $(dirname "$0")))
  script_path=${script_path:-.}
fi

EXPECTED_PATHS="$script_path/util/python/python_include"\
" $script_path/util/python/python_lib"\
" $script_path/third_party/py/numpy/numpy_include"

function main {
  argument="$1"
  shift
  case $argument in
    --check)
      check_python
      exit 0
      ;;
    --setup)
      setup_python "$1"
      exit 0
      ;;
  esac
}

function python_path {
  "$PYTHON_BIN_PATH" - <<END
from __future__ import print_function
import site
import os

try:
  input = raw_input
except NameError:
  pass

python_paths = []
if os.getenv('PYTHONPATH') is not None:
  python_paths = os.getenv('PYTHONPATH').split(':')
try:
  library_paths = site.getsitepackages()
except AttributeError:
 from distutils.sysconfig import get_python_lib
 library_paths = [get_python_lib()]
all_paths = set(python_paths + library_paths)

paths = []
for path in all_paths:
  if os.path.isdir(path):
    paths.append(path)

if len(paths) == 1:
  print(paths[0])
else:
  ret_paths = ",".join(paths)
  print(ret_paths)
END
}

function default_python_path {
  PYTHON_ARG="$1" "$PYTHON_BIN_PATH" - <<END
from __future__ import print_function
import os

default = os.getenv('PYTHON_ARG')
default = str(default)
print(default)
END
}

function setup_python {
  PYTHON_BIN_PATH="$1";

  if [ -z "$PYTHON_BIN_PATH" ]; then
    echo "PYTHON_BIN_PATH was not provided.  Did you run configure?"
    exit 1
  fi
  if [ ! -x "$PYTHON_BIN_PATH" ]  || [ -d "$PYTHON_BIN_PATH" ]; then
    echo "PYTHON_BIN_PATH is not executable.  Is it the python binary?"
    exit 1
  fi

  local python_major_version=$("${PYTHON_BIN_PATH}" -c 'from __future__ import print_function; import sys; print(sys.version_info[0]);')
  if [ "$python_major_version" == "" ]; then
    echo -e "\n\nERROR: Problem getting python version.  Is $PYTHON_BIN_PATH the correct python binary?"
    exit 1
  fi

  local python_include="$("${PYTHON_BIN_PATH}" -c 'from __future__ import print_function; from distutils import sysconfig; print(sysconfig.get_python_inc());')"
  if [ "$python_include" == "" ]; then
    echo -e "\n\nERROR: Problem getting python include path.  Is distutils installed?"
    exit 1
  fi

  if [ -z "$PYTHON_LIB_PATH" ]; then
    local python_lib_path
    # Split python_path into an array of paths, this allows path containing spaces
    IFS=','
    python_lib_path=($(python_path))
    unset IFS

    if [ 1 = "$USE_DEFAULT_PYTHON_LIB_PATH" ]; then
      PYTHON_LIB_PATH="$(default_python_path "${python_lib_path[0]}")"
      echo "Using python library path: $PYTHON_LIB_PATH"

    else
      echo "Found possible Python library paths:"
      for x in "${python_lib_path[@]}"; do
        echo "  $x"
      done
      set -- "${python_lib_path[@]}"
      echo "Please input the desired Python library path to use.  Default is ["$1"]"
      read b || true
      if [ "$b" == "" ]; then
        PYTHON_LIB_PATH="$(default_python_path "${python_lib_path[0]}")"
        echo "Using python library path: $PYTHON_LIB_PATH"
      else
        PYTHON_LIB_PATH="$b"
      fi
    fi
  fi

  if test -d "$PYTHON_LIB_PATH" -a -x "$PYTHON_LIB_PATH"; then
    python_lib="$PYTHON_LIB_PATH"
  else
    echo -e "\n\nERROR: Invalid python library path: ${PYTHON_LIB_PATH}."
    exit 1
  fi

  local numpy_include=$("${PYTHON_BIN_PATH}" -c 'from __future__ import print_function; import numpy; print(numpy.get_include());')
  if [ "$numpy_include" == "" ]; then
    echo -e "\n\nERROR: Problem getting numpy include path.  Is numpy installed?"
    exit 1
  fi

  for x in $EXPECTED_PATHS; do
    if [ -e "$x" ]; then
      rm -rf "$x"
    fi
  done

# ln -sf is actually implemented as copying in msys since creating symbolic
# links is privileged on Windows. But copying is too slow, so invoke mklink
# to create junctions on Windows.
  if is_windows; then
    cmd /c "mklink /J util\\python\\python_include \"${python_include}\""
    cmd /c "mklink /J util\\python\\python_lib \"${python_lib}\""
    cmd /c "mklink /J third_party\\py\\numpy\\numpy_include \"${numpy_include}\""
  else
    ln -sf "${python_include}" util/python/python_include
    ln -sf "${python_lib}" util/python/python_lib
    ln -sf "${numpy_include}" third_party/py/numpy/numpy_include
  fi
  # Convert python path to Windows style before writing into bazel.rc
  if is_windows; then
    PYTHON_BIN_PATH="$(cygpath -m "$PYTHON_BIN_PATH")"
  fi

  # Write tools/bazel.rc
  echo "# Autogenerated by configure: DO NOT EDIT" > tools/bazel.rc
  sed -e "s/\$PYTHON_MAJOR_VERSION/$python_major_version/g" \
      -e "s|\$PYTHON_BINARY|\"$PYTHON_BIN_PATH\"|g" \
      tools/bazel.rc.template >> tools/bazel.rc
  # Write tools/python_bin_path.sh
  echo "export PYTHON_BIN_PATH=\"$PYTHON_BIN_PATH\"" > tools/python_bin_path.sh
}

PLATFORM="$(uname -s | tr 'A-Z' 'a-z')"
function is_windows() {
  # On windows, the shell script is actually running in msys
  if [[ "${PLATFORM}" =~ msys_nt* ]]; then
    true
  else
    false
  fi
}

function check_python {
  for x in $EXPECTED_PATHS; do
    if [ ! -e "$x" ]; then
      echo -e "\n\nERROR: Cannot find '${x}'.  Did you run configure?\n\n" 1>&2
      exit 1
    fi
    # Don't check symbolic link on Windows
    if ! is_windows && [ ! -L "${x}" ]; then
      echo -e "\n\nERROR: '${x}' is not a symbolic link.  Internal error.\n\n" 1>&2
      exit 1
    fi
    if is_windows; then
      # In msys, readlink <path> doesn't work, because no symbolic link on
      # Windows. readlink -f <path> returns the real path of a junction.
      true_path=$(readlink -f "${x}")
    else
      true_path=$(readlink "${x}")
    fi
    if [ ! -d "${true_path}" ]; then
      echo -e "\n\nERROR: '${x}' does not refer to an existing directory: ${true_path}.  Do you need to rerun configure?\n\n" 1>&2
      exit 1
    fi
  done
}

main "$@"
