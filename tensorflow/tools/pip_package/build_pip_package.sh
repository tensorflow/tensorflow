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


set -e

function cp_external() {
  local src_dir=$1
  local dest_dir=$2
  for f in `find "$src_dir" -maxdepth 1 -mindepth 1 ! -name '*local_config_cuda*'`; do
    cp -R "$f" "$dest_dir"
  done
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

function main() {
  if [ $# -lt 1 ] ; then
    echo "No destination dir provided"
    exit 1
  fi

  DEST=$1
  TMPDIR=$(mktemp -d -t tmp.XXXXXXXXXX)

  GPU_FLAG=""
  while true; do
    if [[ "$1" == "--gpu" ]]; then
      GPU_FLAG="--project_name tensorflow_gpu"
    fi
    shift

    if [[ -z "$1" ]]; then
      break
    fi
  done

  echo $(date) : "=== Using tmpdir: ${TMPDIR}"

  if [ ! -d bazel-bin/tensorflow ]; then
    echo "Could not find bazel-bin.  Did you run from the root of the build tree?"
    exit 1
  fi

  if is_windows; then
    rm -rf ./bazel-bin/tensorflow/tools/pip_package/simple_console_for_window_unzip
    mkdir -p ./bazel-bin/tensorflow/tools/pip_package/simple_console_for_window_unzip
    echo "Unzipping simple_console_for_windows.zip to create runfiles tree..."
    unzip -o -q ./bazel-bin/tensorflow/tools/pip_package/simple_console_for_windows.zip -d ./bazel-bin/tensorflow/tools/pip_package/simple_console_for_window_unzip
    echo "Unzip finished."
    # runfiles structure after unzip the python binary
    cp -R \
      bazel-bin/tensorflow/tools/pip_package/simple_console_for_window_unzip/runfiles/org_tensorflow/tensorflow \
      "${TMPDIR}"
    mkdir "${TMPDIR}/external"
    # Note: this makes an extra copy of org_tensorflow.
    cp_external \
      bazel-bin/tensorflow/tools/pip_package/simple_console_for_window_unzip/runfiles \
      "${TMPDIR}/external"
    RUNFILES=bazel-bin/tensorflow/tools/pip_package/simple_console_for_window_unzip/runfiles/org_tensorflow
  elif [ ! -d bazel-bin/tensorflow/tools/pip_package/build_pip_package.runfiles/org_tensorflow ]; then
    # Really old (0.2.1-) runfiles, without workspace name.
    cp -R \
      bazel-bin/tensorflow/tools/pip_package/build_pip_package.runfiles/tensorflow \
      "${TMPDIR}"
    mkdir "${TMPDIR}/external"
    cp_external \
      bazel-bin/tensorflow/tools/pip_package/build_pip_package.runfiles/external \
      "${TMPDIR}/external"
    RUNFILES=bazel-bin/tensorflow/tools/pip_package/build_pip_package.runfiles
    # Copy MKL libs over so they can be loaded at runtime
    if [ -d bazel-bin/tensorflow/tools/pip_package/build_pip_package.runfiles/org_tensorflow/_solib_k8/_U_S_Sthird_Uparty_Smkl_Cintel_Ubinary_Ublob___Uthird_Uparty_Smkl ]; then
      mkdir "${TMPDIR}/_solib_k8"
  		cp -R \
  			bazel-bin/tensorflow/tools/pip_package/build_pip_package.runfiles/org_tensorflow/_solib_k8/_U_S_Sthird_Uparty_Smkl_Cintel_Ubinary_Ublob___Uthird_Uparty_Smkl \
        "${TMPDIR}/_solib_k8"
    fi
  else
    if [ -d bazel-bin/tensorflow/tools/pip_package/build_pip_package.runfiles/org_tensorflow/external ]; then
      # Old-style runfiles structure (--legacy_external_runfiles).
      cp -R \
        bazel-bin/tensorflow/tools/pip_package/build_pip_package.runfiles/org_tensorflow/tensorflow \
        "${TMPDIR}"
      mkdir "${TMPDIR}/external"
      cp_external \
        bazel-bin/tensorflow/tools/pip_package/build_pip_package.runfiles/org_tensorflow/external \
        "${TMPDIR}/external"
      # Copy MKL libs over so they can be loaded at runtime
      if [ -d bazel-bin/tensorflow/tools/pip_package/build_pip_package.runfiles/org_tensorflow/_solib_k8/_U_S_Sthird_Uparty_Smkl_Cintel_Ubinary_Ublob___Uthird_Uparty_Smkl ]; then
        mkdir "${TMPDIR}/_solib_k8"
        cp -R \
          bazel-bin/tensorflow/tools/pip_package/build_pip_package.runfiles/org_tensorflow/_solib_k8/_U_S_Sthird_Uparty_Smkl_Cintel_Ubinary_Ublob___Uthird_Uparty_Smkl \
          "${TMPDIR}/_solib_k8"
      fi
    else
      # New-style runfiles structure (--nolegacy_external_runfiles).
      cp -R \
        bazel-bin/tensorflow/tools/pip_package/build_pip_package.runfiles/org_tensorflow/tensorflow \
        "${TMPDIR}"
      mkdir "${TMPDIR}/external"
      # Note: this makes an extra copy of org_tensorflow.
      cp_external \
        bazel-bin/tensorflow/tools/pip_package/build_pip_package.runfiles \
        "${TMPDIR}/external"
      # Copy MKL libs over so they can be loaded at runtime
      if [ -d bazel-bin/tensorflow/tools/pip_package/build_pip_package.runfiles/org_tensorflow/_solib_k8/_U_S_Sthird_Uparty_Smkl_Cintel_Ubinary_Ublob___Uthird_Uparty_Smkl ]; then
        mkdir "${TMPDIR}/_solib_k8"
    		cp -R \
    			bazel-bin/tensorflow/tools/pip_package/build_pip_package.runfiles/org_tensorflow/_solib_k8/_U_S_Sthird_Uparty_Smkl_Cintel_Ubinary_Ublob___Uthird_Uparty_Smkl \
          "${TMPDIR}/_solib_k8"
      fi
    fi
    RUNFILES=bazel-bin/tensorflow/tools/pip_package/build_pip_package.runfiles/org_tensorflow
  fi

  # protobuf pip package doesn't ship with header files. Copy the headers
  # over so user defined ops can be compiled.
  mkdir -p ${TMPDIR}/google
  mkdir -p ${TMPDIR}/third_party
  pushd ${RUNFILES%org_tensorflow}
  for header in $(find protobuf -name \*.h); do
    mkdir -p "${TMPDIR}/google/$(dirname ${header})"
    cp "$header" "${TMPDIR}/google/$(dirname ${header})/"
  done
  popd
  cp -R $RUNFILES/third_party/eigen3 ${TMPDIR}/third_party

  cp tensorflow/tools/pip_package/MANIFEST.in ${TMPDIR}
  cp tensorflow/tools/pip_package/README ${TMPDIR}
  cp tensorflow/tools/pip_package/setup.py ${TMPDIR}

  # Before we leave the top-level directory, make sure we know how to
  # call python.
  source tools/python_bin_path.sh

  pushd ${TMPDIR}
  rm -f MANIFEST
  echo $(date) : "=== Building wheel"
  "${PYTHON_BIN_PATH:-python}" setup.py bdist_wheel ${GPU_FLAG} >/dev/null
  mkdir -p ${DEST}
  cp dist/* ${DEST}
  popd
  rm -rf ${TMPDIR}
  echo $(date) : "=== Output wheel file is in: ${DEST}"
}

main "$@"
