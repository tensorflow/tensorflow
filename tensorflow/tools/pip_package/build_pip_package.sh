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

function main() {
  if [ $# -lt 1 ] ; then
    echo "No destination dir provided"
    exit 1
  fi

  DEST=$1
  TMPDIR=$(mktemp -d -t tmp.XXXXXXXXXX)

  echo $(date) : "=== Using tmpdir: ${TMPDIR}"

  if [ ! -d bazel-bin/tensorflow ]; then
    echo "Could not find bazel-bin.  Did you run from the root of the build tree?"
    exit 1
  fi

  if [ ! -d bazel-bin/tensorflow/tools/pip_package/build_pip_package.runfiles/org_tensorflow ]; then
    # Really old (0.2.1-) runfiles, without workspace name.
    cp -R \
      bazel-bin/tensorflow/tools/pip_package/build_pip_package.runfiles/tensorflow \
      "${TMPDIR}"
    mkdir "${TMPDIR}/external"
    cp_external \
      bazel-bin/tensorflow/tools/pip_package/build_pip_package.runfiles/external \
      "${TMPDIR}/external"
    RUNFILES=bazel-bin/tensorflow/tools/pip_package/build_pip_package.runfiles
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
    fi
    RUNFILES=bazel-bin/tensorflow/tools/pip_package/build_pip_package.runfiles/org_tensorflow
  fi

  # protobuf pip package doesn't ship with header files. Copy the headers
  # over so user defined ops can be compiled.
  mkdir -p ${TMPDIR}/google
  rsync --include "*/" --include "*.h" --exclude "*" --prune-empty-dirs -a \
    $RUNFILES/external/protobuf ${TMPDIR}/google
  rsync -a $RUNFILES/third_party/eigen3 ${TMPDIR}/third_party

  cp tensorflow/tools/pip_package/MANIFEST.in ${TMPDIR}
  cp tensorflow/tools/pip_package/README ${TMPDIR}
  cp tensorflow/tools/pip_package/setup.py ${TMPDIR}

  # Before we leave the top-level directory, make sure we know how to
  # call python.
  source tools/python_bin_path.sh

  pushd ${TMPDIR}
  rm -f MANIFEST
  echo $(date) : "=== Building wheel"
  ${PYTHON_BIN_PATH:-python} setup.py bdist_wheel >/dev/null
  mkdir -p ${DEST}
  cp dist/* ${DEST}
  popd
  rm -rf ${TMPDIR}
  echo $(date) : "=== Output wheel file is in: ${DEST}"
}

main "$@"
