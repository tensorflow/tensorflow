#!/usr/bin/env bash
# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
# =============================================================================


set -e

BINARY="bazel-bin/tensorflow/contrib/tpu/profiler/capture_tpu_profile"
PACKAGE_NAME="cloud_tpu_profiler"
PIP_PACKAGE="tensorflow/contrib/tpu/profiler/pip_package"

function main() {
  if [ $# -lt 1 ] ; then
    echo "No destination dir provided"
    exit 1
  fi

  DEST=$1
  TMPDIR=$(mktemp -d -t tmp.XXXXXXXXXX)

  echo $(date) : "=== Using tmpdir: ${TMPDIR}"

  if [ ! -f "${BINARY}" ]; then
    echo "Could not find ${BINARY}.  Did you run from the root of the build tree?"
    exit 1
  fi

  cp ${PIP_PACKAGE}/README ${TMPDIR}
  cp ${PIP_PACKAGE}/setup.py ${TMPDIR}
  cp -R ${PIP_PACKAGE}/${PACKAGE_NAME} ${TMPDIR}
  mkdir ${TMPDIR}/${PACKAGE_NAME}/data
  cp ${BINARY} ${TMPDIR}/${PACKAGE_NAME}/data
  echo $(ls $TMPDIR)

  pushd ${TMPDIR}
  echo $(date) : "=== Building wheel"
  echo $(pwd)
  python setup.py bdist_wheel >/dev/null
  python3 setup.py bdist_wheel >/dev/null
  mkdir -p ${DEST}
  cp dist/* ${DEST}
  popd
  rm -rf ${TMPDIR}
  echo $(date) : "=== Output wheel file is in: ${DEST}"
}

main "$@"
