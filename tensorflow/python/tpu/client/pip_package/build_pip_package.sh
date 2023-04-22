#!/usr/bin/env bash
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
# =============================================================================


set -e

if [ "$(uname)" = "Darwin" ]; then
  sedi="sed -i ''"
else
  sedi="sed -i"
fi

PACKAGE_NAME="cloud_tpu_client"
PIP_PACKAGE="tensorflow/python/tpu/client/pip_package"
RUNFILES="bazel-bin/tensorflow/python/tpu/client/pip_package/build_pip_package.runfiles/org_tensorflow/tensorflow/python/tpu/client"

function main() {
  if [ $# -lt 1 ] ; then
    echo "No destination dir provided"
    exit 1
  fi

  DEST=$1
  TMPDIR=$(mktemp -d -t tmp.XXXXXXXXXX)

  echo $(date) : "=== Using tmpdir: ${TMPDIR}"

  cp ${PIP_PACKAGE}/README ${TMPDIR}
  cp ${PIP_PACKAGE}/setup.py ${TMPDIR}
  mkdir ${TMPDIR}/${PACKAGE_NAME}
  cp -a ${RUNFILES}/. ${TMPDIR}/${PACKAGE_NAME}/

  # Fix the import statements to reflect the copied over path.
  find  ${TMPDIR}/${PACKAGE_NAME} -name \*.py |
    xargs $sedi -e '
      s/^from tensorflow.python.tpu.client/from '${PACKAGE_NAME}'/
  '
  echo $(ls $TMPDIR)

  pushd ${TMPDIR}
  echo $(date) : "=== Building wheel"
  echo $(pwd)
  python3 setup.py bdist_wheel >/dev/null
  mkdir -p ${DEST}
  cp dist/* ${DEST}
  popd
  rm -rf ${TMPDIR}
  echo $(date) : "=== Output wheel file is in: ${DEST}"
}

main "$@"
