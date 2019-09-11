#!/usr/bin/env bash
# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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

# TODO(dkovalev): b/140445440 -- Implement test coverage for this script.
set -e

PYTHON="${PYTHON:-python3}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export TENSORFLOW_SRC_ROOT="${SCRIPT_DIR}/../../../.."
export TENSORFLOW_VERSION=$(grep "_VERSION = " "${TENSORFLOW_SRC_ROOT}/tensorflow/tools/pip_package/setup.py" | cut -d'=' -f 2 | sed "s/[ '-]//g");
TFLITE_ROOT="${TENSORFLOW_SRC_ROOT}/tensorflow/lite"
BUILD_ROOT="/tmp/tflite_pip/${PYTHON}"

# Build source tree.
rm -rf "${BUILD_ROOT}"
mkdir -p "${BUILD_ROOT}/tflite_runtime"
cp -r "${TFLITE_ROOT}/tools/pip_package/debian" \
      "${TFLITE_ROOT}/python/interpreter_wrapper" \
      "${TFLITE_ROOT}/tools/pip_package/setup.py" \
      "${TFLITE_ROOT}/tools/pip_package/MANIFEST.in" \
      "${BUILD_ROOT}"
cp "${TFLITE_ROOT}/python/interpreter.py" \
   "${BUILD_ROOT}/tflite_runtime"
touch "${BUILD_ROOT}/tflite_runtime/__init__.py"

# Build python wheel.
cd "${BUILD_ROOT}"
if [[ "${TENSORFLOW_TARGET}" == "rpi" ]]; then
  ${PYTHON} setup.py bdist_wheel --plat-name=linux-armv7l
elif [[ "${TENSORFLOW_TARGET}" == "aarch64" ]]; then
  ${PYTHON} setup.py bdist_wheel --plat-name=linux-aarch64
else
  ${PYTHON} setup.py bdist_wheel
fi

# Build debian package.
if [[ "${BUILD_DEB}" != "y" ]]; then
  exit 0
fi

PYTHON_VERSION=$(${PYTHON} -c "import sys;print(sys.version_info.major)")
if [[ ${PYTHON_VERSION} != 3 ]]; then
  echo "Debian package can only be generated for python3." >&2
  exit 1
fi

DEB_VERSION=$(dpkg-parsechangelog --show-field Version | cut -d- -f1)
if [[ "${DEB_VERSION}" != "${TENSORFLOW_VERSION}" ]]; then
  echo "Debian package version (${DEB_VERSION}) doesn't match TensorFlow version (${TENSORFLOW_VERSION})" >&2
  exit 1
fi

if [[ "${TENSORFLOW_TARGET}" == "rpi" ]]; then
  dpkg-buildpackage -b -rfakeroot -us -uc -tc -d -a armhf
elif [[ "${TENSORFLOW_TARGET}" == "aarch64" ]]; then
  dpkg-buildpackage -b -rfakeroot -us -uc -tc -d -a arm64
else
  dpkg-buildpackage -b -rfakeroot -us -uc -tc
fi
