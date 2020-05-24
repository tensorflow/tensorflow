#!/usr/bin/env bash
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
set -ex

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python3}"
VERSION_SUFFIX=${VERSION_SUFFIX:-}
export TENSORFLOW_DIR="${SCRIPT_DIR}/../../../.."
TENSORFLOW_LITE_DIR="${TENSORFLOW_DIR}/tensorflow/lite"
TENSORFLOW_VERSION=$(grep "_VERSION = " "${TENSORFLOW_DIR}/tensorflow/tools/pip_package/setup.py" | cut -d= -f2 | sed "s/[ '-]//g")
export PACKAGE_VERSION="${TENSORFLOW_VERSION}${VERSION_SUFFIX}"
BUILD_DIR="${SCRIPT_DIR}/gen/tflite_pip/${PYTHON}"
TENSORFLOW_TARGET=$1

# Build source tree.
rm -rf "${BUILD_DIR}" && mkdir -p "${BUILD_DIR}/tflite_runtime"
cp -r "${TENSORFLOW_LITE_DIR}/tools/pip_package/debian" \
      "${TENSORFLOW_LITE_DIR}/tools/pip_package/setup_with_bazel.py" \
      "${TENSORFLOW_LITE_DIR}/tools/pip_package/MANIFEST.in" \
      "${TENSORFLOW_LITE_DIR}/python/interpreter_wrapper" \
      "${BUILD_DIR}"
cp "${TENSORFLOW_LITE_DIR}/python/interpreter.py" \
   "${BUILD_DIR}/tflite_runtime"
echo "__version__ = '${PACKAGE_VERSION}'" >> "${BUILD_DIR}/tflite_runtime/__init__.py"
echo "__git_version__ = '$(git -C "${TENSORFLOW_DIR}" describe)'" >> "${BUILD_DIR}/tflite_runtime/__init__.py"

# Build python interpreter_wrapper.
cd "${BUILD_DIR}"
case "${TENSORFLOW_TARGET}" in
  rpi|armhf)
    BAZEL_FLAGS="--config=elinux_armhf
      --copt=-march=armv7-a --copt=-mfpu=neon-vfpv4
      --copt=-O3 --copt=-fno-tree-pre --copt=-fpermissive
      --define=raspberry_pi_with_neon=true"
    ;;
  aarch64)
    BAZEL_FLAGS="--config=elinux_aarch64
      --copt=-O3"
    ;;
  *)
    ;;
esac

# We need to pass down the environment variable with a possible alternate Python
# include path for Python 3.x builds to work.
export CROSSTOOL_PYTHON_INCLUDE_PATH

bazel build -c opt -s --config=monolithic ${BAZEL_FLAGS} //tensorflow/lite/python/interpreter_wrapper:_pywrap_tensorflow_interpreter_wrapper
cp "${TENSORFLOW_DIR}/bazel-bin/tensorflow/lite/python/interpreter_wrapper/_pywrap_tensorflow_interpreter_wrapper.so" \
   "${BUILD_DIR}/tflite_runtime"

# Build python wheel.
cd "${BUILD_DIR}"
case "${TENSORFLOW_TARGET}" in
  rpi|armhf)
    ${PYTHON} setup_with_bazel.py bdist --plat-name=linux-armv7l \
                       bdist_wheel --plat-name=linux-armv7l
    ;;
  aarch64)
    ${PYTHON} setup_with_bazel.py bdist --plat-name=linux-aarch64 \
                       bdist_wheel --plat-name=linux-aarch64
    ;;
  *)
    if [[ -n "${TENSORFLOW_TARGET}" ]] && [[ -n "${TENSORFLOW_TARGET_ARCH}" ]]; then
      ${PYTHON} setup_with_bazel.py bdist --plat-name=${TENSORFLOW_TARGET}-${TENSORFLOW_TARGET_ARCH} \
                         bdist_wheel --plat-name=${TENSORFLOW_TARGET}-${TENSORFLOW_TARGET_ARCH}
    else
      ${PYTHON} setup_with_bazel.py bdist bdist_wheel
    fi
    ;;
esac

echo "Output can be found here:"
find "${BUILD_DIR}"

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
if [[ "${DEB_VERSION}" != "${PACKAGE_VERSION}" ]]; then
  cat << EOF > "${BUILD_DIR}/debian/changelog"
tflite-runtime (${PACKAGE_VERSION}-1) unstable; urgency=low

  * Bump version to ${PACKAGE_VERSION}.

 -- TensorFlow team <packages@tensorflow.org>  $(date -R)

$(<"${BUILD_DIR}/debian/changelog")
EOF
fi

case "${TENSORFLOW_TARGET}" in
  rpi|armhf)
    dpkg-buildpackage -b -rfakeroot -us -uc -tc -d -a armhf
    ;;
  aarch64)
    dpkg-buildpackage -b -rfakeroot -us -uc -tc -d -a arm64
    ;;
  *)
    dpkg-buildpackage -b -rfakeroot -us -uc -tc -d
    ;;
esac

cat "${BUILD_DIR}/debian/changelog"

