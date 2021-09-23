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
PYTHON="${CI_BUILD_PYTHON:-python3}"
VERSION_SUFFIX=${VERSION_SUFFIX:-}
export TENSORFLOW_DIR="${SCRIPT_DIR}/../../../.."
TENSORFLOW_LITE_DIR="${TENSORFLOW_DIR}/tensorflow/lite"
TENSORFLOW_VERSION=$(grep "_VERSION = " "${TENSORFLOW_DIR}/tensorflow/tools/pip_package/setup.py" | cut -d= -f2 | sed "s/[ '-]//g")
export PACKAGE_VERSION="${TENSORFLOW_VERSION}${VERSION_SUFFIX}"
BUILD_DIR="${SCRIPT_DIR}/gen/tflite_pip/${PYTHON}"
TENSORFLOW_TARGET=${TENSORFLOW_TARGET:-$1}
if [ "${TENSORFLOW_TARGET}" = "rpi" ]; then
  export TENSORFLOW_TARGET="armhf"
fi
export CROSSTOOL_PYTHON_INCLUDE_PATH=$(${PYTHON} -c "from sysconfig import get_paths as gp; print(gp()['include'])")

# Fix container image for cross build.
if [ ! -z "${CI_BUILD_HOME}" ] && [ `pwd` = "/workspace" ]; then
  # Fix for curl build problem in 32-bit, see https://stackoverflow.com/questions/35181744/size-of-array-curl-rule-01-is-negative
  if [ "${TENSORFLOW_TARGET}" = "armhf" ]; then
    sudo sed -i 's/define CURL_SIZEOF_LONG 8/define CURL_SIZEOF_LONG 4/g' /usr/include/curl/curlbuild.h
    sudo sed -i 's/define CURL_SIZEOF_CURL_OFF_T 8/define CURL_SIZEOF_CURL_OFF_T 4/g' /usr/include/curl/curlbuild.h
  fi

  # The system-installed OpenSSL headers get pulled in by the latest BoringSSL
  # release on this configuration, so move them before we build:
  if [ -d /usr/include/openssl ]; then
    sudo mv /usr/include/openssl /usr/include/openssl.original
  fi
fi

# Build source tree.
rm -rf "${BUILD_DIR}" && mkdir -p "${BUILD_DIR}/tflite_runtime"
cp -r "${TENSORFLOW_LITE_DIR}/tools/pip_package/debian" \
      "${TENSORFLOW_LITE_DIR}/tools/pip_package/MANIFEST.in" \
      "${TENSORFLOW_LITE_DIR}/python/interpreter_wrapper" \
      "${BUILD_DIR}"
cp  "${TENSORFLOW_LITE_DIR}/tools/pip_package/setup_with_binary.py" "${BUILD_DIR}/setup.py"
cp "${TENSORFLOW_LITE_DIR}/python/interpreter.py" \
   "${TENSORFLOW_LITE_DIR}/python/metrics_interface.py" \
   "${TENSORFLOW_LITE_DIR}/python/metrics_portable.py" \
   "${BUILD_DIR}/tflite_runtime"
echo "__version__ = '${PACKAGE_VERSION}'" >> "${BUILD_DIR}/tflite_runtime/__init__.py"
echo "__git_version__ = '$(git -C "${TENSORFLOW_DIR}" describe)'" >> "${BUILD_DIR}/tflite_runtime/__init__.py"

# Build python interpreter_wrapper.
cd "${BUILD_DIR}"
case "${TENSORFLOW_TARGET}" in
  armhf)
    BAZEL_FLAGS="--config=elinux_armhf
      --copt=-march=armv7-a --copt=-mfpu=neon-vfpv4
      --copt=-O3 --copt=-fno-tree-pre --copt=-fpermissive
      --define tensorflow_mkldnn_contraction_kernel=0
      --define=raspberry_pi_with_neon=true"
    ;;
  aarch64)
    BAZEL_FLAGS="--config=elinux_aarch64
      --define tensorflow_mkldnn_contraction_kernel=0
      --copt=-O3"
    ;;
  native)
    BAZEL_FLAGS="--copt=-O3 --copt=-march=native"
    ;;
  *)
    BAZEL_FLAGS="--copt=-O3"
    ;;
esac

# We need to pass down the environment variable with a possible alternate Python
# include path for Python 3.x builds to work.
export CROSSTOOL_PYTHON_INCLUDE_PATH

case "${TENSORFLOW_TARGET}" in
  windows)
    LIBRARY_EXTENSION=".pyd"
    ;;
  *)
    LIBRARY_EXTENSION=".so"
    ;;
esac

bazel ${BAZEL_STARTUP_OPTIONS} build -c opt -s --config=monolithic --config=noaws --config=nogcp --config=nohdfs --config=nonccl \
  ${BAZEL_FLAGS} ${CUSTOM_BAZEL_FLAGS} //tensorflow/lite/python/interpreter_wrapper:_pywrap_tensorflow_interpreter_wrapper
cp "${TENSORFLOW_DIR}/bazel-bin/tensorflow/lite/python/interpreter_wrapper/_pywrap_tensorflow_interpreter_wrapper${LIBRARY_EXTENSION}" \
   "${BUILD_DIR}/tflite_runtime"
# Bazel generates the wrapper library with r-x permissions for user.
# At least on Windows, we need write permissions to delete the file.
# Without this, setuptools fails to clean the build directory.
chmod u+w "${BUILD_DIR}/tflite_runtime/_pywrap_tensorflow_interpreter_wrapper${LIBRARY_EXTENSION}"

# Build python wheel.
cd "${BUILD_DIR}"
case "${TENSORFLOW_TARGET}" in
  armhf)
    ${PYTHON} setup.py bdist --plat-name=linux-armv7l \
                       bdist_wheel --plat-name=linux-armv7l
    ;;
  aarch64)
    ${PYTHON} setup.py bdist --plat-name=linux-aarch64 \
                       bdist_wheel --plat-name=linux-aarch64
    ;;
  *)
    if [[ -n "${TENSORFLOW_TARGET}" ]] && [[ -n "${TENSORFLOW_TARGET_ARCH}" ]]; then
      ${PYTHON} setup.py bdist --plat-name=${TENSORFLOW_TARGET}-${TENSORFLOW_TARGET_ARCH} \
                         bdist_wheel --plat-name=${TENSORFLOW_TARGET}-${TENSORFLOW_TARGET_ARCH}
    else
      ${PYTHON} setup.py bdist bdist_wheel
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
  armhf)
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
