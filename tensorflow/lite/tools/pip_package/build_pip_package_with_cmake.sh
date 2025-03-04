#!/usr/bin/env bash
# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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
TENSORFLOW_VERSION=$(grep "TF_VERSION = " "${TENSORFLOW_DIR}/tensorflow/tf_version.bzl" | cut -d= -f2 | sed 's/[ "-]//g')
IFS='.' read -r -a array <<< "$TENSORFLOW_VERSION"
TF_MAJOR=${array[0]}
TF_MINOR=${array[1]}
TF_PATCH=${array[2]}
TF_CXX_FLAGS="-DTF_MAJOR_VERSION=${TF_MAJOR} -DTF_MINOR_VERSION=${TF_MINOR} -DTF_PATCH_VERSION=${TF_PATCH} -DTF_VERSION_SUFFIX=''"
export PACKAGE_VERSION="${TENSORFLOW_VERSION}${VERSION_SUFFIX}"
export PROJECT_NAME=${WHEEL_PROJECT_NAME:-tflite_runtime}
BUILD_DIR="${SCRIPT_DIR}/gen/tflite_pip/${PYTHON}"
TENSORFLOW_TARGET=${TENSORFLOW_TARGET:-$1}
BUILD_NUM_JOBS="${BUILD_NUM_JOBS:-4}"
if [ "${TENSORFLOW_TARGET}" = "rpi" ]; then
  export TENSORFLOW_TARGET="armhf"
fi
PYTHON_INCLUDE=$(${PYTHON} -c "from sysconfig import get_paths as gp; print(gp()['include'])")
PYBIND11_INCLUDE=$(${PYTHON} -c "import pybind11; print (pybind11.get_include())")
NUMPY_INCLUDE=$(${PYTHON} -c "import numpy; print (numpy.get_include())")
export CROSSTOOL_PYTHON_INCLUDE_PATH=${PYTHON_INCLUDE}

# Fix container image for cross build.
if [ ! -z "${CI_BUILD_HOME}" ] && [ `pwd` = "/workspace" ]; then
  # Fix for curl build problem in 32-bit, see https://stackoverflow.com/questions/35181744/size-of-array-curl-rule-01-is-negative
  if [ "${TENSORFLOW_TARGET}" = "armhf" ] && [ -f /usr/include/curl/curlbuild.h ]; then
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
   "${TENSORFLOW_LITE_DIR}/python/metrics/metrics_interface.py" \
   "${TENSORFLOW_LITE_DIR}/python/metrics/metrics_portable.py" \
   "${BUILD_DIR}/tflite_runtime"
echo "__version__ = '${PACKAGE_VERSION}'" >> "${BUILD_DIR}/tflite_runtime/__init__.py"
echo "__git_version__ = '$(git -C "${TENSORFLOW_DIR}" describe)'" >> "${BUILD_DIR}/tflite_runtime/__init__.py"

# Build host tools
if [[ "${TENSORFLOW_TARGET}" != "native" ]]; then
  echo "Building for host tools."
  HOST_BUILD_DIR="${BUILD_DIR}/cmake_build_host"
  mkdir -p "${HOST_BUILD_DIR}"
  pushd "${HOST_BUILD_DIR}"
  cmake "${TENSORFLOW_LITE_DIR}"
  cmake --build . --verbose -j ${BUILD_NUM_JOBS} -t flatbuffers-flatc
  popd
fi

# Build python interpreter_wrapper.
mkdir -p "${BUILD_DIR}/cmake_build"
cd "${BUILD_DIR}/cmake_build"

echo "Building for ${TENSORFLOW_TARGET}"
case "${TENSORFLOW_TARGET}" in
  armhf)
    eval $(${TENSORFLOW_LITE_DIR}/tools/cmake/download_toolchains.sh "${TENSORFLOW_TARGET}")
    ARMCC_FLAGS="${ARMCC_FLAGS} ${TF_CXX_FLAGS} -I${PYBIND11_INCLUDE} -I${NUMPY_INCLUDE}"
    cmake \
      -DCMAKE_C_COMPILER=${ARMCC_PREFIX}gcc \
      -DCMAKE_CXX_COMPILER=${ARMCC_PREFIX}g++ \
      -DCMAKE_C_FLAGS="${ARMCC_FLAGS}" \
      -DCMAKE_CXX_FLAGS="${ARMCC_FLAGS}" \
      -DCMAKE_SYSTEM_NAME=Linux \
      -DCMAKE_SYSTEM_PROCESSOR=armv7 \
      -DTFLITE_ENABLE_XNNPACK=OFF \
      -DTFLITE_HOST_TOOLS_DIR="${HOST_BUILD_DIR}" \
      "${TENSORFLOW_LITE_DIR}"
    ;;
  rpi0)
    eval $(${TENSORFLOW_LITE_DIR}/tools/cmake/download_toolchains.sh "${TENSORFLOW_TARGET}")
    ARMCC_FLAGS="${ARMCC_FLAGS} ${TF_CXX_FLAGS} -I${PYBIND11_INCLUDE} -I${NUMPY_INCLUDE}"
    cmake \
      -DCMAKE_C_COMPILER=${ARMCC_PREFIX}gcc \
      -DCMAKE_CXX_COMPILER=${ARMCC_PREFIX}g++ \
      -DCMAKE_C_FLAGS="${ARMCC_FLAGS}" \
      -DCMAKE_CXX_FLAGS="${ARMCC_FLAGS}" \
      -DCMAKE_SYSTEM_NAME=Linux \
      -DCMAKE_SYSTEM_PROCESSOR=armv6 \
      -DTFLITE_ENABLE_XNNPACK=OFF \
      -DTFLITE_HOST_TOOLS_DIR="${HOST_BUILD_DIR}" \
      "${TENSORFLOW_LITE_DIR}"
    ;;
  aarch64)
    eval $(${TENSORFLOW_LITE_DIR}/tools/cmake/download_toolchains.sh "${TENSORFLOW_TARGET}")
    ARMCC_FLAGS="${ARMCC_FLAGS} ${TF_CXX_FLAGS} -I${PYBIND11_INCLUDE} -I${NUMPY_INCLUDE}"
    cmake \
      -DCMAKE_C_COMPILER=${ARMCC_PREFIX}gcc \
      -DCMAKE_CXX_COMPILER=${ARMCC_PREFIX}g++ \
      -DCMAKE_C_FLAGS="${ARMCC_FLAGS}" \
      -DCMAKE_CXX_FLAGS="${ARMCC_FLAGS}" \
      -DCMAKE_SYSTEM_NAME=Linux \
      -DCMAKE_SYSTEM_PROCESSOR=aarch64 \
      -DXNNPACK_ENABLE_ARM_I8MM=OFF \
      -DTFLITE_HOST_TOOLS_DIR="${HOST_BUILD_DIR}" \
      "${TENSORFLOW_LITE_DIR}"
    ;;
  native)
    BUILD_FLAGS=${BUILD_FLAGS:-"-march=native ${TF_CXX_FLAGS} -I${PYTHON_INCLUDE} -I${PYBIND11_INCLUDE} -I${NUMPY_INCLUDE}"}
    cmake \
      -DCMAKE_C_FLAGS="${BUILD_FLAGS}" \
      -DCMAKE_CXX_FLAGS="${BUILD_FLAGS}" \
      "${TENSORFLOW_LITE_DIR}"
    ;;
  *)
    BUILD_FLAGS=${BUILD_FLAGS:-"${TF_CXX_FLAGS} -I${PYTHON_INCLUDE} -I${PYBIND11_INCLUDE} -I${NUMPY_INCLUDE}"}
    cmake \
      -DCMAKE_C_FLAGS="${BUILD_FLAGS}" \
      -DCMAKE_CXX_FLAGS="${BUILD_FLAGS}" \
      "${TENSORFLOW_LITE_DIR}"
    ;;
esac

cmake --build . --verbose -j ${BUILD_NUM_JOBS} -t _pywrap_tensorflow_interpreter_wrapper
cd "${BUILD_DIR}"

case "${TENSORFLOW_TARGET}" in
  windows)
    LIBRARY_EXTENSION=".pyd"
    ;;
  *)
    LIBRARY_EXTENSION=".so"
    ;;
esac

cp "${BUILD_DIR}/cmake_build/_pywrap_tensorflow_interpreter_wrapper${LIBRARY_EXTENSION}" \
   "${BUILD_DIR}/tflite_runtime"
# Bazel generates the wrapper library with r-x permissions for user.
# At least on Windows, we need write permissions to delete the file.
# Without this, setuptools fails to clean the build directory.
chmod u+w "${BUILD_DIR}/tflite_runtime/_pywrap_tensorflow_interpreter_wrapper${LIBRARY_EXTENSION}"

# Build python wheel.
cd "${BUILD_DIR}"
case "${TENSORFLOW_TARGET}" in
  armhf)
    WHEEL_PLATFORM_NAME="${WHEEL_PLATFORM_NAME:-linux-armv7l}"
    ${PYTHON} setup.py bdist --plat-name=${WHEEL_PLATFORM_NAME} \
                       bdist_wheel --plat-name=${WHEEL_PLATFORM_NAME}
    ;;
  rpi0)
    WHEEL_PLATFORM_NAME="${WHEEL_PLATFORM_NAME:-linux-armv6l}"
    ${PYTHON} setup.py bdist --plat-name=${WHEEL_PLATFORM_NAME} \
                       bdist_wheel --plat-name=${WHEEL_PLATFORM_NAME}
    ;;
  aarch64)
    WHEEL_PLATFORM_NAME="${WHEEL_PLATFORM_NAME:-linux-aarch64}"
    ${PYTHON} setup.py bdist --plat-name=${WHEEL_PLATFORM_NAME} \
                       bdist_wheel --plat-name=${WHEEL_PLATFORM_NAME}
    ;;
  *)
    if [[ -n "${WHEEL_PLATFORM_NAME}" ]]; then
      ${PYTHON} setup.py bdist --plat-name=${WHEEL_PLATFORM_NAME} \
                         bdist_wheel --plat-name=${WHEEL_PLATFORM_NAME}
    else
      ${PYTHON} setup.py bdist bdist_wheel
    fi
    ;;
esac

echo "Output can be found here:"
find "${BUILD_DIR}/dist"

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
  rpi0)
    dpkg-buildpackage -b -rfakeroot -us -uc -tc -d -a armel
    ;;
  aarch64)
    dpkg-buildpackage -b -rfakeroot -us -uc -tc -d -a arm64
    ;;
  *)
    dpkg-buildpackage -b -rfakeroot -us -uc -tc -d
    ;;
esac

cat "${BUILD_DIR}/debian/changelog"

