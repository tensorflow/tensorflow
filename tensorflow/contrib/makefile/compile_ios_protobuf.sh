#!/bin/bash
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
# Builds protobuf 3 for iOS.

set -e

if [[ -n MACOSX_DEPLOYMENT_TARGET ]]; then
    export MACOSX_DEPLOYMENT_TARGET=$(sw_vers -productVersion)
fi

usage() {
  echo "Usage: $(basename "$0") [-a]"
  echo "-a [build_arch] build for specified arch comma separate for multiple archs (eg: x86_64,arm64)"
  echo "default arch x86_64, armv7, armv7s, arm64"
  exit 1
}

BUILD_TARGET="x86_64 armv7 armv7s arm64"
while getopts "a:" opt_name; do
  case "$opt_name" in
    a) BUILD_TARGET="${OPTARG}";;
    *) usage;;
  esac
done
shift $((OPTIND - 1))

IFS=' ' read -r -a build_targets <<< "${BUILD_TARGET}"

SCRIPT_DIR=$(cd `dirname $0` && pwd)
source "${SCRIPT_DIR}/build_helper.subr"

cd ${SCRIPT_DIR}

HOST_GENDIR="$(pwd)/gen/protobuf-host"
mkdir -p "${HOST_GENDIR}"
if [[ ! -f "./downloads/protobuf/autogen.sh" ]]; then
    echo "You need to download dependencies before running this script." 1>&2
    echo "tensorflow/contrib/makefile/download_dependencies.sh" 1>&2
    exit 1
fi

JOB_COUNT="${JOB_COUNT:-$(get_job_count)}"

GENDIR=$(pwd)/gen/protobuf_ios/
LIBDIR=${GENDIR}lib
mkdir -p ${LIBDIR}

OSX_VERSION=darwin14.0.0

IPHONEOS_PLATFORM=$(xcrun --sdk iphoneos --show-sdk-platform-path)
IPHONEOS_SYSROOT=$(xcrun --sdk iphoneos --show-sdk-path)
IPHONESIMULATOR_PLATFORM=$(xcrun --sdk iphonesimulator --show-sdk-platform-path)
IPHONESIMULATOR_SYSROOT=$(xcrun --sdk iphonesimulator --show-sdk-path)
IOS_SDK_VERSION=$(xcrun --sdk iphoneos --show-sdk-version)
MIN_SDK_VERSION=8.0

CFLAGS="-DNDEBUG -Os -pipe -fPIC -fno-exceptions"
CXXFLAGS="${CFLAGS} -std=c++11 -stdlib=libc++"
LDFLAGS="-stdlib=libc++"
LIBS="-lc++ -lc++abi"

cd downloads/protobuf
PROTOC_PATH="${HOST_GENDIR}/bin/protoc"
if [[ ! -f "${PROTOC_PATH}" || ${clean} == true ]]; then
  # Try building compatible protoc first on host
  echo "protoc not found at ${PROTOC_PATH}. Build it first."
  make_host_protoc "${HOST_GENDIR}"
else
  echo "protoc found. Skip building host tools."
fi

# Remove old libs
rm -f ${LIBDIR}/libprotobuf.a
rm -f ${LIBDIR}/libprotobuf-lite.a

./autogen.sh
if [ $? -ne 0 ]
then
  echo "./autogen.sh command failed."
  exit 1
fi

package_pb_library() {
    pb_libs="${LIBDIR}/${1}/lib/libprotobuf.a"
    if [ -f "${LIBDIR}/libprotobuf.a" ]; then
        pb_libs="$pb_libs ${LIBDIR}/libprotobuf.a"
    fi
    lipo \
    $pb_libs \
    -create \
    -output ${LIBDIR}/libprotobuf.a

    pblite_libs="${LIBDIR}/${1}/lib/libprotobuf-lite.a"
    if [ -f "${LIBDIR}/libprotobuf-lite.a" ]; then
        pblite_libs="$pblite_libs ${LIBDIR}/libprotobuf-lite.a"
    fi
    lipo \
    $pblite_libs \
    -create \
    -output ${LIBDIR}/libprotobuf-lite.a
}

build_target() {
case "$1" in
    x86_64) make distclean
        ./configure \
        --host=x86_64-apple-${OSX_VERSION} \
        --disable-shared \
        --enable-cross-compile \
        --with-protoc="${PROTOC_PATH}" \
        --prefix=${LIBDIR}/iossim_x86_64 \
        --exec-prefix=${LIBDIR}/iossim_x86_64 \
        "CFLAGS=${CFLAGS} \
        -mios-simulator-version-min=${MIN_SDK_VERSION} \
        -arch x86_64 \
        -fembed-bitcode \
        -isysroot ${IPHONESIMULATOR_SYSROOT}" \
        "CXX=${CXX}" \
        "CXXFLAGS=${CXXFLAGS} \
        -mios-simulator-version-min=${MIN_SDK_VERSION} \
        -arch x86_64 \
        -fembed-bitcode \
        -isysroot \
        ${IPHONESIMULATOR_SYSROOT}" \
        LDFLAGS="-arch x86_64 \
        -fembed-bitcode \
        -mios-simulator-version-min=${MIN_SDK_VERSION} \
        ${LDFLAGS} \
        -L${IPHONESIMULATOR_SYSROOT}/usr/lib/ \
        -L${IPHONESIMULATOR_SYSROOT}/usr/lib/system" \
        "LIBS=${LIBS}"
        make -j"${JOB_COUNT}"
        make install

        package_pb_library "iossim_x86_64"
        ;;

    armv7) make distclean
        ./configure \
        --host=armv7-apple-${OSX_VERSION} \
        --with-protoc="${PROTOC_PATH}" \
        --disable-shared \
        --prefix=${LIBDIR}/ios_arm7 \
        --exec-prefix=${LIBDIR}/ios_arm7 \
        "CFLAGS=${CFLAGS} \
        -miphoneos-version-min=${MIN_SDK_VERSION} \
        -arch armv7 \
        -fembed-bitcode \
        -isysroot ${IPHONEOS_SYSROOT}" \
        "CXX=${CXX}" \
        "CXXFLAGS=${CXXFLAGS} \
        -miphoneos-version-min=${MIN_SDK_VERSION} \
        -arch armv7 \
        -fembed-bitcode \
        -isysroot ${IPHONEOS_SYSROOT}" \
        LDFLAGS="-arch armv7 \
        -fembed-bitcode \
        -miphoneos-version-min=${MIN_SDK_VERSION} \
        ${LDFLAGS}" \
        "LIBS=${LIBS}"
        make -j"${JOB_COUNT}"
        make install

        package_pb_library "ios_arm7"
        ;;

    armv7s) make distclean
        ./configure \
        --host=armv7s-apple-${OSX_VERSION} \
        --with-protoc="${PROTOC_PATH}" \
        --disable-shared \
        --prefix=${LIBDIR}/ios_arm7s \
        --exec-prefix=${LIBDIR}/ios_arm7s \
        "CFLAGS=${CFLAGS} \
        -miphoneos-version-min=${MIN_SDK_VERSION} \
        -arch armv7s \
        -fembed-bitcode \
        -isysroot ${IPHONEOS_SYSROOT}" \
        "CXX=${CXX}" \
        "CXXFLAGS=${CXXFLAGS} \
        -miphoneos-version-min=${MIN_SDK_VERSION} \
        -arch armv7s \
        -fembed-bitcode \
        -isysroot ${IPHONEOS_SYSROOT}" \
        LDFLAGS="-arch armv7s \
        -fembed-bitcode \
        -miphoneos-version-min=${MIN_SDK_VERSION} \
        ${LDFLAGS}" \
        "LIBS=${LIBS}"
        make -j"${JOB_COUNT}"
        make install

        package_pb_library "ios_arm7s"
        ;;

    arm64) make distclean
        ./configure \
        --host=arm \
        --with-protoc="${PROTOC_PATH}" \
        --disable-shared \
        --prefix=${LIBDIR}/ios_arm64 \
        --exec-prefix=${LIBDIR}/ios_arm64 \
        "CFLAGS=${CFLAGS} \
        -miphoneos-version-min=${MIN_SDK_VERSION} \
        -arch arm64 \
        -fembed-bitcode \
        -isysroot ${IPHONEOS_SYSROOT}" \
        "CXXFLAGS=${CXXFLAGS} \
        -miphoneos-version-min=${MIN_SDK_VERSION} \
        -arch arm64 \
        -fembed-bitcode \
        -isysroot ${IPHONEOS_SYSROOT}" \
        LDFLAGS="-arch arm64 \
        -fembed-bitcode \
        -miphoneos-version-min=${MIN_SDK_VERSION} \
        ${LDFLAGS}" \
        "LIBS=${LIBS}"
        make -j"${JOB_COUNT}"
        make install

        package_pb_library "ios_arm64"
        ;;
    *)
        echo "Unknown ARCH"
        exit 1
        ;;
esac
}

for build_element in "${build_targets[@]}"
do
    echo "$build_element"
    build_target "$build_element"
done

file ${LIBDIR}/libprotobuf.a
file ${LIBDIR}/libprotobuf-lite.a
echo "Done building and packaging the libraries"
