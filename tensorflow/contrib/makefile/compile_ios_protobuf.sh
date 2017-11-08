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

SCRIPT_DIR=$(dirname $0)
source "${SCRIPT_DIR}/build_helper.subr"

cd tensorflow/contrib/makefile

HOST_GENDIR="$(pwd)/gen/protobuf-host"
mkdir -p "${HOST_GENDIR}"
if [[ ! -f "./downloads/protobuf/autogen.sh" ]]; then
    echo "You need to download dependencies before running this script." 1>&2
    echo "tensorflow/contrib/makefile/download_dependencies.sh" 1>&2
    exit 1
fi

JOB_COUNT="${JOB_COUNT:-$(get_job_count)}"

ARCHS="ARMV7 ARMV7S ARM64 I386 X86_64"

USAGE="usage: compile_ios_protobuf.sh [-A architecture]

A script to build protobuf for ios.
This script can only be run on MacOS host platforms.

Options:
-A architecture
Target platforms to compile. The default is: $ARCHS."

while
  ARG="${1-}"
  case "$ARG" in
  -*)  case "$ARG" in -*A*) ARCHS="${2?"$USAGE"}"; shift; esac
       case "$ARG" in -*[!A]*) echo "$USAGE" >&2; exit 2;; esac;;
  "")  break;;
  *)   echo "$USAGE" >&2; exit 2;;
  esac
do
  shift
done

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

./autogen.sh
if [ $? -ne 0 ]
then
  echo "./autogen.sh command failed."
  exit 1
fi

for ARCH in `echo "${ARCHS}" | tr "[:upper:]" "[:lower:]"`; do
  make distclean

  case "$ARCH" in
  i386|x86_64)
    ARCH_PREFIX="${LIBDIR}/iossim_${ARCH}"
    ARCH_SYSROOT="${IPHONESIMULATOR_SYSROOT}"
    ARCH_MIN_SDK_VERSION="-mios-simulator-version-min=${MIN_SDK_VERSION}"
    ARCH_LDFLAGS="-L${ARCH_SYSROOT}/usr/lib/ -L${ARCH_SYSROOT}/usr/lib/system"
    ;;
  *)
    ARCH_PREFIX="${LIBDIR}/ios_${ARCH}"
    ARCH_SYSROOT="${IPHONEOS_SYSROOT}"
    ARCH_MIN_SDK_VERSION="-miphoneos-version-min=${MIN_SDK_VERSION}"
    ARCH_LDFLAGS=
    ;;
  esac

  case "$ARCH" in
  arm64)
    ARCH_HOST="arm";;
  *)
    ARCH_HOST="${ARCH}-apple-${OSX_VERSION}";;
  esac

  ./configure \
--host="${ARCH_HOST}" \
--disable-shared \
--enable-cross-compile \
--with-protoc="${PROTOC_PATH}" \
--prefix="${ARCH_PREFIX}" \
--exec-prefix="${ARCH_PREFIX}" \
"CFLAGS=${CFLAGS} \
${ARCH_MIN_SDK_VERSION} \
-arch ${ARCH} \
-fembed-bitcode \
-isysroot ${ARCH_SYSROOT}" \
"CXX=${CXX}" \
"CXXFLAGS=${CXXFLAGS} \
${ARCH_MIN_SDK_VERSION} \
-arch ${ARCH} \
-fembed-bitcode \
-isysroot ${ARCH_SYSROOT}" \
"LDFLAGS=-arch ${ARCH} \
-fembed-bitcode \
${ARCH_MIN_SDK_VERSION} \
${LDFLAGS} ${ARCH_LDFLAGS}" \
"LIBS=${LIBS}"

  make -j"${JOB_COUNT}"
  if [ $? -ne 0 ]; then
    echo "${ARCH} compilation failed."
    exit 1
  fi
  make install
  
  ARCH_LIBS="${ARCH_LIBS} ${ARCH_PREFIX}/lib/libprotobuf.a"
  ARCH_LIBS_LITE="${ARCH_LIBS_LITE} ${ARCH_PREFIX}/lib/libprotobuf-lite.a"
done

lipo \
${ARCH_LIBS} \
-create \
-output ${LIBDIR}/libprotobuf.a

lipo \
${ARCH_LIBS_LITE} \
-create \
-output ${LIBDIR}/libprotobuf-lite.a
