#!/bin/bash -x
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
# Builds the TensorFlow core library with ARM and x86 architectures for iOS, and
# packs them into a fat file.

ACTUAL_XCODE_VERSION=`xcodebuild -version | head -n 1 | sed 's/Xcode //'`
REQUIRED_XCODE_VERSION=7.3.0
if [ ${ACTUAL_XCODE_VERSION//.} -lt ${REQUIRED_XCODE_VERSION//.} ]
then
    echo "error: Xcode ${REQUIRED_XCODE_VERSION} or later is required."
    exit 1
fi

GENDIR=tensorflow/contrib/makefile/gen/
LIBDIR=${GENDIR}lib
LIB_PREFIX=libtensorflow-core

# TODO(petewarden) - Some new code in Eigen triggers a clang bug, so work
# around it by patching the source.
sed -e 's#static uint32x4_t p4ui_CONJ_XOR = vld1q_u32( conj_XOR_DATA );#static uint32x4_t p4ui_CONJ_XOR; // = vld1q_u32( conj_XOR_DATA ); - Removed by script#' \
-i '' \
tensorflow/contrib/makefile/downloads/eigen-latest/eigen/src/Core/arch/NEON/Complex.h
sed -e 's#static uint32x2_t p2ui_CONJ_XOR = vld1_u32( conj_XOR_DATA );#static uint32x2_t p2ui_CONJ_XOR;// = vld1_u32( conj_XOR_DATA ); - Removed by scripts#' \
-i '' \
tensorflow/contrib/makefile/downloads/eigen-latest/eigen/src/Core/arch/NEON/Complex.h
sed -e 's#static uint64x2_t p2ul_CONJ_XOR = vld1q_u64( p2ul_conj_XOR_DATA );#static uint64x2_t p2ul_CONJ_XOR;// = vld1q_u64( p2ul_conj_XOR_DATA ); - Removed by script#' \
-i '' \
tensorflow/contrib/makefile/downloads/eigen-latest/eigen/src/Core/arch/NEON/Complex.h

make -f tensorflow/contrib/makefile/Makefile cleantarget
make -f tensorflow/contrib/makefile/Makefile \
TARGET=IOS IOS_ARCH=ARMV7 LIB_NAME=${LIB_PREFIX}-armv7.a OPTFLAGS="$1" $2 $3
if [ $? -ne 0 ]
then
  echo "armv7 compilation failed."
  exit 1
fi

make -f tensorflow/contrib/makefile/Makefile cleantarget
make -f tensorflow/contrib/makefile/Makefile \
TARGET=IOS IOS_ARCH=ARMV7S LIB_NAME=${LIB_PREFIX}-armv7s.a OPTFLAGS="$1" $2 $3
if [ $? -ne 0 ]
then
  echo "arm7vs compilation failed."
  exit 1
fi

make -f tensorflow/contrib/makefile/Makefile cleantarget
make -f tensorflow/contrib/makefile/Makefile \
TARGET=IOS IOS_ARCH=ARM64 LIB_NAME=${LIB_PREFIX}-arm64.a OPTFLAGS="$1" $2 $3
if [ $? -ne 0 ]
then
  echo "arm64 compilation failed."
  exit 1
fi

make -f tensorflow/contrib/makefile/Makefile cleantarget
make -f tensorflow/contrib/makefile/Makefile \
TARGET=IOS IOS_ARCH=I386 LIB_NAME=${LIB_PREFIX}-i386.a OPTFLAGS="$1" $2 $3
if [ $? -ne 0 ]
then
  echo "i386 compilation failed."
  exit 1
fi

make -f tensorflow/contrib/makefile/Makefile cleantarget
make -f tensorflow/contrib/makefile/Makefile \
TARGET=IOS IOS_ARCH=X86_64 LIB_NAME=${LIB_PREFIX}-x86_64.a OPTFLAGS="$1" $2 $3
if [ $? -ne 0 ]
then
  echo "x86_64 compilation failed."
  exit 1
fi

lipo \
${LIBDIR}/${LIB_PREFIX}-armv7.a \
${LIBDIR}/${LIB_PREFIX}-armv7s.a \
${LIBDIR}/${LIB_PREFIX}-arm64.a \
${LIBDIR}/${LIB_PREFIX}-i386.a \
${LIBDIR}/${LIB_PREFIX}-x86_64.a \
-create \
-output ${LIBDIR}/${LIB_PREFIX}.a
