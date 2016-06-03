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
# Builds protobuf 3 for Android. Pass in the location of your NDK as the first
# argument to the script, for example:
# tensorflow/contrib/makefile/compile_android_protobuf.sh \
# ${HOME}/toolchains/clang-21-stl-gnu

if [[ $# -ne 1 ]]
then
  echo "You need to pass in the Android NDK as the first argument, e.g:"
  echo "tensorflow/contrib/makefile/compile_android_protobuf.sh \
 ${HOME}/toolchains/clang-21-stl-gnu"
  exit 1
fi

cd tensorflow/contrib/makefile

GENDIR=`pwd`/gen/protobuf/
LIBDIR=${GENDIR}lib
mkdir -p ${LIBDIR}

export NDK=$1
export PATH=${NDK}/bin:$PATH
export SYSROOT=${NDK}/sysroot
export CC="arm-linux-androideabi-gcc --sysroot $SYSROOT"
export CXX="arm-linux-androideabi-g++ --sysroot $SYSROOT"
export CXXSTL=$NDK/sources/cxx-stl/gnu-libstdc++/4.6
 
cd downloads/protobuf

mkdir build

./autogen.sh
if [ $? -ne 0 ]
then
  echo "./autogen.sh command failed."
  exit 1
fi
 
./configure --prefix=$(pwd)/build \
--host=arm-linux-androideabi \
--with-sysroot=$SYSROOT \
--disable-shared \
--enable-cross-compile \
--with-protoc=protoc \
CFLAGS="-march=armv7-a" \
CXXFLAGS="-march=armv7-a -I$CXXSTL/include -I$CXXSTL/libs/armeabi-v7a/include"
if [ $? -ne 0 ]
then
  echo "./configure command failed."
  exit 1
fi

make
if [ $? -ne 0 ]
then
  echo "make command failed."
  exit 1
fi

cp src/.libs/* ${LIBDIR}
if [ $? -ne 0 ]
then
  echo "cp command failed."
  exit 1
fi
