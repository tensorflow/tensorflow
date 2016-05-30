#!/bin/bash
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
