#!/bin/bash -x
# Builds the TensorFlow core library with ARM and x86 architectures for iOS, and
# packs them into a fat file.

GENDIR=tensorflow/contrib/makefile/gen/
LIBDIR=${GENDIR}lib
LIB_PREFIX=libtensorflow-core

make -f tensorflow/contrib/makefile/Makefile cleantarget
make -f tensorflow/contrib/makefile/Makefile \
TARGET=IOS IOS_ARCH=ARMV7 LIB_NAME=${LIB_PREFIX}-armv7.a OPTFLAGS="$1"
if [ $? -ne 0 ]
then
  echo "armv7 compilation failed."
  exit 1
fi

make -f tensorflow/contrib/makefile/Makefile cleantarget
make -f tensorflow/contrib/makefile/Makefile \
TARGET=IOS IOS_ARCH=ARMV7S LIB_NAME=${LIB_PREFIX}-armv7s.a OPTFLAGS=$1
if [ $? -ne 0 ]
then
  echo "arm7vs compilation failed."
  exit 1
fi

make -f tensorflow/contrib/makefile/Makefile cleantarget
make -f tensorflow/contrib/makefile/Makefile \
TARGET=IOS IOS_ARCH=ARM64 LIB_NAME=${LIB_PREFIX}-arm64.a OPTFLAGS="$1"
if [ $? -ne 0 ]
then
  echo "arm64 compilation failed."
  exit 1
fi

make -f tensorflow/contrib/makefile/Makefile cleantarget
make -f tensorflow/contrib/makefile/Makefile \
TARGET=IOS IOS_ARCH=I386 LIB_NAME=${LIB_PREFIX}-i386.a OPTFLAGS="$1"
if [ $? -ne 0 ]
then
  echo "i386 compilation failed."
  exit 1
fi

make -f tensorflow/contrib/makefile/Makefile cleantarget
make -f tensorflow/contrib/makefile/Makefile \
TARGET=IOS IOS_ARCH=X86_64 LIB_NAME=${LIB_PREFIX}-x86_64.a OPTFLAGS="$1"
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
