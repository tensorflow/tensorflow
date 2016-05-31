#!/bin/bash -x
# Builds protobuf 3 for iOS.

cd tensorflow/contrib/makefile

GENDIR=`pwd`/gen/protobuf_pi/
LIBDIR=${GENDIR}
mkdir -p ${LIBDIR}

CXX=arm-linux-gnueabihf-g++

cd downloads/protobuf

./autogen.sh
if [ $? -ne 0 ]
then
  echo "./autogen.sh command failed."
  exit 1
fi

make distclean
./configure \
--build=i686-pc-linux-gnu \
--host=arm-linux \
--target=arm-linux \
--disable-shared \
--enable-cross-compile \
--with-protoc=protoc \
--prefix=${LIBDIR} \
--exec-prefix=${LIBDIR} \
"CXX=${CXX}" \
make
make install
