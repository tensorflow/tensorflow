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
# Builds protobuf 3 for iOS.

cd tensorflow/contrib/makefile || exit 1

GENDIR=$(pwd)/gen/protobuf_pi/
LIBDIR=${GENDIR}
mkdir -p ${LIBDIR}

CXX=arm-linux-gnueabihf-g++

cd downloads/protobuf || exit 1

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
