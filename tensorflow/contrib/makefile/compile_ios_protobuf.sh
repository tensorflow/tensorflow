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

cd tensorflow/contrib/makefile

GENDIR=`pwd`/gen/protobuf_ios/
LIBDIR=${GENDIR}lib
mkdir -p ${LIBDIR}

OSX_VERSION=darwin14.0.0

IPHONEOS_PLATFORM=`xcrun --sdk iphoneos --show-sdk-platform-path`
IPHONEOS_SYSROOT=`xcrun --sdk iphoneos --show-sdk-path`
IPHONESIMULATOR_PLATFORM=`xcrun --sdk iphonesimulator --show-sdk-platform-path`
IPHONESIMULATOR_SYSROOT=`xcrun --sdk iphonesimulator --show-sdk-path`
IOS_SDK_VERSION=`xcrun --sdk iphoneos --show-sdk-version`
MIN_SDK_VERSION=9.2

CFLAGS="-DNDEBUG -g -O0 -pipe -fPIC -fcxx-exceptions"
CXXFLAGS="${CFLAGS} -std=c++11 -stdlib=libc++"
LDFLAGS="-stdlib=libc++"
LIBS="-lc++ -lc++abi"

cd downloads/protobuf

./autogen.sh
if [ $? -ne 0 ]
then
  echo "./autogen.sh command failed."
  exit 1
fi

make distclean
./configure \
--build=x86_64-apple-${OSX_VERSION} \
--host=i386-apple-${OSX_VERSION} \
--disable-shared \
--enable-cross-compile \
--with-protoc=protoc \
--prefix=${LIBDIR}/iossim_386 \
--exec-prefix=${LIBDIR}/iossim_386 \
"CFLAGS=${CFLAGS} \
-mios-simulator-version-min=${MIN_SDK_VERSION} \
-arch i386 \
-isysroot ${IPHONESIMULATOR_SYSROOT}" \
"CXX=${CXX}" \
"CXXFLAGS=${CXXFLAGS} \
-mios-simulator-version-min=${MIN_SDK_VERSION} \
-arch i386 \
-isysroot \
${IPHONESIMULATOR_SYSROOT}" \
LDFLAGS="-arch i386 \
-mios-simulator-version-min=${MIN_SDK_VERSION} \
${LDFLAGS} \
-L${IPHONESIMULATOR_SYSROOT}/usr/lib/ \
-L${IPHONESIMULATOR_SYSROOT}/usr/lib/system" \
"LIBS=${LIBS}"
make
make install

make distclean
./configure \
--build=x86_64-apple-${OSX_VERSION} \
--host=x86_64-apple-${OSX_VERSION} \
--disable-shared \
--enable-cross-compile \
--with-protoc=protoc \
--prefix=${LIBDIR}/iossim_x86_64 \
--exec-prefix=${LIBDIR}/iossim_x86_64 \
"CFLAGS=${CFLAGS} \
-mios-simulator-version-min=${MIN_SDK_VERSION} \
-arch x86_64 \
-isysroot ${IPHONESIMULATOR_SYSROOT}" \
"CXX=${CXX}" \
"CXXFLAGS=${CXXFLAGS} \
-mios-simulator-version-min=${MIN_SDK_VERSION} \
-arch x86_64 \
-isysroot \
${IPHONESIMULATOR_SYSROOT}" \
LDFLAGS="-arch x86_64 \
-mios-simulator-version-min=${MIN_SDK_VERSION} \
${LDFLAGS} \
-L${IPHONESIMULATOR_SYSROOT}/usr/lib/ \
-L${IPHONESIMULATOR_SYSROOT}/usr/lib/system" \
"LIBS=${LIBS}"
make
make install

make distclean
./configure \
--build=x86_64-apple-${OSX_VERSION} \
--host=armv7-apple-${OSX_VERSION} \
--with-protoc=protoc \
--disable-shared \
--prefix=${LIBDIR}/ios_arm7 \
--exec-prefix=${LIBDIR}/ios_arm7 \
"CFLAGS=${CFLAGS} \
-miphoneos-version-min=${MIN_SDK_VERSION} \
-arch armv7 \
-isysroot ${IPHONEOS_SYSROOT}" \
"CXX=${CXX}" \
"CXXFLAGS=${CXXFLAGS} \
-miphoneos-version-min=${MIN_SDK_VERSION} \
-arch armv7 \
-isysroot ${IPHONEOS_SYSROOT}" \
LDFLAGS="-arch armv7 \
-miphoneos-version-min=${MIN_SDK_VERSION} \
${LDFLAGS}" \
"LIBS=${LIBS}"
make
make install

make distclean
./configure \
--build=x86_64-apple-${OSX_VERSION} \
--host=armv7s-apple-${OSX_VERSION} \
--with-protoc=protoc \
--disable-shared \
--prefix=${LIBDIR}/ios_arm7s \
--exec-prefix=${LIBDIR}/ios_arm7s \
"CFLAGS=${CFLAGS} \
-miphoneos-version-min=${MIN_SDK_VERSION} \
-arch armv7s \
-isysroot ${IPHONEOS_SYSROOT}" \
"CXX=${CXX}" \
"CXXFLAGS=${CXXFLAGS} \
-miphoneos-version-min=${MIN_SDK_VERSION} \
-arch armv7s \
-isysroot ${IPHONEOS_SYSROOT}" \
LDFLAGS="-arch armv7s \
-miphoneos-version-min=${MIN_SDK_VERSION} \
${LDFLAGS}" \
"LIBS=${LIBS}"
make
make install

make distclean
./configure \
--build=x86_64-apple-${OSX_VERSION} \
--host=arm \
--with-protoc=protoc \
--disable-shared \
--prefix=${LIBDIR}/ios_arm64 \
--exec-prefix=${LIBDIR}/ios_arm64 \
"CFLAGS=${CFLAGS} \
-miphoneos-version-min=${MIN_SDK_VERSION} \
-arch arm64 \
-isysroot ${IPHONEOS_SYSROOT}" \
"CXXFLAGS=${CXXFLAGS} \
-miphoneos-version-min=${MIN_SDK_VERSION} \
-arch arm64 \
-isysroot ${IPHONEOS_SYSROOT}" \
LDFLAGS="-arch arm64 \
-miphoneos-version-min=${MIN_SDK_VERSION} \
${LDFLAGS}" \
"LIBS=${LIBS}"
make
make install

lipo \
${LIBDIR}/iossim_386/lib/libprotobuf.a \
${LIBDIR}/iossim_x86_64/lib/libprotobuf.a \
${LIBDIR}/ios_arm7/lib/libprotobuf.a \
${LIBDIR}/ios_arm7s/lib/libprotobuf.a \
${LIBDIR}/ios_arm64/lib/libprotobuf.a \
-create \
-output ${LIBDIR}/libprotobuf.a

lipo \
${LIBDIR}/iossim_386/lib/libprotobuf-lite.a \
${LIBDIR}/iossim_x86_64/lib/libprotobuf-lite.a \
${LIBDIR}/ios_arm7/lib/libprotobuf-lite.a \
${LIBDIR}/ios_arm7s/lib/libprotobuf-lite.a \
${LIBDIR}/ios_arm64/lib/libprotobuf-lite.a \
-create \
-output ${LIBDIR}/libprotobuf-lite.a
