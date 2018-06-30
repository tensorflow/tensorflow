#!/bin/bash
# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
set -e
set -x

TMPDIR=`mktemp -d`
trap "rm -rf $TMPDIR" EXIT

VERSION=1.0

BUILDER=bazel
BASEDIR=tensorflow/contrib/lite
CROSSTOOL="//external:android/crosstool"
HOST_CROSSTOOL="@bazel_tools//tools/cpp:toolchain"

BUILD_OPTS="--cxxopt=--std=c++11 -c opt"
CROSSTOOL_OPTS="--crosstool_top=$CROSSTOOL --host_crosstool_top=$HOST_CROSSTOOL"

test -d $BASEDIR || (echo "Aborting: not at top-level build directory"; exit 1)

function build_basic_aar() {
  local OUTDIR=$1
  $BUILDER build $BUILD_OPTS $BASEDIR/java:tensorflowlite.aar
  unzip -d $OUTDIR $BUILDER-bin/$BASEDIR/java/tensorflowlite.aar
  # targetSdkVersion is here to prevent the app from requesting spurious
  # permissions, such as permission to make phone calls. It worked for v1.0,
  # but minSdkVersion might be the preferred way to handle this.
  sed -i -e 's/<application>/<uses-sdk android:targetSdkVersion="25"\/><application>/' $OUTDIR/AndroidManifest.xml
}

function build_arch() {
  local ARCH=$1
  local CONFIG=$2
  local OUTDIR=$3
  mkdir -p $OUTDIR/jni/$ARCH/
  $BUILDER build $BUILD_OPTS $CROSSTOOL_OPTS --cpu=$CONFIG \
    $BASEDIR/java:libtensorflowlite_jni.so
  cp $BUILDER-bin/$BASEDIR/java/libtensorflowlite_jni.so $OUTDIR/jni/$ARCH/
}

rm -rf $TMPDIR
mkdir -p $TMPDIR/jni

build_basic_aar $TMPDIR
build_arch arm64-v8a arm64-v8a $TMPDIR
build_arch armeabi-v7a armeabi-v7a $TMPDIR
build_arch x86 x86 $TMPDIR
build_arch x86_64 x86_64 $TMPDIR

AAR_FILE=`realpath tflite-${VERSION}.aar`
(cd $TMPDIR && zip $AAR_FILE -r *)
echo "New AAR file is $AAR_FILE"

