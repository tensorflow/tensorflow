#!/usr/bin/env bash
# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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

set -ex

# Sanity test for the binary artifacts for the TensorFlow Java API.
# - Unarchive
# - Compile a trivial Java file that excercises the Java API and underlying
#   native library.
# - Run it

# Tools needed: java, javac, tar
JAVA="${JAVA}"
JAVAC="${JAVAC}"
TAR="${TAR}"

[ -z "${JAVA}" ] && JAVA="java"
[ -z "${JAVAC}" ] && JAVAC="javac"
[ -z "${TAR}"] && TAR="tar"

# bazel tests run with ${PWD} set to the root of the bazel workspace
TARFILE="${PWD}/tensorflow/tools/lib_package/libtensorflow_jni.tar.gz"
JAVAFILE="${PWD}/tensorflow/tools/lib_package/LibTensorFlowTest.java"
JARFILE="${PWD}/tensorflow/java/libtensorflow.jar"

cd ${TEST_TMPDIR}

# Extract the archive into a subdirectory 'jni'
mkdir jni
${TAR} -xzf ${TARFILE} -Cjni

# Compile and run the .java file
${JAVAC} -cp ${JARFILE} -d . ${JAVAFILE}
OUTPUT=$(${JAVA} \
  -cp "${JARFILE}:." \
  -Djava.library.path=jni \
  LibTensorFlowTest)
if [ -z "${OUTPUT}" ]
then
  echo "Empty output, expecting version number"
  exit 1
fi
