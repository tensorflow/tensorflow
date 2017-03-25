#!/usr/bin/env bash
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
#
# This script is intended to be run inside a docker container to provide a
# hermetic process. See release.sh for the expected invocation.


RELEASE_URL_PREFIX="https://storage.googleapis.com/tensorflow/libtensorflow"
IS_SNAPSHOT="false"
if [[ "${TF_VERSION}" == *"-SNAPSHOT" ]]; then
  IS_SNAPSHOT="true"
fi

set -ex

clean() {
  # Clean up any existing artifacts
  # (though if run inside a clean docker container, there won't be any dirty
  # artifacts lying around)
  mvn -q clean
  rm -rf libtensorflow_jni/src libtensorflow_jni/target libtensorflow/src libtensorflow/target
}

update_version_in_pom() {
  mvn versions:set -DnewVersion="${TF_VERSION}"
}

download_libtensorflow() {
  if [[ "${IS_SNAPSHOT}" == "true" ]]; then
    URL="http://ci.tensorflow.org/view/Nightly/job/nightly-libtensorflow/TYPE=cpu-slave/lastSuccessfulBuild/artifact/lib_package/libtensorflow-src.jar"
  else
    URL="${RELEASE_URL_PREFIX}/libtensorflow-src-${TF_VERSION}.jar"
  fi
  curl -L "${URL}" -o /tmp/src.jar
  cd "${DIR}/libtensorflow"
  jar -xvf /tmp/src.jar
  rm -rf META-INF
  cd "${DIR}"
}

download_libtensorflow_jni() {
  NATIVE_DIR="${DIR}/libtensorflow_jni/src/main/resources/org/tensorflow/native"
  mkdir -p "${NATIVE_DIR}"
  cd "${NATIVE_DIR}"

  mkdir linux-x86_64
  mkdir windows-x86_64
  mkdir darwin-x86_64

  if [[ "${IS_SNAPSHOT}" == "true" ]]; then
    # Nightly builds from http://ci.tensorflow.org/view/Nightly/job/nightly-libtensorflow/
    # and http://ci.tensorflow.org/view/Nightly/job/nightly-libtensorflow-windows/
    curl -L "http://ci.tensorflow.org/view/Nightly/job/nightly-libtensorflow/TYPE=cpu-slave/lastSuccessfulBuild/artifact/lib_package/libtensorflow_jni-cpu-linux-x86_64.tar.gz" | tar -xvz -C linux-x86_64
    curl -L "http://ci.tensorflow.org/view/Nightly/job/nightly-libtensorflow/TYPE=mac-slave/lastSuccessfulBuild/artifact/lib_package/libtensorflow_jni-cpu-darwin-x86_64.tar.gz" | tar -xvz -C darwin-x86_64
    curl -L "http://ci.tensorflow.org/view/Nightly/job/nightly-libtensorflow-windows/lastSuccessfulBuild/artifact/lib_package/libtensorflow_jni-cpu-windows-x86_64.zip" -o /tmp/windows.zip
  else
    curl -L "${RELEASE_URL_PREFIX}/libtensorflow_jni-cpu-linux-x86_64-${TF_VERSION}.tar.gz" | tar -xvz -C linux-x86_64
    curl -L "${RELEASE_URL_PREFIX}/libtensorflow_jni-cpu-darwin-x86_64-${TF_VERSION}.tar.gz" | tar -xvz -C darwin-x86_64
#    curl -L "${RELEASE_URL_PREFIX}/libtensorflow_jni-cpu-windows-x86_64-${TF_VERSION}.zip" -o /tmp/windows.zip
  fi

#  unzip /tmp/windows.zip -d windows-x86_64
#  rm -f /tmp/windows.zip
  # Updated timestamps seem to be required to get Maven to pick up the file.
  touch linux-x86_64/*
  touch darwin-x86_64/*
  touch windows-x86_64/*
  cd "${DIR}"
}

if [ -z "${TF_VERSION}" ]
then
  echo "Must set the TF_VERSION environment variable"
  exit 1
fi

DIR="$(realpath $(dirname $0))"
cd "${DIR}"

# The meat of the script.
# Comment lines out appropriately if debugging/tinkering with the release
# process.
# gnupg2 is required for signing
apt-get -qq update && apt-get -qqq install -y gnupg2
clean
update_version_in_pom
download_libtensorflow
download_libtensorflow_jni
# Build the release artifacts
mvn verify
# If successfully built, try to deploy.
# If successfully deployed, clean.
# If deployment fails, debug with
#   ./release.sh ${TF_VERSION} ${SETTINGS_XML} bash
# To get a shell to poke around the maven artifacts with.
mvn deploy && clean

set +ex
if [[ "${IS_SNAPSHOT}" == "false" ]]; then
  echo "Uploaded to the staging repository"
  echo "After validating the release: "
  echo "1. Login to https://oss.sonatype.org/#stagingRepositories"
  echo "2. Find the 'org.tensorflow' staging release and click either 'Release' to release or 'Drop' to abort"
else
  echo "Uploaded to the snapshot repository"
fi
