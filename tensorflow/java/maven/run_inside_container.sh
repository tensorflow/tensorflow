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
TF_ECOSYSTEM_URL="https://github.com/tensorflow/ecosystem.git"

# By default we deploy to both ossrh and bintray. These two
# environment variables can be set to skip either repository.
DEPLOY_BINTRAY="${DEPLOY_BINTRAY:-true}"
DEPLOY_OSSRH="${DEPLOY_OSSRH:-true}"

PROTOC_RELEASE_URL="https://github.com/google/protobuf/releases/download/v3.5.1/protoc-3.5.1-linux-x86_64.zip"
if [[ "${DEPLOY_BINTRAY}" != "true" && "${DEPLOY_OSSRH}" != "true" ]]; then
  echo "Must deploy to at least one of Bintray or OSSRH" >&2
  exit 2
fi

set -ex

clean() {
  # Clean up any existing artifacts
  # (though if run inside a clean docker container, there won't be any dirty
  # artifacts lying around)
  mvn -q clean
  rm -rf libtensorflow_jni/src libtensorflow_jni/target libtensorflow_jni_gpu/src libtensorflow_jni_gpu/target \
    libtensorflow/src libtensorflow/target tensorflow-android/target proto/src proto/target \
    hadoop/src hadoop/target spark-connector/src spark-connector/target
}

update_version_in_pom() {
  mvn versions:set -DnewVersion="${TF_VERSION}"
}

# Fetch a property from pom files for a given profile.
# Arguments:
#   profile - name of the selected profile.
#   property - name of the property to be retrieved.
# Output:
#   Echo property value to stdout
mvn_property() {
  local profile="$1"
  local prop="$2"
  mvn -q --non-recursive exec:exec -P "${profile}" \
    -Dexec.executable='echo' \
    -Dexec.args="\${${prop}}"
}

download_libtensorflow() {
  URL="${RELEASE_URL_PREFIX}/libtensorflow-src-${TF_VERSION}.jar"
  curl -L "${URL}" -o /tmp/src.jar
  cd "${DIR}/libtensorflow"
  jar -xvf /tmp/src.jar
  rm -rf META-INF
  cd "${DIR}"
}

# Fetch the android aar artifact from the CI build system, and update
# its associated pom file.
update_tensorflow_android() {
  TARGET_DIR="${DIR}/tensorflow-android/target"
  mkdir -p "${TARGET_DIR}"
  python "${DIR}/tensorflow-android/update.py" \
    --version "${TF_VERSION}" \
    --template "${DIR}/tensorflow-android/pom-android.xml.template" \
    --dir "${TARGET_DIR}"
}

download_libtensorflow_jni() {
  NATIVE_DIR="${DIR}/libtensorflow_jni/src/main/resources/org/tensorflow/native"
  mkdir -p "${NATIVE_DIR}"
  cd "${NATIVE_DIR}"

  mkdir linux-x86_64
  mkdir windows-x86_64
  mkdir darwin-x86_64

  curl -L "${RELEASE_URL_PREFIX}/libtensorflow_jni-cpu-linux-x86_64-${TF_VERSION}.tar.gz" | tar -xvz -C linux-x86_64
  curl -L "${RELEASE_URL_PREFIX}/libtensorflow_jni-cpu-darwin-x86_64-${TF_VERSION}.tar.gz" | tar -xvz -C darwin-x86_64
  curl -L "${RELEASE_URL_PREFIX}/libtensorflow_jni-cpu-windows-x86_64-${TF_VERSION}.zip" -o /tmp/windows.zip

  unzip /tmp/windows.zip -d windows-x86_64
  rm -f /tmp/windows.zip
  # Updated timestamps seem to be required to get Maven to pick up the file.
  touch linux-x86_64/*
  touch darwin-x86_64/*
  touch windows-x86_64/*
  cd "${DIR}"
}

download_libtensorflow_jni_gpu() {
  NATIVE_DIR="${DIR}/libtensorflow_jni_gpu/src/main/resources/org/tensorflow/native"
  mkdir -p "${NATIVE_DIR}"
  cd "${NATIVE_DIR}"

  mkdir linux-x86_64

  curl -L "${RELEASE_URL_PREFIX}/libtensorflow_jni-gpu-linux-x86_64-${TF_VERSION}.tar.gz" | tar -xvz -C linux-x86_64

  # Updated timestamps seem to be required to get Maven to pick up the file.
  touch linux-x86_64/*
  cd "${DIR}"
}

# Ideally, the .jar for generated Java code for TensorFlow protocol buffer files
# would have been produced by bazel rules. However, protocol buffer library
# support in bazel is in flux. Once
# https://github.com/bazelbuild/bazel/issues/2626 has been resolved, perhaps
# TensorFlow can move to something like
# https://bazel.build/blog/2017/02/27/protocol-buffers.html
# for generating C++, Java and Python code for protocol buffers.
#
# At that point, perhaps the libtensorflow build scripts
# (tensorflow/tools/ci_build/builds/libtensorflow.sh) can build .jars for
# generated code and this function would not need to download protoc to generate
# code.
generate_java_protos() {
  # Clean any previous attempts
  rm -rf "${DIR}/proto/tmp"

  # Download protoc
  curl -L "${PROTOC_RELEASE_URL}" -o "/tmp/protoc.zip"
  mkdir -p "${DIR}/proto/tmp/protoc"
  unzip -d "${DIR}/proto/tmp/protoc" "/tmp/protoc.zip"
  rm -f "/tmp/protoc.zip"

  # Download the release archive of TensorFlow protos.
  URL="${RELEASE_URL_PREFIX}/libtensorflow_proto-${TF_VERSION}.zip"
  curl -L "${URL}" -o /tmp/libtensorflow_proto.zip
  mkdir -p "${DIR}/proto/tmp/src"
  unzip -d "${DIR}/proto/tmp/src" "/tmp/libtensorflow_proto.zip"
  rm -f "/tmp/libtensorflow_proto.zip"

  # Generate Java code
  mkdir -p "${DIR}/proto/src/main/java"
  find "${DIR}/proto/tmp/src" -name "*.proto" | xargs \
  ${DIR}/proto/tmp/protoc/bin/protoc \
    --proto_path="${DIR}/proto/tmp/src" \
    --java_out="${DIR}/proto/src/main/java"

  # Cleanup
  rm -rf "${DIR}/proto/tmp"
}


# Download the TensorFlow ecosystem source from git.
# The pom files from this repo do not inherit from the parent pom so the maven version
# is updated for each module.
download_tf_ecosystem() {
  ECOSYSTEM_DIR="/tmp/tensorflow-ecosystem"
  HADOOP_DIR="${DIR}/hadoop"
  SPARK_DIR="${DIR}/spark-connector"

  # Clean any previous attempts
  rm -rf "${ECOSYSTEM_DIR}"

  # Clone the TensorFlow ecosystem project
  mkdir -p  "${ECOSYSTEM_DIR}"
  cd "${ECOSYSTEM_DIR}"
  git clone "${TF_ECOSYSTEM_URL}"
  cd ecosystem
  # TF_VERSION is a semver string (<major>.<minor>.<patch>[-suffix])
  # but the branch is just (r<major>.<minor>).
  RELEASE_BRANCH=$(echo "${TF_VERSION}" | sed -e 's/\([0-9]\+\.[0-9]\+\)\.[0-9]\+.*/\1/')
  git checkout r${RELEASE_BRANCH}

  # Copy the TensorFlow Hadoop source
  cp -r "${ECOSYSTEM_DIR}/ecosystem/hadoop/src" "${HADOOP_DIR}"
  cp "${ECOSYSTEM_DIR}/ecosystem/hadoop/pom.xml" "${HADOOP_DIR}"
  cd "${HADOOP_DIR}"
  update_version_in_pom

  # Copy the TensorFlow Spark connector source
  cp -r "${ECOSYSTEM_DIR}/ecosystem/spark/spark-tensorflow-connector/src" "${SPARK_DIR}"
  cp "${ECOSYSTEM_DIR}/ecosystem/spark/spark-tensorflow-connector/pom.xml" "${SPARK_DIR}"
  cd "${SPARK_DIR}"
  update_version_in_pom

  # Cleanup
  rm -rf "${ECOSYSTEM_DIR}"

  cd "${DIR}"
}

# Deploy artifacts using a specific profile.
# Arguments:
#   profile - name of selected profile.
# Outputs:
#   n/a
deploy_profile() {
  local profile="$1"
  # Deploy the non-android pieces.
  mvn deploy -P"${profile}"
  # Determine the correct pom file property to use
  # for the repository url.
  local rtype
  rtype='repository'
  local url=$(mvn_property "${profile}" "project.distributionManagement.${rtype}.url")
  local repositoryId=$(mvn_property "${profile}" "project.distributionManagement.${rtype}.id")
  mvn gpg:sign-and-deploy-file \
    -Dfile="${DIR}/tensorflow-android/target/tensorflow.aar" \
    -DpomFile="${DIR}/tensorflow-android/target/pom-android.xml" \
    -Durl="${url}" \
    -DrepositoryId="${repositoryId}"
}

# If successfully built, try to deploy.
# If successfully deployed, clean.
# If deployment fails, debug with
#   ./release.sh ${TF_VERSION} ${SETTINGS_XML} bash
# To get a shell to poke around the maven artifacts with.
deploy_artifacts() {
  # Deploy artifacts to ossrh if requested.
  if [[ "${DEPLOY_OSSRH}" == "true" ]]; then
    deploy_profile 'ossrh'
  fi
  # Deploy artifacts to bintray if requested.
  if [[ "${DEPLOY_BINTRAY}" == "true" ]]; then
    deploy_profile 'bintray'
  fi
  # Clean up when everything works
  clean
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
apt-get -qq update && apt-get -qqq install -y gnupg2 git

clean
update_version_in_pom
download_libtensorflow
download_libtensorflow_jni
download_libtensorflow_jni_gpu
update_tensorflow_android
generate_java_protos
download_tf_ecosystem

# Build the release artifacts
mvn verify
# Push artifacts to repository
deploy_artifacts

set +ex
echo "Uploaded to the staging repository"
echo "After validating the release: "
if [[ "${DEPLOY_OSSRH}" == "true" ]]; then
  echo "* Login to https://oss.sonatype.org/#stagingRepositories"
  echo "* Find the 'org.tensorflow' staging release and click either 'Release' to release or 'Drop' to abort"
fi
if [[ "${DEPLOY_BINTRAY}" == "true" ]]; then
  echo "* Login to https://bintray.com/google/tensorflow/tensorflow"
  echo "* Either 'Publish' unpublished items to release, or 'Discard' to abort"
fi
