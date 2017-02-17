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
#
# Automatically update TensorFlow version in source files
#
# Usage:  update_version.sh <new_major_ver>.<new_minor_ver>.<new_patch_ver>
#         e.g.,
#           update_version.sh 0.7.2
#

# Helper functions
die() {
  echo $1
  exit 1
}

check_existence() {
  # Usage: check_exists (dir|file) <path>

  if [[ "$1" == "dir" ]]; then
    test -d "$2" ||
      die "ERROR: Cannot find directory ${2}. "\
"Are you under the TensorFlow source root directory?"
  else
    test -f "$2" ||
      die "ERROR: Cannot find file ${2}. "\
"Are you under the TensorFlow source root directory?"
  fi
}


TF_SRC_DIR="tensorflow"
check_existence dir "${TF_SRC_DIR}"

# Process command-line arguments
if [[ $# != 1 ]]; then
  die "Usage: $(basename $0) <new_major_ver>.<new_minor_ver>.<new_patch_ver>"
fi
NEW_VER=$1

# Check validity of new version string
echo "${NEW_VER}" | grep -q -E "[0-9]+\.[0-9]+\.[[:alnum:]]+"
if [[ $? != "0" ]]; then
  die "ERROR: Invalid new version: \"${NEW_VER}\""
fi

# Extract major, minor and patch versions
MAJOR=$(echo "${NEW_VER}" | cut -d \. -f 1)
MINOR=$(echo "${NEW_VER}" | cut -d \. -f 2)
PATCH=$(echo "${NEW_VER}" | cut -d \. -f 3)
PIP_PATCH="${PATCH//-}"

# Update tensorflow/core/public/version.h
VERSION_H="${TF_SRC_DIR}/core/public/version.h"
check_existence file "${VERSION_H}"

OLD_MAJOR=$(cat ${VERSION_H} | grep -E "^#define TF_MAJOR_VERSION [0-9]+" | \
cut -d ' ' -f 3)
OLD_MINOR=$(cat ${VERSION_H} | grep -E "^#define TF_MINOR_VERSION [0-9]+" | \
cut -d ' ' -f 3)
OLD_PATCH=$(cat ${VERSION_H} | grep -E "^#define TF_PATCH_VERSION [[:alnum:]-]+" | \
cut -d ' ' -f 3)

sed -i -e "s/^#define TF_MAJOR_VERSION ${OLD_MAJOR}/#define TF_MAJOR_VERSION ${MAJOR}/g" ${VERSION_H}
sed -i -e "s/^#define TF_MINOR_VERSION ${OLD_MINOR}/#define TF_MINOR_VERSION ${MINOR}/g" ${VERSION_H}
sed -i -e "s/^#define TF_PATCH_VERSION ${OLD_PATCH}/#define TF_PATCH_VERSION ${PATCH}/g" "${VERSION_H}"


# Update setup.py
SETUP_PY="${TF_SRC_DIR}/tools/pip_package/setup.py"
check_existence file "${SETUP_PY}"

sed -i -e "s/^\_VERSION = [\'\"].*[\'\"]/\_VERSION = \'${MAJOR}.${MINOR}.${PATCH}\'/g" "${SETUP_PY}"

# Update os_setup.md
OS_SETUP="${TF_SRC_DIR}/g3doc/get_started/os_setup.md"
check_existence file "${OS_SETUP}"

sed -i -r -e "s/(.*pip[0-9]* install .*tensorflow-)([0-9]+\.[0-9]+\.[[:alnum:]]+)(-.*\.whl)/\1${MAJOR}.${MINOR}.${PIP_PATCH}\3/g" "${OS_SETUP}"
sed -i -r -e "s/(.*pip[0-9]* install .*tensorflow_gpu-)([0-9]+\.[0-9]+\.[[:alnum:]]+)(-.*\.whl)/\1${MAJOR}.${MINOR}.${PIP_PATCH}\3/g" "${OS_SETUP}"
sed -i -r -e "s/(.*export TF_BINARY_URL.*tensorflow-)([0-9]+\.[0-9]+\.[[:alnum:]]+)(-.*\.whl)/\1${MAJOR}.${MINOR}.${PIP_PATCH}\3/g" "${OS_SETUP}"
sed -i -r -e "s/(.*export TF_BINARY_URL.*tensorflow_gpu-)([0-9]+\.[0-9]+\.[[:alnum:]]+)(-.*\.whl)/\1${MAJOR}.${MINOR}.${PIP_PATCH}\3/g" "${OS_SETUP}"
sed -i -r -e "s/(.*\`)([0-9]+\.[0-9]+\.[[:alnum:]-]+)(-gpu.*)/\1${MAJOR}.${MINOR}.${PATCH}\3/g" "${OS_SETUP}"


# Update README.md
README_MD="./README.md"
check_existence file "${README_MD}"

sed -i -r -e "s/${OLD_MAJOR}\.${OLD_MINOR}\.([[:alnum:]]+)-/${MAJOR}.${MINOR}.${PIP_PATCH}-/g" "${README_MD}"


# Updates to be made if there are major / minor version changes
MAJOR_MINOR_CHANGE=0
if [[ ${OLD_MAJOR} != ${MAJOR} ]] || [[ ${OLD_MINOR} != ${MINOR} ]]; then
  MAJOR_MINOR_CHANGE=1

  OLD_R_MAJOR_MINOR="r${OLD_MAJOR}\.${OLD_MINOR}"
  R_MAJOR_MINOR="r${MAJOR}\.${MINOR}"

  echo "Detected Major.Minor change. "\
"Updating pattern ${OLD_R_MAJOR_MINOR} to ${R_MAJOR_MINOR} in additional files"

  # Update tensorflow/tensorboard/README.md
  TENSORBOARD_README_MD="${TF_SRC_DIR}/tensorboard/README.md"
  check_existence file "${TENSORBOARD_README_MD}"
  sed -i -r -e "s/${OLD_R_MAJOR_MINOR}/${R_MAJOR_MINOR}/g" \
      "${TENSORBOARD_README_MD}"

  # Update dockerfiles
  DEVEL_DOCKERFILE="${TF_SRC_DIR}/tools/docker/Dockerfile.devel"
  check_existence file "${DEVEL_DOCKERFILE}"
  sed -i -r -e "s/${OLD_R_MAJOR_MINOR}/${R_MAJOR_MINOR}/g" "${DEVEL_DOCKERFILE}"

  GPU_DEVEL_DOCKERFILE="${TF_SRC_DIR}/tools/docker/Dockerfile.devel-gpu"
  check_existence file "${GPU_DEVEL_DOCKERFILE}"
  sed -i -r -e "s/${OLD_R_MAJOR_MINOR}/${R_MAJOR_MINOR}/g" \
      "${GPU_DEVEL_DOCKERFILE}"
fi

echo "Major: ${OLD_MAJOR} -> ${MAJOR}"
echo "Minor: ${OLD_MINOR} -> ${MINOR}"
echo "Patch: ${OLD_PATCH} -> ${PATCH}"
echo ""

# Look for potentially lingering old version strings in TensorFlow source files
declare -a OLD_PATCHES=(${OLD_PATCH} $(echo "${OLD_PATCH//-}"))
for i in "${OLD_PATCHES[@]}"
do
  OLD_VER="${OLD_MAJOR}\.${OLD_MINOR}\.$i"
  LINGER_STRS=$(grep -rnoH "${OLD_VER}" "${TF_SRC_DIR}")

  if [[ ! -z "${LINGER_STRS}" ]]; then
    echo "WARNING: Below are potentially instances of lingering old version "\
  "string (${OLD_VER}) in source directory \"${TF_SRC_DIR}/\" that are not "\
  "updated by this script. Please check them manually!"
    for LINGER_STR in ${LINGER_STRS}; do
      echo "${LINGER_STR}"
    done
  else
    echo "No lingering old version strings \"${OLD_VER}\" found in source directory "\
  "\"${TF_SRC_DIR}/\". Good."
  fi
done

if [[ ${MAJOR_MINOR_CHANGE} == "1" ]]; then
  LINGER_R_STRS=$(grep -rnoH "${OLD_R_MAJOR_MINOR}" "${TF_SRC_DIR}")

  if [[ ! -z "${LINGER_R_STRS}" ]]; then
    echo "WARNING: Below are potentially instances of lingering old "\
"major.minor release string (${OLD_R_MAJOR_MINOR}) in source directory "\
"\"${TF_SRC_DIR}/\" that are not updated by this script. "\
"Please check them manually!"
    for LINGER_R_STR in ${LINGER_R_STRS}; do
      echo "${LINGER_R_STR}"
    done
  else
    echo "No lingering old instances of ${OLD_R_MAJOR_MINOR} found in source "\
"directory \"${TF_SRC_DIR}/\". Good."
  fi
fi
