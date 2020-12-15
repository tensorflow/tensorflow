#!/bin/bash
# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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

# Utility script that handles downloading and extracting portable version of Renode for testing purposes.
# Called with one argument:
# 1 - Path to new folder to unpack the package into.


if [ $# -ne 1 ]; then
    echo "Usage: download_renode.sh PATH"
    echo "    PATH is a path where Renode should be unpacked"
    echo ""
    echo "E.g: ./download_renode.sh /tmp/renode"
    exit 1
fi

# Colours
ORANGE="\033[33m"
RED="\033[31m"
NC="\033[0m"

# Target version
RENODE_VERSION='1.11.0'
# Get target path
TARGET_PATH=$1
mkdir -p "${TARGET_PATH}" || exit 1

echo "Downloading Renode portable in version ${RENODE_VERSION}"

# Get link to requested version
RELEASES_JSON=`curl https://api.github.com/repos/renode/renode/releases 2>/dev/null`
LINUX_PORTABLE_URL=`echo "${RELEASES_JSON}" |grep 'browser_download_url'|\
    grep --extended-regexp --only-matching "https://.*${RENODE_VERSION}.*linux-portable.*tar.gz"`
if [ -z "${LINUX_PORTABLE_URL}" ]; then
  echo -e "${RED}Portable version of release v${RENODE_VERSION} not found. Please make sure you use correct version format ('[0-9]+.[0-9]+.[0-9]+')${NC}"
  exit 1
fi

# Check if newer version available
LATEST_RENODE_VERSION=`echo "${RELEASES_JSON}" |grep 'tag_name' |\
    head --lines 1 | grep --extended-regexp --only-matching '[0-9]+\.[0-9]+\.[0-9]+'`
if [ "${RENODE_VERSION}" != "${LATEST_RENODE_VERSION}" ]; then
  echo -e "${ORANGE}Latest available version is ${LATEST_RENODE_VERSION}, please consider using it.${NC}"
fi
echo "Downloading from url: ${LINUX_PORTABLE_URL}"

# Get portable & unpack
wget --quiet --output-document - "${LINUX_PORTABLE_URL}" |\
    tar xz --strip-components=1 --directory "${TARGET_PATH}"
echo "Unpacked to directory: ${TARGET_PATH}"
