#!/usr/bin/env bash
#
# Copyright 2026 The TensorFlow Authors. All Rights Reserved.
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
# Sets up the CMake toolchain (toolchain PPA, Clang/LLD symlinks, and CMake in
# /usr/local).
set -euo pipefail

# Prevent apt install tzinfo from asking our location (assumes UTC)
export DEBIAN_FRONTEND=noninteractive

# Install `software-properties-common` for `add-apt-repository`.
apt-get update
apt-get install -y software-properties-common

# Add Ubuntu Toolchain PPA for newer GCC/libstdc++.
add-apt-repository -y ppa:ubuntu-toolchain-r/test

# Set up symlinks for Clang and LLD.
ln -sf /usr/bin/clang-18 /usr/bin/clang
ln -sf /usr/bin/clang++-18 /usr/bin/clang++
ln -sf /usr/bin/lld-18 /usr/bin/lld

# Install CMake from Kitware's release archive because LLVM requires
# CMake >= 3.31.0 and Ubuntu 22.04 apt ships 3.22.1. Pin the latest 3.31.x
# release to stay on CMake 3.x policies while meeting LLVM's minimum.
readonly CMAKE_VERSION="3.31.12"
readonly CMAKE_SHA256="0dc2e9a6860f06bf10bd8fadc03e35d9eeb4df46e33763a7e480e987758f385c"
readonly CMAKE_ARCHIVE="cmake-${CMAKE_VERSION}-linux-x86_64.tar.gz"
readonly CMAKE_URL="https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}/${CMAKE_ARCHIVE}"
# Bounded connect/transfer timeouts and exponential-backoff retries for the
# ~55 MiB GitHub release tarball.
readonly CURL_CONNECT_TIMEOUT_SECONDS=15
readonly CURL_MAX_TIME_SECONDS=120
readonly CURL_MAX_RETRIES=3
readonly CURL_RETRY_MAX_TIME_SECONDS=300

tmp_dir="$(mktemp -d)"
trap 'rm -rf "${tmp_dir}"' EXIT
curl --fail --show-error --silent --location \
  --connect-timeout "${CURL_CONNECT_TIMEOUT_SECONDS}" \
  --max-time "${CURL_MAX_TIME_SECONDS}" \
  --retry "${CURL_MAX_RETRIES}" \
  --retry-max-time "${CURL_RETRY_MAX_TIME_SECONDS}" \
  "${CMAKE_URL}" -o "${tmp_dir}/${CMAKE_ARCHIVE}"
echo "${CMAKE_SHA256}  ${tmp_dir}/${CMAKE_ARCHIVE}" | sha256sum --check
tar -zxf "${tmp_dir}/${CMAKE_ARCHIVE}" \
  --no-same-owner \
  --strip-components=1 \
  -C /usr/local \
  "cmake-${CMAKE_VERSION}-linux-x86_64/bin/cmake" \
  "cmake-${CMAKE_VERSION}-linux-x86_64/bin/ctest" \
  "cmake-${CMAKE_VERSION}-linux-x86_64/bin/cpack" \
  "cmake-${CMAKE_VERSION}-linux-x86_64/share"
/usr/local/bin/cmake --version
