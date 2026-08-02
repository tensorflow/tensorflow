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
# Sets up custom apt sources for our TF images.

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
