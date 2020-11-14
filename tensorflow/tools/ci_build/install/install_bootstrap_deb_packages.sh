#!/usr/bin/env bash
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

set -e

# Retry on connection timeout.
bash -c "echo 'APT::Acquire::Retries \"3\";' > /etc/apt/apt.conf.d/80-retries"

# Install bootstrap dependencies from ubuntu deb repository.
apt-get update
apt-get install -y --no-install-recommends \
    apt-transport-https ca-certificates software-properties-common
apt-get clean
rm -rf /var/lib/apt/lists/*
