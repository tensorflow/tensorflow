#!/bin/bash
# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
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
# This script dumps some information about the environment. It's most useful
# for verifying changes to the TFCI scripts system, and most users won't need
# to interact with it at all.
source "${BASH_SOURCE%/*}/utilities/setup.sh"

echo "==TFCI== env outside of tfrun:"
env
echo "==TFCI== env inside of tfrun:"
tfrun env
