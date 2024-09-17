#!/bin/bash
# Copyright 2024 The TensorFlow Authors. All Rights Reserved.
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
# Windows-specific utilities.
#

# Docker on Windows has difficulty using volumes other than C:\, when it comes
# to setting up up volume mappings.
# Thus, the drive letter is replaced with C:\, in case it's
# something else (ex. T:), which is frequently the case inside Kokoro jobs.
function replace_drive_letter_with_c () {
  sed -E "s|^[a-zA-Z]:|C:|g" <<< $1
}
