# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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
"""External-only build rules for mini-benchmark."""

def libjpeg_deps():
    """Returns the deps for the jpeg lib used in the mini-benchmark."""
    return ["//tensorflow/core/platform:jpeg"]

def jpeg_copts():
    """Returns copts for jpeg decoding code used in the mini-benchmark."""
    return []
