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
# =============================================================================

"""Loads the Hexagon NN Header files library, used by TF Lite."""

load("//third_party:repo.bzl", "tf_http_archive")

# Note: Use libhexagon_nn_skel version 1.20.0.1 Only with the current version.
# This comment will be updated with compatible version.
def repo():
    tf_http_archive(
        name = "hexagon_nn",
        sha256 = "f577b4c150b72e11e9dfb3f9d14f9772ba8fe460f7d65c84a7327ea9bef44d8e",
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/storage.cloud.google.com/download.tensorflow.org/tflite/hexagon_nn_headers_v1.20.0.9.tgz",
            # Repeated to bypass 'at least two urls' check. TODO(karimnosseir): add original source of this package.
            "https://storage.googleapis.com/mirror.tensorflow.org/storage.cloud.google.com/download.tensorflow.org/tflite/hexagon_nn_headers_v1.20.0.9.tgz",
        ],
        build_file = "//third_party/hexagon:hexagon.BUILD",
    )
