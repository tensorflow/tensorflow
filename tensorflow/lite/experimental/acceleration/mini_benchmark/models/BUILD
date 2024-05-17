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

# Model files for mini-benchmark tests and examples.

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [
        "//tensorflow/lite/experimental/acceleration/mini_benchmark:__subpackages__",
        "//tensorflow/lite/tools/benchmark:__subpackages__",
    ],
    licenses = ["notice"],
)

exports_files(["blazeface_mlkit_v1.tfl"])

filegroup(
    name = "add.bin",
    srcs = [
        "//tensorflow/lite:testdata/add.bin",
    ],
)

filegroup(
    name = "mobilenet_v1_1.0_224.tflite",
    srcs = [
        "@tflite_mobilenet_float//:mobilenet_v1_1.0_224.tflite",
    ],
)

filegroup(
    name = "mobilenet_v1_1.0_224_quant.tflite",
    srcs = [
        "@tflite_mobilenet_quant//:mobilenet_v1_1.0_224_quant.tflite",
    ],
)
