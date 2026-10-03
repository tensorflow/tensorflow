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

"""External versions of build rules that differ outside of Google."""

def flex_portable_tensorflow_deps():
    """Returns dependencies for building portable tensorflow in Flex delegate."""

    return [
        "//third_party/fft2d:fft2d_headers",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:optional",
        "@eigen_archive//:eigen3",
        "@gemmlowp",
        "@icu//:common",
        "//third_party/icu/data:conversion_data",
    ]

def tflite_copts_extra():
    """Defines extra compile time flags for tflite_copts(). Currently empty."""
    return []
