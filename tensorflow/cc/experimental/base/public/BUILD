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

# Experimental C++ APIs for TensorFlow.
# New TF C++ APIs under the tensorflow::cc namespace aim to guarantee ABI stability.
# Users are expected to compile against public c++ headers, and link against
# libtensorflow (https://www.tensorflow.org/install/lang_c).
# We aim to achieve ABI stability in new C++ APIs by only using types
# on the API surface that:
# 1. Have a header-only implementation
# 2. Are std:: types
# 3. Wrap an opaque C type

load("//tensorflow/core/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    # This is intentionally public
    default_visibility = [
        "//visibility:public",
    ],
    licenses = ["notice"],
)

cc_library(
    name = "runtime",
    hdrs = [
        "runtime.h",
    ],
    deps = [
        ":status",
        "//tensorflow/c/eager:c_api",
        "//tensorflow/c/eager:c_api_experimental",
    ],
)

cc_library(
    name = "runtime_builder",
    hdrs = [
        "runtime_builder.h",
    ],
    deps = [
        ":runtime",
        ":status",
        "//tensorflow/c/eager:c_api",
        "//tensorflow/c/eager:c_api_experimental",
    ],
)

cc_library(
    name = "status",
    hdrs = [
        "status.h",
    ],
    deps = [
        "//tensorflow/c:tf_status",
    ],
)

cc_library(
    name = "tensor",
    hdrs = [
        "tensor.h",
    ],
    deps = [
        ":status",
        "//tensorflow/c:tf_datatype",
        "//tensorflow/c:tf_tensor",
    ],
)

cc_library(
    name = "tensorhandle",
    hdrs = [
        "tensorhandle.h",
    ],
    deps = [
        ":runtime",
        ":status",
        ":tensor",
        "//tensorflow/c/eager:c_api",
        "//tensorflow/c/eager:c_api_experimental",
    ],
)
