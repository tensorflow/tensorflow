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

# User ops.

load(
    "//tensorflow:tensorflow.bzl",
    "tf_copts",
)
load(
    "//tensorflow/core/platform:rules_cc.bzl",
    "cc_library",
)

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [
        "//tensorflow/core:__pkg__",
    ],
    licenses = ["notice"],
)

# One target for all user ops.
cc_library(
    name = "user_ops_op_lib",
    srcs = glob(["*.cc"]),
    copts = tf_copts(),
    linkstatic = 1,
    visibility = ["//tensorflow/core:__pkg__"],
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:portable_gif_internal",
    ],
    alwayslink = 1,
)
