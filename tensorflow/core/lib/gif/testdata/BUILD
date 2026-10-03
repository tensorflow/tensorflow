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

# Description:
# gif test data packages.

load("//tensorflow:tensorflow.default.bzl", "filegroup")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    licenses = ["notice"],
)

filegroup(
    name = "gif_testdata",
    srcs = [
        # GIF data
        "lena.gif",
        "scan.gif",
        "red_black.gif",
        "squares.gif",
        "3g_multiframe.gif",
        "pendulum_sm.gif",
        # Add groundtruth frames for `pendulum_sm.gif`.
        # PNG format because it's lossless.
        "pendulum_sm_frame0.png",
        "pendulum_sm_frame1.png",
        "pendulum_sm_frame2.png",
        # GIF data with optimization
        "optimized.gif",
    ],
    visibility = ["//visibility:public"],
)
