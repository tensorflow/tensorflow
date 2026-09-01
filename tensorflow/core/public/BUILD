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

load("//tensorflow:tensorflow.default.bzl", "filegroup")
load(
    "//tensorflow:tf_version.bzl",
    "MAJOR_VERSION",
    "MINOR_VERSION",
    "PATCH_VERSION",
    "TF_SEMANTIC_VERSION_SUFFIX",
)
load("//tensorflow/core/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [
        "//tensorflow/core:__subpackages__",
    ],
    licenses = ["notice"],
)

exports_files(
    srcs = [
        "session.h",
        "session_options.h",
        "version.h",
    ],
    visibility = ["//visibility:public"],
)

# Export source files needed for mobile builds, which do not use granular targets.
filegroup(
    name = "mobile_hdrs_no_runtime",
    srcs = [
        "version.h",
    ],
    visibility = ["//tensorflow/core:__pkg__"],
)

filegroup(
    name = "mobile_hdrs_only_runtime",
    srcs = [
        "session.h",
        "session_options.h",
    ],
    visibility = ["//tensorflow/core:__pkg__"],
)

cc_library(
    name = "version",
    hdrs = ["version.h"],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "release_version",
    hdrs = ["release_version.h"],
    defines = [
        "TF_MAJOR_VERSION={}".format(MAJOR_VERSION),
        "TF_MINOR_VERSION={}".format(MINOR_VERSION),
        "TF_PATCH_VERSION={}".format(PATCH_VERSION),
        'TF_VERSION_SUFFIX=\\"{}\\"'.format(TF_SEMANTIC_VERSION_SUFFIX),
    ],
    visibility = ["//visibility:public"],
)
