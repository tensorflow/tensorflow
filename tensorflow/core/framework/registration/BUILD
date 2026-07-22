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

load("@rules_cc//cc:cc_library.bzl", "cc_library")
load(
    "//tensorflow:tensorflow.bzl",
    "tf_cc_test",
    "tf_gen_options_header",
)
load("//tensorflow:tensorflow.default.bzl", "tf_selective_registration_deps")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [
        "//tensorflow/core:__subpackages__",
    ],
    licenses = ["notice"],
)

exports_files(
    [
        "registration.h",
    ],
    visibility = ["//tensorflow/core:__subpackages__"],
)

tf_gen_options_header(
    name = "gen_registration_options",
    build_settings = {
        "//tensorflow:enable_registration_v2": "REGISTRATION_V2",
    },
    output_header = "options.h",
    template = "options.h.tpl",
)

cc_library(
    name = "registration",
    hdrs = [
        "options.h",
        "registration.h",
    ],
    deps = tf_selective_registration_deps(),
)

tf_cc_test(
    name = "registration_test",
    size = "small",
    srcs = [
        "registration_test.cc",
    ],
    deps = [
        ":registration",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
        "@com_google_googletest//:gtest_main",
    ],
)
