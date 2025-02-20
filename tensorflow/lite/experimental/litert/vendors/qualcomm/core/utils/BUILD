# Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

load("//tensorflow/lite/experimental/litert/build_common:litert_build_defs.bzl", "litert_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//tensorflow/lite/experimental/litert:__subpackages__"],
)

cc_library(
    name = "utils",
    srcs = select({
        "//tensorflow:android": ["log_android.cc",],
        "//conditions:default": ["log_default.cc",],
    }),
    hdrs = ["log.h"],
    tags = [
        # Don't build/test in OS until qnn is available.
        "nobuilder",
    ],
    linkopts = select({
        "//tensorflow:android": ["-landroid"],
        "//conditions:default": [],
    }),
    deps = [
        "//third_party/qairt/latest:qnn_lib_headers",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/experimental/litert/c:litert_model",
    ],
)

litert_test(
    name = "utils_test",
    srcs = [
        "utils_test.cc",
    ],
    tags = [
        # Tests with ungrte deps do not currently work on forge.
        "no-remote-exec",
        "notap",
        # Don't build/test in OS until qnn is available.
        "nobuilder",
        "no_oss",
        # Sanitizer runtime doesn't work with anything that loads libQnnHtp.so.
        "nosan",
    ],
    deps = [
        ":utils",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/strings:string_view",
    ],
)
