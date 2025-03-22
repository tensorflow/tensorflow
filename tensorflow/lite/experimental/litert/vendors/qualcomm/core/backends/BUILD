# Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//tensorflow/lite/experimental/litert/vendors/qualcomm:__subpackages__"],
)

cc_library(
    name = "hexagon_backend_utils",
    hdrs = ["hexagon_backend_utils.h"],
)

cc_library(
    name = "perf_control",
    srcs = ["perf_control.cc"],
    hdrs = ["perf_control.h"],
    deps = [
        # copybara:uncomment "//third_party/qairt/latest:qnn_lib_headers",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm/core/backends:hexagon_backend_utils",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm/core:common",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm/core/utils:log",
    ],
)
