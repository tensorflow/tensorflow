# Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//tensorflow/lite/experimental/litert/vendors/qualcomm:__subpackages__"],
)

cc_library(
    name = "soc_table",
    hdrs = ["soc_table.h"],
    srcs = ["soc_table.cc"],
    tags = [
        # Don't build/test in OS until qnn is available.
        "nobuilder",
    ],
    deps = [
        "//tensorflow/lite/experimental/litert/vendors/qualcomm/core/utils:log",
    ],
)
