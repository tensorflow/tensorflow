# Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//tensorflow/lite/experimental/litert/vendors/qualcomm:__subpackages__"],
)

cc_library(
    name = "device_custom_config",
    srcs = ["device_custom_config.cc"],
    hdrs = ["device_custom_config.h"],
    tags = [
        # Don't build/test in OS until qnn is available.
        "nobuilder",
    ],
    deps = [
        # copybara:uncomment "//third_party/qairt/latest:qnn_lib_headers",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm/core/schema:soc_table",
    ],
)

cc_library(
    name = "device_platform_config",
    srcs = ["device_platform_config.cc"],
    hdrs = ["device_platform_config.h"],
    tags = [
        # Don't build/test in OS until qnn is available.
        "nobuilder",
    ],
    deps = [
        # copybara:uncomment "//third_party/qairt/latest:qnn_lib_headers",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm/core/schema:soc_table",
    ],
)
