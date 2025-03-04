# Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//tensorflow/lite/experimental/litert/vendors/qualcomm:__subpackages__"],
)

cc_test(
    name = "op_wrapper_test",
    srcs = [
        "op_wrapper_test.cc",
    ],
    deps = [
        "//tensorflow/lite/experimental/litert/vendors/qualcomm/core/wrappers:op_wrapper",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm/core/wrappers:param_wrapper",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm/core/wrappers:tensor_wrapper",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "tensor_wrapper_test",
    srcs = [
        "tensor_wrapper_test.cc",
    ],
    deps = [
        "//tensorflow/lite/experimental/litert/vendors/qualcomm/core/wrappers:tensor_wrapper",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "param_wrapper_test",
    srcs = [
        "param_wrapper_test.cc",
    ],
    deps = [
        "//tensorflow/lite/experimental/litert/vendors/qualcomm/core/wrappers:param_wrapper",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm/core/wrappers:tensor_wrapper",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "quantize_params_wrapper_test",
    srcs = [
        "quantize_params_wrapper_test.cc",
    ],
    deps = [
        "//tensorflow/lite/experimental/litert/vendors/qualcomm/core/wrappers:quantize_params_wrapper",
        "@com_google_googletest//:gtest_main",
    ],
)
