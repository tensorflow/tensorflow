load("//tensorflow:tensorflow.default.bzl", "get_compatible_with_portable")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

cc_library(
    name = "resource",
    srcs = [
        "initialization_status.cc",
        "resource_variable.cc",
        "static_hashtable.cc",
    ],
    hdrs = [
        "initialization_status.h",
        "lookup_interfaces.h",
        "lookup_util.h",
        "resource_base.h",
        "resource_variable.h",
        "static_hashtable.h",
    ],
    compatible_with = get_compatible_with_portable(),
    deps = [
        "//tensorflow/lite:string_util",
        "//tensorflow/lite/core/c:c_api_types",
        "//tensorflow/lite/core/c:common",
        "//tensorflow/lite/kernels/internal:tensor",
    ],
)

cc_test(
    name = "resource_variable_test",
    srcs = [
        "resource_variable_test.cc",
    ],
    deps = [
        ":resource",
        "//tensorflow/lite:util",
        "//tensorflow/lite/core/c:c_api_types",
        "//tensorflow/lite/core/c:common",
        "//tensorflow/lite/kernels:test_util",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ],
)
