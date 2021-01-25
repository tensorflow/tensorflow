load("//tensorflow:tensorflow.bzl", "get_compatible_with_portable")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],  # Apache 2.0
)

cc_library(
    name = "resource",
    srcs = [
        "resource_variable.cc",
        "static_hashtable.cc",
    ],
    hdrs = [
        "lookup_interfaces.h",
        "lookup_util.h",
        "resource_base.h",
        "resource_variable.h",
        "static_hashtable.h",
    ],
    compatible_with = get_compatible_with_portable(),
    deps = [
        "//tensorflow/lite:string_util",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/kernels/internal:compatibility",
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
        "//tensorflow/lite/c:common",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ],
)
