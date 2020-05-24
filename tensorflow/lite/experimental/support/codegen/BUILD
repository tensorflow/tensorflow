# The tools for generating wrapper classes for a TFLite model with metadata.

package(
    default_visibility = [
        "//visibility:public",
    ],
    licenses = ["notice"],  # Apache 2.0
)

cc_library(
    name = "utils",
    srcs = [
        "utils.cc",
    ],
    hdrs = [
        "utils.h",
    ],
    deps = [
    ],
)

cc_library(
    name = "code_generator",
    srcs = [
        "code_generator.cc",
    ],
    hdrs = [
        "code_generator.h",
    ],
    deps = [
        ":utils",
        "//tensorflow/lite/experimental/support/metadata:metadata_schema_cc",
    ],
)

cc_library(
    name = "metadata_helper",
    srcs = [
        "metadata_helper.cc",
    ],
    hdrs = [
        "metadata_helper.h",
    ],
    deps = [
        ":utils",
        "//tensorflow/lite/experimental/support/metadata:metadata_schema_cc",
        "//tensorflow/lite/schema:schema_fbs",
    ],
)

cc_library(
    name = "android_java_generator",
    srcs = [
        "android_java_generator.cc",
    ],
    hdrs = [
        "android_java_generator.h",
    ],
    deps = [
        ":code_generator",
        ":metadata_helper",
        ":utils",
        "//tensorflow/core/platform:logging",
        "//tensorflow/lite/experimental/support/metadata:metadata_schema_cc",
        "//tensorflow/lite/schema:schema_fbs",
    ],
)

cc_test(
    name = "code_generator_test",
    size = "small",
    srcs = ["code_generator_test.cc"],
    data = ["//tensorflow/lite/experimental/support/metadata:metadata_schema.fbs"],
    deps = [
        ":code_generator",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "utils_test",
    srcs = ["utils_test.cc"],
    deps = [
        ":utils",
        "@com_google_googletest//:gtest_main",
    ],
)
