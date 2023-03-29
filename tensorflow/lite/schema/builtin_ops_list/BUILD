# This package contains code to auto-generate the contents of the file
#   tensorflow/lite/core/shims:builtin_ops_list.inc
# from the BuiltinOperator enum in the FlatBuffer schema,
# and a test to verify that the checked-in copy remains up-to-date.

# TODO(b/184934065): consider merging the code in this directory with
# the code in ../builtin_ops_header/, i.e. have a single tool generate
# both the builtin_ops.h header and the builtin_ops_list.inc file?

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [
        "//visibility:public",
    ],
    licenses = ["notice"],
)

cc_library(
    name = "generator",
    srcs = ["generator.cc"],
    hdrs = ["generator.h"],
    deps = [
        "//tensorflow/lite/schema:schema_fbs",
    ],
)

cc_binary(
    name = "generate",
    srcs = ["generate.cc"],
    deps = [
        ":generator",
    ],
)

cc_test(
    name = "generator_test",
    srcs = ["generator_test.cc"],
    deps = [
        ":generator",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "consistency_test",
    srcs = ["consistency_test.cc"],
    data = [
        "//tensorflow/lite/kernels:builtin_ops_list.inc",
    ],
    deps = [
        ":generator",
        "@com_google_googletest//:gtest_main",
    ],
)
