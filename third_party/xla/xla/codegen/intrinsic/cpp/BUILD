load("//xla:xla.default.bzl", "xla_cc_test")
load("//xla/tsl/platform:rules_cc.bzl", "cc_library")
load(":cc_to_llvm_ir.bzl", "cc_ir_header")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [":friends"],
    licenses = ["notice"],
)

package_group(
    name = "friends",
    includes = [
        "//xla:friends",
    ],
)

# This empty target provides the C++ compilation context for the C++ source file.
cc_library(
    name = "cpp_context_provider",
    srcs = [],
)

cc_library(
    name = "tanh",
    srcs = ["tanh.cc"],
    hdrs = ["tanh.h"],
    deps = [
        ":vector_ops",
    ],
)

cc_library(
    name = "vector_ops",
    hdrs = ["vector_ops.h"],
    deps = [
        "@eigen_archive//:eigen3",
    ],
)

cc_ir_header(
    name = "tanh_ll",
    src = "tanh.cc",
    deps = [
        ":cpp_context_provider",
        ":tanh",
        ":vector_ops",
    ],
)

xla_cc_test(
    name = "tanh_test",
    srcs = ["tanh_test.cc"],
    deps = [
        ":tanh_ll",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "eigen_unary_lib",
    srcs = ["eigen_unary.cc"],
    hdrs = ["eigen_unary.h"],
    deps = [
        ":vector_ops",
        "@eigen_archive//:eigen3",
    ],
)

cc_ir_header(
    name = "eigen_unary_ll",
    src = "eigen_unary.cc",
    deps = [
        ":cpp_context_provider",
        ":eigen_unary_lib",
        ":vector_ops",
        "@eigen_archive//:eigen3",
    ],
)

xla_cc_test(
    name = "eigen_unary_test",
    srcs = ["eigen_unary_test.cc"],
    deps = [
        ":eigen_unary_lib",
        ":eigen_unary_ll",
        ":vector_ops",
        "//xla/codegen/intrinsic:test_matchers",
        "@com_google_googletest//:gtest_main",
    ],
)
