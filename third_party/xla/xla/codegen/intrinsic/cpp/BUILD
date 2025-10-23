load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("//xla:xla.default.bzl", "xla_cc_test")
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
)

cpu_features = select({
    ":is_x86_64": {
        "avx2": [
            "-mavx2",
            "-mno-avx512f",
        ],
        "avx512": [
            "-mavx2",
            "-mavx512f",
        ],
    },
    ":is_aarch64": {
        "neon": [],
    },
    "//conditions:default": {},
})

cc_ir_header(
    name = "tanh_ll",
    src = "tanh.cc",
    cpu_features = cpu_features,
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
    cpu_features = cpu_features,
    deps = [
        ":cpp_context_provider",
        ":eigen_unary_lib",
        ":vector_ops",
    ],
)

xla_cc_test(
    name = "eigen_unary_test",
    srcs = ["eigen_unary_test.cc"],
    deps = [
        ":eigen_unary_ll",
        "@com_google_googletest//:gtest_main",
    ],
)

config_setting(
    name = "is_x86_64",
    constraint_values = [
        "@platforms//cpu:x86_64",
    ],
)

config_setting(
    name = "is_aarch64",
    constraint_values = [
        "@platforms//cpu:aarch64",
    ],
)
