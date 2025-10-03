load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("//xla:xla.default.bzl", "xla_cc_test")
load(":cc_to_llvm_ir.bzl", "cc_to_llvm_ir")

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

cc_to_llvm_ir(
    name = "tanh_ir_file",
    src = "tanh.cc",
    out = "tanh.ll",
    deps = [
        ":cpp_context_provider",
        ":tanh",
        ":vector_ops",
    ],
)

genrule(
    name = "tanh_ll_header",
    srcs = [":tanh_ir_file"],
    outs = ["tanh_ll.h"],
    # This command wraps the IR text in a C++ raw string literal.
    cmd = """
        echo 'constexpr const char* tanh_ir_string = R"gen_ir(' > $@ && \\
        cat $(location :tanh_ir_file) >> $@ && \\
        echo ')gen_ir";' >> $@
    """,
)

cc_library(
    name = "tanh_ll_header_lib",
    hdrs = [":tanh_ll_header"],
)

xla_cc_test(
    name = "tanh_test",
    srcs = ["tanh_test.cc"],
    deps = [
        ":tanh",
        ":tanh_ll_header_lib",
        ":vector_ops",
        "@com_google_googletest//:gtest_main",
        "@local_tsl//tsl/platform:status_matchers",
        "@local_tsl//tsl/platform:test",
    ],
)

cc_library(
    name = "eigen_unary",
    srcs = ["eigen_unary.cc"],
    hdrs = ["eigen_unary.h"],
    deps = [
        ":vector_ops",
        "@eigen_archive//:eigen3",
    ],
)

cc_to_llvm_ir(
    name = "eigen_unary_ir_file",
    src = "eigen_unary.cc",
    out = "eigen_unary.ll",
    deps = [
        ":cpp_context_provider",
        ":eigen_unary",
    ],
)

cc_library(
    name = "eigen_unary_ll_header_lib",
    hdrs = [":eigen_unary_ll_header"],
)

genrule(
    name = "eigen_unary_ll_header",
    srcs = [":eigen_unary_ir_file"],
    outs = ["eigen_unary_ll.h"],
    cmd = """
        echo 'constexpr const char* eigen_unary_ir_string = R"gen_ir(' > $@ && \\
        cat $(location :eigen_unary_ir_file) >> $@ && \\
        echo ')gen_ir";' >> $@
    """,
)

xla_cc_test(
    name = "eigen_unary_test",
    srcs = ["eigen_unary_test.cc"],
    deps = [
        ":eigen_unary",
        ":eigen_unary_ll_header_lib",
        "@com_google_googletest//:gtest_main",
        "@eigen_archive//:eigen3",
    ],
)
