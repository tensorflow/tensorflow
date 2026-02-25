load("//xla:xla.default.bzl", "xla_cc_test")
load("//xla/service:xla_compile.bzl", "xla_aot_compile_cpu")
load("//xla/tsl/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

exports_files(glob(["hlos/**"]))

exports_files(glob(["*.textproto"]))

xla_aot_compile_cpu(
    name = "add_aot",
    module = ":hlos/add.hlo",
)

xla_aot_compile_cpu(
    name = "dot_aot",
    module = ":hlos/dot.hlo",
)

# Used to test that libraries can handle loading the aot files.
# This test is here mostly as an example of how one might want to use the generated files.
cc_library(
    name = "add_aot_example_lib",
    testonly = True,
    srcs = ["add_aot_example_lib.cc"],
    hdrs = ["add_aot_example_lib.h"],
    data = [
        ":add_aot",
    ],
    deps = [
        "//xla:xla_data_proto_cc",
        "//xla/backends/cpu/lite_aot:xla_aot_function",
        "//xla/tsl/platform:status_macros",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status:statusor",
        "@tsl//tsl/platform",
        "@tsl//tsl/platform:path",
    ],
)

xla_cc_test(
    name = "xla_aot_function_test",
    srcs = ["xla_aot_function_test.cc"],
    data = [
        ":dot_aot",
    ],
    tags = [
        "not_run:arm",
    ],
    deps = [
        ":add_aot_example_lib",
        "//xla:xla_data_proto_cc",
        "//xla/backends/cpu:alignment",
        "//xla/backends/cpu/lite_aot:xla_aot_function",
        "//xla/tsl/platform:env",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_googletest//:gtest_main",
        "@tsl//tsl/platform",
        "@tsl//tsl/platform:path",
    ],
)
