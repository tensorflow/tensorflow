load("@rules_shell//shell:sh_test.bzl", "sh_test")
load("//xla:xla.default.bzl", "xla_cc_test")
load(
    "//xla/tsl/platform:rules_cc.bzl",
    "cc_binary",
    "cc_library",
)

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

cc_library(
    name = "xla_aot_function",
    srcs = ["xla_aot_function.cc"],
    hdrs = ["xla_aot_function.h"],
    deps = [
        "//xla:literal",
        "//xla:shape_util",
        "//xla:xla_data_proto_cc",
        "//xla/backends/cpu/nanort:nanort_executable",
        "//xla/service:executable",
        "//xla/service/cpu:executable_proto_cc",
        "//xla/tsl/concurrency:async_value",
        "//xla/tsl/platform:status_macros",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@tsl//tsl/platform:path",
    ],
)

cc_library(
    name = "infer_lite_aot_deps_main_lib",
    srcs = ["infer_lite_aot_deps_main_lib.cc"],
    hdrs = ["infer_lite_aot_deps_main_lib.h"],
    deps = [
        "//xla/backends/cpu/runtime:thunk_proto_cc",
        "//xla/service/cpu:executable_proto_cc",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
    ],
)

cc_binary(
    name = "infer_lite_aot_deps_main",
    srcs = ["infer_lite_aot_deps_main.cc"],
    deps = [
        ":infer_lite_aot_deps_main_lib",
        "//xla/service/cpu:executable_proto_cc",
        "//xla/tsl/platform:env",
        "//xla/tsl/platform:logging",
        "//xla/tsl/util:command_line_flags",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@tsl//tsl/platform:platform_port",
    ],
)

xla_cc_test(
    name = "infer_lite_aot_deps_main_lib_test",
    srcs = ["infer_lite_aot_deps_main_lib_test.cc"],
    deps = [
        ":infer_lite_aot_deps_main_lib",
        "//xla/backends/cpu/runtime:thunk_proto_cc",
        "//xla/service:xla_compile_result_proto_cc",
        "//xla/service/cpu:executable_proto_cc",
        "//xla/tsl/platform:statusor",
        "@com_google_googletest//:gtest_main",
    ],
)

sh_test(
    name = "infer_lite_aot_deps_test",
    srcs = ["infer_lite_aot_deps_test.sh"],
    args = [
        "$(location :infer_lite_aot_deps_main)",
        "$(location //third_party/tensorflow/compiler/xla/backends/cpu/lite_aot/tests:dot_aot)",
        "\"//xla/backends/cpu/runtime/thunk_serdes:dot_thunk_serdes\"",
    ],
    data = [
        ":infer_lite_aot_deps_main",
        "//xla/backends/cpu/lite_aot/tests:dot_aot",
    ],
)
