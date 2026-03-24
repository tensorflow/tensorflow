load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("//xla:xla.default.bzl", "xla_cc_test")
load("//xla/tsl:tsl.default.bzl", "get_compatible_with_libtpu_portable", "get_compatible_with_portable")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

cc_library(
    name = "dump",
    srcs = ["dump.cc"],
    hdrs = ["dump.h"],
    compatible_with = (get_compatible_with_libtpu_portable() + get_compatible_with_portable()),
    deps = [
        ":mlir",
        "//xla/pjrt:pjrt_compiler",
        "//xla/pjrt:pjrt_executable",
        "//xla/tsl/platform:env",
        "//xla/tsl/platform:errors",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/time",
        "@llvm-project//mlir:IR",
        "@tsl//tsl/platform:path",
    ],
)

xla_cc_test(
    name = "dump_test",
    srcs = ["dump_test.cc"],
    deps = [
        ":dump",
        "//xla:shape_util",
        "//xla:xla_data_proto_cc",
        "//xla/pjrt:pjrt_compiler",
        "//xla/pjrt:pjrt_device_description",
        "//xla/pjrt:pjrt_executable",
        "//xla/pjrt/proto:topology_description_proto_cc",
        "//xla/service/llvm_ir:llvm_util",
        "//xla/tsl/lib/core:status_test_util",
        "//xla/tsl/platform:env",
        "//xla/tsl/platform:statusor",
        "//xla/tsl/platform:test",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
        "@llvm-project//mlir:IR",
        "@tsl//tsl/platform:path",
    ],
)

cc_library(
    name = "mlir",
    srcs = ["mlir.cc"],
    hdrs = ["mlir.h"],
    compatible_with = (get_compatible_with_libtpu_portable() + get_compatible_with_portable()),
    deps = [
        "//xla/tsl/platform:env",
        "@com_google_absl//absl/status",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
    ],
)

xla_cc_test(
    name = "mlir_test",
    srcs = ["mlir_test.cc"],
    compatible_with = (get_compatible_with_libtpu_portable() + get_compatible_with_portable()),
    deps = [
        ":mlir",
        "//xla/mlir_hlo",
        "//xla/service/llvm_ir:llvm_util",
        "//xla/tsl/lib/core:status_test_util",
        "//xla/tsl/platform:env",
        "//xla/tsl/platform:test",
        "@com_google_googletest//:gtest_main",
        "@llvm-project//mlir:IR",
        "@tsl//tsl/platform:path",
    ],
)
