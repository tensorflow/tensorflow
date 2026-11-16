load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("//tensorflow:tensorflow.bzl", "tf_cc_test")
load("//tensorflow:tensorflow.default.bzl", "get_compatible_with_portable")
load(
    "//tensorflow/core/platform:build_config.bzl",
    "tf_proto_library",
)

# copybara:uncomment package(default_applicable_licenses = ["//tensorflow:LICENSE"])

# OSS only: This target is header-only. Link `debug_options_proto_cc_impl` only to
# `libtensorflow_framework.so` via `lib_internal_impl`. Do NOT link `debug_options_proto_cc_impl`
# directly unless the target does not link `libtensorflow_framework.so`.
tf_proto_library(
    name = "debug_options_proto",
    srcs = ["debug_options.proto"],
    make_default_target_header_only = True,
    visibility = ["//visibility:public"],
)

cc_library(
    name = "debug",
    srcs = ["debug.cc"],
    hdrs = ["debug.h"],
    visibility = ["//tensorflow/compiler/mlir/lite:__subpackages__"],
    deps = [
        ":debug_options_proto_cc",
        "//tensorflow/compiler/mlir/lite/metrics:error_collector_inst",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/time",
        "@com_googlesource_code_re2//:re2",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:Transforms",
        "@tsl//tsl/platform:path",
        "@xla//xla/tsl/lib/io:buffered_file",
        "@xla//xla/tsl/platform:env",
    ],
)

tf_cc_test(
    name = "debug_test",
    srcs = ["debug_test.cc"],
    deps = [
        ":debug",
        ":debug_options_proto_cc",
        "//tensorflow/compiler/mlir/lite:tensorflow_lite",
        "//tensorflow/core:portable_gif_internal",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@tsl//tsl/platform:path",
        "@xla//xla/tsl/lib/core:status_test_util",
        "@xla//xla/tsl/platform:env",
    ],
)

# TSL-free version of `debug` for the StableHLO to TFLite converter. Only
# supports dump directories on the local file system.
cc_library(
    name = "pass_manager_debug",
    srcs = ["pass_manager_debug.cc"],
    hdrs = ["pass_manager_debug.h"],
    compatible_with = get_compatible_with_portable(),
    visibility = ["//tensorflow/compiler/mlir/lite:__subpackages__"],
    deps = [
        ":debug_options_proto_cc",
        "//tensorflow/compiler/mlir/lite:file_utils",
        "//tensorflow/compiler/mlir/lite/metrics:error_collector_inst",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/time",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
    ],
)

tf_cc_test(
    name = "pass_manager_debug_test",
    srcs = ["pass_manager_debug_test.cc"],
    deps = [
        ":debug_options_proto_cc",
        ":pass_manager_debug",
        "//tensorflow/compiler/mlir/lite:file_utils",
        "//tensorflow/compiler/mlir/lite:tensorflow_lite_ops",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest_main",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
    ],
)
