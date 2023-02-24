load("//tensorflow/tsl/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [":friends"],
    licenses = ["notice"],
)

package_group(
    name = "friends",
    includes = [
        "//tensorflow/compiler/xla:friends",
    ],
)

cc_library(
    name = "collectives",
    srcs = ["collectives.cc"],
    hdrs = ["collectives.h"],
    deps = [
        "//tensorflow/compiler/xla:executable_run_options",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla/runtime:custom_call",
        "//tensorflow/compiler/xla/runtime:custom_call_registry",
        "//tensorflow/compiler/xla/runtime:executable",
        "//tensorflow/compiler/xla/service/cpu:cpu_runtime",
        "@llvm-project//mlir:Support",
    ],
)

cc_library(
    name = "custom_call",
    srcs = ["custom_call.cc"],
    hdrs = ["custom_call.h"],
    deps = [
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:xla_proto_cc",
        "//tensorflow/compiler/xla/runtime:custom_call",
        "//tensorflow/compiler/xla/runtime:custom_call_registry",
        "//tensorflow/compiler/xla/runtime:executable",
        "//tensorflow/compiler/xla/service:custom_call_status_internal",
        "//tensorflow/compiler/xla/service:custom_call_target_registry",
        "//tensorflow/compiler/xla/service:hlo_proto_cc",
        "@llvm-project//mlir:Support",
    ],
)

cc_library(
    name = "fft_call",
    srcs = ["fft_call.cc"],
    hdrs = ["fft_call.h"],
    deps = [
        "//tensorflow/compiler/xla:executable_run_options",
        "//tensorflow/compiler/xla:xla_proto_cc",
        "//tensorflow/compiler/xla/runtime:custom_call",
        "//tensorflow/compiler/xla/runtime:custom_call_registry",
        "//tensorflow/compiler/xla/runtime:executable",
        "//tensorflow/compiler/xla/service:hlo_proto_cc",
        "//tensorflow/compiler/xla/service/cpu:runtime_fft",
        "@com_google_absl//absl/status",
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "xfeed",
    srcs = ["xfeed.cc"],
    hdrs = ["xfeed.h"],
    deps = [
        "//tensorflow/compiler/xla:executable_run_options",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla/runtime:custom_call",
        "//tensorflow/compiler/xla/runtime:custom_call_registry",
        "//tensorflow/compiler/xla/runtime:executable",
        "//tensorflow/compiler/xla/service/cpu:cpu_runtime",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
    ],
)

cc_library(
    name = "rng",
    srcs = ["rng.cc"],
    hdrs = ["rng.h"],
    deps = [
        "//tensorflow/compiler/xla:executable_run_options",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla/runtime:custom_call",
        "//tensorflow/compiler/xla/runtime:custom_call_registry",
        "//tensorflow/compiler/xla/runtime:executable",
        "@com_google_absl//absl/status",
        "@llvm-project//mlir:Support",
    ],
)
