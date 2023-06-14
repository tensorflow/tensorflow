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
    name = "retain",
    srcs = ["retain.cc"],
    visibility = ["//visibility:public"],
    alwayslink = 1,
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
    name = "convolution",
    srcs = ["convolution.cc"],
    hdrs = ["convolution.h"],
    deps = [
        "//tensorflow/compiler/xla:executable_run_options",
        "//tensorflow/compiler/xla/runtime:memref_view",
        "//tensorflow/compiler/xla/service/cpu:runtime_conv2d",
        "//tensorflow/compiler/xla/service/cpu:runtime_conv3d",
        "//tensorflow/compiler/xla/service/cpu:runtime_fft",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "convolution_ffi",
    srcs = ["convolution_ffi.cc"],
    hdrs = ["convolution_ffi.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":convolution",
        "//tensorflow/compiler/xla/runtime:aot_ffi",
        "//tensorflow/compiler/xla/runtime:aot_ffi_execution_context",
        "//tensorflow/compiler/xla/runtime/ffi:ffi_api",
    ],
)

cc_library(
    name = "convolution_call",
    srcs = ["convolution_call.cc"],
    hdrs = ["convolution_call.h"],
    deps = [
        ":convolution",
        "//tensorflow/compiler/xla:executable_run_options",
        "//tensorflow/compiler/xla/runtime:custom_call",
        "//tensorflow/compiler/xla/runtime:custom_call_registry",
        "//tensorflow/compiler/xla/runtime:executable",
        "@com_google_absl//absl/types:span",
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
        "//tensorflow/compiler/xla/runtime:memref_view",
        "@com_google_absl//absl/status",
    ],
)

cc_library(
    name = "rng_call",
    srcs = ["rng_call.cc"],
    hdrs = ["rng_call.h"],
    deps = [
        ":rng",
        "//tensorflow/compiler/xla:executable_run_options",
        "//tensorflow/compiler/xla/runtime:custom_call",
        "//tensorflow/compiler/xla/runtime:custom_call_registry",
        "//tensorflow/compiler/xla/runtime:executable",
    ],
)

cc_library(
    name = "rng_ffi",
    srcs = ["rng_ffi.cc"],
    hdrs = ["rng_ffi.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":rng",
        "//tensorflow/compiler/xla/runtime:aot_ffi",
        "//tensorflow/compiler/xla/runtime:aot_ffi_execution_context",
        "//tensorflow/compiler/xla/runtime/ffi:ffi_api",
    ],
)
