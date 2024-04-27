load("@local_tsl//tsl/platform:rules_cc.bzl", "cc_library")

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
        "//xla:executable_run_options",
        "//xla:shape_util",
        "//xla:xla_data_proto_cc",
        "//xla/runtime:custom_call",
        "//xla/runtime:custom_call_registry",
        "//xla/runtime:executable",
        "//xla/runtime:memref_view",
        "//xla/service/cpu:cpu_runtime",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:Support",
    ],
)

cc_library(
    name = "convolution",
    srcs = ["convolution.cc"],
    hdrs = ["convolution.h"],
    deps = [
        "//xla:executable_run_options",
        "//xla:xla_data_proto_cc",
        "//xla/runtime:memref_view",
        "//xla/service/cpu:runtime_conv2d",
        "//xla/service/cpu:runtime_conv3d",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/types:span",
        "@eigen_archive//:eigen3",
    ],
)

cc_library(
    name = "convolution_ffi",
    srcs = ["convolution_ffi.cc"],
    hdrs = ["convolution_ffi.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":convolution",
        "//xla:xla_data_proto_cc",
        "//xla/runtime:aot_ffi",
        "//xla/runtime:aot_ffi_execution_context",
        "//xla/runtime:memref_view",
        "//xla/runtime/ffi:ffi_api",
        "//xla/runtime/ffi:ffi_c_api_hdrs",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "convolution_call",
    srcs = ["convolution_call.cc"],
    hdrs = ["convolution_call.h"],
    deps = [
        ":convolution",
        "//xla:executable_run_options",
        "//xla/runtime:custom_call",
        "//xla/runtime:custom_call_registry",
        "//xla/runtime:executable",
        "//xla/runtime:memref_view",
        "@com_google_absl//absl/types:span",
        "@llvm-project//mlir:Support",
    ],
)

cc_library(
    name = "custom_call",
    srcs = ["custom_call.cc"],
    hdrs = ["custom_call.h"],
    deps = [
        "//xla:shape_util",
        "//xla:xla_proto_cc",
        "//xla/runtime:custom_call",
        "//xla/runtime:custom_call_registry",
        "//xla/runtime:executable",
        "//xla/runtime:memref_view",
        "//xla/service:custom_call_status_internal",
        "//xla/service:custom_call_status_public_headers",
        "//xla/service:custom_call_target_registry",
        "//xla/service:hlo_proto_cc",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:Support",
    ],
)

cc_library(
    name = "fft_call",
    srcs = ["fft_call.cc"],
    hdrs = ["fft_call.h"],
    deps = [
        "//xla:executable_run_options",
        "//xla:xla_data_proto_cc",
        "//xla:xla_proto_cc",
        "//xla/runtime:custom_call",
        "//xla/runtime:custom_call_registry",
        "//xla/runtime:executable",
        "//xla/runtime:memref_view",
        "//xla/service:hlo_proto_cc",
        "//xla/service/cpu:runtime_fft",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@llvm-project//mlir:Support",
    ],
)

cc_library(
    name = "xfeed",
    srcs = ["xfeed.cc"],
    hdrs = ["xfeed.h"],
    deps = [
        "//xla:executable_run_options",
        "//xla:shape_util",
        "//xla:xla_data_proto_cc",
        "//xla/runtime:custom_call",
        "//xla/runtime:custom_call_registry",
        "//xla/runtime:executable",
        "//xla/runtime:memref_view",
        "//xla/service/cpu:cpu_runtime",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/types:span",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:Support",
    ],
)

cc_library(
    name = "rng",
    srcs = ["rng.cc"],
    hdrs = ["rng.h"],
    deps = [
        "//xla:executable_run_options",
        "//xla:xla_data_proto_cc",
        "//xla/runtime:memref_view",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "rng_call",
    srcs = ["rng_call.cc"],
    hdrs = ["rng_call.h"],
    deps = [
        ":rng",
        "//xla:executable_run_options",
        "//xla/runtime:custom_call",
        "//xla/runtime:custom_call_registry",
        "//xla/runtime:executable",
        "//xla/runtime:memref_view",
        "@llvm-project//mlir:Support",
    ],
)

cc_library(
    name = "rng_ffi",
    srcs = ["rng_ffi.cc"],
    hdrs = ["rng_ffi.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":rng",
        "//xla:xla_data_proto_cc",
        "//xla/runtime:aot_ffi",
        "//xla/runtime:aot_ffi_execution_context",
        "//xla/runtime:memref_view",
        "//xla/runtime/ffi:ffi_api",
        "//xla/runtime/ffi:ffi_c_api_hdrs",
        "@com_google_absl//absl/status",
    ],
)
