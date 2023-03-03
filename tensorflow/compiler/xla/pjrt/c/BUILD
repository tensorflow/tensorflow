load("//tensorflow/tsl/platform:rules_cc.bzl", "cc_library")
load("//tensorflow/compiler/xla:xla.bzl", "xla_cc_test")

# copybara:uncomment package(default_applicable_licenses = ["//tensorflow:license"])

cc_library(
    name = "pjrt_c_api_hdrs",
    hdrs = ["pjrt_c_api.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/compiler/xla/stream_executor/tpu:c_api_decl",  # TODO(b/238999986): Remove this.
    ],
)

cc_library(
    name = "pjrt_c_api_tpu_hdrs",
    hdrs = ["pjrt_c_api_tpu.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":pjrt_c_api_hdrs",
    ],
)

# TODO(269360974): remove this target after pjrt_c_api_client no longer needs
# to access wrapped objects.
cc_library(
    name = "pjrt_c_api_wrapper_impl_header",
    hdrs = ["pjrt_c_api_wrapper_impl.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":pjrt_c_api_hdrs",
        "//tensorflow/compiler/xla/pjrt:pjrt_client",
        "//tensorflow/compiler/xla/pjrt:pjrt_compiler",
        "//tensorflow/compiler/xla/pjrt:pjrt_future",
    ],
)

cc_library(
    name = "pjrt_c_api_wrapper_impl",
    srcs = ["pjrt_c_api_wrapper_impl.cc"],
    hdrs = ["pjrt_c_api_wrapper_impl.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":pjrt_c_api_hdrs",
        ":pjrt_c_api_helpers",
        "//tensorflow/compiler/xla:literal",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:status",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla/client:xla_computation",
        "//tensorflow/compiler/xla/hlo/ir:hlo",
        "//tensorflow/compiler/xla/pjrt:mlir_to_hlo",
        "//tensorflow/compiler/xla/pjrt:pjrt_client",
        "//tensorflow/compiler/xla/pjrt:pjrt_compiler",
        "//tensorflow/compiler/xla/pjrt:pjrt_executable",
        "//tensorflow/compiler/xla/pjrt:pjrt_future",
        "//tensorflow/compiler/xla/service:hlo_proto_cc",
        "//tensorflow/compiler/xla/stream_executor/tpu:c_api_conversions",  # TODO(b/238999986): Remove this.
        "//tensorflow/tsl/platform:errors",
        "@com_google_absl//absl/synchronization",
    ],
)

cc_library(
    name = "pjrt_c_api_helpers",
    srcs = ["pjrt_c_api_helpers.cc"],
    hdrs = ["pjrt_c_api_helpers.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":pjrt_c_api_hdrs",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:status",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla/pjrt:pjrt_client",
        "//tensorflow/compiler/xla/pjrt:pjrt_future",
    ],
)

cc_library(
    name = "pjrt_c_api_cpu",
    srcs = ["pjrt_c_api_cpu.cc"],
    hdrs = ["pjrt_c_api_cpu.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":pjrt_c_api_hdrs",
        ":pjrt_c_api_helpers",
        ":pjrt_c_api_wrapper_impl",
        "//tensorflow/compiler/xla/pjrt:tfrt_cpu_pjrt_client",
    ],
)

xla_cc_test(
    name = "pjrt_c_api_helpers_test",
    srcs = ["pjrt_c_api_helpers_test.cc"],
    deps = [
        ":pjrt_c_api_helpers",
        "//tensorflow/compiler/xla/pjrt:pjrt_client",
        "//tensorflow/tsl/lib/core:status_test_util",
        "//tensorflow/tsl/platform:status",
        "@com_google_googletest//:gtest_main",
    ],
)

xla_cc_test(
    name = "pjrt_c_api_cpu_test",
    srcs = ["pjrt_c_api_cpu_test.cc"],
    deps = [
        ":pjrt_c_api_cpu",
        ":pjrt_c_api_hdrs",
        ":pjrt_c_api_wrapper_impl",
        "//tensorflow/compiler/xla/pjrt:pjrt_client",
        "@com_google_googletest//:gtest_main",
    ],
)
