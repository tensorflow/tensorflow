load("//third_party/tensorflow/compiler/xla:xla.google.bzl", "xla_cc_binary", "xla_cc_test")
load(
    "//third_party/tensorflow/tsl/platform:build_config_root.bzl",
    "tf_cuda_tests_tags",
)
load(
    "//third_party/tensorflow/tsl/platform/google:cuda_build_defs.bzl",
    "if_cuda_is_configured",
)
load("//third_party/tensorflow/tsl/platform:rules_cc.bzl", "cc_library")

package(default_applicable_licenses = ["//third_party/tensorflow:license"])  # copybara:comment

cc_library(
    name = "pjrt_c_api_hdrs",
    hdrs = ["pjrt_c_api.h"],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "pjrt_c_api_tpu_hdrs",
    hdrs = ["pjrt_c_api_tpu.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":pjrt_c_api_hdrs",
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
        "//third_party/absl/base:core_headers",
        "//third_party/absl/container:flat_hash_map",
        "//third_party/absl/log",
        "//third_party/absl/log:check",
        "//third_party/absl/status",
        "//third_party/absl/strings",
        "//third_party/absl/synchronization",
        "//third_party/absl/time",
        "//third_party/absl/types:span",
        "//third_party/llvm/llvm-project/mlir:IR",
        "//third_party/tensorflow/compiler/xla:literal",
        "//third_party/tensorflow/compiler/xla:shape_util",
        "//third_party/tensorflow/compiler/xla:status",
        "//third_party/tensorflow/compiler/xla:statusor",
        "//third_party/tensorflow/compiler/xla:util",
        "//third_party/tensorflow/compiler/xla:xla_data_proto_cc",
        "//third_party/tensorflow/compiler/xla:xla_proto_cc",
        "//third_party/tensorflow/compiler/xla/client:xla_computation",
        "//third_party/tensorflow/compiler/xla/hlo/ir:hlo",
        "//third_party/tensorflow/compiler/xla/pjrt:compile_options_proto_cc",
        "//third_party/tensorflow/compiler/xla/pjrt:mlir_to_hlo",
        "//third_party/tensorflow/compiler/xla/pjrt:pjrt_client",
        "//third_party/tensorflow/compiler/xla/pjrt:pjrt_common",
        "//third_party/tensorflow/compiler/xla/pjrt:pjrt_compiler",
        "//third_party/tensorflow/compiler/xla/pjrt:pjrt_device_description",
        "//third_party/tensorflow/compiler/xla/pjrt:pjrt_executable",
        "//third_party/tensorflow/compiler/xla/pjrt:pjrt_future",
        "//third_party/tensorflow/compiler/xla/service:computation_placer_hdr",
        "//third_party/tensorflow/compiler/xla/service:hlo_proto_cc",
        "//third_party/tensorflow/tsl/framework:allocator",
        "//third_party/tensorflow/tsl/platform:errors",
        "//third_party/tensorflow/tsl/platform:statusor",
    ],
)

cc_library(
    name = "pjrt_c_api_helpers",
    srcs = ["pjrt_c_api_helpers.cc"],
    hdrs = ["pjrt_c_api_helpers.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":pjrt_c_api_hdrs",
        "//third_party/absl/status",
        "//third_party/absl/strings",
        "//third_party/absl/types:span",
        "//third_party/tensorflow/compiler/xla:shape_util",
        "//third_party/tensorflow/compiler/xla:status",
        "//third_party/tensorflow/compiler/xla:statusor",
        "//third_party/tensorflow/compiler/xla:xla_data_proto_cc",
        "//third_party/tensorflow/compiler/xla/pjrt:pjrt_client",
        "//third_party/tensorflow/compiler/xla/pjrt:pjrt_future",
    ],
)

cc_library(
    name = "pjrt_c_api_cpu_internal",
    srcs = ["pjrt_c_api_cpu_internal.cc"],
    hdrs = ["pjrt_c_api_cpu_internal.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":pjrt_c_api_hdrs",
        ":pjrt_c_api_helpers",
        ":pjrt_c_api_wrapper_impl",
        "//third_party/tensorflow/compiler/xla/pjrt:pjrt_client",
        "//third_party/tensorflow/compiler/xla/pjrt:tfrt_cpu_pjrt_client",
    ],
)

cc_library(
    name = "pjrt_c_api_cpu",
    srcs = ["pjrt_c_api_cpu.cc"],
    hdrs = ["pjrt_c_api_cpu.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":pjrt_c_api_cpu_internal",
        ":pjrt_c_api_hdrs",
    ],
)

cc_library(
    name = "pjrt_c_api_gpu_internal",
    srcs = ["pjrt_c_api_gpu_internal.cc"],
    hdrs = ["pjrt_c_api_gpu_internal.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":pjrt_c_api_hdrs",
        ":pjrt_c_api_helpers",
        ":pjrt_c_api_wrapper_impl",
        "//third_party/absl/container:flat_hash_map",
        "//third_party/tensorflow/compiler/xla/pjrt:pjrt_client",
        "//third_party/tensorflow/compiler/xla/pjrt:pjrt_common",
        "//third_party/tensorflow/compiler/xla/pjrt/gpu:gpu_helpers",
        "//third_party/tensorflow/compiler/xla/pjrt/gpu:se_gpu_pjrt_client",
        "//third_party/tensorflow/tsl/platform:errors",
    ],
)

cc_library(
    name = "pjrt_c_api_gpu",
    srcs = ["pjrt_c_api_gpu.cc"],
    hdrs = ["pjrt_c_api_gpu.h"],
    deps = [
        ":pjrt_c_api_gpu_internal",
        ":pjrt_c_api_hdrs",
    ],
)

# PJRT GPU plugin. Can be configured to be built for CUDA or ROCM.
xla_cc_binary(
    name = "pjrt_c_api_gpu_plugin.so",
    linkopts = [
        "-Wl,--version-script,$(location :pjrt_c_api_gpu_version_script.lds)",
        "-Wl,--no-undefined",
    ],
    linkshared = True,
    tags = [
        "noasan",
        "nomsan",
        "notsan",
    ],
    deps = [
        ":pjrt_c_api_gpu",
        ":pjrt_c_api_gpu_version_script.lds",
        "//third_party/tensorflow/compiler/xla/service:gpu_plugin",
    ] + if_cuda_is_configured([
        "//third_party/tensorflow/compiler/xla/stream_executor:cuda_platform",
    ]),
)

cc_library(
    name = "pjrt_c_api_test_base",
    testonly = 1,
    srcs = ["pjrt_c_api_test_base.cc"],
    hdrs = ["pjrt_c_api_test_base.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":pjrt_c_api_hdrs",
        ":pjrt_c_api_wrapper_impl",
        "//testing/base/public:gunit_main",
        "//third_party/tensorflow/compiler/xla/pjrt:pjrt_client",
    ],
)

xla_cc_test(
    name = "pjrt_c_api_gpu_test",
    srcs = ["pjrt_c_api_gpu_test.cc"],
    tags = tf_cuda_tests_tags(),
    deps = [
        ":pjrt_c_api_gpu",
        ":pjrt_c_api_hdrs",
        ":pjrt_c_api_helpers",
        ":pjrt_c_api_test_base",
        ":pjrt_c_api_test_common",
        ":pjrt_c_api_wrapper_impl",
        "//testing/base/public:gunit_main",
        "//third_party/absl/container:flat_hash_map",
        "//third_party/absl/log:check",
        "//third_party/absl/status",
        "//third_party/absl/status:statusor",
        "//third_party/absl/strings",
        "//third_party/absl/synchronization",
        "//third_party/absl/time",
        "//third_party/tensorflow/compiler/xla:status",
        "//third_party/tensorflow/compiler/xla:statusor",
        "//third_party/tensorflow/compiler/xla/pjrt:pjrt_client",
        "//third_party/tensorflow/compiler/xla/pjrt:pjrt_common",
        "//third_party/tensorflow/compiler/xla/pjrt/gpu:gpu_helpers",
        "//third_party/tensorflow/compiler/xla/service:gpu_plugin",
        "//third_party/tensorflow/tsl/platform:status",
        "//third_party/tensorflow/tsl/platform:statusor",
    ],
)

xla_cc_test(
    name = "pjrt_c_api_helpers_test",
    srcs = ["pjrt_c_api_helpers_test.cc"],
    deps = [
        ":pjrt_c_api_hdrs",
        ":pjrt_c_api_helpers",
        ":pjrt_c_api_wrapper_impl",
        "//testing/base/public:gunit_main",
        "//third_party/absl/time",
        "//third_party/tensorflow/compiler/xla:shape_util",
        "//third_party/tensorflow/tsl/lib/core:status_test_util",
        "//third_party/tensorflow/tsl/platform:status",
    ],
)

xla_cc_test(
    name = "pjrt_c_api_cpu_test",
    srcs = ["pjrt_c_api_cpu_test.cc"],
    deps = [
        ":pjrt_c_api_cpu",
        ":pjrt_c_api_test_common",
        ":pjrt_c_api_wrapper_impl",
        "//testing/base/public:gunit_main",
    ],
)

cc_library(
    name = "pjrt_c_api_test_common",
    testonly = 1,
    srcs = ["pjrt_c_api_test.cc"],
    hdrs = ["pjrt_c_api_test.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":pjrt_c_api_hdrs",
        ":pjrt_c_api_helpers",
        ":pjrt_c_api_test_base",
        "//testing/base/public:gunit_main",
        "//third_party/absl/base:core_headers",
        "//third_party/absl/log:check",
        "//third_party/absl/status",
        "//third_party/absl/status:statusor",
        "//third_party/absl/strings",
        "//third_party/absl/synchronization",
        "//third_party/absl/types:span",
        "//third_party/tensorflow/compiler/xla:literal",
        "//third_party/tensorflow/compiler/xla:literal_util",
        "//third_party/tensorflow/compiler/xla:shape_util",
        "//third_party/tensorflow/compiler/xla:xla_data_proto_cc",
        "//third_party/tensorflow/compiler/xla:xla_proto_cc",
        "//third_party/tensorflow/compiler/xla/client:executable_build_options",
        "//third_party/tensorflow/compiler/xla/hlo/ir:hlo",
        "//third_party/tensorflow/compiler/xla/pjrt:compile_options_proto_cc",
        "//third_party/tensorflow/compiler/xla/pjrt:pjrt_client",
        "//third_party/tensorflow/compiler/xla/pjrt:pjrt_executable",
        "//third_party/tensorflow/compiler/xla/pjrt:pjrt_future",
        "//third_party/tensorflow/compiler/xla/service:computation_placer_hdr",
        "//third_party/tensorflow/compiler/xla/service:hlo_parser",
        "//third_party/tensorflow/compiler/xla/service:hlo_proto_cc",
        "//third_party/tensorflow/compiler/xla/tests:literal_test_util",
        "//third_party/tensorflow/tsl/platform:errors",
        "//third_party/tensorflow/tsl/platform:status",
        "//third_party/tensorflow/tsl/platform:statusor",
    ],
)
