load("//tensorflow:tensorflow.bzl", "tf_cc_test", "tf_cuda_cc_test")
load(
    "//tensorflow/core/platform:build_config_root.bzl",
    "tf_cuda_tests_tags",
)
load("//tensorflow/core/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//learning/brain:__subpackages__"],
    licenses = ["notice"],
)

cc_library(
    name = "raw_api_test_lib",
    testonly = 1,
    srcs = [
        "raw_api_test.cc",
    ],
    deps = [
        "//tensorflow/cc:cc_ops",
        "//tensorflow/cc:client_session",
        "//tensorflow/cc:ops",
        "//tensorflow/cc:scope",
        "//tensorflow/compiler/tf2xla:common",
        "//tensorflow/compiler/tf2xla:xla_compiler",
        "//tensorflow/compiler/xrt:xrt_proto_cc",
        "//tensorflow/compiler/xrt:xrt_server",
        "//tensorflow/compiler/xrt/cc:xrt_ops",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core:tensorflow_opensource",
        "//tensorflow/core:test",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@local_tsl//tsl/platform:status",
        "@local_xla//xla:literal",
        "@local_xla//xla:literal_util",
        "@local_xla//xla:shape_util",
        "@local_xla//xla:xla_data_proto_cc",
        "@local_xla//xla/client:client_library",
        "@local_xla//xla/client:executable_build_options",
        "@local_xla//xla/client:local_client",
        "@local_xla//xla/client:padding",
        "@local_xla//xla/client:xla_builder",
        "@local_xla//xla/client:xla_computation",
        "@local_xla//xla/client/lib:arithmetic",
        "@local_xla//xla/client/lib:constants",
        "@local_xla//xla/service:platform_util",
        "@local_xla//xla/stream_executor:platform",
    ],
)

tf_cc_test(
    name = "raw_api_test_cpu",
    size = "medium",
    srcs = [],
    args = [
        "--xla_test_device=XLA_CPU",
        "--xla_platform=CPU",
    ],
    deps = [
        ":raw_api_test_lib",
        "//tensorflow/compiler/jit:xla_cpu_device",
    ],
)

tf_cuda_cc_test(
    name = "raw_api_test_gpu",
    size = "medium",
    srcs = [],
    args = [
        "--xla_test_device=XLA_GPU",
        "--xla_platform=GPU",
    ],
    tags = tf_cuda_tests_tags() + [
        "no_cuda_asan",  # TODO(b/171319142): re-enable.
    ],
    deps = [
        ":raw_api_test_lib",
        "//tensorflow/compiler/jit:xla_gpu_device",
    ],
)
