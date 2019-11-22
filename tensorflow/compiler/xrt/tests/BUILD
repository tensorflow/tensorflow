load("//tensorflow:tensorflow.bzl", "tf_cc_test", "tf_cuda_cc_test")
load(
    "//tensorflow/core/platform:default/build_config_root.bzl",
    "tf_cuda_tests_tags",
)

package(
    default_visibility = [
        "//learning/brain:__subpackages__",
        "//tensorflow/compiler:__subpackages__",
    ],
    licenses = ["notice"],  # Apache 2.0
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
        "//tensorflow/compiler/xla:literal",
        "//tensorflow/compiler/xla:literal_util",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla/client:client_library",
        "//tensorflow/compiler/xla/client:local_client",
        "//tensorflow/compiler/xla/client:xla_builder",
        "//tensorflow/compiler/xla/client:xla_computation",
        "//tensorflow/compiler/xla/client/lib:arithmetic",
        "//tensorflow/compiler/xla/client/lib:constants",
        "//tensorflow/compiler/xla/service:platform_util",
        "//tensorflow/compiler/xrt:xrt_proto_cc",
        "//tensorflow/compiler/xrt:xrt_server",
        "//tensorflow/compiler/xrt/cc:xrt_ops",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core:tensorflow_opensource",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "@com_google_absl//absl/strings",
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
    tags = tf_cuda_tests_tags(),
    deps = [
        ":raw_api_test_lib",
        "//tensorflow/compiler/jit:xla_gpu_device",
    ],
)
