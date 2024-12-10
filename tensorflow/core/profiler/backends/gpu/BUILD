load(
    "@local_xla//xla/tsl/platform/default:cuda_build_defs.bzl",
    "if_cuda_is_configured",
)
load("//tensorflow:tensorflow.default.bzl", "tf_cuda_cc_test")
load("//tensorflow/core/platform:build_config_root.bzl", "tf_cuda_tests_tags")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    licenses = ["notice"],
)

tf_cuda_cc_test(
    name = "device_tracer_test",
    size = "small",
    srcs = ["device_tracer_test.cc"],
    args = ["--heap_check="],
    tags = tf_cuda_tests_tags() + [
        "gpu_cupti",
        "nomac",
    ],
    deps = [
        "//tensorflow/cc:cc_ops",
        "//tensorflow/core:all_kernels",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:direct_session",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
        "//tensorflow/core/common_runtime:direct_session_internal",
        "//tensorflow/core/framework:graph_proto_cc",
        "//tensorflow/core/framework:types_proto_cc",
        "//tensorflow/core/kernels:ops_util",
        "//tensorflow/core/profiler/lib:profiler_interface",
        "//tensorflow/core/profiler/lib:profiler_session",
        "//tensorflow/core/profiler/utils:xplane_schema",
        "//tensorflow/core/profiler/utils:xplane_utils",
        "//tensorflow/core/profiler/utils:xplane_visitor",
        "@com_google_absl//absl/strings",
        "@local_xla//xla/backends/profiler/gpu:cuda_test",
        "@local_xla//xla/backends/profiler/gpu:cupti_collector",
        "@local_xla//xla/backends/profiler/gpu:device_tracer",
        "@local_xla//xla/tsl/profiler/utils:tf_xplane_visitor",
    ] + if_cuda_is_configured([
        "@local_config_cuda//cuda:cuda_headers",
        "@local_config_cuda//cuda:cupti_headers",
    ]),
)
