load("//tensorflow:tensorflow.bzl", "tf_cc_test")
load("//tensorflow/core/platform:rules_cc.bzl", "cc_library")

licenses(["notice"])

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//visibility:private"],
)

cc_library(
    name = "auto_clustering_test_helper",
    testonly = True,
    srcs = ["auto_clustering_test_helper.cc"],
    hdrs = ["auto_clustering_test_helper.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/compiler/jit:compilation_passes",
        "//tensorflow/compiler/jit:jit_compilation_passes",
        "//tensorflow/compiler/jit:xla_cluster_util",
        "//tensorflow/compiler/jit:xla_cpu_jit",
        "//tensorflow/compiler/jit:xla_gpu_jit",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:graph",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/tools/optimization:optimization_pass_runner_lib",
        "@com_google_absl//absl/strings",
        "@local_xla//xla:status_macros",
        "@local_xla//xla:statusor",
    ],
)

tf_cc_test(
    name = "auto_clustering_test",
    srcs = ["auto_clustering_test.cc"],
    data = [
        "keras_imagenet_main.golden_summary",
        "keras_imagenet_main.pbtxt",
        "keras_imagenet_main_graph_mode.golden_summary",
        "keras_imagenet_main_graph_mode.pbtxt",
        "opens2s_gnmt_mixed_precision.golden_summary",
        "opens2s_gnmt_mixed_precision.pbtxt.gz",
    ],
    tags = ["config-cuda-only"],
    deps = [
        ":auto_clustering_test_helper",
        "//tensorflow/core:test",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "device_compiler_test_helper",
    testonly = True,
    srcs = ["device_compiler_test_helper.cc"],
    hdrs = ["device_compiler_test_helper.h"],
    visibility = [
        "//tensorflow/compiler/jit:__pkg__",
    ],
    deps = [
        "//tensorflow/compiler/jit:xla_activity_listener",
        "//tensorflow/compiler/jit:xla_compilation_cache_proto_cc",
        "//tensorflow/compiler/jit:xla_cpu_jit",
        "//tensorflow/compiler/jit:xla_gpu_device",
        "//tensorflow/compiler/jit:xla_gpu_jit",
        "//tensorflow/core:all_kernels",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:direct_session",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:ops",
        "//tensorflow/core:test",
        "//tensorflow/core/platform:path",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@local_xla//xla/service:hlo_proto_cc",
    ],
)

tf_cc_test(
    name = "device_compiler_serialize_test",
    srcs = [
        "device_compiler_serialize_test.cc",
    ],
    tags = [
        "config-cuda-only",
        "no_oss",  # This test only runs with GPU.
        "requires-gpu-nvidia",
        "xla",
    ],
    deps = [
        ":device_compiler_test_helper",
        "//tensorflow/compiler/jit:compilation_passes",
        "//tensorflow/compiler/jit:flags",
        "//tensorflow/core:test",
    ],
)

tf_cc_test(
    name = "device_compiler_serialize_options_test",
    srcs = [
        "device_compiler_serialize_options_test.cc",
    ],
    tags = [
        "config-cuda-only",
        "no_oss",  # This test only runs with GPU.
        "requires-gpu-nvidia",
        "xla",
    ],
    deps = [
        ":device_compiler_test_helper",
        "//tensorflow/compiler/jit:compilation_passes",
        "//tensorflow/compiler/jit:flags",
        "//tensorflow/core:test",
    ],
)
