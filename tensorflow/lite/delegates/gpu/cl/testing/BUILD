package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

cc_binary(
    name = "delegate_testing",
    srcs = ["delegate_testing.cc"],
    tags = [
        "nobuilder",
        "notap",
    ],
    deps = [
        "//tensorflow/lite/delegates/gpu:delegate",
        "//tensorflow/lite/delegates/gpu/cl:gpu_api_delegate",
        "//tensorflow/lite/delegates/gpu/common:model_builder",
        "//tensorflow/lite/delegates/gpu/common:status",
        "//tensorflow/lite/kernels:builtin_ops",
        "//tensorflow/lite/kernels:kernel_util",
        "@com_google_absl//absl/time",
    ],
)

cc_test(
    name = "gpu_model_test",
    srcs = ["gpu_model_test.cc"],
    tags = [
        "noasan",
        "nomsan",
        "notsan",
        "requires-gpu-nvidia",
    ],
    deps = [
        "//tensorflow/lite/delegates/gpu/cl/kernels:cl_test",
        "//tensorflow/lite/delegates/gpu/common:gpu_model_test_util",
        "//tensorflow/lite/delegates/gpu/common:status",
        "@com_google_googletest//:gtest_main_no_heapcheck",  # constant buffers leak on nvidia
    ],
)

cc_binary(
    name = "internal_api_samples",
    srcs = ["internal_api_samples.cc"],
    tags = [
        "nobuilder",
        "notap",
    ],
    deps = [
        "//tensorflow/lite/delegates/gpu:api",
        "//tensorflow/lite/delegates/gpu/cl:api",
        "//tensorflow/lite/delegates/gpu/cl:environment",
        "//tensorflow/lite/delegates/gpu/cl:inference_context",
        "//tensorflow/lite/delegates/gpu/common:model",
        "//tensorflow/lite/delegates/gpu/common:model_builder",
        "//tensorflow/lite/delegates/gpu/common:status",
        "//tensorflow/lite/kernels:builtin_ops",
        "//tensorflow/lite/kernels:kernel_util",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:span",
    ],
)

cc_binary(
    name = "memory_sharing_sample",
    srcs = ["memory_sharing_sample.cc"],
    deps = [
        "//tensorflow/lite/core/kernels:builtin_ops",
        "//tensorflow/lite/delegates/gpu/cl:environment",
        "//tensorflow/lite/delegates/gpu/cl:inference_context",
        "//tensorflow/lite/delegates/gpu/common:model",
        "//tensorflow/lite/delegates/gpu/common:model_builder",
        "//tensorflow/lite/delegates/gpu/common:status",
        "@com_google_absl//absl/time",
    ],
)

cc_binary(
    name = "performance_profiling",
    srcs = ["performance_profiling.cc"],
    deps = [
        "//tensorflow/lite/core/kernels:builtin_ops",
        "//tensorflow/lite/delegates/gpu/cl:environment",
        "//tensorflow/lite/delegates/gpu/cl:inference_context",
        "//tensorflow/lite/delegates/gpu/common:model",
        "//tensorflow/lite/delegates/gpu/common:model_builder",
        "//tensorflow/lite/delegates/gpu/common:status",
        "@com_google_absl//absl/time",
    ],
)
