package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],  # Apache 2.0
)

cc_binary(
    name = "performance_profiling",
    srcs = ["performance_profiling.cc"],
    deps = [
        "//tensorflow/lite/core/api",
        "//tensorflow/lite/delegates/gpu/cl:cl_command_queue",
        "//tensorflow/lite/delegates/gpu/cl:environment",
        "//tensorflow/lite/delegates/gpu/cl:inference_context",
        "//tensorflow/lite/delegates/gpu/cl:model_hints",
        "//tensorflow/lite/delegates/gpu/cl:opencl_wrapper",
        "//tensorflow/lite/delegates/gpu/cl:precision",
        "//tensorflow/lite/delegates/gpu/cl:tensor_type",
        "//tensorflow/lite/delegates/gpu/common:model",
        "//tensorflow/lite/delegates/gpu/common:model_builder",
        "//tensorflow/lite/delegates/gpu/common:model_transformer",
        "//tensorflow/lite/delegates/gpu/common:status",
        "//tensorflow/lite/delegates/gpu/common/transformations:general_transformations",
        "//tensorflow/lite/delegates/gpu/common/transformations:merge_padding_with",
        "//tensorflow/lite/kernels:builtin_ops",
        "@com_google_absl//absl/time",
    ],
)
