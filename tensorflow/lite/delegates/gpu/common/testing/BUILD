package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],  # Apache 2.0
)

cc_library(
    name = "interpreter_utils",
    testonly = 1,
    srcs = ["interpreter_utils.cc"],
    hdrs = ["interpreter_utils.h"],
    deps = [
        "//tensorflow/lite:framework",
        "//tensorflow/lite/core/api",
        "//tensorflow/lite/delegates/gpu/common:status",
        "//tensorflow/lite/delegates/gpu/common:tensor",
        "//tensorflow/lite/kernels:builtin_ops",
        "//tensorflow/lite/schema:schema_fbs",
        "@com_google_absl//absl/memory",
    ],
)

cc_library(
    name = "tflite_model_reader",
    srcs = ["tflite_model_reader.cc"],
    hdrs = ["tflite_model_reader.h"],
    deps = [
        "//tensorflow/lite:framework_lib",
        "//tensorflow/lite:kernel_api",
        "//tensorflow/lite:minimal_logging",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/delegates/gpu/cl:api",
        "//tensorflow/lite/delegates/gpu/cl:opencl_wrapper",
        "//tensorflow/lite/delegates/gpu/cl:tensor_type_util",
        "//tensorflow/lite/delegates/gpu/common:model",
        "//tensorflow/lite/delegates/gpu/common:model_builder",
        "//tensorflow/lite/delegates/gpu/common:model_transformer",
        "//tensorflow/lite/delegates/gpu/common:status",
        "//tensorflow/lite/delegates/gpu/common/transformations:general_transformations",
        "//tensorflow/lite/delegates/gpu/gl:api2",
        "//tensorflow/lite/kernels:builtin_ops",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
)
