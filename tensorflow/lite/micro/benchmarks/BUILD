licenses(["notice"])  # Apache 2.0

cc_library(
    name = "micro_benchmark",
    hdrs = [
        "micro_benchmark.h",
    ],
    deps = [
        "//tensorflow/lite/micro:micro_error_reporter",
        "//tensorflow/lite/micro:micro_framework",
        "//tensorflow/lite/micro:micro_time",
        "//tensorflow/lite/micro:op_resolvers",
    ],
)

cc_binary(
    name = "conv_benchmark",
    srcs = [
        "conv_benchmark.cc",
    ],
    deps = [
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/micro:micro_time",
        "//tensorflow/lite/micro/kernels:micro_ops",
        "//tensorflow/lite/micro/testing:micro_test",
    ],
)

cc_binary(
    name = "depthwise_conv_benchmark",
    srcs = [
        "depthwise_conv_benchmark.cc",
    ],
    deps = [
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/micro:micro_time",
        "//tensorflow/lite/micro/kernels:micro_ops",
        "//tensorflow/lite/micro/testing:micro_test",
    ],
)

cc_library(
    name = "keyword_scrambled_model_data",
    srcs = [
        "keyword_scrambled_model_data.cc",
    ],
    hdrs = [
        "keyword_scrambled_model_data.h",
    ],
)

cc_binary(
    name = "keyword_benchmark",
    srcs = ["keyword_benchmark.cc"],
    deps = [
        ":keyword_scrambled_model_data",
        ":micro_benchmark",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/micro:micro_error_reporter",
        "//tensorflow/lite/micro:micro_framework",
        "//tensorflow/lite/micro:op_resolvers",
        "//tensorflow/lite/micro/kernels:micro_ops",
    ],
)

cc_binary(
    name = "person_detection_benchmark",
    srcs = ["person_detection_benchmark.cc"],
    deps = [
        ":micro_benchmark",
        "//tensorflow/lite:schema_fbs_version",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/micro:micro_error_reporter",
        "//tensorflow/lite/micro:micro_framework",
        "//tensorflow/lite/micro:micro_utils",
        "//tensorflow/lite/micro:op_resolvers",
        "//tensorflow/lite/micro/examples/person_detection:model_settings",
        "//tensorflow/lite/micro/examples/person_detection:person_detect_model_data",
        "//tensorflow/lite/micro/examples/person_detection:simple_images_test_data",
        "//tensorflow/lite/micro/kernels:micro_ops",
        "//tensorflow/lite/schema:schema_fbs",
    ],
)
