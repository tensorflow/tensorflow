# Benchmarks for the ambient music detection model.

package(
    licenses = ["notice"],  # Apache 2.0
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
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/micro:micro_framework",
        "//tensorflow/lite/micro/kernels:micro_ops",
        "//tensorflow/lite/micro/testing:micro_benchmark",
    ],
)
