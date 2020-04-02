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
        "//tensorflow/lite/micro/kernels:micro_ops",
        "//tensorflow/lite/micro/kernels:micro_utils",
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
        "//tensorflow/lite/micro/kernels:micro_ops",
        "//tensorflow/lite/micro/kernels:micro_utils",
        "//tensorflow/lite/micro/testing:micro_test",
    ],
)
