package(default_visibility = [
    "//visibility:private",
])

licenses(["notice"])  # Apache 2.0

py_library(
    name = "ops",
    srcs = [
        "ops.py",
    ],
    deps = [
        "//tensorflow:tensorflow_py",
    ],
)

py_test(
    name = "ops_test",
    srcs = ["ops_test.py"],
    tags = [
        # The test fails with CUDA on CPU. A lot of other tensorflow tests
        # also use this flag. Unsure if this is a real problem or false
        # alarm.
        # TODO(ycling): Investigate if this is a real problem before open
        # sourcing.
        "no_cuda_on_cpu_tap",
    ],
    deps = [
        ":ops",
        "//tensorflow/contrib/lite/experimental/pb2lite/python:converter_wrapper",
    ],
)
