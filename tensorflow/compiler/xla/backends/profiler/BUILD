load("//tensorflow:tensorflow.bzl", "if_libtpu", "tf_cuda_library")

package_group(
    name = "friends",
    packages = [
        "//tensorflow/compiler/xla/backends/profiler/cpu",
        "//tensorflow/compiler/xla/backends/profiler/tpu",
    ],
)

tf_cuda_library(
    name = "profiler_backends",
    visibility = ["//tensorflow:internal"],
    deps = if_libtpu([
        "//tensorflow/compiler/xla/backends/profiler/tpu:tpu_tracer",
    ]),
    alwayslink = True,
)
