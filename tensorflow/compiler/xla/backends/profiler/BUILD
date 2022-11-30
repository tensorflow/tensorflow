load("//tensorflow:tensorflow.bzl", "if_libtpu", "tf_cuda_library")

# copybara:uncomment package(default_applicable_licenses = ["//tensorflow:license"])

package_group(
    name = "friends",
    packages = [
        "//tensorflow/compiler/xla/backends/profiler/cpu",
        "//tensorflow/compiler/xla/backends/profiler/gpu",
        "//tensorflow/compiler/xla/backends/profiler/tpu",
    ],
)

tf_cuda_library(
    name = "profiler_backends",
    cuda_deps = [
        "//tensorflow/compiler/xla/backends/profiler/gpu:device_tracer",
    ],
    visibility = ["//tensorflow:internal"],
    deps = if_libtpu([
        "//tensorflow/compiler/xla/backends/profiler/tpu:tpu_tracer",
    ]),
    alwayslink = True,
)
