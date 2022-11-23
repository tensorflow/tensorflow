load("//tensorflow:tensorflow.bzl", "tf_cuda_library")
load("//tensorflow/tsl:tsl.bzl", "if_libtpu")

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
    deps = [
        "//tensorflow/compiler/xla/backends/profiler/cpu:host_tracer",
        "//tensorflow/compiler/xla/backends/profiler/cpu:metadata_collector",
    ] + if_libtpu([
        "//tensorflow/compiler/xla/backends/profiler/tpu:tpu_tracer",
    ]),
    alwayslink = True,
)
