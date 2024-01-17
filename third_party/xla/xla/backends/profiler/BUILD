load("@local_tsl//tsl:tsl.bzl", "if_with_tpu_support", "tsl_gpu_library")

# copybara:uncomment package(default_applicable_licenses = ["//tensorflow:license"])

package_group(
    name = "friends",
    packages = [
        "//xla/backends/profiler/cpu",
        "//xla/backends/profiler/gpu",
        "//xla/backends/profiler/tpu",
    ],
)

tsl_gpu_library(
    name = "profiler_backends",
    visibility = ["//visibility:public"],
    deps = [
        "//xla/backends/profiler/cpu:host_tracer",
        "//xla/backends/profiler/cpu:metadata_collector",
    ] + if_with_tpu_support(
        [
            "//xla/backends/profiler/tpu:tpu_tracer",
        ],
    ),
    alwayslink = True,
)
