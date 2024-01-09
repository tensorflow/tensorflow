# copybara:uncomment package(default_applicable_licenses = ["//tensorflow:license"])

package_group(
    name = "friends",
    includes = ["//tsl:internal"],
)

package_group(
    name = "internal",
    packages = [
        "//tensorflow/core/profiler/...",
        "//tensorflow/python/eager/...",
        "//tensorflow/python/profiler/...",
        "//tensorflow/python/tpu/profiler/...",
        "//tsl/profiler/...",
        "//xla/backends/profiler/...",
    ],
)

package_group(
    name = "xla_profiler_backends",
    packages = ["//xla/backends/profiler/..."],
)

package_group(
    name = "xla_internal",
    packages = [
        "//xla/...",
    ],
)
