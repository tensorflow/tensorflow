load("//tensorflow:tensorflow.default.bzl", "cuda_py_strict_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//tensorflow:internal"],
    licenses = ["notice"],
)

cuda_py_strict_test(
    name = "kpi_benchmark_test",
    size = "medium",
    srcs = ["kpi_benchmark_test.py"],
    python_version = "PY3",
    tags = [
        "no_windows",  #  b/141617449
        "optonly",
    ],
    deps = [
        "//tensorflow:tensorflow_py_no_contrib",
        "//tensorflow/python/eager:benchmarks_test_base",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/profiler:trace",
    ],
)
