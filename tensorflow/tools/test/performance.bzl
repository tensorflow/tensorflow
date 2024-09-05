"""
Benchmark-related macros.
"""

load(
    "//tensorflow:tensorflow.default.bzl",
    "cuda_py_strict_test",
    "tf_py_strict_test",
)

# Create a benchmark test target of a TensorFlow C++ test (tf_cc_*_test)
def tf_cc_logged_benchmark(
        name = None,
        target = None,
        benchmarks = "..",
        tags = [],
        benchmark_type = "cpp_microbenchmark",
        **kwargs):
    if not name:
        fail("Must provide a name")
    if not target:
        fail("Must provide a target")
    if (not ":" in target or
        not target.startswith("//") or
        target.endswith(":all") or
        target.endswith(".")):
        fail(" ".join((
            "Target must be a single well-defined test, e.g.,",
            "//path/to:test. Received: %s" % target,
        )))

    all_tags = tags + ["benchmark-test", "local", "manual", "regression-test"]

    tf_py_strict_test(
        name = name,
        tags = all_tags,
        size = "large",
        srcs = ["//tensorflow/tools/test:run_and_gather_logs"],
        args = [
            "--name=//%s:%s" % (native.package_name(), name),
            "--test_name=" + target,
            "--test_args=--benchmarks=%s" % benchmarks,
            "--benchmark_type=%s" % benchmark_type,
        ],
        data = [
            target,
        ],
        main = "//tensorflow/tools/test:run_and_gather_logs.py",
        deps = [
            "@absl_py//absl:app",
            "@absl_py//absl/flags",
            "@org_tensorflow//tensorflow/core:protos_all_py",
            "//tensorflow/python/platform:gfile",
            "//tensorflow/python/platform:test",
            "//tensorflow/python/platform:tf_logging",
            "//tensorflow/tools/test:run_and_gather_logs_main_lib",
        ],
        **kwargs
    )

def add_benchmark_tag_to_kwargs(kwargs):
    """Adds the `benchmark-test` tag to the kwargs, if not already present.

    Notes:
      For benchmarks which are not technically tests, but whose class methods
      can still be discovered, and run as such via `bazel run`.
    Args:
      kwargs: kwargs to be passed to a test wrapper/rule further down.
    Returns:
      kwargs: kwargs with the tags including the `benchmark-test` tags.
    """
    benchmark_tag = "benchmark-test"
    if "tags" in kwargs and kwargs["tags"] != None:
        if benchmark_tag not in kwargs["tags"]:
            kwargs["tags"].append(benchmark_tag)
    else:
        kwargs["tags"] = [benchmark_tag]
    return kwargs

def tf_py_benchmark_test(**kwargs):
    kwargs = add_benchmark_tag_to_kwargs(kwargs)
    tf_py_strict_test(**kwargs)

def cuda_py_benchmark_test(**kwargs):
    kwargs = add_benchmark_tag_to_kwargs(kwargs)
    cuda_py_strict_test(**kwargs)

# Create a benchmark test target of a TensorFlow python test (*py_tests)
def tf_py_logged_benchmark(
        name = None,
        target = None,
        benchmarks = "..",
        tags = [],
        **kwargs):
    # For now generating a py benchmark is the same as generating a C++
    # benchmark target. In the future this may change, so we have
    # two macros just in case
    tf_cc_logged_benchmark(
        name = name,
        target = target,
        benchmarks = benchmarks,
        tags = tags,
        benchmark_type = "python_benchmark",
        **kwargs
    )
