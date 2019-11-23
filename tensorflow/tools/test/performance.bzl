load("//tensorflow:tensorflow.bzl", "tf_py_test")

# Create a benchmark test target of a TensorFlow C++ test (tf_cc_*_test)
def tf_cc_logged_benchmark(
        name = None,
        target = None,
        benchmarks = "..",
        tags = [],
        test_log_output_prefix = "",
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

    tf_py_test(
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
        main = "run_and_gather_logs.py",
        additional_deps = [
            "//tensorflow/tools/test:run_and_gather_logs",
        ],
        **kwargs
    )

# Create a benchmark test target of a TensorFlow python test (*py_tests)
def tf_py_logged_benchmark(
        name = None,
        target = None,
        benchmarks = "..",
        tags = [],
        test_log_output_prefix = "",
        **kwargs):
    # For now generating a py benchmark is the same as generating a C++
    # benchmark target. In the future this may change, so we have
    # two macros just in case
    tf_cc_logged_benchmark(
        name = name,
        target = target,
        benchmarks = benchmarks,
        tags = tags,
        test_log_output_prefix = test_log_output_prefix,
        benchmark_type = "python_benchmark",
        **kwargs
    )
