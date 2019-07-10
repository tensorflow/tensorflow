load("//tensorflow:tensorflow.bzl", "py_test")
load("//tensorflow/tools/test:performance.bzl", "tf_py_logged_benchmark")

package(
    licenses = ["notice"],  # Apache 2.0
)

py_library(
    name = "benchmark_base",
    srcs = [
        "benchmark_base.py",
    ],
    deps = [
        "//tensorflow:tensorflow_py",
    ],
)

py_test(
    name = "cartpole_benchmark",
    size = "enormous",
    srcs = ["cartpole_benchmark.py"],
    python_version = "PY2",
    tags = [
        "local",
        "manual",
        "no_oss",
        "notap",
        "nozapfhahn",
    ],
    deps = [
        ":benchmark_base",
        # Note: required gym dependency may need to be added here.
    ],
)

tf_py_logged_benchmark(
    name = "cartpole_logged_benchmark",
    target = "//tensorflow/contrib/autograph/examples/benchmarks:cartpole_benchmark",
)
