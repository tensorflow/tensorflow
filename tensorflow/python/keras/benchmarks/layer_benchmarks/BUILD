# Description:
#   Implementation of benchmarks on Keras layers.

load("//tensorflow:tensorflow.bzl", "tf_py_test")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

filegroup(
    name = "all_py_srcs",
    srcs = glob(["*.py"]),
    visibility = ["//tensorflow/python/keras/google/private_tf_api_test:__pkg__"],
)

BECHMARK_TAGS = [
    "no_oss_py38",  # TODO(b/162044699)
    "no_pip",  # TODO(b/161253163)
    "no_windows",  # TODO(b/160628318)
]

# To run CPU benchmarks:
#   bazel run -c opt benchmarks_test -- --benchmarks=.

# To run GPU benchmarks:
#   bazel run -c opt --config=cuda benchmarks_test -- \
#     --benchmarks=.

# To run benchmarks with TFRT:
#   bazel run -c opt --config=cuda --test_env=EXPERIMENTAL_ENABLE_TFRT=1 benchmarks_test -- \
#     --benchmarks=.

# To run a subset of benchmarks using --benchmarks flag.
# --benchmarks: the list of benchmarks to run. The specified value is interpreted
# as a regular expression and any benchmark whose name contains a partial match
# to the regular expression is executed.
# e.g. --benchmarks=".*lstm*." will run all lstm layer related benchmarks.

py_library(
    name = "run_xprof",
    srcs = ["run_xprof.py"],
    srcs_version = "PY3",
    visibility = ["//tensorflow/python/keras:__subpackages__"],
)

py_library(
    name = "layer_benchmarks_test_base",
    srcs = ["layer_benchmarks_test_base.py"],
    srcs_version = "PY3",
    visibility = ["//tensorflow/python/keras:__subpackages__"],
    deps = [
        ":run_xprof",
        "//tensorflow:tensorflow_py_no_contrib",
        "//tensorflow/python/keras/benchmarks:profiler_lib",
    ],
)

tf_py_test(
    name = "layer_benchmarks_test",
    srcs = ["layer_benchmarks_test.py"],
    python_version = "PY3",
    tags = BECHMARK_TAGS,
    deps = [
        ":layer_benchmarks_test_base",
        "//tensorflow:tensorflow_py_no_contrib",
        "//tensorflow/python/keras/benchmarks:benchmark_util",
    ],
)
