# Description:
#   Implementation of Keras benchmarks.

load("//tensorflow:tensorflow.bzl", "cuda_py_test")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

filegroup(
    name = "all_py_srcs",
    srcs = glob(["*.py"]),
    visibility = ["//tensorflow/python/keras/google/private_tf_api_test:__pkg__"],
)

# To run CPU benchmarks:
#   bazel run -c opt benchmarks_test -- --benchmarks=.

# To run GPU benchmarks:
#   bazel run --config=cuda -c opt --copt="-mavx" benchmarks_test -- \
#     --benchmarks=.

# To run a subset of benchmarks using --benchmarks flag.
# --benchmarks: the list of benchmarks to run. The specified value is interpreted
# as a regular expression and any benchmark whose name contains a partial match
# to the regular expression is executed.
# e.g. --benchmarks=".*lstm*." will run all lstm layer related benchmarks.

# Add all benchmarks related utils here for pip testing dependencis.
py_library(
    name = "keras_benchmark_lib_pip",
    srcs_version = "PY3",
    deps = [
        ":benchmark_util",
        ":distribution_util",
        "//tensorflow/python/keras/benchmarks/saved_model_benchmarks:saved_model_benchmark_util",
    ],
)

# This lib is mainly for running benchmarks on mlcompass infra.
py_library(
    name = "profiler_lib",
    srcs_version = "PY3",
    visibility = [
        "//learning/brain/contrib/keras_benchmark:__subpackages__",
        "//tensorflow/python/keras:__subpackages__",
    ],
)

COMMON_TAGS = [
    "no_pip",  # b/161253163
    "no_windows",  # b/160628318
]

py_test(
    name = "keras_cpu_benchmark_test",
    size = "large",
    srcs = ["keras_cpu_benchmark_test.py"],
    python_version = "PY3",
    tags = COMMON_TAGS,
    deps = [
        ":benchmark_util",
        ":profiler_lib",
        "//tensorflow:tensorflow_py_no_contrib",
        "//third_party/py/numpy",
    ],
)

cuda_py_test(
    name = "eager_microbenchmarks_test",
    size = "medium",
    srcs = ["eager_microbenchmarks_test.py"],
    python_version = "PY3",
    tags = COMMON_TAGS + [
        "no_oss_py38",  # TODO(b/162044699)
    ],
    deps = [
        ":profiler_lib",
        "//tensorflow:tensorflow_py_no_contrib",
        "//tensorflow/python/keras/utils:tf_inspect",
    ],
)

cuda_py_test(
    name = "model_components_benchmarks_test",
    srcs = ["model_components_benchmarks_test.py"],
    python_version = "PY3",
    deps = [
        ":profiler_lib",
        "//tensorflow:tensorflow_py_no_contrib",
    ],
)

py_library(
    name = "benchmark_util",
    srcs = ["benchmark_util.py"],
    srcs_version = "PY3",
    deps = [
        ":distribution_util",
        "//tensorflow:tensorflow_py_no_contrib",
        "//third_party/py/numpy",
    ],
)

py_test(
    name = "benchmark_util_test",
    srcs = ["benchmark_util_test.py"],
    python_version = "PY3",
    tags = COMMON_TAGS,
    deps = [
        ":benchmark_util",
        "//tensorflow:tensorflow_py_no_contrib",
    ],
)

py_library(
    name = "distribution_util",
    srcs = ["distribution_util.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow:tensorflow_py_no_contrib",
    ],
)

py_test(
    name = "optimizer_benchmarks_test",
    srcs = ["optimizer_benchmarks_test.py"],
    python_version = "PY3",
    tags = COMMON_TAGS + [
        "no_oss_py38",  # TODO(b/162044699)
    ],
    deps = [
        ":benchmark_util",
        ":profiler_lib",
        "//tensorflow:tensorflow_py_no_contrib",
        "//tensorflow/python/keras/optimizer_v2",
    ],
)

# Run memory profiler on Keras model.
# Please make sure `memory_profiler` is installed.
# To run the memory profiler:
# With CPU:
#   bazel run -c opt model_memory_profile -- --model=YOUR_MODEL_NAME
# With GPU:
#   bazel run -c opt --config=cuda model_memory_profile -- --model=YOUR_MODEL_NAME
py_binary(
    name = "model_memory_profile",
    srcs = ["model_memory_profile.py"],
    python_version = "PY3",
    tags = ["no_oss"],
    deps = ["//tensorflow:tensorflow_py_no_contrib"],
)

py_test(
    name = "metrics_memory_benchmark_test",
    srcs = ["metrics_memory_benchmark_test.py"],
    python_version = "PY3",
    tags = COMMON_TAGS,
    deps = [
        "//tensorflow:tensorflow_py_no_contrib",
        "//third_party/py/numpy",
    ],
)
