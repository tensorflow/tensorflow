# Build simple_hash_table custom ops example, which is similar to,
# but simpler than, tf.lookup.experimental.MutableHashTable

load("//tensorflow:tensorflow.bzl", "py_strict_library")
load("//tensorflow:tensorflow.bzl", "tf_custom_op_library")
load("//tensorflow:tensorflow.bzl", "tf_py_test")

licenses(["notice"])

tf_custom_op_library(
    name = "simple_hash_table_kernel.so",
    srcs = [
        "simple_hash_table_kernel.cc",
        "simple_hash_table_op.cc",
    ],
    deps = [
        "//tensorflow/core/lib/gtl:map_util",
        "//tensorflow/core/platform:strcat",
        "@com_google_absl//absl/container:flat_hash_map",
    ],
)

py_strict_library(
    name = "simple_hash_table_op",
    srcs = ["simple_hash_table_op.py"],
    data = ["simple_hash_table_kernel.so"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow:tensorflow_py",
    ],
)

py_strict_library(
    name = "simple_hash_table",
    srcs = ["simple_hash_table.py"],
    srcs_version = "PY3",
    deps = [
        ":simple_hash_table_op",
        "//tensorflow:tensorflow_py",
    ],
)

tf_py_test(
    name = "simple_hash_table_test",
    size = "small",
    srcs = ["simple_hash_table_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    tags = [
        "no_mac",  # TODO(b/216321151): Re-enable this test.
        "no_pip",
    ],
    deps = [
        ":simple_hash_table",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python/framework:errors",
        "//tensorflow/python/framework:test_lib",
        "//third_party/py/numpy",
    ],
)
