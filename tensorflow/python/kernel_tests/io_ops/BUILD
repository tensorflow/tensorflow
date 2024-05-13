# Tests of TensorFlow IO kernels written using the Python API.

load("//tensorflow:tensorflow.default.bzl", "tf_py_strict_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    licenses = ["notice"],
)

tf_py_strict_test(
    name = "checkpoint_ops_test",
    size = "medium",
    srcs = ["checkpoint_ops_test.py"],
    deps = [
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:errors",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/ops:checkpoint_ops_gen",
        "//tensorflow/python/ops:partitioned_variables",
        "//tensorflow/python/ops:variable_scope",
        "//tensorflow/python/ops:variables",
        "//tensorflow/python/platform:client_testlib",
        "//tensorflow/python/platform:flags",
        "//tensorflow/python/training:saver",
        "//third_party/py/numpy",
    ],
)

tf_py_strict_test(
    name = "decode_csv_op_test",
    size = "small",
    srcs = ["decode_csv_op_test.py"],
    deps = [
        "//tensorflow/python/eager:context",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:errors",
        "//tensorflow/python/ops:parsing_ops",
        "//tensorflow/python/platform:client_testlib",
        "//third_party/py/numpy",
    ],
)

tf_py_strict_test(
    name = "io_ops_test",
    size = "small",
    srcs = ["io_ops_test.py"],
    deps = [
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/ops:io_ops",
        "//tensorflow/python/platform:client_testlib",
        "//tensorflow/python/util:compat",
    ],
)

tf_py_strict_test(
    name = "parse_single_example_op_test",
    size = "small",
    srcs = ["parse_single_example_op_test.py"],
    deps = [
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python/framework:errors",
        "//tensorflow/python/framework:for_generated_wrappers",
        "//tensorflow/python/framework:sparse_tensor",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/ops:parsing_ops",
        "//tensorflow/python/platform:client_testlib",
        "//tensorflow/python/platform:tf_logging",
        "//third_party/py/numpy",
    ],
)

tf_py_strict_test(
    name = "parsing_ops_test",
    size = "medium",
    srcs = ["parsing_ops_test.py"],
    deps = [
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:errors",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:sparse_tensor",
        "//tensorflow/python/framework:tensor",
        "//tensorflow/python/framework:tensor_shape",
        "//tensorflow/python/framework:tensor_util",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:parsing_ops",
        "//tensorflow/python/ops/ragged:ragged_concat_ops",
        "//tensorflow/python/ops/ragged:ragged_factory_ops",
        "//tensorflow/python/ops/ragged:ragged_tensor",
        "//tensorflow/python/platform:client_testlib",
        "//tensorflow/python/platform:tf_logging",
        "//third_party/py/numpy",
    ],
)

tf_py_strict_test(
    name = "reader_ops_test",
    size = "small",
    srcs = ["reader_ops_test.py"],
    deps = [
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python/framework:errors",
        "//tensorflow/python/framework:for_generated_wrappers",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/lib/io:tf_record",
        "//tensorflow/python/ops:data_flow_ops",
        "//tensorflow/python/ops:io_ops",
        "//tensorflow/python/ops:variables",
        "//tensorflow/python/platform:client_testlib",
        "//tensorflow/python/util:compat",
    ],
)

tf_py_strict_test(
    name = "record_input_test",
    size = "medium",
    srcs = ["record_input_test.py"],
    deps = [
        "//tensorflow/python/framework:errors",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/lib/io:tf_record",
        "//tensorflow/python/ops:data_flow_ops",
        "//tensorflow/python/ops:variables",
        "//tensorflow/python/platform:client_testlib",
    ],
)

tf_py_strict_test(
    name = "save_restore_ops_test",
    size = "small",
    srcs = ["save_restore_ops_test.py"],
    deps = [
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python/client:session",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/ops:io_ops",
        "//tensorflow/python/ops:io_ops_gen",
        "//tensorflow/python/platform:client_testlib",
        "@absl_py//absl/testing:parameterized",
    ],
)
