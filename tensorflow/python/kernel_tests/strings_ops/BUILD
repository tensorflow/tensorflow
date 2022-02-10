# Tests of TensorFlow math kernels written using the Python API.

load("//tensorflow:tensorflow.bzl", "cuda_py_test")

# buildifier: disable=same-origin-load
load("//tensorflow:tensorflow.bzl", "tf_py_test")

package(
    default_visibility = ["//tensorflow:internal"],
    licenses = ["notice"],
)

tf_py_test(
    name = "as_string_op_test",
    size = "small",
    srcs = ["as_string_op_test.py"],
    tags = ["no_windows"],
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:string_ops",
        "//third_party/py/numpy",
    ],
)

tf_py_test(
    name = "base64_ops_test",
    size = "small",
    srcs = ["base64_ops_test.py"],
    tags = ["nomac"],  # b/35468214
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:string_ops",
        "//third_party/py/numpy",
    ],
)

cuda_py_test(
    name = "reduce_join_op_test",
    size = "small",
    srcs = ["reduce_join_op_test.py"],
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:string_ops",
        "//third_party/py/numpy",
    ],
)

tf_py_test(
    name = "regex_full_match_op_test",
    size = "small",
    srcs = ["regex_full_match_op_test.py"],
    deps = [
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:string_ops",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_py_test(
    name = "regex_replace_op_test",
    size = "small",
    srcs = ["regex_replace_op_test.py"],
    deps = [
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:string_ops",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_py_test(
    name = "string_bytes_split_op_test",
    size = "small",
    srcs = ["string_bytes_split_op_test.py"],
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:string_ops",
        "//tensorflow/python/ops/ragged",
        "//tensorflow/python/ops/ragged:ragged_factory_ops",
        "//tensorflow/python/ops/ragged:ragged_string_ops",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_py_test(
    name = "string_format_op_test",
    size = "small",
    srcs = ["string_format_op_test.py"],
    deps = [
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:string_ops",
    ],
)

tf_py_test(
    name = "string_join_op_test",
    size = "small",
    srcs = ["string_join_op_test.py"],
    deps = [
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:string_ops",
    ],
)

tf_py_test(
    name = "string_length_op_test",
    size = "small",
    srcs = ["string_length_op_test.py"],
    deps = [
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:string_ops",
    ],
)

tf_py_test(
    name = "string_lower_op_test",
    size = "small",
    srcs = ["string_lower_op_test.py"],
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:string_ops",
        "//third_party/py/numpy",
    ],
)

tf_py_test(
    name = "string_split_op_test",
    size = "small",
    srcs = ["string_split_op_test.py"],
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:string_ops",
        "//tensorflow/python:util",
        "//tensorflow/python/ops/ragged:ragged_factory_ops",
        "//tensorflow/python/ops/ragged:ragged_string_ops",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_py_test(
    name = "string_strip_op_test",
    size = "small",
    srcs = ["string_strip_op_test.py"],
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:string_ops",
        "//third_party/py/numpy",
    ],
)

cuda_py_test(
    name = "string_to_hash_bucket_op_test",
    size = "small",
    srcs = ["string_to_hash_bucket_op_test.py"],
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:string_ops",
    ],
)

cuda_py_test(
    name = "string_to_number_op_test",
    size = "small",
    srcs = ["string_to_number_op_test.py"],
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:parsing_ops",
    ],
)

tf_py_test(
    name = "string_upper_op_test",
    size = "small",
    srcs = ["string_upper_op_test.py"],
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:string_ops",
        "//third_party/py/numpy",
    ],
)

tf_py_test(
    name = "substr_op_test",
    size = "small",
    srcs = ["substr_op_test.py"],
    deps = [
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:errors",
        "//tensorflow/python:string_ops",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_py_test(
    name = "unicode_decode_op_test",
    size = "small",
    srcs = ["unicode_decode_op_test.py"],
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:sparse_tensor",
        "//tensorflow/python:string_ops",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/ops/ragged",
        "//tensorflow/python/ops/ragged:ragged_factory_ops",
        "//tensorflow/python/ops/ragged:ragged_string_ops",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_py_test(
    name = "unicode_encode_op_test",
    size = "small",
    srcs = ["unicode_encode_op_test.py"],
    deps = [
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python/ops/ragged:ragged_factory_ops",
        "//tensorflow/python/ops/ragged:ragged_string_ops",
        "//tensorflow/python/ops/ragged:ragged_tensor",
        "//tensorflow/python/ops/ragged:ragged_tensor_value",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_py_test(
    name = "unicode_script_op_test",
    size = "small",
    srcs = ["unicode_script_op_test.py"],
    deps = [
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:string_ops",
    ],
)

tf_py_test(
    name = "unicode_transcode_op_test",
    size = "small",
    srcs = ["unicode_transcode_op_test.py"],
    deps = [
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:string_ops",
        "@absl_py//absl/testing:parameterized",
    ],
)

cuda_py_test(
    name = "unsorted_segment_join_op_test",
    size = "small",
    srcs = ["unsorted_segment_join_op_test.py"],
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:string_ops",
        "//third_party/py/numpy",
    ],
)
