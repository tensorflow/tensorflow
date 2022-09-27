# Tests of TensorFlow image kernels written using the Python API.

load("//tensorflow:tensorflow.bzl", "cuda_py_test")

# buildifier: disable=same-origin-load
load("//tensorflow:tensorflow.bzl", "tf_py_test")

package(licenses = ["notice"])

tf_py_test(
    name = "attention_ops_test",
    size = "small",
    srcs = ["attention_ops_test.py"],
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:image_ops",
        "//third_party/py/numpy",
    ],
)

tf_py_test(
    name = "decode_bmp_op_test",
    size = "small",
    srcs = ["decode_bmp_op_test.py"],
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:image_ops",
        "//tensorflow/python:nn_grad",
    ],
)

tf_py_test(
    name = "decode_compressed_op_test",
    size = "small",
    srcs = ["decode_compressed_op_test.py"],
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:parsing_ops",
        "//third_party/py/numpy",
    ],
)

tf_py_test(
    name = "decode_image_op_test",
    size = "small",
    srcs = ["decode_image_op_test.py"],
    data = ["//tensorflow/core:image_testdata"],
    deps = [
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:errors",
        "//tensorflow/python:image_ops",
        "//tensorflow/python:io_ops",
        "//tensorflow/python:nn_grad",
        "//third_party/py/numpy",
    ],
)

tf_py_test(
    name = "decode_jpeg_op_test",
    srcs = ["decode_jpeg_op_test.py"],
    data = ["//tensorflow/core:image_testdata"],
    deps = [
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:image_ops",
        "//tensorflow/python:platform",
    ],
)

tf_py_test(
    name = "decode_png_op_test",
    size = "small",
    srcs = ["decode_png_op_test.py"],
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:image_ops",
        "//tensorflow/python:nn_grad",
    ],
)

tf_py_test(
    name = "decode_raw_op_test",
    size = "small",
    srcs = ["decode_raw_op_test.py"],
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:parsing_ops",
        "//third_party/py/numpy",
    ],
)

tf_py_test(
    name = "draw_bounding_box_op_test",
    size = "small",
    srcs = ["draw_bounding_box_op_test.py"],
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:image_ops",
        "//tensorflow/python:math_ops",
        "//third_party/py/numpy",
    ],
)

# TODO(gpapan): Revisit the gradient of extract_image_patches_op to resolve
# http://b/31080670.
cuda_py_test(
    name = "extract_image_patches_grad_test",
    size = "medium",
    srcs = ["extract_image_patches_grad_test.py"],
    shard_count = 15,
    tags = [
        "no_oss",  # b/241024908
        "no_rocm",
        "nomac",  # b/181799478
        "notap",  # b/31080670
    ],
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//third_party/py/numpy",
    ],
)

cuda_py_test(
    name = "extract_image_patches_op_test",
    size = "small",
    srcs = ["extract_image_patches_op_test.py"],
    # TODO(b/144432983): S32 convolutions should not be auto-clustered.
    xla_enable_strict_auto_jit = False,
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//third_party/py/numpy",
    ],
)

cuda_py_test(
    name = "extract_volume_patches_grad_test",
    size = "medium",
    srcs = ["extract_volume_patches_grad_test.py"],
    shard_count = 50,
    tags = [
        "no_gpu",  # b/171837334
        "no_oss",  # Test times out on oss-nightly cpu builds
        "no_pip",
        "nomac",  # b/139946976
        "notap",  # b/31080670
    ],
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//third_party/py/numpy",
    ],
)

cuda_py_test(
    name = "extract_volume_patches_op_test",
    size = "small",
    srcs = ["extract_volume_patches_op_test.py"],
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//third_party/py/numpy",
    ],
)
