# Description:
#   Contains ops for tensor2tensor acceleration.

# This should do 3 things:
# * Compile and link ops/t2t_ops.cc and kernels/* 
# * Generate a python wrapper gen_t2t_ops.py out of ops/t2t_ops.cc [this exposes the op]
# * Copy python/ops/t2t_ops.py into release [this registers the gradient of the op]
# Unfortunately, the Bazel interface I have to use is essentially undocumented, 
# and I couldn't find a way to do this in fewer than 5 calls
# Ideally all C++/CUDA code should be statically linked into libtensorflow_framework.so, 
# but I don't see a way to do that either. Instead it all goes into a separate _t2t_ops.so.

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

package(default_visibility = ["//visibility:public"])

load(
    "//tensorflow:tensorflow.bzl",
    "tf_cc_test",
    "tf_custom_op_library",
    "tf_custom_op_py_library",
    "tf_gen_op_libs",
    "tf_gen_op_wrapper_py",
    "tf_kernel_library",
    "tf_py_test",
)
load("//tensorflow:tensorflow.bzl", "cuda_py_test")
load("//tensorflow:tensorflow.bzl", "tf_custom_op_py_library")


tf_custom_op_library(
    name = "python/ops/_t2t_ops.so",
    srcs = [
        "kernels/t2t_ops.cc",
        "kernels/t2t_ops.h",
        "ops/t2t_ops.cc",
    ],
    gpu_srcs = [
        "kernels/t2t_ops_gpu.cu.cc",
        "kernels/t2t_ops.h",
    ],
)

tf_custom_op_py_library(
    name = "t2t_py",
    srcs = [ "__init__.py", "python/ops/t2t_ops.py"],
    srcs_version="PY2AND3",
    dso = [":python/ops/_t2t_ops.so"],
    kernels = [":t2t_ops_kernels", ":t2t_ops_op_lib"],
    deps= [ ":gen_t2t_ops" ]
)

tf_gen_op_libs(op_lib_names = ["t2t_ops"] )

tf_kernel_library(
    name="t2t_ops_kernels",
    prefix="kernels/t2t_ops"
)

tf_gen_op_wrapper_py(
    name="gen_t2t_ops",
    out="python/ops/gen_t2t_ops.py",
    deps=[":t2t_ops_op_lib"]
) 

cuda_py_test(
    name = "t2t_test",
    size = "small",
    srcs = ["python/kernel_tests/t2t_support_test.py"]
)
