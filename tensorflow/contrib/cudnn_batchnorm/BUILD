licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

package(default_visibility = ["//tensorflow:__subpackages__"])

load("//tensorflow:tensorflow.bzl", "tf_custom_op_library")
load("//tensorflow:tensorflow.bzl", "tf_gen_op_libs")
load("//tensorflow:tensorflow.bzl", "tf_gen_op_wrapper_py")
load("//tensorflow:tensorflow.bzl", "tf_kernel_library")

tf_custom_op_library(
    name = "_cudnn_batchnorm.so",
    srcs = [
        "ops/cudnn_batchnorm.cc",
    ],
    deps = [
        "//tensorflow/contrib/cudnn_batchnorm/kernels:cudnn_batchnorm",
    ],
)

tf_gen_op_libs(
    op_lib_names = ["cudnn_batchnorm"],
)

tf_gen_op_wrapper_py(
  name = "cudnn_batchnorm_op",
  out = "gen_cudnn_batchnorm.py",
  require_shape_functions=False,
  hidden=[
    "BatchNormTrainingGrad"
  ],
  deps = ["//tensorflow/contrib/cudnn_batchnorm:cudnn_batchnorm_op_lib"]
)

cc_library(
    name = "cudnn_batchnorm_kernels",

    deps = [
        "//tensorflow/core/kernels:ops_util",
        "//tensorflow/core/kernels:bounds_check",
        "//tensorflow/core/kernels:conv_2d",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//third_party/eigen3",

        "//tensorflow/contrib/cudnn_batchnorm/kernels:cudnn_batchnorm",
    ],
    alwayslink = 1,
)

py_library(
    name = "cudnn_batchnorm_py" ,
    srcs = ["__init__.py", "load_cudnn_batchnorm_so.py"] + glob(["python/ops/*.py"]),
    data = [":_cudnn_batchnorm.so"],
    srcs_version = "PY2AND3",
    deps = [":cudnn_batchnorm_op"],
)

filegroup(
    name = "all_files",
    srcs = glob(
        ["**/*"],
        exclude = [
            "**/METADATA",
            "**/OWNERS",
        ],
    ),
)
