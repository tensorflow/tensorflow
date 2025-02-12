# Description:
#   Run sample models with TensorRT through TF-TRT bridge. Test TensorRT
#   numerics and latency.

load("//tensorflow:strict.default.bzl", "py_strict_binary", "py_strict_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

exports_files(glob([
    "models/*",
]))

py_strict_library(
    name = "model_handler",
    srcs = ["model_handler.py"],
    deps = [
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python/client:session",
        "//tensorflow/python/compiler/tensorrt:trt_convert_py",
        "//tensorflow/python/framework:convert_to_constants",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:importer",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/ops:random_ops",
        "//tensorflow/python/platform:tf_logging",
        "//tensorflow/python/saved_model:load",
        "//tensorflow/python/saved_model:loader",
        "//tensorflow/python/saved_model:signature_constants",
        "//tensorflow/python/saved_model:tag_constants",
        "//third_party/py/numpy",
    ],
)

py_strict_library(
    name = "result_analyzer",
    srcs = ["result_analyzer.py"],
    deps = [
        ":model_handler",
        "//tensorflow/python/compiler/tensorrt:trt_convert_py",
        "//tensorflow/python/platform:tf_logging",
        "//third_party/py/numpy",
    ],
)

py_strict_binary(
    name = "run_models",
    srcs = ["run_models.py"],
    data = ["sample_model/saved_model.pb"],
    deps = [
        ":model_handler",
        ":result_analyzer",
        "//tensorflow/python/compiler/tensorrt:trt_convert_py",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/framework:config",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/platform:client_testlib",
        "//tensorflow/python/platform:gfile",
        "//tensorflow/python/platform:tf_logging",
        "//tensorflow/python/saved_model:signature_constants",
        "//tensorflow/python/saved_model:tag_constants",
        "@absl_py//absl:app",
        "@absl_py//absl/flags",
    ],
)
