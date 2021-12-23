load("//tensorflow:tensorflow.bzl", "pybind_extension")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

cc_library(
    name = "calibration_wrapper_lib",
    srcs = ["calibration_wrapper.cc"],
    hdrs = ["calibration_wrapper.h"],
    deps = [
        "//tensorflow/lite:framework",
        "//tensorflow/lite:shared_library",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/kernels:builtin_ops",
        "//tensorflow/lite/python/interpreter_wrapper:numpy",
        "//tensorflow/lite/python/interpreter_wrapper:python_error_reporter",
        "//tensorflow/lite/python/interpreter_wrapper:python_utils",
        "//tensorflow/lite/tools/optimize:quantization_wrapper_utils",
        "//tensorflow/lite/tools/optimize:quantize_model",
        "//tensorflow/lite/tools/optimize/calibration:calibration_reader",
        "//tensorflow/lite/tools/optimize/calibration:calibrator_lib",
        "//third_party/python_runtime:headers",  # buildcleaner: keep
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:optional",
    ],
)

pybind_extension(
    name = "_pywrap_tensorflow_lite_calibration_wrapper",
    srcs = [
        "calibration_wrapper_pybind11.cc",
    ],
    hdrs = ["calibration_wrapper.h"],
    link_in_framework = True,
    module_name = "_pywrap_tensorflow_lite_calibration_wrapper",
    # TODO(b/146808376): Reduce the dependencies to those that are really needed.
    static_deps = [
        "@arm_neon_2_x86_sse//:__subpackages__",
        "@bazel_tools//:__subpackages__",
        "@boringssl//:__subpackages__",
        "@clog//:__subpackages__",
        "@com_github_googlecloudplatform_tensorflow_gcp_tools//:__subpackages__",
        "@com_google_absl//:__subpackages__",
        "@com_google_googleapis//:__subpackages__",
        "@com_google_protobuf//:__subpackages__",
        "@com_googlesource_code_re2//:__subpackages__",
        "@cpuinfo//:__subpackages__",
        "@curl//:__subpackages__",
        "@double_conversion//:__subpackages__",
        "@eigen_archive//:__subpackages__",
        "@farmhash_archive//:__subpackages__",
        "@fft2d//:__subpackages__",
        "@flatbuffers//:__subpackages__",
        "@FP16//:__subpackages__",
        "@FXdiv//:__subpackages__",
        "@gemmlowp//:__subpackages__",
        "@gif//:__subpackages__",
        "@highwayhash//:__subpackages__",
        "@hwloc//:__subpackages__",
        "@jsoncpp_git//:__subpackages__",
        "@libjpeg_turbo//:__subpackages__",
        "@libxsmm_archive//:__subpackages__",
        "@llvm_openmp//:__subpackages__",
        "@llvm-project//:__subpackages__",
        "@llvm_terminfo//:__subpackages__",
        "@llvm_zlib//:__subpackages__",
        "@local_config_cuda//:__subpackages__",
        "@local_config_git//:__subpackages__",
        "@local_config_python//:__subpackages__",
        "@local_config_rocm//:__subpackages__",
        "@local_config_tensorrt//:__subpackages__",
        "@local_execution_config_platform//:__subpackages__",
        "@nsync//:__subpackages__",
        "@platforms//:__subpackages__",
        "@pthreadpool//:__subpackages__",
        "@pybind11//:__subpackages__",
        "@ruy//:__subpackages__",
        "@snappy//:__subpackages__",
        "//:__subpackages__",
        "@XNNPACK//:__subpackages__",
        "@zlib//:__subpackages__",
    ],
    deps = [
        ":calibration_wrapper_lib",
        "//tensorflow/lite:framework_lib",
        "//tensorflow/python:pybind11_lib",
        "//third_party/python_runtime:headers",
        "@pybind11",
    ],
)

py_library(
    name = "calibrator",
    srcs = [
        "calibrator.py",
    ],
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [
        ":_pywrap_tensorflow_lite_calibration_wrapper",  # buildcleaner: keep
        "//tensorflow/lite/python:convert_phase",
        "//tensorflow/lite/python:interpreter",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:util",
        "//third_party/py/numpy",
    ],
)

py_test(
    name = "calibrator_test",
    srcs = ["calibrator_test.py"],
    data = [
        ":test_data",
        "//tensorflow/lite:testdata/multi_add.bin",
    ],
    python_version = "PY3",
    srcs_version = "PY3",
    tags = ["no_oss"],
    deps = [
        ":calibrator",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
        "@six_archive//:six",
    ],
)
