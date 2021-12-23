load("//tensorflow:tensorflow.bzl", "pybind_extension")

# buildifier: disable=same-origin-load
load("//tensorflow:tensorflow.bzl", "get_compatible_with_portable")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

cc_library(
    name = "numpy",
    srcs = ["numpy.cc"],
    hdrs = ["numpy.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        "//tensorflow/lite:string_util",
        "//tensorflow/lite/c:common",
        "//third_party/py/numpy:headers",
        "//third_party/python_runtime:headers",  # buildcleaner: keep
    ],
)

cc_library(
    name = "interpreter_wrapper_lib",
    srcs = ["interpreter_wrapper.cc"],
    hdrs = [
        "interpreter_wrapper.h",
    ],
    compatible_with = get_compatible_with_portable(),
    deps = [
        ":numpy",
        ":python_error_reporter",
        ":python_utils",
        "//tensorflow/lite:framework",
        "//tensorflow/lite:shared_library",
        "//tensorflow/lite:string_util",
        "//tensorflow/lite:util",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/core/api",
        "//tensorflow/lite/kernels:builtin_ops",
        "//tensorflow/lite/kernels:reference_ops",
        "//tensorflow/lite/kernels/internal:compatibility",
        "//third_party/python_runtime:headers",  # buildcleaner: keep
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings:str_format",
    ],
)

cc_library(
    name = "python_error_reporter",
    srcs = ["python_error_reporter.cc"],
    hdrs = ["python_error_reporter.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        "//tensorflow/lite:stateful_error_reporter",
        "//third_party/python_runtime:headers",  # buildcleaner: keep
    ],
)

cc_library(
    name = "python_utils",
    srcs = ["python_utils.cc"],
    hdrs = ["python_utils.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        "//tensorflow/lite:framework",
        "//tensorflow/lite:string_util",
        "//third_party/python_runtime:headers",  # buildcleaner: keep
    ],
)

config_setting(
    name = "tflite_pip_with_flex",
    define_values = {
        "tflite_pip_with_flex": "true",
    },
)

pybind_extension(
    name = "_pywrap_tensorflow_interpreter_wrapper",
    srcs = [
        "interpreter_wrapper_pybind11.cc",
    ],
    hdrs = ["interpreter_wrapper.h"],
    compatible_with = get_compatible_with_portable(),
    link_in_framework = True,
    module_name = "_pywrap_tensorflow_interpreter_wrapper",
    # TODO(b/146808376): Reduce the dependencies to those that are really needed.
    static_deps = [
        "@arm_neon_2_x86_sse//:__subpackages__",
        "@bazel_tools//:__subpackages__",
        "@boringssl//:__subpackages__",
        "@clog//:__subpackages__",
        "@com_github_cares_cares//:__subpackages__",
        "@com_github_googlecloudplatform_tensorflow_gcp_tools//:__subpackages__",
        "@com_github_grpc_grpc//:__subpackages__",
        "@com_google_absl//:__subpackages__",
        "@com_google_googleapis//:__subpackages__",
        "@com_google_protobuf//:__subpackages__",
        "@com_googlesource_code_re2//:__subpackages__",
        "@compute_library//:__subpackages__",
        "@cpuinfo//:__subpackages__",
        "@curl//:__subpackages__",
        "@double_conversion//:__subpackages__",
        "@eigen_archive//:__subpackages__",
        "@farmhash_archive//:__subpackages__",
        "@farmhash_gpu_archive//:__subpackages__",
        "@fft2d//:__subpackages__",
        "@flatbuffers//:__subpackages__",
        "@FP16//:__subpackages__",
        "@FXdiv//:__subpackages__",
        "@gemmlowp//:__subpackages__",
        "@gif//:__subpackages__",
        "@highwayhash//:__subpackages__",
        "@hwloc//:__subpackages__",
        "@icu//:__subpackages__",
        "@jsoncpp_git//:__subpackages__",
        "@libjpeg_turbo//:__subpackages__",
        "@libxsmm_archive//:__subpackages__",
        "@llvm_openmp//:__subpackages__",
        "@llvm-project//:__subpackages__",
        "@llvm_terminfo//:__subpackages__",
        "@llvm_zlib//:__subpackages__",
        "@lmdb//:__subpackages__",
        "@local_config_cuda//:__subpackages__",
        "@local_config_git//:__subpackages__",
        "@local_config_nccl//:__subpackages__",
        "@local_config_python//:__subpackages__",
        "@local_config_rocm//:__subpackages__",
        "@local_config_tensorrt//:__subpackages__",
        "@local_execution_config_platform//:__subpackages__",
        "@mkl_dnn_acl_compatible//:__subpackages__",
        "@mkl_dnn_v1//:__subpackages__",
        "@nsync//:__subpackages__",
        "@org_sqlite//:__subpackages__",
        "@platforms//:__subpackages__",
        "@png//:__subpackages__",
        "@pthreadpool//:__subpackages__",
        "@pybind11//:__subpackages__",
        "@ruy//:__subpackages__",
        "@snappy//:__subpackages__",
        "@sobol_data//:__subpackages__",
        "//:__subpackages__",
        "@upb//:__subpackages__",
        "@XNNPACK//:__subpackages__",
        "@zlib//:__subpackages__",
    ],
    deps = [
        ":interpreter_wrapper_lib",
        "@pybind11",
        "//third_party/python_runtime:headers",
        "//tensorflow/lite:framework_lib",
        "//tensorflow/python:pybind11_lib",
    ] + select({
        ":tflite_pip_with_flex": ["//tensorflow/lite/delegates/flex:delegate"],
        "//conditions:default": [],
    }),
)
