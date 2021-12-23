# buildifier: disable=same-origin-load
load("//tensorflow:tensorflow.bzl", "get_compatible_with_portable")

# buildifier: disable=same-origin-load
load("//tensorflow:tensorflow.bzl", "pybind_extension")

package(
    default_visibility = [
        "//visibility:public",
    ],
    licenses = ["notice"],
)

cc_library(
    name = "bcast_grad_args_op",
    srcs = [
        "bcast_grad_args.cc",
    ],
    hdrs = [
        "bcast_grad_args.h",
    ],
    compatible_with = get_compatible_with_portable(),
    deps = [
        "//tensorflow/lite:framework",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/kernels:kernel_util",
        "//tensorflow/lite/kernels:padding",
        "//tensorflow/lite/kernels/internal:common",
        "//tensorflow/lite/kernels/internal:compatibility",
        "//tensorflow/lite/kernels/internal:tensor",
        "//tensorflow/lite/kernels/internal:tensor_utils",
        "//tensorflow/lite/kernels/internal:types",
        "@flatbuffers",
    ],
)

cc_library(
    name = "gradient_ops",
    srcs = [
        "gradient_ops.cc",
    ],
    hdrs = [
        "gradient_ops.h",
    ],
    compatible_with = get_compatible_with_portable(),
    deps = [
        ":bcast_grad_args_op",
        "//tensorflow/lite:framework",
    ],
)

cc_test(
    name = "bcast_grad_args_op_test",
    size = "small",
    srcs = [
        "bcast_grad_args_test.cc",
    ],
    deps = [
        ":bcast_grad_args_op",
        "//tensorflow/lite:framework",
        "//tensorflow/lite/kernels:test_main",
        "//tensorflow/lite/kernels:test_util",
        "//tensorflow/lite/testing:util",
        "@flatbuffers",
    ],
)

pybind_extension(
    name = "pywrap_gradient_ops",
    srcs = [
        "gradient_ops_wrapper.cc",
    ],
    hdrs = ["gradient_ops.h"],
    additional_exported_symbols = ["GradientOpsRegisterer"],
    link_in_framework = True,
    module_name = "pywrap_gradient_ops",
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
        "@pybind11//:__subpackages__",
        "@ruy//:__subpackages__",
        "@snappy//:__subpackages__",
        "//:__subpackages__",
        "@XNNPACK//:__subpackages__",
        "@zlib//:__subpackages__",
    ],
    deps = [
        ":gradient_ops",
        "//tensorflow/lite:mutable_op_resolver",
        "//third_party/python_runtime:headers",
        "@pybind11",
    ],
)
