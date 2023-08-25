"""OSS versions of Bazel macros that can't be migrated to TSL."""

load(
    "//tensorflow/tsl:tsl.bzl",
    "clean_dep",
    "if_libtpu",
)
load("@local_config_cuda//cuda:build_defs.bzl", "if_cuda")
load("@local_config_rocm//rocm:build_defs.bzl", "if_rocm")
load(
    "//third_party/mkl:build_defs.bzl",
    "if_mkl_ml",
)

def tf_tpu_dependencies():
    return if_libtpu(["//tensorflow/core/tpu/kernels"])

def tf_dtensor_tpu_dependencies():
    return if_libtpu(["//tensorflow/dtensor/cc:dtensor_tpu_kernels"])

def tf_additional_binary_deps():
    return [
        clean_dep("@nsync//:nsync_cpp"),
        # TODO(allenl): Split these out into their own shared objects. They are
        # here because they are shared between contrib/ op shared objects and
        # core.
        clean_dep("//tensorflow/core/kernels:lookup_util"),
        clean_dep("//tensorflow/core/util/tensor_bundle"),
    ] + if_cuda(
        [
            clean_dep("//tensorflow/compiler/xla/stream_executor:cuda_platform"),
        ],
    ) + if_rocm(
        [
            clean_dep("//tensorflow/compiler/xla/stream_executor:rocm_platform"),
            clean_dep("//tensorflow/compiler/xla/stream_executor/rocm:rocm_rpath"),
        ],
    ) + if_mkl_ml(
        [
            clean_dep("//third_party/mkl:intel_binary_blob"),
        ],
    )
