"""OSS versions of Bazel macros that can't be migrated to TSL."""

load("@local_config_cuda//cuda:build_defs.bzl", "if_cuda")
load("@local_config_rocm//rocm:build_defs.bzl", "if_rocm")
load(
    "@local_xla//xla/tsl:tsl.bzl",
    "if_libtpu",
)
load(
    "@local_xla//xla/tsl/mkl:build_defs.bzl",
    "if_mkl_ml",
)
load(
    "//tensorflow/core/platform:build_config_root.bzl",
    "if_static",
)

def tf_tpu_dependencies():
    return if_libtpu([Label("//tensorflow/core/tpu/kernels")])

def tf_dtensor_tpu_dependencies():
    return if_libtpu([Label("//tensorflow/dtensor/cc:dtensor_tpu_kernels")])

def tf_additional_binary_deps():
    return [
        # TODO(allenl): Split these out into their own shared objects. They are
        # here because they are shared between contrib/ op shared objects and
        # core.
        Label("//tensorflow/core/kernels:lookup_util"),
        Label("//tensorflow/core/util/tensor_bundle"),
    ] + if_cuda([
        Label("@local_xla//xla/stream_executor:cuda_platform"),
    ]) + if_rocm([
        "@local_xla//xla/stream_executor:rocm_platform",
        "@local_xla//xla/stream_executor/rocm:rocm_rpath",
    ]) + if_mkl_ml([
        Label("@local_xla//xla/tsl/mkl:intel_binary_blob"),
    ])

def tf_protos_all():
    return if_static(
        extra_deps = [
            Label("//tensorflow/core/protobuf:conv_autotuning_proto_cc_impl"),
            Label("//tensorflow/core:protos_all_cc_impl"),
            "@local_xla//xla:autotune_results_proto_cc_impl",
            "@local_xla//xla:autotuning_proto_cc_impl",
            "@local_xla//xla/tsl/protobuf:protos_all_cc_impl",
        ],
        otherwise = [Label("//tensorflow/core:protos_all_cc")],
    )
