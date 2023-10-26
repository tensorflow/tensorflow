"""OSS versions of Bazel macros that can't be migrated to TSL."""

load(
    "//tensorflow/core/platform:build_config_root.bzl",
    "if_static",
)
load(
    "@local_xla//xla:xla.bzl",
    _xla_clean_dep = "clean_dep",
)
load(
    "@local_tsl//tsl:tsl.bzl",
    "if_libtpu",
    _tsl_clean_dep = "clean_dep",
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
        str(Label("@nsync//:nsync_cpp")),
        # TODO(allenl): Split these out into their own shared objects. They are
        # here because they are shared between contrib/ op shared objects and
        # core.
        str(Label("//tensorflow/core/kernels:lookup_util")),
        str(Label("//tensorflow/core/util/tensor_bundle")),
    ] + if_cuda(
        [
            str(Label("@local_xla//xla/stream_executor:cuda_platform")),
        ],
    ) + if_rocm(
        [
            str(Label("@local_xla//xla/stream_executor:rocm_platform")),
            str(Label("@local_xla//xla/stream_executor/rocm:rocm_rpath")),
        ],
    ) + if_mkl_ml(
        [
            str(Label("//third_party/mkl:intel_binary_blob")),
        ],
    )

def tf_protos_all():
    return if_static(
        extra_deps = [
            str(Label("//tensorflow/core/protobuf:conv_autotuning_proto_cc_impl")),
            str(Label("//tensorflow/core:protos_all_cc_impl")),
            _xla_clean_dep("@local_xla//xla:autotune_results_proto_cc_impl"),
            _xla_clean_dep("@local_xla//xla:autotuning_proto_cc_impl"),
            _tsl_clean_dep("@local_tsl//tsl/protobuf:protos_all_cc_impl"),
        ],
        otherwise = [str(Label("//tensorflow/core:protos_all_cc"))],
    )
