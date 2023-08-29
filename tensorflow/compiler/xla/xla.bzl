"""Wrapper around proto libraries used inside the XLA codebase."""

load(
    "//tensorflow/tsl:tsl.bzl",
    "if_tsl_link_protobuf",
    "tsl_copts",
    _tsl_clean_dep = "clean_dep",
)
load(
    "//tensorflow/tsl/platform/default:cuda_build_defs.bzl",
    "if_cuda_is_configured",
)
load(
    "@local_config_rocm//rocm:build_defs.bzl",
    "if_rocm_is_configured",
)
load(
    "//tensorflow/tsl/platform:build_config_root.bzl",
    "tf_exec_properties",
)

def register_extension_info(**kwargs):
    pass

def clean_dep(target):
    """Returns string to 'target' in @{org_tensorflow,xla} repository.

    This is distinct from the clean_dep which appears in @{org_tensorflow,tsl}.
    TODO(ddunleavy,jakeharmon): figure out what to do with this after vendoring.
    """

    # A repo-relative label is resolved relative to the file in which the
    # Label() call appears, i.e. @org_tensorflow.
    return str(Label(target))

def xla_py_proto_library(**kwargs):
    # Note: we don't currently define a proto library target for Python in OSS.
    _ignore = kwargs
    pass

def xla_py_grpc_library(**kwargs):
    # Note: we don't currently define any special targets for Python GRPC in OSS.
    _ignore = kwargs
    pass

ORC_JIT_MEMORY_MAPPER_TARGETS = []

def xla_py_test_deps():
    return []

def xla_cc_binary(deps = None, copts = tsl_copts(), **kwargs):
    if not deps:
        deps = []

    # TODO(ddunleavy): some of these should be removed from here and added to
    # specific targets.
    deps += [
        _tsl_clean_dep("@com_google_protobuf//:protobuf"),
        "//tensorflow/compiler/xla:xla_proto_cc_impl",
        "//tensorflow/compiler/xla:xla_data_proto_cc_impl",
        "//tensorflow/compiler/xla/service:hlo_proto_cc_impl",
        "//tensorflow/compiler/xla/service:memory_space_assignment_proto_cc_impl",
        "//tensorflow/compiler/xla/service/gpu:backend_configs_cc_impl",
        "//tensorflow/compiler/xla/service/gpu:hlo_op_profile_proto_cc_impl",
        "//tensorflow/compiler/xla/stream_executor:dnn_proto_cc_impl",
        "//tensorflow/tsl/platform:env_impl",
        "//tensorflow/tsl/platform:tensor_float_32_utils",
        "//tensorflow/tsl/profiler/utils:time_utils_impl",
        "//tensorflow/tsl/profiler/backends/cpu:annotation_stack_impl",
        "//tensorflow/tsl/profiler/backends/cpu:traceme_recorder_impl",
        "//tensorflow/compiler/xla:autotuning_proto_cc_impl",
        "//tensorflow/tsl/protobuf:protos_all_cc_impl",
        "//tensorflow/tsl/protobuf:dnn_proto_cc_impl",
        "//tensorflow/tsl/framework:allocator",
        "//tensorflow/tsl/framework:allocator_registry_impl",
        "//tensorflow/tsl/util:determinism",
    ]
    native.cc_binary(deps = deps, copts = copts, **kwargs)

def xla_cc_test(
        name,
        deps = [],
        **kwargs):
    native.cc_test(
        name = name,
        deps = deps + if_tsl_link_protobuf(
                   [],
                   [
                       _tsl_clean_dep("@com_google_protobuf//:protobuf"),
                       # TODO(zacmustin): remove these in favor of more granular dependencies in each test.
                       clean_dep("//tensorflow/compiler/xla:xla_proto_cc_impl"),
                       clean_dep("//tensorflow/compiler/xla:xla_data_proto_cc_impl"),
                       clean_dep("//tensorflow/compiler/xla/service:hlo_proto_cc_impl"),
                       clean_dep("//tensorflow/compiler/xla/service:memory_space_assignment_proto_cc_impl"),
                       clean_dep("//tensorflow/compiler/xla/service/gpu:backend_configs_cc_impl"),
                       clean_dep("//tensorflow/compiler/xla/service/gpu:hlo_op_profile_proto_cc_impl"),
                       clean_dep("//tensorflow/compiler/xla/stream_executor:dnn_proto_cc_impl"),
                       clean_dep("//tensorflow/compiler/xla/stream_executor:stream_executor_impl"),
                       clean_dep("//tensorflow/compiler/xla/stream_executor:device_id_utils"),
                       clean_dep("//tensorflow/compiler/xla/stream_executor/gpu:gpu_cudamallocasync_allocator"),
                       clean_dep("//tensorflow/compiler/xla/stream_executor/gpu:gpu_init_impl"),
                       clean_dep("//tensorflow/tsl/profiler/utils:time_utils_impl"),
                       clean_dep("//tensorflow/tsl/profiler/backends/cpu:annotation_stack_impl"),
                       clean_dep("//tensorflow/tsl/profiler/backends/cpu:traceme_recorder_impl"),
                       clean_dep("//tensorflow/tsl/profiler/protobuf:xplane_proto_cc_impl"),
                       clean_dep("//tensorflow/compiler/xla:autotuning_proto_cc_impl"),
                       clean_dep("//tensorflow/tsl/protobuf:dnn_proto_cc_impl"),
                       clean_dep("//tensorflow/tsl/protobuf:protos_all_cc_impl"),
                       clean_dep("//tensorflow/tsl/platform:env_impl"),
                       clean_dep("//tensorflow/tsl/framework:allocator"),
                       clean_dep("//tensorflow/tsl/framework:allocator_registry_impl"),
                       clean_dep("//tensorflow/tsl/util:determinism"),
                   ],
               ) +
               if_cuda_is_configured([
                   clean_dep("//tensorflow/compiler/xla/stream_executor/cuda:cuda_stream"),
                   clean_dep("//tensorflow/compiler/xla/stream_executor/cuda:all_runtime"),
                   clean_dep("//tensorflow/compiler/xla/stream_executor/cuda:stream_executor_cuda"),
               ]) +
               if_rocm_is_configured([
                   clean_dep("//tensorflow/compiler/xla/stream_executor/gpu:gpu_stream"),
                   clean_dep("//tensorflow/compiler/xla/stream_executor/rocm:all_runtime"),
                   clean_dep("//tensorflow/compiler/xla/stream_executor/rocm:stream_executor_rocm"),
               ]),
        exec_properties = tf_exec_properties(kwargs),
        **kwargs
    )

def auto_sharding_deps():
    return ["//tensorflow/compiler/xla/hlo/experimental/auto_sharding:auto_sharding_impl"]

register_extension_info(
    extension = xla_cc_test,
    label_regex_for_dep = "{extension_name}",
)
