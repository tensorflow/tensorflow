"""Wrapper around proto libraries used inside the XLA codebase."""

load("@bazel_skylib//:bzl_library.bzl", "bzl_library")
load(
    "@local_config_rocm//rocm:build_defs.bzl",
    "if_rocm_is_configured",
)
load(
    "//xla/tsl:tsl.bzl",
    "tsl_copts",
)
load(
    "//xla/tsl/platform:build_config_root.bzl",
    "if_static",
    "tf_exec_properties",
)
load(
    "//xla/tsl/platform/default:cuda_build_defs.bzl",
    "if_cuda_is_configured",
)

def xla_py_proto_library(**_kwargs):
    # Note: we don't currently define a proto library target for Python in OSS.
    pass

def xla_py_test_deps():
    return []

# TODO(ddunleavy): some of these should be removed from here and added to
# specific targets.
# We actually shouldn't need this anymore post vendoring. If we build without
# `framework_shared_object` in the bazelrc all of this should be able to go
# away. The problem is making sure that all these impl deps are `if_static`'d
# appropriately throughout XLA.
_XLA_SHARED_OBJECT_SENSITIVE_DEPS = if_static(extra_deps = [], otherwise = [
    Label("//xla:autotune_results_proto_cc_impl"),
    Label("//xla:autotuning_proto_cc_impl"),
    Label("//xla:xla_data_proto_cc_impl"),
    Label("//xla:xla_proto_cc_impl"),
    Label("//xla/service:buffer_assignment_proto_cc_impl"),
    Label("//xla/service:hlo_proto_cc_impl"),
    Label("//xla/service:metrics_proto_cc_impl"),
    Label("//xla/service/gpu:backend_configs_cc_impl"),
    Label("//xla/service/gpu/model:hlo_op_profile_proto_cc_impl"),
    Label("//xla/service/memory_space_assignment:memory_space_assignment_proto_cc_impl"),
    Label("//xla/stream_executor:device_description_proto_cc_impl"),
    Label("//xla/stream_executor:stream_executor_impl"),
    Label("//xla/stream_executor/cuda:cuda_compute_capability_proto_cc_impl"),
    Label("//xla/stream_executor/gpu:gpu_init_impl"),
    Label("//xla/backends/cpu/runtime:thunk_proto_cc_impl"),
    "@com_google_protobuf//:protobuf",
    "//xla/tsl/framework:allocator_registry_impl",
    "//xla/tsl/framework:allocator",
    "@local_tsl//tsl/platform:env_impl",
    "//xla/tsl/profiler/backends/cpu:annotation_stack_impl",
    "//xla/tsl/profiler/backends/cpu:traceme_recorder_impl",
    "@local_tsl//tsl/profiler/protobuf:profiler_options_proto_cc_impl",
    "@local_tsl//tsl/profiler/protobuf:xplane_proto_cc_impl",
    "//xla/tsl/profiler/utils:time_utils_impl",
    "//xla/tsl/protobuf:protos_all_cc_impl",
]) + if_cuda_is_configured([
    Label("//xla/stream_executor/cuda:all_runtime"),
    Label("//xla/stream_executor/cuda:stream_executor_cuda"),
]) + if_rocm_is_configured([
    Label("//xla/stream_executor/gpu:gpu_stream"),
    Label("//xla/stream_executor/rocm:all_runtime"),
    Label("//xla/stream_executor/rocm:stream_executor_rocm"),
    "//xla/tsl/util:determinism",
])

def xla_cc_binary(deps = [], copts = tsl_copts(), **kwargs):
    native.cc_binary(deps = deps + _XLA_SHARED_OBJECT_SENSITIVE_DEPS, copts = copts, **kwargs)

def xla_cc_test(name, deps = [], **kwargs):
    native.cc_test(
        name = name,
        deps = deps + _XLA_SHARED_OBJECT_SENSITIVE_DEPS,
        exec_properties = tf_exec_properties(kwargs),
        **kwargs
    )

def xla_internal(targets, otherwise = []):
    _ = targets  # buildifier: disable=unused-variable
    return otherwise

def tests_build_defs_bzl_deps():
    return []

def xla_bzl_library(name = "xla_bzl_library"):
    bzl_library(
        name = "xla_bzl",
        srcs = ["xla.bzl"],
        deps = [
            "//xla/tsl:tsl_bzl",
            "@local_config_rocm//rocm:build_defs_bzl",
            "//xla/tsl/platform:build_config_root_bzl",
            "//xla/tsl/platform/default:cuda_build_defs_bzl",
            "@bazel_skylib//:bzl_library",
        ],
    )
