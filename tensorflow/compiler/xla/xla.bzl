"""Wrapper around proto libraries used inside the XLA codebase."""

load(
    "//tensorflow/tsl:tsl.bzl",
    "clean_dep",
    "if_tsl_link_protobuf",
)
load("//tensorflow/tsl/platform:build_config.bzl", "tsl_cc_test")

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

def xla_cc_binary(deps = None, **kwargs):
    if not deps:
        deps = []

    # TODO(ddunleavy): some of these should be removed from here and added to
    # specific targets.
    deps += [
        clean_dep("@com_google_protobuf//:protobuf"),
        "//tensorflow/compiler/xla:xla_proto_cc_impl",
        "//tensorflow/compiler/xla:xla_data_proto_cc_impl",
        "//tensorflow/compiler/xla/service:hlo_proto_cc_impl",
        "//tensorflow/compiler/xla/service/gpu:backend_configs_cc_impl",
        "//tensorflow/compiler/xla/stream_executor:dnn_proto_cc_impl",
        "//tensorflow/tsl/platform:env_impl",
        "//tensorflow/tsl/profiler/utils:time_utils_impl",
        "//tensorflow/tsl/profiler/backends/cpu:traceme_recorder_impl",
        "//tensorflow/tsl/protobuf:protos_all_cc_impl",
    ]
    native.cc_binary(deps = deps, **kwargs)

def xla_cc_test(
        name,
        deps = [],
        **kwargs):
    tsl_cc_test(
        name = name,
        deps = deps + if_tsl_link_protobuf(
            [],
            [
                # clean_dep("@com_google_protobuf//:protobuf"),
                # TODO(zacmustin): remove these in favor of more granular dependencies in each test.
                "//tensorflow/compiler/xla:xla_proto_cc_impl",
                "//tensorflow/compiler/xla:xla_data_proto_cc_impl",
                "//tensorflow/compiler/xla/service:hlo_proto_cc_impl",
                "//tensorflow/compiler/xla/service/gpu:backend_configs_cc_impl",
                "//tensorflow/compiler/xla/stream_executor:dnn_proto_cc_impl",
                "//tensorflow/tsl/profiler/utils:time_utils_impl",
                "//tensorflow/tsl/profiler/backends/cpu:traceme_recorder_impl",
                "//tensorflow/tsl/protobuf:protos_all_cc_impl",
            ],
        ),
        **kwargs
    )
