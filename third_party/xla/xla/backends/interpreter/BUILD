load(
    "@local_tsl//tsl/platform:build_config_root.bzl",
    "if_static",
)
load("@local_tsl//tsl/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

cc_library(
    name = "interpreter_transfer_manager",
    srcs = ["interpreter_transfer_manager.cc"],
    hdrs = ["interpreter_transfer_manager.h"],
    deps = [
        ":platform_id",
        "//xla/service:generic_transfer_manager",
        "//xla/service:transfer_manager",
    ],
    alwayslink = True,  # Contains per-platform transfer manager registration
)

cc_library(
    name = "compiler",
    srcs = ["compiler.cc"],
    hdrs = ["compiler.h"],
    deps = [
        ":executable",
        ":platform_id",
        "//xla:literal",
        "//xla:status_macros",
        "//xla:statusor",
        "//xla:util",
        "//xla/hlo/evaluator:hlo_evaluator",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/ir:hlo_module_group",
        "//xla/service:batchnorm_expander",
        "//xla/service:cholesky_expander",
        "//xla/service:comparison_expander",
        "//xla/service:compiler",
        "//xla/service:computation_placer",
        "//xla/service:custom_call_target_registry",
        "//xla/service:dynamic_dimension_inference",
        "//xla/service:dynamic_index_splitter",
        "//xla/service:eigh_expander",
        "//xla/service:executable",
        "//xla/service:hlo_cost_analysis",
        "//xla/service:hlo_module_config",
        "//xla/service:hlo_pass_pipeline",
        "//xla/service:layout_assignment",
        "//xla/service:qr_expander",
        "//xla/service:topk_rewriter",
        "//xla/service:triangular_solve_expander",
        "//xla/stream_executor",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/types:span",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:status",
        "@local_tsl//tsl/platform:statusor",
    ],
    alwayslink = True,  # Contains compiler registration
)

cc_library(
    name = "platform_id",
    srcs = ["platform_id.cc"],
    hdrs = ["platform_id.h"],
    deps = ["//xla/stream_executor"] + if_static(
        ["@com_google_protobuf//:protobuf"],
        ["@com_google_protobuf//:protobuf_headers"],
    ),
)

cc_library(
    name = "executable_base",
    srcs = ["executable_base.cc"],
    hdrs = ["executable_base.h"],
    deps = [
        "//xla:literal",
        "//xla:shape_tree",
        "//xla:shape_util",
        "//xla:statusor",
        "//xla:xla_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/service:dynamic_dimension_inference",
        "//xla/service:executable",
        "//xla/service:hlo_execution_profile",
        "//xla/service:maybe_owning_device_memory",
        "//xla/service:shaped_buffer",
        "//xla/service:transfer_manager",
        "//xla/stream_executor",
        "@local_tsl//tsl/platform:statusor",
    ],
)

cc_library(
    name = "executable",
    srcs = ["executable.cc"],
    hdrs = ["executable.h"],
    deps = [
        ":executable_base",
        ":executor",
        "//xla:literal",
        "//xla:shape_util",
        "//xla:status_macros",
        "//xla:statusor",
        "//xla:types",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/evaluator:hlo_evaluator",
        "//xla/hlo/ir:hlo",
        "//xla/service:executable",
        "//xla/service:hlo_cost_analysis",
        "//xla/service:hlo_execution_profile",
        "//xla/service:hlo_module_config",
        "//xla/service:maybe_owning_device_memory",
        "//xla/service:shaped_buffer",
        "//xla/service:transfer_manager",
        "//xla/stream_executor",
        "@com_google_absl//absl/types:span",
        "@local_tsl//tsl/platform:env",
        "@local_tsl//tsl/platform:errors",
    ],
)

cc_library(
    name = "platform",
    srcs = ["platform.cc"],
    hdrs = ["platform.h"],
    deps = [
        ":executor",
        ":platform_id",
        "//xla/stream_executor",
        "//xla/stream_executor/platform",
        "@com_google_absl//absl/strings:str_format",
        "@local_tsl//tsl/platform:status",
        "@local_tsl//tsl/platform:statusor",
    ],
    alwayslink = True,  # Registers itself with the PlatformManager.
)

cc_library(
    name = "executor",
    srcs = ["executor.cc"],
    hdrs = ["executor.h"],
    deps = [
        "//xla:shape_util",
        "//xla:status_macros",
        "//xla:xla_data_proto_cc",
        "//xla/stream_executor",
        "//xla/stream_executor:stream_executor_interface",
        "//xla/stream_executor/host:host_stream",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/types:span",
    ],
)
