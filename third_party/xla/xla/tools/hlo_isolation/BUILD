load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_shell//shell:sh_test.bzl", "sh_test")
load(
    "//xla:xla.default.bzl",
    "xla_cc_binary",
    "xla_cc_test",
    "xla_py_proto_library",
)
load("//xla/tsl/platform:build_config.bzl", "tf_proto_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//visibility:public"],
)

tf_proto_library(
    name = "hlo_isolation_proto",
    srcs = ["hlo_isolation.proto"],
    visibility = ["//visibility:public"],
    deps = [
        "@com_google_protobuf//:any_proto",
    ],
)

xla_py_proto_library(
    name = "hlo_isolation_py_pb2",
    visibility = ["//visibility:public"],
    deps = [":hlo_isolation_proto"],
)

cc_library(
    name = "hlo_isolation_api",
    testonly = True,
    srcs = ["hlo_isolation_api.cc"],
    hdrs = ["hlo_isolation_api.h"],
    deps = [
        ":hlo_isolation_proto_cc",
        "//xla:comparison_util",
        "//xla:error_spec",
        "//xla:literal",
        "//xla:literal_comparison",
        "//xla:shape_util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/pass:hlo_pass_pipeline",
        "//xla/hlo/transforms:defuser",
        "//xla/hlo/transforms:despecializer",
        "//xla/hlo/transforms/simplifiers:hlo_memory_scheduler",
        "//xla/pjrt:pjrt_executable",
        "//xla/service:hlo_runner_interface",
        "//xla/tests:test_utils",
        "//xla/tools:hlo_decomposer_lib",
        "//xla/tools:hlo_module_loader",
        "//xla/tsl/platform:env",
        "//xla/tsl/platform:status_macros",
        "//xla/tsl/platform:test",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/container:btree",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/flags:commandlineflag",
        "@com_google_absl//absl/flags:reflection",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:span",
        "@com_google_protobuf//:protobuf_lite",
        "@com_googlesource_code_re2//:re2",
        "@tsl//tsl/platform:path",
    ],
)

cc_library(
    name = "hlo_isolation_test_base",
    testonly = True,
    hdrs = ["hlo_isolation_test_base.h"],
    deps = [
        ":hlo_isolation_api",
        "//xla/hlo/ir:hlo",
        "//xla/tsl/lib/core:status_test_util",
        "//xla/tsl/platform:test",
        "@com_google_absl//absl/log",
    ],
)

xla_cc_test(
    name = "hlo_isolation_test_base_test",
    srcs = ["hlo_isolation_test_base_test.cc"],
    deps = [
        ":hlo_isolation_api",
        ":hlo_isolation_proto_cc",
        ":hlo_isolation_test_base",
        "//xla:array2d",
        "//xla:literal",
        "//xla:literal_util",
        "//xla:shape_util",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/parser:hlo_parser",
        "//xla/service:computation_placer",
        "//xla/service:hlo_runner_interface",
        "//xla/tests:hlo_interpreter_reference_mixin",
        "//xla/tests:hlo_test_base",
        "//xla/tests:pjrt_client_registry",  # fixdeps: keep
        "//xla/tests:pjrt_interpreter_client_registry",
        "//xla/tests:test_utils",
        "//xla/tests:xla_internal_test_main",  # fixdeps: keep
        "//xla/tsl/platform:env",
        "//xla/tsl/platform:status_macros",
        "//xla/tsl/platform:test",
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest",
        "@tsl//tsl/platform:path",
    ],
)

sh_test(
    name = "hlo_isolation_cli_test",
    srcs = ["hlo_isolation_cli_test.sh"],
    args = [
        "$(location :hlo_isolation_test)",
        "$(location @llvm-project//llvm:FileCheck)",
        "$(location :hlo_isolation_cli_test_data)",
    ],
    data = [
        ":hlo_isolation_cli_test_data",
        ":hlo_isolation_test",
        "@llvm-project//llvm:FileCheck",
    ],
)

xla_cc_binary(
    name = "hlo_isolation_test",
    testonly = True,
    srcs = ["hlo_isolation_cli.cc"],
    deps = [
        ":hlo_isolation_api",
        ":hlo_isolation_proto_cc",
        "//xla/pjrt:pjrt_client",
        "//xla/service:cpu_plugin",
        "//xla/service:gpu_plugin",
        "//xla/service:hlo_runner",
        "//xla/service:hlo_runner_interface",
        "//xla/service:interpreter_plugin",
        "//xla/tools:run_hlo_module_lib",
        "//xla/tsl/platform:status_macros",
        "//xla/tsl/util:command_line_flags",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@tsl//tsl/platform:platform_port",
    ],
)

filegroup(
    name = "hlo_isolation_cli_test_data",
    srcs = ["//xla/tools:data/hlo_isolation_cli_test.hlo"],
)
