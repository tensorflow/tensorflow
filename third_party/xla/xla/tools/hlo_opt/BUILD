load(
    "@local_config_rocm//rocm:build_defs.bzl",
    "if_rocm_is_configured",
)
load(
    "@local_tsl//tsl/platform:build_config_root.bzl",
    "tf_cuda_tests_tags",
)

# hlo-opt tool.
load(
    "@local_tsl//tsl/platform/default:cuda_build_defs.bzl",
    "if_cuda_is_configured",
)
load("//xla:lit.bzl", "enforce_glob", "lit_test_suite")
load(
    "//xla/stream_executor:build_defs.bzl",
    "if_gpu_is_configured",
)
load("//xla/tsl:tsl.default.bzl", "filegroup")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//xla:internal"],
    licenses = ["notice"],
)

# Includes a macro to register a provider.
cc_library(
    name = "opt_lib",
    srcs = ["opt_lib.cc"],
    hdrs = ["opt_lib.h"],
    deps = [
        "//xla:debug_options_flags",
        "//xla:statusor",
        "//xla:types",
        "//xla/hlo/ir:hlo",
        "//xla/service:compiler",
        "//xla/service:executable",
        "//xla/service:hlo_graph_dumper",
        "//xla/service:platform_util",
        "//xla/stream_executor",
        "//xla/stream_executor:platform",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/strings",
        "@local_tsl//tsl/platform:statusor",
    ],
)

cc_library(
    name = "gpu_opt",
    testonly = True,
    srcs = if_gpu_is_configured(["gpu_opt.cc"]),
    deps = [
        ":opt_lib",
        "//xla:debug_options_flags",
        "//xla:statusor",
        "//xla:types",
        "//xla/hlo/ir:hlo",
        "//xla/service:buffer_value",
        "//xla/service:compiler",
        "//xla/service:dump",
        "//xla/service:executable",
        "//xla/service:hlo_graph_dumper",
        "//xla/service:platform_util",
        "//xla/service/gpu:buffer_sharing",
        "//xla/service/gpu:compile_module_to_llvm_ir",
        "//xla/service/gpu:executable_proto_cc",
        "//xla/service/gpu:gpu_compiler",
        "//xla/service/gpu:gpu_hlo_schedule",
        "//xla/service/llvm_ir:llvm_util",
        "//xla/stream_executor",
        "//xla/stream_executor/platform",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings:string_view",
        "@llvm-project//llvm:ir_headers",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:statusor",
    ] + if_gpu_is_configured([
        "//xla/service:gpu_plugin",
        "//xla/service/gpu:gpu_executable",
    ]) + if_cuda_is_configured([
        "//xla/stream_executor:cuda_platform",
    ]) + if_rocm_is_configured([
        "//xla/stream_executor:rocm_platform",
    ]),
    alwayslink = True,  # Initializer needs to run.
)

cc_library(
    name = "cpu_opt",
    testonly = True,
    srcs = ["cpu_opt.cc"],
    deps = [
        ":opt_lib",
        "//xla/service:cpu_plugin",
        "//xla/service:hlo_graph_dumper",
        "//xla/stream_executor/host:host_platform",
        "//xla/stream_executor/platform",
    ],
    alwayslink = True,  # Initializer needs to run.
)

cc_library(
    name = "opt_main",
    testonly = True,
    srcs = ["opt_main.cc"],
    deps = [
        "cpu_opt",
        ":opt_lib",
        "//xla:debug_options_flags",
        "//xla:status",
        "//xla:statusor",
        "//xla/hlo/ir:hlo",
        "//xla/service:hlo_runner",
        "//xla/service:platform_util",
        "//xla/tools:hlo_module_loader",
        "//xla/tools:run_hlo_module_lib",
        "//xla/tsl/util:command_line_flags",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@local_tsl//tsl/platform:env",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:logging",
        "@local_tsl//tsl/platform:path",
        "@local_tsl//tsl/platform:platform_port",
        "@local_tsl//tsl/platform:status",
        "@local_tsl//tsl/platform:statusor",
        "@local_tsl//tsl/platform:test",
    ] + if_gpu_is_configured([
        ":gpu_opt",
    ]) + if_cuda_is_configured([
        "//xla/stream_executor:cuda_platform",
    ]) + if_rocm_is_configured([
        "//xla/stream_executor:rocm_platform",
    ]),
)

lit_test_suite(
    name = "hlo_opt_tests",
    srcs = enforce_glob(
        [
            "cpu_hlo.hlo",
            "gpu_hlo.hlo",
            "gpu_hlo_backend.hlo",
            "gpu_hlo_buffers.hlo",
            "gpu_hlo_llvm.hlo",
            "gpu_hlo_ptx.hlo",
            "gpu_hlo_unoptimized_llvm.hlo",
        ],
        include = [
            "*.hlo",
        ],
    ),
    args = if_cuda_is_configured([
        "--param=PTX=PTX",
        "--param=GPU=a100_80",
    ]) + if_rocm_is_configured([
        "--param=PTX=GCN",
        "--param=GPU=mi200",
    ]),
    cfg = "//xla:lit.cfg.py",
    data = [":test_utilities"],
    default_tags = tf_cuda_tests_tags(),
    tags_override = {
        "gpu_hlo_ptx.hlo": ["no_rocm"],
    },
    tools = [
        "//xla/tools:hlo-opt",
        "@llvm-project//llvm:FileCheck",
    ],
)

# Bundle together all of the test utilities that are used by tests.
filegroup(
    name = "test_utilities",
    testonly = True,
    data = [
        "gpu_specs/a100_80.txtpb",
        "gpu_specs/mi200.txtpb",
        "//xla/tools:hlo-opt",
        "@llvm-project//llvm:FileCheck",
    ],
)

filegroup(
    name = "all_gpu_specs",
    data = glob(["gpu_specs/*.txtpb"]),
)

exports_files(glob([
    "gpu_specs/*.txtpb",
]))
