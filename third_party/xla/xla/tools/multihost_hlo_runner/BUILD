load("@bazel_skylib//rules:build_test.bzl", "build_test")
load("@local_config_cuda//cuda:build_defs.bzl", "if_cuda")
load("@local_config_rocm//rocm:build_defs.bzl", "if_rocm")
load("//xla:xla.default.bzl", "xla_cc_binary", "xla_py_strict_test")
load("//xla/tests:build_defs.bzl", "xla_test")
load("//xla/tsl:tsl.bzl", "if_cuda_or_rocm", "if_google")
load("//xla/tsl:tsl.default.bzl", "tsl_pybind_extension")
load("//xla/tsl/platform:build_config_root.bzl", "tf_gpu_tests_tags")
load("//xla/tsl/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//xla:internal"],
    licenses = ["notice"],
)

build_test(
    name = "hlo_runner_main_build_test",
    tags = [
        "cpu",
        "gpu",  # TODO(ddunleavy): this configuration of tags will only build on GPU in OSS.
    ],
    targets = [
        ":hlo_runner_main",
    ],
)

cc_library(
    name = "profiler_interface",
    hdrs = ["profiler_interface.h"],
    deps = [
        "@com_google_absl//absl/status:statusor",
    ],
)

cc_library(
    name = "hlo_runner_main_lib",
    testonly = True,
    srcs = ["hlo_runner_main.cc"],
    compatible_with = None,
    tags = ["no_mac"],
    deps = [
        ":create_client",
        ":functional_hlo_runner",
        ":profiler_interface",
        "//xla:debug_options_flags",
        "//xla:status_macros",
        "//xla:xla_data_proto_cc",
        "//xla/pjrt:pjrt_client",
        "//xla/pjrt/distributed",
        "//xla/pjrt/distributed:client",
        "//xla/pjrt/distributed:key_value_store_interface",
        "//xla/pjrt/distributed:service",
        "//xla/pjrt/plugin/xla_gpu:xla_gpu_allocator_config",
        "//xla/pjrt/plugin/xla_gpu:xla_gpu_client_options",
        "//xla/service:cpu_plugin",
        "//xla/service:hlo_module_util",
        "//xla/tsl/platform:statusor",
        "//xla/tsl/util:command_line_flags",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/time",
        "@tsl//tsl/platform:errors",
        "@tsl//tsl/platform:logging",
        "@tsl//tsl/platform:platform_port",
        "@tsl//tsl/platform:status",
        "@tsl//tsl/platform:statusor",
    ] + if_cuda_or_rocm([
        "//xla/service:gpu_plugin",
        "//xla/backends/profiler/gpu:device_tracer",
    ]) + if_cuda([
        "//xla/stream_executor:cuda_platform",
        "//xla/backends/profiler/gpu:cupti_tracer",
    ] + if_google(
        [
            "//third_party/py/jax/jaxlib/cuda:cuda_gpu_kernels",  # fixdeps: keep
        ],
    )) + if_rocm([
        "//xla/stream_executor:rocm_platform",
    ]),
    alwayslink = True,
)

xla_cc_binary(
    name = "hlo_runner_main",
    testonly = True,
    tags = [
        "no_mac",
    ],
    deps = [
        ":hlo_runner_main_lib",
    ],
)

xla_cc_binary(
    name = "hlo_runner_main_gpu",
    testonly = True,
    tags = [
        "cuda-only",
        "gpu",
        "no_mac",
    ] + tf_gpu_tests_tags(),
    deps = [
        ":hlo_runner_main_lib",
        "//xla/service:gpu_plugin",
        "//xla/stream_executor:cuda_platform",
    ] + if_google([
        "//third_party/py/jax/jaxlib/cuda:cuda_gpu_kernels",  # fixdeps: keep
    ]),
)

cc_library(
    name = "create_client",
    srcs = ["create_client.cc"],
    hdrs = ["create_client.h"],
    deps = [
        "//xla:status_macros",
        "//xla:xla_proto_cc",
        "//xla/pjrt:pjrt_client",
        "//xla/pjrt:pjrt_compiler",
        "//xla/pjrt/distributed",
        "//xla/pjrt/distributed:client",
        "//xla/pjrt/distributed:key_value_store_interface",
        "//xla/pjrt/distributed:service",
        "//xla/pjrt/plugin/xla_cpu:cpu_client_options",
        "//xla/pjrt/plugin/xla_cpu:xla_cpu_pjrt_client",
        "//xla/pjrt/plugin/xla_gpu:xla_gpu_client_options",
        "//xla/pjrt/plugin/xla_gpu:xla_gpu_pjrt_client",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/time",
        "@tsl//tsl/platform:status",
        "@tsl//tsl/platform:statusor",
    ],
)

cc_library(
    name = "hlo_input_output_format",
    srcs = ["hlo_input_output_format.cc"],
    hdrs = ["hlo_input_output_format.h"],
    visibility = [
        "//devtools/ml/mlcompass:__subpackages__",
        "//xla:internal",
    ],
    deps = [
        "//xla/tsl/util:fixed_option_set_flag",
        "@com_google_absl//absl/strings:string_view",
    ],
)

cc_library(
    name = "functional_hlo_runner",
    srcs = ["functional_hlo_runner.cc"],
    hdrs = ["functional_hlo_runner.h"],
    deps = [
        ":hlo_input_output_format",
        ":profiler_interface",
        "//xla:future",
        "//xla:literal",
        "//xla:literal_util",
        "//xla:shape_layout",
        "//xla:shape_util",
        "//xla:status_macros",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla:xla_proto_cc",
        "//xla/backends/profiler:profiler_backends",  # To register the Host Tracers for GPU Plugin.
        "//xla/backends/profiler/plugin:plugin_tracer",  # To register the GPU Tracers with the GPU Plugin.
        "//xla/client:executable_build_options",
        "//xla/hlo/builder:xla_computation",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/parser:hlo_parser",
        "//xla/hlo/pass:hlo_pass_pipeline",
        "//xla/hlo/transforms:while_loop_trip_count_annotator",
        "//xla/hlo/translate:stablehlo",
        "//xla/hlo/translate/hlo_to_mhlo:translate",
        "//xla/pjrt:host_memory_spaces",
        "//xla/pjrt:pjrt_client",
        "//xla/pjrt:pjrt_compiler",
        "//xla/pjrt:pjrt_executable",
        "//xla/pjrt:pjrt_layout",
        "//xla/pjrt/distributed:key_value_store_interface",
        "//xla/runtime/large_hlo_snapshot_serialization:serialization",
        "//xla/service:computation_layout",
        "//xla/service:computation_placer_hdr",
        "//xla/service:hlo_module_config",
        "//xla/service:hlo_module_util",
        "//xla/service:hlo_proto_cc",
        "//xla/service:slow_operation_alarm",
        "//xla/tests:test_utils",
        "//xla/tools:hlo_control_flow_flattening",
        "//xla/tsl/platform:env",
        "//xla/tsl/platform:errors",
        "//xla/tsl/platform:statusor",
        "//xla/tsl/util:fixed_option_set_flag",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:btree",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:span",
        "@llvm-project//mlir:FuncExtensions",
        "@tsl//tsl/profiler/lib:profiler_factory_impl",
        "@tsl//tsl/profiler/lib:profiler_session",
        "@tsl//tsl/profiler/lib:profiler_session_impl",
        "@tsl//tsl/profiler/protobuf:profiler_options_proto_cc",
        "@tsl//tsl/profiler/protobuf:xplane_proto_cc",
    ],
)

filegroup(
    name = "hlo_file",
    srcs = glob(["data/**"]),
)

xla_test(
    name = "functional_hlo_runner_test",
    srcs = ["functional_hlo_runner_test.cc"],
    backend_tags = {
        "gpu": [
            "multi_gpu",
            "no_oss",
            "nomsan",
        ],
        "nvgpu_any": [
            "broken",
            "no_oss",
        ],
    },
    backends = [
        "cpu",
        "gpu",
    ],
    data = [":hlo_file"],
    tags = ["no_mac"],
    deps = [
        ":create_client",
        ":functional_hlo_runner",
        ":hlo_input_output_format",
        ":profiler_interface",
        "//xla:debug_options_flags",
        "//xla:literal",
        "//xla:shape_layout",
        "//xla:shape_util",
        "//xla:status_macros",
        "//xla:xla_data_proto_cc",
        "//xla:xla_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/testlib:filecheck",
        "//xla/pjrt:pjrt_client",
        "//xla/pjrt:pjrt_executable",
        "//xla/pjrt/plugin/xla_gpu:xla_gpu_allocator_config",
        "//xla/pjrt/plugin/xla_gpu:xla_gpu_client_options",
        "//xla/runtime/large_hlo_snapshot_serialization:serialization",
        "//xla/service:computation_layout",
        "//xla/service:hlo_proto_cc",
        "//xla/service:platform_util",
        "//xla/stream_executor:platform",
        "//xla/stream_executor:platform_manager",
        "//xla/tests:test_utils",
        "//xla/tests:xla_test_backend_predicates",
        "//xla/tsl/lib/core:status_test_util",
        "//xla/tsl/platform:env",
        "//xla/tsl/platform:errors",
        "//xla/tsl/platform:statusor",
        "//xla/tsl/platform:subprocess",
        "//xla/tsl/platform:test",
        "//xla/tsl/testing:temporary_directory",
        "//xla/tsl/util:command_line_flags",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest",
        "@tsl//tsl/platform",
        "@tsl//tsl/platform:path",
        "@tsl//tsl/platform:protobuf",
    ] + if_cuda(["//xla/tsl/cuda:nvml"]),
)

tsl_pybind_extension(
    name = "py_hlo_multihost_runner",
    srcs = ["python_hlo_runner.cc"],
    deps = [
        ":create_client",
        ":functional_hlo_runner",
        ":hlo_input_output_format",
        ":profiler_interface",
        "//xla:debug_options_flags",
        "//xla:status_macros",
        "//xla:xla_data_proto_cc",
        "//xla/ffi",
        "//xla/ffi:ffi_api",
        "//xla/ffi/api:c_api",
        "//xla/pjrt:pjrt_client",
        "//xla/pjrt:status_casters",
        "//xla/pjrt/distributed",
        "//xla/pjrt/distributed:client",
        "//xla/pjrt/distributed:key_value_store_interface",
        "//xla/pjrt/distributed:service",
        "//xla/pjrt/plugin/xla_gpu:xla_gpu_allocator_config",
        "//xla/pjrt/plugin/xla_gpu:xla_gpu_client_options",
        "//xla/python:logging",
        "//xla/service:cpu_plugin",
        "//xla/service:custom_call_target_registry",
        "//xla/service:hlo_module_util",
        "//xla/tsl/platform:statusor",
        "//xla/tsl/util:command_line_flags",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/time",
        "@nanobind",
        "@tsl//tsl/platform:errors",
        "@tsl//tsl/platform:logging",
        "@tsl//tsl/platform:platform_port",
        "@tsl//tsl/platform:status",
        "@tsl//tsl/platform:statusor",
    ] + if_cuda_or_rocm([
        "//xla/service:gpu_plugin",
        "//xla/backends/profiler/gpu:device_tracer",
    ]) + if_cuda([
        "//xla/stream_executor:cuda_platform",
        "//xla/backends/profiler/gpu:cupti_tracer",
    ] + if_google(
        [
            "//third_party/py/jax/jaxlib/cuda:cuda_gpu_kernels",  # fixdeps: keep
        ],
    )) + if_rocm([
        "//xla/stream_executor:rocm_platform",
    ]),
)

xla_py_strict_test(
    name = "python_hlo_runner_test",
    srcs = ["python_hlo_runner_test.py"],
    data = [
        ":hlo_file",
    ],
    # Transformer engine dlopens several cuda libraries and so requires them as data dependencies.
    need_cuda_libs = True,
    tags = [
        "gpu",
        # Transformer engine takes a long time to compile. Disabling it for CI tests.
        "no_oss",
        "requires-gpu-sm90-only",
    ],
    deps = [
        ":py_hlo_multihost_runner",
        "@absl_py//absl/testing:absltest",
        "@transformer_engine//:transformer_engine_jax",
    ] + if_cuda([
        "//xla/stream_executor:cuda_platform",
    ]),
)
