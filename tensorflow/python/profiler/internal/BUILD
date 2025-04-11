load("@local_xla//xla/tsl:tsl.bzl", "if_macos", "internal_visibility")
load("@local_xla//xla/tsl:tsl.default.bzl", "tsl_pybind_extension")
load("//tensorflow:strict.default.bzl", "py_strict_library", "py_strict_test")
load("//tensorflow:tensorflow.default.bzl", "cuda_py_strict_test", "get_compatible_with_portable", "tf_python_pybind_extension")
load("//tensorflow/core/profiler/builds:build_config.bzl", "tf_profiler_copts")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//tensorflow/python/profiler:__subpackages__"],
    licenses = ["notice"],
)

py_strict_library(
    name = "flops_registry",
    srcs = ["flops_registry.py"],
    deps = [
        "//tensorflow/python/framework:graph_util",
        "//tensorflow/python/framework:ops",
        "//third_party/py/numpy",
    ],
)

py_strict_test(
    name = "flops_registry_test",
    srcs = ["flops_registry_test.py"],
    deps = [
        ":flops_registry",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/ops:random_ops",
        "//tensorflow/python/ops:variables",
        "//tensorflow/python/platform:client_testlib",
    ],
)

py_strict_library(
    name = "model_analyzer_testlib",
    srcs = ["model_analyzer_testlib.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/python/framework:for_generated_wrappers",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:init_ops",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/ops:nn_grad",
        "//tensorflow/python/ops:nn_ops",
        "//tensorflow/python/ops:rnn",
        "//tensorflow/python/ops:rnn_cell",
        "//tensorflow/python/ops:tensor_array_grad",
        "//tensorflow/python/ops:variable_scope",
        "//tensorflow/python/profiler:model_analyzer",
        "//tensorflow/python/training:gradient_descent",
        "//tensorflow/python/util:_pywrap_tfprof",
        "//tensorflow/python/util:compat",
    ],
)

py_strict_test(
    name = "print_model_analysis_test",
    srcs = ["print_model_analysis_test.py"],
    deps = [
        "//tensorflow/python/framework:for_generated_wrappers",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:init_ops",
        "//tensorflow/python/ops:nn_ops",
        "//tensorflow/python/ops:variable_scope",
        "//tensorflow/python/platform:client_testlib",
    ],
)

cuda_py_strict_test(
    name = "run_metadata_test",
    srcs = ["run_metadata_test.py"],
    tags = [
        "no_gpu",  # b/138442728
        "no_pip",
    ],
    xla_enable_strict_auto_jit = False,  # Node names are different with autojit
    deps = [
        ":model_analyzer_testlib",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python/client:session",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/ops:random_ops",
        "//tensorflow/python/ops:variables",
        "//tensorflow/python/platform:client_testlib",
        "//tensorflow/python/profiler:model_analyzer",
        "//tensorflow/python/profiler:option_builder",
    ],
)

tf_python_pybind_extension(
    name = "_pywrap_traceme",
    srcs = ["traceme_wrapper.cc"],
    copts = tf_profiler_copts(),
    enable_stub_generation = True,
    pytype_srcs = [
        "_pywrap_traceme.pyi",
    ],
    visibility = [
        "//perftools/accelerators/xprof/xprofilez/integration_tests:__pkg__",
        "//tensorflow/python:__pkg__",
        "//tensorflow/python/profiler:__subpackages__",
        "//tensorflow/tools/pip_package:__subpackages__",
    ],
    deps = [
        "@local_xla//xla/python/profiler/internal:traceme_wrapper",
        "@pybind11",
    ],
)

tf_python_pybind_extension(
    name = "_pywrap_profiler",
    srcs = ["profiler_wrapper.cc"],
    copts = tf_profiler_copts(),
    enable_stub_generation = True,
    pytype_srcs = [
        "_pywrap_profiler.pyi",
    ],
    visibility = [
        "//tensorflow/core/profiler:internal",
        "//tensorflow/python:__pkg__",
        "//tensorflow/python/eager:__pkg__",
        "//tensorflow/python/profiler:__pkg__",
        "//tensorflow/tools/pip_package:__subpackages__",
    ],
    deps = [
        ":profiler_pywrap_impl",
        "//tensorflow/core/profiler/rpc:profiler_server_for_pybind",
        "//tensorflow/python/lib/core:pybind11_status",
        "@com_google_absl//absl/status",
        "@org_xprof//xprof/convert:repository",
        "@org_xprof//xprof/convert:tool_options",
        "@org_xprof//xprof/convert:xplane_to_tools_data",
        "@pybind11",
    ],
)

cc_library(
    name = "python_hooks",
    hdrs = ["python_hooks.h"],
    compatible_with = get_compatible_with_portable(),
    copts = tf_profiler_copts() + ["-fexceptions"],
    features = ["-use_header_modules"],  # Incompatible with -fexceptions.
    visibility = ["//visibility:private"],
    deps = [
        "//tensorflow/core:lib",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/memory",
        "@local_tsl//tsl/profiler/protobuf:xplane_proto_cc",
        "@local_xla//xla/python/profiler/internal:python_hooks",
        "@pybind11",
    ],
    alwayslink = True,
)

cc_library(
    name = "profiler_pywrap_impl",
    srcs = ["profiler_pywrap_impl.cc"],
    hdrs = ["profiler_pywrap_impl.h"],
    visibility = ["//visibility:private"],
    deps = [
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core/profiler/lib:profiler_session_for_pybind",
        "//tensorflow/core/profiler/rpc:profiler_server_for_pybind",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:variant",
        "@local_tsl//tsl/profiler/lib:profiler_session",
        "@local_tsl//tsl/profiler/protobuf:xplane_proto_cc",
        "@local_xla//xla/tsl/profiler/convert:xplane_to_trace_events",
        "@local_xla//xla/tsl/profiler/rpc/client:capture_profile",
        "@local_xla//xla/tsl/profiler/utils:session_manager",
    ],
)

tsl_pybind_extension(
    name = "_pywrap_profiler_plugin",
    srcs = ["pywrap_profiler_plugin.cc"],
    pytype_srcs = [
        "_pywrap_profiler_plugin.pyi",
    ],
    visibility = internal_visibility([
        "//tensorflow/core/profiler:internal",
        "//tensorflow/python/eager:__pkg__",
        "//tensorflow/python/profiler:__pkg__",
    ]),
    deps = [
        "//tensorflow/core:protos_all_cc_impl",
        "//tensorflow/core/framework:attr_value_proto_cc_impl",
        "//tensorflow/core/framework:op",
        "//tensorflow/core/framework:tensor",
        "//tensorflow/python/lib/core:py_exception_registry",
        "//tensorflow/python/lib/core:pybind11_status",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_protobuf//:protobuf",
        "@local_tsl//tsl/profiler/protobuf:profiler_analysis_proto_cc_impl",
        "@local_tsl//tsl/profiler/protobuf:profiler_options_proto_cc_impl",
        "@local_tsl//tsl/profiler/protobuf:profiler_service_monitor_result_proto_cc_impl",
        "@local_tsl//tsl/profiler/protobuf:profiler_service_proto_cc_impl",
        "@local_xla//xla:autotune_results_proto_cc_impl",
        "@local_xla//xla:autotuning_proto_cc_impl",
        "@local_xla//xla:xla_data_proto_cc_impl",
        "@local_xla//xla:xla_proto_cc_impl",
        "@local_xla//xla/pjrt:status_casters",
        "@local_xla//xla/service:hlo_proto_cc_impl",
        "@local_xla//xla/service:metrics_proto_cc_impl",
        "@local_xla//xla/service/gpu:backend_configs_cc_impl",
        "@local_xla//xla/service/gpu/model:hlo_op_profile_proto_cc_impl",
        "@local_xla//xla/stream_executor:device_description_proto_cc_impl",
        "@local_xla//xla/stream_executor/cuda:cuda_compute_capability_proto_cc_impl",
        "@local_xla//xla/tsl/framework:allocator_registry_impl",
        "@local_xla//xla/tsl/lib/io:table",
        "@local_xla//xla/tsl/platform:env_impl",
        "@local_xla//xla/tsl/platform:types",
        "@local_xla//xla/tsl/platform/cloud:gcs_file_system",
        "@local_xla//xla/tsl/profiler/backends/cpu:traceme_recorder_impl",
        "@local_xla//xla/tsl/profiler/rpc:profiler_server_impl",
        "@local_xla//xla/tsl/profiler/rpc:profiler_service_impl",
        "@local_xla//xla/tsl/profiler/rpc/client:capture_profile",
        "@local_xla//xla/tsl/profiler/rpc/client:profiler_client_impl",
        "@local_xla//xla/tsl/profiler/utils:session_manager",
        "@local_xla//xla/tsl/profiler/utils:time_utils_impl",
        "@local_xla//xla/tsl/protobuf:bfc_memory_map_proto_cc_impl",
        "@local_xla//xla/tsl/protobuf:dnn_proto_cc_impl",
        "@local_xla//xla/tsl/protobuf:error_codes_proto_impl_cc_impl",
        "@local_xla//xla/tsl/protobuf:histogram_proto_cc_impl",
        "@local_xla//xla/tsl/protobuf:rpc_options_proto_cc_impl",
        "@local_xla//xla/tsl/protobuf:test_log_proto_cc_impl",
        "@org_xprof//xprof/convert:tool_options",
        "@org_xprof//xprof/pywrap:profiler_plugin_impl",
        "@pybind11",
    ] + if_macos([
        "@local_xla//xla/tsl/lib/histogram",
        "@local_xla//xla/tsl/lib/io:record_writer",
        "@local_xla//xla/tsl/lib/monitoring:collection_registry",
        "@local_xla//xla/tsl/lib/monitoring:sampler",
        "//tensorflow/core:framework_internal_impl",
        "//tensorflow/core/common_runtime/gpu:gpu_id_impl",
        "//tensorflow/core/lib/core:arena",
        "//tensorflow/core/lib/strings:ordered_code",
        "//tensorflow/core/platform:platform_strings",
        "@local_tsl//tsl/platform:random",
        "@local_xla//xla/tsl/platform:resource",
        "@local_tsl//tsl/profiler/lib:profiler_factory_impl",
        "@local_tsl//tsl/profiler/lib:profiler_session_impl",
    ]),
)
