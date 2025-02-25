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
        "//tensorflow/core/profiler/convert:repository",
        "//tensorflow/core/profiler/convert:tool_options",
        "//tensorflow/core/profiler/convert:xplane_to_tools_data",
        "//tensorflow/core/profiler/rpc:profiler_server_for_pybind",
        "//tensorflow/python/lib/core:pybind11_status",
        "@com_google_absl//absl/status",
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
        "//tensorflow/core/profiler/protobuf:xplane_proto_cc",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/memory",
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
        # copybara:uncomment "//cloud/ai/platform/tensorboard/webserver/filesystem:hypercomputer_file_system",
        "@local_xla//xla/tsl/profiler/convert:xplane_to_trace_events",
        "@local_xla//xla/tsl/profiler/rpc/client:capture_profile",
        "@local_xla//xla/tsl/profiler/utils:session_manager",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core/profiler/convert:xplane_to_tools_data",
        "//tensorflow/core/profiler/lib:profiler_session_for_pybind",
        "//tensorflow/core/profiler/protobuf:xplane_proto_cc",
        "//tensorflow/core/profiler/rpc:profiler_server_for_pybind",
        "//tensorflow/core/profiler/rpc/client:save_profile",
        "@local_tsl//tsl/profiler/protobuf:xplane_proto_cc",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:variant",
    ],
)

py_strict_library(
    name = "_pywrap_profiler_plugin",
    srcs = ["_pywrap_profiler_plugin.py"],
    data = ["@org_xprof//plugin/mlprofiler/pywrap:_pywrap_profiler_plugin"],
    visibility = ["//visibility:public"],
    deps = ["@org_xprof//plugin/mlprofiler/pywrap:_pywrap_profiler_plugin"],
)

py_strict_test(
    name = "_pywrap_test",
    srcs = ["_pywrap_test.py"],
    deps = [
        ":_pywrap_profiler_plugin",
        "//tensorflow/python/platform:client_testlib",
    ],
)
