load("//tensorflow:tensorflow.default.bzl", "cuda_py_test", "get_compatible_with_cloud", "tf_python_pybind_extension")
load("//tensorflow:tensorflow.bzl", "py_test")
load("//tensorflow/core/profiler/builds:build_config.bzl", "tf_profiler_copts")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//tensorflow/python/profiler:__subpackages__"],
    licenses = ["notice"],
)

py_library(
    name = "flops_registry",
    srcs = ["flops_registry.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:graph_util",
    ],
)

py_test(
    name = "flops_registry_test",
    srcs = ["flops_registry_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":flops_registry",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:random_ops",
        "//tensorflow/python:variables",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:test_lib",
    ],
)

py_library(
    name = "model_analyzer_testlib",
    srcs = ["model_analyzer_testlib.py"],
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:init_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:nn_grad",
        "//tensorflow/python:nn_ops",
        "//tensorflow/python:rnn",
        "//tensorflow/python:rnn_cell",
        "//tensorflow/python:tensor_array_grad",
        "//tensorflow/python:training",
        "//tensorflow/python:variable_scope",
    ],
)

py_test(
    name = "print_model_analysis_test",
    srcs = ["print_model_analysis_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        "//tensorflow/core/profiler:protos_all_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:init_ops",
        "//tensorflow/python:nn_ops",
        "//tensorflow/python:pywrap_tensorflow",
        "//tensorflow/python:variable_scope",
    ],
)

cuda_py_test(
    name = "run_metadata_test",
    srcs = ["run_metadata_test.py"],
    python_version = "PY3",
    tags = [
        "no_gpu",  # b/138442728
        "no_pip",
    ],
    xla_enable_strict_auto_jit = False,  # Node names are different with autojit
    deps = [
        ":model_analyzer_testlib",
        "//tensorflow/core/profiler:protos_all_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:random_ops",
        "//tensorflow/python/profiler:model_analyzer",
    ],
)

tf_python_pybind_extension(
    name = "_pywrap_traceme",
    srcs = ["traceme_wrapper.cc"],
    copts = tf_profiler_copts(),
    visibility = [
        "//perftools/accelerators/xprof/xprofilez/integration_tests:__pkg__",
        "//tensorflow/python/profiler:__subpackages__",
    ],
    deps = [
        "//tensorflow/compiler/xla/python/profiler/internal:traceme_wrapper",
        "@pybind11",
    ],
)

tf_python_pybind_extension(
    name = "_pywrap_profiler",
    srcs = ["profiler_wrapper.cc"],
    copts = tf_profiler_copts(),
    visibility = [
        "//tensorflow/core/profiler:internal",
        "//tensorflow/python/eager:__pkg__",
        "//tensorflow/python/profiler:__pkg__",
    ],
    deps = [
        ":profiler_pywrap_impl",
        "//tensorflow/core/platform:env",
        "//tensorflow/core/profiler/convert:repository",
        "//tensorflow/core/profiler/convert:tool_options",
        "//tensorflow/core/profiler/convert:xplane_to_tools_data",
        "//tensorflow/core/profiler/rpc:profiler_server_for_pybind",
        "//tensorflow/python:pybind11_status",
        "@com_google_absl//absl/types:variant",
        "@pybind11",
    ],
)

cc_library(
    name = "python_hooks",
    hdrs = ["python_hooks.h"],
    compatible_with = get_compatible_with_cloud(),
    copts = tf_profiler_copts() + ["-fexceptions"],
    features = ["-use_header_modules"],  # Incompatible with -fexceptions.
    visibility = ["//tensorflow/core/profiler:__subpackages__"],
    deps = [
        "//tensorflow/compiler/xla/python/profiler/internal:python_hooks",
        "//tensorflow/core:lib",
        "//tensorflow/core/profiler/protobuf:xplane_proto_cc",
        "//tensorflow/core/profiler/utils:time_utils",
        "//tensorflow/core/profiler/utils:xplane_builder",
        "//tensorflow/core/profiler/utils:xplane_schema",
        "//tensorflow/core/profiler/utils:xplane_utils",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@pybind11",
    ],
    alwayslink = True,
)

cc_library(
    name = "profiler_pywrap_impl",
    srcs = ["profiler_pywrap_impl.cc"],
    hdrs = ["profiler_pywrap_impl.h"],
    visibility = [
        # Do not change this visibility field.
        # This library is referred in OSS world by github.com/pytorch/xla
        "//tensorflow/compiler/xla:__subpackages__",
    ],
    deps = [
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core/profiler/convert:xplane_to_tools_data",
        "//tensorflow/core/profiler/lib:profiler_session_for_pybind",
        "//tensorflow/core/profiler/protobuf:xplane_proto_cc",
        "//tensorflow/core/profiler/rpc:profiler_server_for_pybind",
        "//tensorflow/core/profiler/rpc/client:capture_profile",
        "//tensorflow/core/profiler/rpc/client:save_profile",
        "//tensorflow/tsl/profiler/convert:xplane_to_trace_events",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:variant",
    ],
)
