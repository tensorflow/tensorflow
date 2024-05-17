load("//tensorflow:strict.default.bzl", "py_strict_library")
load("//tensorflow:tensorflow.bzl", "if_google", "tf_cc_shared_test")
load("//tensorflow:tensorflow.default.bzl", "tf_py_strict_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    licenses = ["notice"],
)

cc_library(
    name = "tfrt_session_init",
    hdrs = ["tfrt_session_init.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/core:lib",
        "//tensorflow/core/common_runtime:local_session_selection",
    ],
)

cc_library(
    name = "tfrt_session",
    srcs = ["tfrt_session.cc"],
    hdrs = ["tfrt_session.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/compiler/mlir/tfrt:backend_compiler",
        "//tensorflow/compiler/mlir/tfrt:tfrt_compile_options",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_lite",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/common_runtime:local_session_selection",
        "//tensorflow/core/common_runtime:process_util",
        "//tensorflow/core/common_runtime:session_factory",
        "//tensorflow/core/platform:errors",
        "//tensorflow/core/platform:status",
        "//tensorflow/core/platform:statusor",
        "//tensorflow/core/platform:threadpool_interface",
        "//tensorflow/core/platform:threadpool_options",
        "//tensorflow/core/runtime_fallback/runtime:runtime_fallback_alwayslink",
        "//tensorflow/core/tfrt/fallback:fallback_state",
        "//tensorflow/core/tfrt/graph_executor",
        "//tensorflow/core/tfrt/graph_executor:graph_execution_options",
        "//tensorflow/core/tfrt/mlrt/interpreter:context",
        "//tensorflow/core/tfrt/mlrt/kernel",
        "//tensorflow/core/tfrt/mlrt/kernel:batch_kernel",
        "//tensorflow/core/tfrt/run_handler_thread_pool:run_handler_concurrent_work_queue",
        "//tensorflow/core/tfrt/runtime",
        "//tensorflow/core/tfrt/runtime:tf_threadpool_concurrent_work_queue",
        "//tensorflow/core/tfrt/runtime:work_queue_interface",
        "//tensorflow/core/tfrt/utils",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/log:die_if_null",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
        "@eigen_archive//:eigen3",
        "@llvm-project//llvm:Support",
        "@local_tsl//tsl/platform:errors",
        "@tf_runtime//:core_runtime",
        "@tf_runtime//:hostcontext",
    ] + if_google([
        "//learning/brain/tfrt/tpu:tfrttpu_alwayslink",
        "//learning/brain/tfrt/tpu/fused_kernel:tpu_fused_kernel_alwayslink",
        "//learning/brain/tfrt/support:export_mlir",
    ]),
    alwayslink = 1,
)

tf_cc_shared_test(
    name = "tfrt_session_test",
    size = "small",
    srcs = ["tfrt_session_test.cc"],
    data = [
        "//tensorflow/core/tfrt/saved_model/tests:toy_v1/1/saved_model.pb",
        "//tensorflow/core/tfrt/saved_model/tests:toy_v1/1/variables/variables.data-00000-of-00001",
        "//tensorflow/core/tfrt/saved_model/tests:toy_v1/1/variables/variables.index",
    ],
    tags = ["no_oss"],
    deps = [
        ":tfrt_session",
        "//tensorflow/cc:array_ops",
        "//tensorflow/cc:cc_ops",
        "//tensorflow/cc:const_op",
        "//tensorflow/cc:ops",
        "//tensorflow/cc:scope",
        "//tensorflow/cc/saved_model:reader",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:core_cpu_base",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_lite",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core/common_runtime:local_session_selection",
        "//tensorflow/core/common_runtime:process_util",
        "//tensorflow/core/common_runtime:session_factory",
        "//tensorflow/core/framework:tensor_testutil",
        "//tensorflow/core/platform:resource_loader",
        "//tensorflow/core/platform:status",
        "//tensorflow/core/platform:threadpool_options",
        "//tensorflow/core/tfrt/runtime",
        "//tensorflow/core/tfrt/saved_model:saved_model_testutil",
        "//tensorflow/core/tfrt/utils:thread_pool",
        "//tensorflow/python/framework:test_ops_kernels",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/time",
        "@com_google_googletest//:gtest",
        "@local_tsl//tsl/platform:protobuf",
    ],
)

# copybara:uncomment_begin(C++ Global registration not working in python, b/290988709)
# py_strict_library(
#     name = "tfrt_session_py",
#     srcs_version = "PY3",
#     visibility = ["//visibility:public"],
#     deps = [
#         "//tensorflow/core/tfrt/tfrt_session",
#     ],
# )
#
# tf_py_strict_test(
#     name = "tfrt_session_python_test",
#     srcs = ["tfrt_session_python_test.py"],
#     exec_properties = select({
#         "//tools/cpp:asan_build": {"cpp_link.mem": "16g"},
#         "//conditions:default": None,
#     }),
#     python_version = "PY3",
#     deps = [
#         ":tfrt_session_py",
#         "//tensorflow/core:protos_all_py",
#         "//tensorflow/python/client:session",
#         "//tensorflow/python/framework:constant_op",
#         "//tensorflow/python/framework:ops",
#         "//tensorflow/python/framework:test_lib",
#         "//tensorflow/python/ops:math_ops",
#         "//tensorflow/python/platform:test",
#     ],
# )
# copybara:uncomment_end
