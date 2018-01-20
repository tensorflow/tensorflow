# Description:
# TensorFlow is a computational framework, primarily for use in machine
# learning applications.

package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])  # Apache 2.0

load(
    "//tensorflow:tensorflow.bzl",
    "tf_cc_test",
    "tf_cc_binary",
    "tf_copts",
    "tf_gen_op_wrappers_cc",
    "cc_library_with_android_deps",
)

cc_library(
    name = "gradients",
    srcs = [
        "framework/gradients.cc",
        "framework/while_gradients.cc",
        "framework/while_gradients.h",
    ],
    hdrs = ["framework/gradients.h"],
    deps = [
        ":cc_ops",
        ":cc_ops_internal",
        ":grad_op_registry",
        ":ops",
        ":scope",
        ":scope_internal",
        ":while_loop",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
    ],
)

tf_cc_test(
    name = "framework_gradients_test",
    srcs = ["framework/gradients_test.cc"],
    deps = [
        ":cc_ops",
        ":client_session",
        ":grad_op_registry",
        ":grad_ops",
        ":gradients",
        ":testutil",
        "//tensorflow/core:all_kernels",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
    ],
)

tf_cc_test(
    name = "framework_while_gradients_test",
    size = "small",
    srcs = ["framework/while_gradients_test.cc"],
    deps = [
        ":cc_ops",
        ":client_session",
        ":grad_op_registry",
        ":grad_ops",
        ":gradients",
        ":testutil",
        ":while_loop",
        "//tensorflow/core:all_kernels",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
    ],
)

cc_library(
    name = "gradient_checker",
    srcs = ["framework/gradient_checker.cc"],
    hdrs = ["framework/gradient_checker.h"],
    deps = [
        ":cc_ops",
        ":client_session",
        ":gradients",
        ":ops",
        ":scope",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
    ],
)

tf_cc_test(
    name = "framework_gradient_checker_test",
    srcs = ["framework/gradient_checker_test.cc"],
    deps = [
        ":cc_ops",
        ":grad_op_registry",
        ":grad_ops",
        ":gradient_checker",
        ":testutil",
        "//tensorflow/core:all_kernels",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
    ],
)

cc_library(
    name = "grad_ops",
    deps = [
        ":array_grad",
        ":data_flow_grad",
        ":math_grad",
        ":nn_grad",
    ],
)

cc_library(
    name = "grad_testutil",
    testonly = 1,
    srcs = ["gradients/grad_testutil.cc"],
    hdrs = ["gradients/grad_testutil.h"],
    deps = [
        ":grad_op_registry",
        ":ops",
        ":scope",
    ],
)

cc_library_with_android_deps(
    name = "ops",
    srcs = ["framework/ops.cc"],
    hdrs = ["framework/ops.h"],
    android_deps = ["//tensorflow/core:android_tensorflow_lib"],
    deps = [
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:protos_all_cc",
    ],
)

cc_library_with_android_deps(
    name = "scope",
    srcs = [
        "framework/scope.cc",
        "framework/scope_internal.h",
    ],
    hdrs = ["framework/scope.h"],
    android_deps = ["//tensorflow/core:android_tensorflow_lib"],
    common_deps = [
        ":ops",
    ],
    deps = [
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
    ],
)

cc_library_with_android_deps(
    name = "scope_internal",
    hdrs = ["framework/scope_internal.h"],
    common_deps = [
        ":scope",
    ],
    deps = [],
)

tf_cc_test(
    name = "framework_scope_test",
    srcs = ["framework/scope_test.cc"],
    deps = [
        ":ops",
        ":scope",
        "//tensorflow/core:framework",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
    ],
)

cc_library_with_android_deps(
    name = "client_session",
    srcs = ["client/client_session.cc"],
    hdrs = ["client/client_session.h"],
    android_deps = ["//tensorflow/core:android_tensorflow_lib"],
    common_deps = [
        ":ops",
        ":scope",
    ],
    deps = [
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
    ],
)

tf_cc_test(
    name = "client_client_session_test",
    srcs = ["client/client_session_test.cc"],
    deps = [
        ":cc_ops",
        ":client_session",
        "//tensorflow/core:all_kernels",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:tensorflow",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
    ],
)

cc_library_with_android_deps(
    name = "const_op",
    srcs = ["ops/const_op.cc"],
    hdrs = ["ops/const_op.h"],
    android_deps = [
        "//tensorflow/core:android_tensorflow_lib",
    ],
    common_deps = [
        ":ops",
        ":scope",
    ],
    deps = [
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
    ],
)

tf_cc_test(
    name = "ops_const_op_test",
    srcs = ["ops/const_op_test.cc"],
    deps = [
        ":const_op",
        "//tensorflow/core:framework",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
    ],
)

cc_library_with_android_deps(
    name = "while_loop",
    srcs = ["ops/while_loop.cc"],
    hdrs = ["ops/while_loop.h"],
    android_deps = [
        "//tensorflow/core:android_tensorflow_lib",
    ],
    common_deps = [
        ":cc_ops",
        ":cc_ops_internal",
        ":ops",
        ":scope",
        ":scope_internal",
    ],
    deps = [
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
    ],
)

tf_cc_test(
    name = "ops_while_loop_test",
    size = "small",
    srcs = ["ops/while_loop_test.cc"],
    deps = [
        ":cc_ops",
        ":client_session",
        ":testutil",
        ":while_loop",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
    ],
)

cc_library(
    name = "grad_op_registry",
    srcs = ["framework/grad_op_registry.cc"],
    hdrs = ["framework/grad_op_registry.h"],
    deps = [
        ":ops",
        ":scope",
    ],
)

cc_library(
    name = "array_grad",
    srcs = ["gradients/array_grad.cc"],
    deps = [
        ":cc_ops",
        ":cc_ops_internal",
        ":grad_op_registry",
        ":gradients",
        "//tensorflow/core:lib_proto_parsing",
    ],
    alwayslink = 1,
)

tf_cc_test(
    name = "gradients_array_grad_test",
    srcs = ["gradients/array_grad_test.cc"],
    deps = [
        ":array_grad",
        ":cc_ops",
        ":cc_ops_internal",
        ":grad_op_registry",
        ":grad_testutil",
        ":gradient_checker",
        ":testutil",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
    ],
)

cc_library(
    name = "math_grad",
    srcs = ["gradients/math_grad.cc"],
    deps = [
        ":cc_ops",
        ":cc_ops_internal",
        ":grad_op_registry",
        ":gradients",
    ],
    alwayslink = 1,
)

tf_cc_test(
    name = "gradients_math_grad_test",
    srcs = ["gradients/math_grad_test.cc"],
    deps = [
        ":cc_ops",
        ":grad_op_registry",
        ":grad_testutil",
        ":gradient_checker",
        ":math_grad",
        ":testutil",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
    ],
)

cc_library(
    name = "nn_grad",
    srcs = ["gradients/nn_grad.cc"],
    deps = [
        ":cc_ops",
        ":cc_ops_internal",
        ":grad_op_registry",
        ":gradients",
    ],
    alwayslink = 1,
)

tf_cc_test(
    name = "gradients_nn_grad_test",
    srcs = ["gradients/nn_grad_test.cc"],
    deps = [
        ":cc_ops",
        ":grad_op_registry",
        ":grad_testutil",
        ":gradient_checker",
        ":nn_grad",
        ":testutil",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
    ],
)

cc_library(
    name = "data_flow_grad",
    srcs = ["gradients/data_flow_grad.cc"],
    deps = [
        ":cc_ops",
        ":cc_ops_internal",
        ":grad_op_registry",
        ":gradients",
    ],
    alwayslink = 1,
)

tf_cc_test(
    name = "gradients_data_flow_grad_test",
    size = "small",
    srcs = ["gradients/data_flow_grad_test.cc"],
    deps = [
        ":cc_ops",
        ":data_flow_grad",
        ":grad_op_registry",
        ":grad_testutil",
        ":gradient_checker",
        ":testutil",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
    ],
)

tf_gen_op_wrappers_cc(
    name = "cc_ops",
    api_def_srcs = ["//tensorflow/core/api_def:base_api_def"],
    op_lib_names = [
        "array_ops",
        "audio_ops",
        "candidate_sampling_ops",
        "control_flow_ops",
        "data_flow_ops",
        "image_ops",
        "io_ops",
        "linalg_ops",
        "logging_ops",
        "lookup_ops",
        "math_ops",
        "nn_ops",
        "no_op",
        "parsing_ops",
        "random_ops",
        "sparse_ops",
        "state_ops",
        "string_ops",
        "training_ops",
        "user_ops",
    ],
    other_hdrs = [
        "ops/const_op.h",
        "ops/standard_ops.h",
    ],
    pkg = "//tensorflow/core",
)

tf_cc_test(
    name = "framework_cc_ops_test",
    srcs = ["framework/cc_ops_test.cc"],
    deps = [
        ":cc_ops",
        ":client_session",
        ":test_op",
        ":test_op_op_lib",
        ":testutil",
        "//tensorflow/core:framework",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
    ],
)

tf_gen_op_wrappers_cc(
    name = "sendrecv_ops",
    include_internal_ops = 1,
    op_lib_names = [
        "sendrecv_ops",
    ],
    pkg = "//tensorflow/core",
)

tf_gen_op_wrappers_cc(
    name = "function_ops",
    include_internal_ops = 1,
    op_lib_names = [
        "function_ops",
    ],
    pkg = "//tensorflow/core",
    visibility = ["//tensorflow:internal"],
)

tf_gen_op_wrappers_cc(
    name = "functional_ops",
    include_internal_ops = 1,
    op_lib_names = [
        "functional_ops",
    ],
    pkg = "//tensorflow/core",
    visibility = ["//tensorflow:internal"],
)

tf_gen_op_wrappers_cc(
    name = "resource_variable_ops",
    include_internal_ops = 1,
    op_lib_names = [
        "resource_variable_ops",
    ],
    pkg = "//tensorflow/core",
    visibility = ["//tensorflow:internal"],
)

tf_gen_op_wrappers_cc(
    name = "remote_fused_graph_ops",
    op_lib_names = [
        "remote_fused_graph_ops",
    ],
    pkg = "//tensorflow/core",
)

cc_library_with_android_deps(
    name = "cc_op_gen_main",
    srcs = [
        "framework/cc_op_gen.cc",
        "framework/cc_op_gen.h",
        "framework/cc_op_gen_main.cc",
    ],
    android_deps = [
        "//tensorflow/core:android_tensorflow_lib",
    ],
    copts = tf_copts(),
    data = [
        "//tensorflow/core/api_def:base_api_def",
    ],
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:op_gen_lib",
        "//tensorflow/core:proto_text",
        "//tensorflow/core:protos_all_cc",
    ],
)

tf_cc_test(
    name = "cc_op_gen_test",
    srcs = [
        "framework/cc_op_gen.cc",
        "framework/cc_op_gen.h",
        "framework/cc_op_gen_test.cc",
    ],
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:op_gen_lib",
        "//tensorflow/core:proto_text",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
    ],
)

cc_library(
    name = "test_op_op_lib",
    srcs = ["framework/test_op.cc"],
    linkstatic = 1,
    deps = ["//tensorflow/core:framework"],
    alwayslink = 1,
)

cc_library(
    name = "testutil",
    testonly = 1,
    srcs = ["framework/testutil.cc"],
    hdrs = ["framework/testutil.h"],
    deps = [
        ":client_session",
        ":ops",
        ":scope",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:tensorflow",
        "//tensorflow/core:testlib",
    ],
)

tf_gen_op_wrappers_cc(
    name = "test_op",
    op_lib_names = [
        "test_op",
    ],
)

tf_cc_binary(
    name = "tutorials_example_trainer",
    srcs = ["tutorials/example_trainer.cc"],
    copts = tf_copts(),
    linkopts = select({
        "//tensorflow:windows": [],
        "//tensorflow:windows_msvc": [],
        "//tensorflow:darwin": [
            "-lm",
            "-lpthread",
        ],
        "//tensorflow:ios": [
            "-lm",
            "-lpthread",
        ],
        "//conditions:default": [
            "-lm",
            "-lpthread",
            "-lrt",
        ],
    }),
    deps = [
        ":cc_ops",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:tensorflow",
    ],
)

filegroup(
    name = "all_files",
    srcs = glob(
        ["**/*"],
        exclude = [
            "**/METADATA",
            "**/OWNERS",
        ],
    ),
    visibility = ["//tensorflow:__subpackages__"],
)

cc_library(
    name = "queue_runner",
    srcs = ["training/queue_runner.cc"],
    hdrs = ["training/queue_runner.h"],
    deps = [
        ":coordinator",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/kernels:ops_util",
    ],
)

tf_cc_test(
    name = "queue_runner_test",
    srcs = ["training/queue_runner_test.cc"],
    deps = [
        "coordinator",
        ":cc_ops",
        ":queue_runner",
        ":scope",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:tensorflow",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
    ],
)

cc_library(
    name = "coordinator",
    srcs = ["training/coordinator.cc"],
    hdrs = ["training/coordinator.h"],
    deps = [
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:tensorflow",
    ],
)

tf_cc_test(
    name = "coordinator_test",
    srcs = ["training/coordinator_test.cc"],
    deps = [
        ":cc_ops",
        ":coordinator",
        ":queue_runner",
        ":scope",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:tensorflow",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
    ],
)
