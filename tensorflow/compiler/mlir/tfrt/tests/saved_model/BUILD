load("//tensorflow:tensorflow.bzl", "tf_cc_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    licenses = ["notice"],
)

tf_cc_test(
    name = "saved_model_test",
    srcs = ["saved_model_test.cc"],
    data = [
        "testdata/test.mlir",
        "testdata/xla_launch.mlir",
        "testdata/xla_launch_xla_reduce_window.mlir",
    ],
    tags = ["no_oss"],
    deps = [
        "//tensorflow/compiler/mlir/tensorflow",
        "//tensorflow/compiler/mlir/tfrt:import_model",
        "//tensorflow/compiler/mlir/tfrt:saved_model",
        "//tensorflow/compiler/mlir/tfrt:tfrt_compile_options",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core/platform:resource_loader",
        "//tensorflow/core/tfrt/fallback:fallback_state",
        "//tensorflow/core/tfrt/graph_executor:graph_execution_options",
        "//tensorflow/core/tfrt/runtime",
        "@com_google_googletest//:gtest_main",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Parser",
        "@local_xla//xla/tsl/platform:statusor",
        "@tf_runtime//:bef",
        "@tf_runtime//:hostcontext",
    ],
)

filegroup(
    name = "testdata",
    srcs = glob(
        ["testdata/**"],
    ),
    visibility = ["//visibility:public"],
)
