load("@bazel_skylib//rules:common_settings.bzl", "bool_flag")
load("//tensorflow:tensorflow.bzl", "tf_cc_test")
load("//tensorflow:tensorflow.default.bzl", "filegroup")
load("//tensorflow/core/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

bool_flag(
    name = "disable_mlir",
    build_setting_default = False,
)

config_setting(
    name = "disable_mlir_config",
    flag_values = {":disable_mlir": "True"},
    visibility = ["//visibility:public"],
)

cc_library(
    name = "mlir",
    srcs = ["mlir.cc"],
    hdrs = ["mlir.h"],
    deps = [
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "//tensorflow/cc/saved_model:bundle_v2",
        "//tensorflow/cc/saved_model:loader",
        "//tensorflow/compiler/mlir/tensorflow/translate/tools:parsers",
        "//tensorflow/compiler/mlir/quantization/stablehlo:bridge_passes",
        "@com_google_absl//absl/strings",
        "@llvm-project//mlir:FuncExtensions",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:AllPassesAndDialects",
        "@llvm-project//mlir:BytecodeWriter",
        "@llvm-project//mlir:ShapeDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Parser",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:RegisterAllPasses",
        "//tensorflow/core/common_runtime:graph_constructor",
        "@llvm-project//mlir:Support",
        "@stablehlo//:register",
        "//tensorflow/c:tf_status",
        "//tensorflow/c:tf_status_helper",
        "//tensorflow/compiler/mlir/tf2xla/api/v2:graph_to_tf_executor",
        "//tensorflow/c/eager:c_api",
        "//tensorflow/c/eager:tfe_context_internal",
        "//tensorflow/compiler/mlir/tensorflow:mlir_import_options",
        "@local_xla//xla/mlir/framework/transforms:passes",
        "@local_xla//xla/mlir_hlo:all_passes",
        "//tensorflow/compiler/mlir/tensorflow",
        "//tensorflow/compiler/mlir/tensorflow:import_model",
        "//tensorflow/compiler/mlir/tensorflow:error_util",
        "//tensorflow/compiler/mlir/tensorflow:import_utils",
        "//tensorflow/compiler/mlir/tensorflow:mlir_roundtrip_flags",
        "//tensorflow/compiler/mlir/tensorflow:mlprogram_util",
        "//tensorflow/compiler/mlir/tensorflow/transforms:tensorflow_passes",
        "//tensorflow/compiler/mlir/tensorflow/transforms:tf_saved_model_passes",
        "//tensorflow/compiler/mlir/tensorflow:translate_lib",
        "//tensorflow/compiler/mlir/tf2xla/transforms:xla_legalize_tf",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib_proto_parsing",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:tflite_portable_logging",
        "//tensorflow/core/common_runtime/eager:context",
        # (yongtang) The graph_optimization_pass_registration needs to be part
        # of a shared object that will be loaded whenever `import tensorflow`
        # is run. The natural place is libtensorflow_framework.so.
        # While adding graph_optimization_pass_registration to
        # libtensorflow_framework.so is possible with some modification in
        # dependency, many tests will fail due to multiple copies of LLVM.
        # See https://github.com/tensorflow/tensorflow/pull/39231 for details.
        # Alternatively, we place graph_optimization_pass_registration here
        # because:
        # - tensorflow/python/_pywrap_mlir.so already depends on LLVM anyway
        # - tensorflow/python/_pywrap_mlir.so always loaded as part of python
        #   binding
        # TODO: It might be still preferrable to place graph_optimization_pass
        # as part of the libtensorflow_framework.so, as it is the central
        # place for core related components.
        "//tensorflow/compiler/mlir/tensorflow/transforms:graph_optimization_pass_registration",  # buildcleaner: keep
    ],
    alwayslink = 1,
)

tf_cc_test(
    name = "mlir_test",
    srcs = ["mlir_test.cc"],
    deps = [
        ":mlir",
        "//tensorflow/c:safe_ptr",
        "//tensorflow/c:tf_status_headers",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest_main",
    ],
)

filegroup(
    name = "pywrap_mlir_hdrs",
    srcs = [
        "mlir.h",
    ],
    visibility = [
        "//tensorflow/python:__pkg__",
    ],
)
