load("//tensorflow:tensorflow.bzl", "tf_cc_test")
load("//tensorflow/core/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [
        "//tensorflow/compiler/mlir/tf2xla/internal:__subpackages__",
    ],
    licenses = ["notice"],
)

cc_library(
    name = "dialect_detection_utils",
    srcs = [
        "dialect_detection_utils.cc",
    ],
    hdrs = [
        "dialect_detection_utils.h",
    ],
    deps = [
        "//tensorflow/compiler/mlir/tensorflow",
        "//tensorflow/core:framework",
        "//tensorflow/core/transforms/toposort:Pass",
        "@com_google_absl//absl/strings",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:Transforms",
    ],
)

tf_cc_test(
    name = "dialect_detection_utils_test",
    srcs = ["dialect_detection_utils_test.cc"],
    deps = [
        ":dialect_detection_utils",
        "//tensorflow/compiler/mlir/tensorflow:tensorflow_types",
        "@com_google_googletest//:gtest_main",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@stablehlo//:chlo_ops",
    ],
)

cc_library(
    name = "test_metadata_config",
    testonly = True,
    srcs = ["test_metadata_config.cc"],
    hdrs = ["test_metadata_config.h"],
    visibility = [
        "//tensorflow/compiler/mlir/tf2xla/api:__subpackages__",
    ],
    deps = [
        "//tensorflow/compiler/jit",
        "//tensorflow/compiler/jit:xla_tpu_device",
        "//tensorflow/compiler/mlir:register_common_dialects",
        "//tensorflow/compiler/mlir/tensorflow:convert_type",
        "//tensorflow/compiler/mlir/tensorflow:serialize_mlir_module_utils",
        "//tensorflow/compiler/tf2xla:layout_util",
        "//tensorflow/compiler/tf2xla:xla_compiler",
        "//tensorflow/compiler/tf2xla:xla_helpers",
        "//tensorflow/compiler/tf2xla:xla_tpu_backend_registration",
        "//tensorflow/compiler/tf2xla/kernels:xla_ops",
        "//tensorflow/core:framework",
        "//tensorflow/core:test_main",
        "//tensorflow/core/protobuf/tpu:compile_metadata_proto_cc",
        "//tensorflow/core/tpu/kernels/xla:host_compute_ops",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:string_view",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@local_xla//xla:shape_util",
        "@local_xla//xla/hlo/translate/mhlo_to_hlo:type_to_shape",
        "@local_xla//xla/mlir_hlo:hlo_dialect_registration",
        "@local_xla//xla/tsl/platform:errors",
    ],
)
