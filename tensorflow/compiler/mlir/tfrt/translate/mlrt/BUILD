load(
    "//tensorflow:tensorflow.bzl",
    "tf_cc_test",
)

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [
        # copybara:uncomment "//learning/brain/experimental/tfrt:__subpackages__",
        # copybara:uncomment "//learning/infra/mira/distributed:__subpackages__",
        # copybara:uncomment "//smartass/brain/ops/tfrt_kernels:__subpackages__",
        "//tensorflow/compiler/mlir/tfrt/transforms/mlrt:__subpackages__",
        "//tensorflow/core/tfrt:__subpackages__",
    ],
)

cc_library(
    name = "mlir_to_bytecode",
    srcs = ["mlir_to_bytecode.cc"],
    hdrs = ["mlir_to_bytecode.h"],
    deps = [
        "//tensorflow/core/tfrt/mlrt/bytecode",
        "//tensorflow/core/tfrt/mlrt/bytecode:executable",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status:statusor",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
    ],
)

tf_cc_test(
    name = "mlir_to_bytecode_test",
    srcs = ["mlir_to_bytecode_test.cc"],
    data = glob(["testdata/**"]),
    deps = [
        ":mlir_to_bytecode",
        "//tensorflow/core/tfrt/mlrt/bytecode:executable",
        "//tensorflow/core/tfrt/mlrt/interpreter:attribute_span",
        "//tensorflow/tsl/platform:resource_loader",
        "@com_google_googletest//:gtest_main",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:Parser",
    ],
)

cc_library(
    name = "test_utils",
    testonly = 1,
    srcs = ["test_utils.cc"],
    hdrs = ["test_utils.h"],
    deps = [
        "//learning/brain/experimental/tfrt/native_lowering/kernels:sync_context",
        "//tensorflow/core:core_cpu_base",
        "//tensorflow/core:framework",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/tfrt/mlrt/attribute",
        "//tensorflow/core/tfrt/mlrt/bytecode",
        "//tensorflow/core/tfrt/mlrt/bytecode:kernel",
        "//tensorflow/core/tfrt/mlrt/interpreter:context",
        "//tensorflow/core/tfrt/mlrt/interpreter:interpreter_testutil",
        "//tensorflow/core/tfrt/mlrt/interpreter:value",
        "//tensorflow/core/tfrt/utils:tensor_util",
        "//tensorflow/tsl/platform:errors",
        "@com_google_absl//absl/status:statusor",
        "@tf_runtime//:hostcontext",
        "@tf_runtime//:tensor",
    ],
)
