package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [
        # copybara:uncomment "//learning/brain/experimental/tfrt/native_lowering:__subpackages__",
        # copybara:uncomment "//learning/brain/tfrt/native_lowering:__subpackages__",
        "//tensorflow/compiler/mlir/tfrt/translate/mlrt:__subpackages__",
        "//tensorflow/core/tfrt:__subpackages__",
    ],
    licenses = ["notice"],
)

cc_library(
    name = "tfrt_native_lowering_stub",
    srcs = ["tfrt_native_lowering_stub.cc"],
    hdrs = ["tfrt_native_lowering_stub.h"],
    deps = [
        "//tensorflow/core/tfrt/graph_executor:executable_context",
        "//tensorflow/core/tfrt/graph_executor:sync_resource_state",
        "//tensorflow/core/tfrt/mlrt/bytecode:executable",
        "//tensorflow/core/tfrt/mlrt/interpreter:context",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@tf_runtime//:hostcontext",
    ],
)
