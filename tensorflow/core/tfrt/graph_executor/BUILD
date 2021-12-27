package(
    default_visibility = [":friends"],
    licenses = ["notice"],
)

package_group(
    name = "friends",
    packages = [
        "//tensorflow/core/tfrt/saved_model/...",
    ],
)

cc_library(
    name = "graph_execution_options",
    hdrs = ["graph_execution_options.h"],
    deps = [
        "//tensorflow/compiler/mlir/tfrt:tfrt_compile_options",
        "//tensorflow/core/protobuf:for_core_protos_cc",
        "//tensorflow/core/tfrt/runtime",
        "@com_google_absl//absl/types:optional",
    ],
)
