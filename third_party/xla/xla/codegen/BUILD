load("//xla/tsl/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [":friends"],
    licenses = ["notice"],
)

package_group(
    name = "friends",
    includes = [
        "//xla:friends",
    ],
)

cc_library(
    name = "kernel_emitter",
    hdrs = ["kernel_emitter.h"],
    deps = [
        ":kernel_spec",
        "@com_google_absl//absl/status:statusor",
    ],
)

cc_library(
    name = "kernel_spec",
    srcs = ["kernel_spec.cc"],
    hdrs = ["kernel_spec.h"],
    deps = [
        "//xla/runtime:buffer_use",
        "//xla/stream_executor:launch_dim",
        "@com_google_absl//absl/container:inlined_vector",
    ],
)

cc_library(
    name = "llvm_ir_kernel_source",
    srcs = ["llvm_ir_kernel_source.cc"],
    hdrs = ["llvm_ir_kernel_source.h"],
    deps = [
        ":kernel_spec",
        "//xla/service/llvm_ir:llvm_util",
        "@llvm-project//llvm:Core",
        "@llvm-project//llvm:JITLink",
    ],
)
