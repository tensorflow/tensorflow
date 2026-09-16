load("@rules_cc//cc:cc_library.bzl", "cc_library")

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
    name = "conversion",
    srcs = ["conversion.cc"],
    hdrs = ["conversion.h"],
    deps = [
        "//xla:comparison_util",
        "//xla:literal",
        "//xla:literal_util",
        "//xla:shape_util",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/service/llvm_ir:llvm_util",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:SideEffectInterfaces",
        "@llvm-project//mlir:Support",
        "@tensor_ir//:NVTensorIRDialect",
    ],
)

cc_library(
    name = "support",
    srcs = ["support.cc"],
    hdrs = ["support.h"],
    deps = [
        "//xla:comparison_util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/service:decision",
        "@com_google_absl//absl/strings",
    ],
)
