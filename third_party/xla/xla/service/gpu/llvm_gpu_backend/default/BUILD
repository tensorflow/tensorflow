# Description:
#   Default implementations for llvm_gpu_backend

load("@rules_cc//cc:cc_library.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//xla/service/gpu/llvm_gpu_backend:__pkg__"],
    licenses = ["notice"],
)

cc_library(
    name = "nvptx_libdevice_path",
    srcs = ["nvptx_libdevice_path.cc"],
    deps = [
        "//xla/service/gpu/llvm_gpu_backend:nvptx_utils",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/synchronization",
        "@tsl//tsl/platform:cuda_root_path",
        "@tsl//tsl/platform:env",
        "@tsl//tsl/platform:logging",
        "@tsl//tsl/platform:path",
    ],
    alwayslink = True,
)

cc_library(
    name = "load_ir_module",
    srcs = ["load_ir_module.cc"],
    deps = [
        "@com_google_absl//absl/log",
        "@llvm-project//llvm:IRReader",
        "@llvm-project//llvm:Support",
        "@llvm-project//llvm:ir_headers",
        "@tsl//tsl/platform:logging",
    ],
    alwayslink = True,
)
