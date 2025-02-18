# Description:
#   Default implementations for llvm_gpu_backend

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
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/synchronization",
        "@local_tsl//tsl/platform:cuda_root_path",
        "@local_tsl//tsl/platform:env",
        "@local_tsl//tsl/platform:logging",
        "@local_tsl//tsl/platform:path",
    ],
    alwayslink = True,
)

cc_library(
    name = "load_ir_module",
    srcs = ["load_ir_module.cc"],
    deps = [
        "@llvm-project//llvm:IRReader",
        "@llvm-project//llvm:Support",
        "@llvm-project//llvm:ir_headers",
        "@local_tsl//tsl/platform:logging",
    ],
    alwayslink = True,
)
