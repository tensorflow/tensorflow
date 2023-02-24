load("//tensorflow/compiler/xla:xla.bzl", "xla_cc_test")
load(
    "@local_config_rocm//rocm:build_defs.bzl",
    "if_rocm_is_configured",
)

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [":friends"],
    licenses = ["notice"],
)

package_group(
    name = "friends",
    includes = [
        "//tensorflow/compiler/xla:friends",
    ],
)

cc_library(
    name = "llvm_gpu_backend",
    srcs = [
        "gpu_backend_lib.cc",
        "utils.cc",
    ],
    hdrs = [
        "gpu_backend_lib.h",
        "utils.h",
    ],
    deps = [
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "//tensorflow/tsl/platform:env",
        "//tensorflow/tsl/platform:logging",
        "@llvm-project//llvm:Analysis",
        "@llvm-project//llvm:BitReader",
        "@llvm-project//llvm:BitWriter",
        "@llvm-project//llvm:CodeGen",
        "@llvm-project//llvm:Core",
        "@llvm-project//llvm:IPO",
        "@llvm-project//llvm:IRReader",
        "@llvm-project//llvm:MC",
        "@llvm-project//llvm:Linker",
        "@llvm-project//llvm:Passes",
        "@llvm-project//llvm:NVPTXCodeGen",  # buildcleaner: keep
        "@llvm-project//llvm:ObjCARC",  # buildcleaner: keep
        "@llvm-project//llvm:Scalar",
        "@llvm-project//llvm:Support",
        "@llvm-project//llvm:Target",
        "//tensorflow/compiler/xla/service/gpu:metrics",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/tsl/platform:cuda_libdevice_path",
        "//tensorflow/compiler/xla/service:hlo_module_config",
        "//tensorflow/compiler/xla/service/gpu:gpu_types",
        "//tensorflow/compiler/xla/service/llvm_ir:llvm_command_line_options",
        "//tensorflow/compiler/xla/service/llvm_ir:llvm_type_conversion_util",
        "//tensorflow/tsl/platform:random",
        "//tensorflow/tsl/profiler/lib:traceme",
        "//tensorflow/tsl/platform:path",
        "//tensorflow/tsl/util:env_var",
    ] + if_rocm_is_configured([
        "@local_config_rocm//rocm:rocm_headers",
        "@llvm-project//llvm:AMDGPUCodeGen",
    ]),
)

xla_cc_test(
    name = "utils_test",
    size = "small",
    srcs = ["utils_test.cc"],
    data = [
        "tests_data/saxpy.ll",
    ],
    deps = [
        ":llvm_gpu_backend",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
        "//tensorflow/tsl/platform:logging",
        "//tensorflow/tsl/platform:path",
        "//tensorflow/tsl/platform:resource_loader",
        "//tensorflow/tsl/platform:test",
        "@llvm-project//llvm:Core",
        "@llvm-project//llvm:Support",
    ],
)
