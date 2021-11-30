load("//tensorflow:tensorflow.bzl", "tf_cc_test")
load(
    "@local_config_rocm//rocm:build_defs.bzl",
    "if_rocm_is_configured",
)

package(
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
        "dump_ir_pass.cc",
        "gpu_backend_lib.cc",
        "utils.cc",
    ],
    hdrs = [
        "dump_ir_pass.h",
        "gpu_backend_lib.h",
        "utils.h",
    ],
    deps = [
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@llvm-project//llvm:Analysis",
        "@llvm-project//llvm:BitReader",
        "@llvm-project//llvm:BitWriter",
        "@llvm-project//llvm:CodeGen",
        "@llvm-project//llvm:Core",
        "@llvm-project//llvm:IPO",
        "@llvm-project//llvm:IRReader",
        "@llvm-project//llvm:MC",
        "@llvm-project//llvm:Linker",
        "@llvm-project//llvm:NVPTXCodeGen",  # buildcleaner: keep
        "@llvm-project//llvm:ObjCARC",  # buildcleaner: keep
        "@llvm-project//llvm:Scalar",
        "@llvm-project//llvm:Support",
        "@llvm-project//llvm:Target",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla/service:hlo_module_config",
        "//tensorflow/compiler/xla/service/gpu:gpu_types",
        "//tensorflow/compiler/xla/service/llvm_ir:llvm_command_line_options",
        "//tensorflow/compiler/xla/service/llvm_ir:llvm_type_conversion_util",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core/profiler/lib:traceme",
    ] + if_rocm_is_configured([
        "@local_config_rocm//rocm:rocm_headers",
        "@llvm-project//llvm:AMDGPUCodeGen",
    ]),
)

tf_cc_test(
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
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "//tensorflow/core/platform:resource_loader",
        "@llvm-project//llvm:Core",
        "@llvm-project//llvm:Support",
    ],
)
