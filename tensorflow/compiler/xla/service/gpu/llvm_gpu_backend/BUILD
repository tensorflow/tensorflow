load("//tensorflow:tensorflow.bzl", "tf_cc_test")
load("@local_config_cuda//cuda:build_defs.bzl", "if_cuda_is_configured")
load("@local_config_rocm//rocm:build_defs.bzl", "if_rocm_is_configured")


package(
    default_visibility = [":friends"],
    licenses = ["notice"],  # Apache 2.0
)

package_group(
    name = "friends",
    includes = [
        "//tensorflow/compiler/xla:friends",
    ],
)

LLVM_DEPS =  [
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla/service:hlo_module_config",
        "//tensorflow/compiler/xla/service/llvm_ir:llvm_util",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core/profiler/lib:traceme",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@llvm//:amdgpu_code_gen",
        "@llvm//:analysis",
        "@llvm//:bit_reader",
        "@llvm//:bit_writer",
        "@llvm//:code_gen",
        "@llvm//:core",
        "@llvm//:ipo",
        "@llvm//:ir_reader",
        "@llvm//:linker",
        "@llvm//:nvptx_code_gen",  # buildcleaner: keep
        "@llvm//:objc_arc",  # buildcleaner: keep
        "@llvm//:scalar",
        "@llvm//:support",
        "@llvm//:target",
]

cc_library(
    name = "nvptx_llvm_backend",
    srcs = [
        "dump_ir_pass.cc",
        "utils.cc",
        "nvptx_backend_lib.cc",
    ],
    hdrs = [
        "dump_ir_pass.h",
        "utils.h",
        "nvptx_backend_lib.h",
    ],
    deps = LLVM_DEPS,
)

cc_library(
    name = "amdgpu_llvm_backend",
    srcs = [
        "dump_ir_pass.cc",
        "utils.cc",
        "amdgpu_backend_lib.cc",
    ],
    hdrs = [
        "dump_ir_pass.h",
        "utils.h",
        "amdgpu_backend_lib.h",
    ],
    deps = LLVM_DEPS,
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
        "@llvm//:core",
        "@llvm//:support",
    ],
)
