licenses(["notice"])  # Apache 2.0

package(
    default_visibility = [":friends"],
)

package_group(
    name = "friends",
    includes = [
        "//tensorflow/compiler/xla:friends",
    ],
)

load("//tensorflow:tensorflow.bzl", "tf_cc_test")

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
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla/service:hlo_module_config",
        "//tensorflow/compiler/xla/service/llvm_ir:llvm_util",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
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
        "@llvm//:transform_utils",
    ],
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
