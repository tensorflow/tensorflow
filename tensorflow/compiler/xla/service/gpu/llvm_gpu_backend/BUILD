load("//tensorflow:tensorflow.bzl", "tf_cc_test")

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
        "//tensorflow/compiler/xla/service/gpu:gpu_types",
        "//tensorflow/compiler/xla/service/llvm_ir:llvm_util",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core/profiler/lib:traceme",
        "//third_party/llvm/llvm:analysis",
        "//third_party/llvm/llvm:bit_reader",
        "//third_party/llvm/llvm:bit_writer",
        "//third_party/llvm/llvm:codegen",
        "//third_party/llvm/llvm:ipo_transforms",
        "//third_party/llvm/llvm:ir",
        "//third_party/llvm/llvm:ir_reader",
        "//third_party/llvm/llvm:linker",
        "//third_party/llvm/llvm:nvptx_target",  # buildcleaner: keep
        "//third_party/llvm/llvm:objcarc_transforms",  # buildcleaner: keep
        "//third_party/llvm/llvm:scalar_transforms",
        "//third_party/llvm/llvm:support",
        "//third_party/llvm/llvm:target_base",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@llvm//:amdgpu_code_gen",
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
        "//third_party/llvm/llvm:ir",
        "//third_party/llvm/llvm:support",
    ],
)
