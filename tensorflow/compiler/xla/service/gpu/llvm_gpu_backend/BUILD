licenses(["notice"])  # Apache 2.0

package(
    default_visibility = [":friends"],
    features = [
        "-parse_headers",
        "no_layering_check",
    ],
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
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla/legacy_flags:gpu_backend_lib_flags",
        "//tensorflow/compiler/xla/service:hlo_module_config",
        "//tensorflow/compiler/xla/service/llvm_ir:llvm_util",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "@llvm//:analysis",
        "@llvm//:asm_printer",
        "@llvm//:bit_reader",
        "@llvm//:bit_writer",
        "@llvm//:code_gen",
        "@llvm//:core",
        "@llvm//:instrumentation",
        "@llvm//:ipo",
        "@llvm//:ir_reader",
        "@llvm//:linker",
        "@llvm//:mc",
        "@llvm//:nvptx_code_gen",
        "@llvm//:objc_arc",
        "@llvm//:support",
        "@llvm//:target",
        "@llvm//:transform_utils",
    ],
)

cc_test(
    name = "utils_test",
    size = "small",
    srcs = ["utils_test.cc"],
    data = [
        "tests_data/saxpy.ll",
    ],
    deps = [
        ":llvm_gpu_backend",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "@llvm//:core",
        "@llvm//:support",
    ],
)

# -----------------------------------------------------------------------------

filegroup(
    name = "all_files",
    srcs = glob(
        ["**/*"],
        exclude = [
            "**/METADATA",
            "**/OWNERS",
        ],
    ),
    visibility = ["//tensorflow:__subpackages__"],
)
