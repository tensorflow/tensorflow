# Description: GPU-specific XLA tests. For example, codegen tests that
# verify the IR emitted.
#
# TODO(jlebar): None of these tests actually use the GPU, so they should not
# need to run on machines with GPUs present.

licenses(["notice"])  # Apache 2.0

package(default_visibility = [":friends"])

package_group(
    name = "friends",
    includes = [
        "//tensorflow/compiler/xla:friends",
    ],
)

# Filegroup used to collect source files for dependency checking.
filegroup(
    name = "c_srcs",
    data = glob([
        "**/*.cc",
        "**/*.h",
    ]),
)

load("//tensorflow:tensorflow.bzl", "tf_cc_test")

cc_library(
    name = "gpu_codegen_test",
    testonly = True,
    srcs = ["gpu_codegen_test.cc"],
    hdrs = ["gpu_codegen_test.h"],
    tags = [
        "requires-gpu-sm35",
    ],
    deps = [
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla/legacy_flags:debug_options_flags",
        "//tensorflow/compiler/xla/service:gpu_plugin",
        "//tensorflow/compiler/xla/service/gpu:gpu_executable",
        "//tensorflow/compiler/xla/tests:filecheck",
        "//tensorflow/compiler/xla/tests:llvm_irgen_test_base",
        "//tensorflow/core:lib",
    ],
)

tf_cc_test(
    name = "gpu_copy_test",
    srcs = ["gpu_copy_test.cc"],
    tags = [
        "requires-gpu-sm35",
    ],
    deps = [
        ":gpu_codegen_test",
        "//tensorflow/compiler/xla:literal",
        "//tensorflow/compiler/xla:literal_util",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla:xla_data_proto",
        "//tensorflow/compiler/xla/service:hlo",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
    ],
)

tf_cc_test(
    name = "gpu_ftz_test",
    srcs = ["gpu_ftz_test.cc"],
    tags = [
        "requires-gpu-sm35",
    ],
    deps = [
        ":gpu_codegen_test",
        "//tensorflow/core:test_main",
    ],
)

tf_cc_test(
    name = "gpu_index_test",
    srcs = ["gpu_index_test.cc"],
    tags = [
        "requires-gpu-sm35",
    ],
    deps = [
        ":gpu_codegen_test",
        "//tensorflow/compiler/xla:literal",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla:xla_data_proto",
        "//tensorflow/compiler/xla:xla_proto",
        "//tensorflow/compiler/xla/service:hlo",
        "//tensorflow/compiler/xla/service:hlo_module_config",
        "//tensorflow/compiler/xla/service:hlo_parser",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
    ],
)

tf_cc_test(
    name = "gpu_infeed_test",
    srcs = ["infeed_test.cc"],
    tags = [
        "requires-gpu-sm35",
    ],
    deps = [
        ":gpu_codegen_test",
        "//tensorflow/compiler/xla:literal",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:test_helpers",
        "//tensorflow/compiler/xla:xla_data_proto",
        "//tensorflow/compiler/xla/client:global_data",
        "//tensorflow/compiler/xla/client:local_client",
        "//tensorflow/compiler/xla/client/lib:arithmetic",
        "//tensorflow/compiler/xla/client/xla_client:xla_builder",
        "//tensorflow/compiler/xla/tests:client_library_test_base",
        "//tensorflow/compiler/xla/tests:literal_test_util",
        "//tensorflow/core:lib",
        "//tensorflow/core:test_main",
    ],
)

tf_cc_test(
    name = "gpu_kernel_tiling_test",
    srcs = ["gpu_kernel_tiling_test.cc"],
    tags = [
        "requires-gpu-sm35",
    ],
    deps = [
        ":gpu_codegen_test",
        "//tensorflow/compiler/xla/service:hlo",
        "//tensorflow/compiler/xla/service:hlo_module_config",
        "//tensorflow/compiler/xla/service:hlo_parser",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
    ],
)

tf_cc_test(
    name = "gpu_ldg_test",
    srcs = ["gpu_ldg_test.cc"],
    tags = ["requires-gpu-sm35"],
    deps = [
        ":gpu_codegen_test",
        "//tensorflow/compiler/xla:literal",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla:xla_data_proto",
        "//tensorflow/compiler/xla/service:hlo",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
    ],
)

tf_cc_test(
    name = "gpu_noalias_test",
    srcs = ["gpu_noalias_test.cc"],
    tags = [
        "requires-gpu-sm35",
    ],
    deps = [
        ":gpu_codegen_test",
        "//tensorflow/compiler/xla:literal",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla:xla_data_proto",
        "//tensorflow/compiler/xla/service:hlo",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
    ],
)

tf_cc_test(
    name = "gpu_fusion_test",
    srcs = ["gpu_fusion_test.cc"],
    tags = [
        "requires-gpu-sm35",
    ],
    deps = [
        ":gpu_codegen_test",
        "//tensorflow/compiler/xla/service:hlo_module_config",
        "//tensorflow/compiler/xla/service:hlo_parser",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
    ],
)

tf_cc_test(
    name = "gpu_unrolling_test",
    srcs = ["gpu_unrolling_test.cc"],
    tags = [
        "requires-gpu-sm35",
    ],
    deps = [
        ":gpu_codegen_test",
        "//tensorflow/compiler/xla/service:hlo_module_config",
        "//tensorflow/compiler/xla/service:hlo_parser",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
    ],
)

tf_cc_test(
    name = "gpu_alignment_test",
    testonly = True,
    srcs = ["gpu_alignment_test.cc"],
    tags = [
        "requires-gpu-sm35",
    ],
    deps = [
        ":gpu_codegen_test",
        "//tensorflow/compiler/xla/service:gpu_plugin",
        "//tensorflow/compiler/xla/service/cpu:custom_call_target_registry",
        "//tensorflow/compiler/xla/service/llvm_ir:alias_analysis",
        "//tensorflow/compiler/xla/tests:filecheck",
        "//tensorflow/compiler/xla/tests:llvm_irgen_test_base",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
    ],
)
