load("//xla:lit.bzl", "lit_test_suite")
load("//xla:xla.default.bzl", "xla_cc_binary")
load("//xla/tsl:tsl.bzl", "if_google")
load("//xla/tsl/platform/default:build_config_root.bzl", "tf_gpu_tests_tags")
load("//xla/tsl/platform/default:cuda_build_defs.bzl", "if_cuda_is_configured")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    licenses = ["notice"],
)

xla_cc_binary(
    name = "ptx_opt",
    srcs = ["ptx_opt.cc"],
    # We want to use this tool for lit tests. Due to hermetic cuda, we need to
    # set linkopts in such a way that dynamic libraries are found, which are
    # symlinked from the lit_lib directory.
    linkopts = ["-Wl,-rpath,$$ORIGIN/../lit_lib"],
    tags = [
        "cuda-only",
        "gpu",
    ],
    deps = [
        "//xla:debug_options_flags",
        "//xla:xla_proto_cc",
        "//xla/service/gpu/llvm_gpu_backend:load_ir_module",
        "//xla/service/gpu/llvm_gpu_backend:nvptx_backend",
        "//xla/stream_executor:device_description",
        "//xla/stream_executor/cuda:cuda_compute_capability",
        "//xla/tsl/util:command_line_flags",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@llvm-project//llvm:Core",
        "@llvm-project//llvm:NVPTXCodeGen",  # buildcleaner: keep
        "@llvm-project//llvm:ObjCARC",  # buildcleaner: keep
        "@tsl//tsl/platform:platform_port",
    ] + if_cuda_is_configured([
        "//xla/stream_executor/cuda:cuda_platform",
    ]),
)

lit_test_suite(
    name = "ptx_opt_tests",
    # It fails in OSS to load the module correctly.
    srcs = if_google(
        glob(["**/*.ll"]),
        [],
    ),
    args = if_cuda_is_configured([
        "--param=PTX=PTX",
        "--param=GPU=a100_pcie_80",
    ]),
    cfg = "//xla:lit.cfg.py",
    default_tags = tf_gpu_tests_tags(),
    hermetic_cuda_data_dir = "%S/../../../../../cuda_nvcc",
    tags = [
        "cuda-only",
        "gpu",
    ],
    tools = [
        ":ptx_opt",
        "@llvm-project//llvm:FileCheck",
    ],
)
