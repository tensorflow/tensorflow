load("//xla:lit.bzl", "lit_test_suite")
load("//xla/tests:build_defs.bzl", "xla_test")
load("//xla/tsl:tsl.bzl", "if_google")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    licenses = ["notice"],
)

lit_test_suite(
    name = "lit_tests",
    srcs = glob(["**/*.hlo"]),
    cfg = "//xla:lit.cfg.py",
    default_tags = [
        "cuda-only",
        "gpu",
    ],
    tags = [
        "cuda-only",
        "gpu",
    ],
    tools = [
        "//xla/backends/gpu/codegen/tools:fusion_to_tensorir",
        "@llvm-project//llvm:FileCheck",
    ],
)

# Checks the shape of what the TensorIR emitter hands to the runtime. Compiles
# only, never launches, so unlike the numerics tests below it does not need a
# driver that can load a TensorIR kernel.
xla_test(
    name = "fusion_emitter_thunk_test",
    srcs = ["fusion_emitter_thunk_test.cc"],
    backend_tags = {
        # The emitter compiles for whatever device the test runs on, and
        # `IsSupportedComputeCapability` rejects anything before Hopper.
        "gpu": ["requires-gpu-sm90"],
    },
    backends = ["gpu"],
    tags = [
        "cuda-only",
        "no_mac",
    ],
    deps = [
        "//xla:xla_proto_cc",
        "//xla/backends/gpu/runtime:custom_kernel_thunk",
        "//xla/backends/gpu/runtime:thunk",
        "//xla/backends/gpu/runtime:thunk_proto_cc",
        "//xla/backends/gpu/tests:gpu_pjrt_codegen_test",
        "//xla/hlo/testlib:verified_hlo_module",
        "//xla/service:executable",
        "//xla/service/gpu:gpu_executable",
        "//xla/service/gpu:launch_dimensions",
        "//xla/stream_executor:launch_dim",
        "//xla/tests:xla_internal_test_main",  # fixdeps: keep
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest",
    ],
)

# End-to-end numerics tests for the TensorIR fusion emitter.
#
# This target is `manual` because it needs a newer toolchain than CI has, not
# because it fails. It passes on an H100 with a CUDA 13.3 toolkit.
#
# Why CUDA 13.3. The emitter asks the compiler for a cubin, which it assembles
# by running `tileiras` (found next to the toolkit's `nvcc`, in
# `--xla_gpu_cuda_data_dir`, or on `PATH`). CUDA 13.0 ships no `tileiras`, and
# the 13.1 and 13.2 releases lack the `--list-versions` probe TensorIR uses to
# check bytecode compatibility, so 13.3 is the first release the compiler can
# assemble with. Without an assembler the compiler hands back Tile IR bytecode
# for the driver to JIT. TensorIR emits the CUDA 13.3 bytecode format by
# default (`max(compatibility, 13.3)`, see the tensor_ir README), which needs
# the matching r610 or newer driver; the r590 (CUDA 13.1) JIT only takes the
# `compatibility` format, which the emitter does not request.
#
# Why Hopper. The compiler refuses to assemble a cubin below sm_90
# (`CudaTileCompileOptions::validateDerived`).
#
# Why the driver matters even for an assembled cubin. Loading a Tile IR cubin
# makes the driver dlopen() libnvidia-gpucomp.so; a driver install without it
# fails with CUDA_ERROR_JIT_COMPILER_NOT_FOUND.
#
# TODO: drop the `manual` tag once CI builds against CUDA 13.3 or newer.
xla_test(
    name = "fusion_emitter_device_test",
    srcs = ["fusion_emitter_device_test.cc"],
    backend_tags = {
        "gpu": ["requires-gpu-sm90"],
    },
    backends = ["gpu"],
    tags = [
        "cuda-only",
        "manual",
        "no_mac",
    ] + if_google(["notap"]),
    deps = [
        "//xla:error_spec",
        "//xla:literal",
        "//xla:literal_util",
        "//xla:xla_proto_cc",
        "//xla/backends/gpu/tests:gpu_pjrt_codegen_test",
        "//xla/hlo/testlib:verified_hlo_module",
        "//xla/tests:hlo_interpreter_reference_mixin",
        "//xla/tests:xla_internal_test_main",  # fixdeps: keep
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest",
    ] + if_google([
        # Forge drivers lack libnvidia-gpucomp.so and the compat library supplies it.
        "@local_config_cuda//cuda:cuda_compat_tileir",
    ]),
)
