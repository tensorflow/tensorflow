load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("//xla/tsl:tsl.bzl", "if_google")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [":friends"],
    licenses = ["notice"],
)

package_group(
    name = "friends",
    includes = [
        "//xla:friends",
    ],
)

cc_library(
    name = "cpu_codegen_backend",
    hdrs = ["cpu_codegen_backend.h"],
    tags = if_google(["avoid_dep"]),
    deps = ["//xla/backends/cpu/core/autotuner:cpu_codegen_backend"],
)

cc_library(
    name = "cpu_profiler",
    hdrs = ["cpu_profiler.h"],
    tags = if_google(["avoid_dep"]),
    deps = ["//xla/backends/cpu/core/autotuner:cpu_profiler"],
)

cc_library(
    name = "llvm_kernel_backend",
    hdrs = ["llvm_kernel_backend.h"],
    tags = if_google(["avoid_dep"]),
    deps = ["//xla/backends/cpu/core/autotuner:llvm_kernel_backend"],
)

cc_library(
    name = "ynnpack_backend",
    hdrs = ["ynnpack_backend.h"],
    tags = if_google(["avoid_dep"]),
    deps = ["//xla/backends/cpu/core/autotuner:ynnpack_backend"],
)

cc_library(
    name = "cpu_autotuner",
    hdrs = ["cpu_autotuner.h"],
    tags = if_google(["avoid_dep"]),
    deps = ["//xla/backends/cpu/core/autotuner:cpu_autotuner"],
)

cc_library(
    name = "block_level_emitter_backend",
    hdrs = ["block_level_emitter_backend.h"],
    tags = if_google(["avoid_dep"]),
    deps = ["//xla/backends/cpu/core/autotuner:block_level_emitter_backend"],
)
