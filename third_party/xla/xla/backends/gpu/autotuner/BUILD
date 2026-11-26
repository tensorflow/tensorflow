load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("//xla/tsl:tsl.bzl", "if_google")
load("//xla/tsl:tsl.default.bzl", "get_compatible_with_portable")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [
        "//xla/backends/autotuner:__subpackages__",
        "//xla/backends/gpu:__pkg__",
        "//xla/backends/gpu/hlo/transforms:__subpackages__",
        "//xla/service/gpu:__subpackages__",
    ],
    licenses = ["notice"],
)

package_group(
    name = "friends",
    includes = [
        "//xla:friends",
    ],
)

cc_library(
    name = "gpu_codegen_backend",
    hdrs = ["gpu_codegen_backend.h"],
    tags = if_google(["avoid_dep"]),
    deps = ["//xla/backends/gpu/core/autotuner:gpu_codegen_backend"],
)

cc_library(
    name = "block_level_emitter",
    hdrs = ["block_level_emitter.h"],
    tags = [
        "gpu",
    ] + if_google(["avoid_dep"]),
    deps = ["//xla/backends/gpu/core/autotuner:block_level_emitter"],
)

cc_library(
    name = "tensor_ir",
    hdrs = ["tensor_ir.h"],
    tags = [
        "gpu",
    ] + if_google(["avoid_dep"]),
    deps = ["//xla/backends/gpu/core/autotuner:tensor_ir"],
)

cc_library(
    name = "cublaslt",
    hdrs = ["cublaslt.h"],
    tags = if_google(["avoid_dep"]),
    deps = ["//xla/backends/gpu/core/autotuner:cublaslt"],
)

cc_library(
    name = "cudnn",
    hdrs = ["cudnn.h"],
    tags = [
        "cuda-only",
        "gpu",
    ] + if_google(["avoid_dep"]),
    deps = ["//xla/backends/gpu/core/autotuner:cudnn"],
)

cc_library(
    name = "triton",
    hdrs = ["triton.h"],
    tags = [
        "gpu",
    ] + if_google(["avoid_dep"]),
    deps = ["//xla/backends/gpu/core/autotuner:triton"],
)

cc_library(
    name = "gpu_profiler",
    hdrs = ["gpu_profiler.h"],
    tags = if_google(["avoid_dep"]),
    deps = ["//xla/backends/gpu/core/autotuner:gpu_profiler"],
)

cc_library(
    name = "native_emitter",
    hdrs = ["native_emitter.h"],
    tags = if_google(["avoid_dep"]),
    deps = ["//xla/backends/gpu/core/autotuner:native_emitter"],
)

cc_library(
    name = "factory_cuda",
    tags = [
        "cuda-only",
        "gpu",
    ] + if_google(["avoid_dep"]),
    deps = ["//xla/backends/gpu/core/autotuner:factory_cuda"],
)

cc_library(
    name = "hipblaslt",
    hdrs = ["hipblaslt.h"],
    tags = [
        "gpu",
        "rocm-only",
    ] + if_google(["avoid_dep"]),
    deps = ["//xla/backends/gpu/core/autotuner:hipblaslt"],
)

cc_library(
    name = "factory_rocm",
    tags = [
        "gpu",
        "rocm-only",
    ] + if_google(["avoid_dep"]),
    deps = ["//xla/backends/gpu/core/autotuner:factory_rocm"],
)

cc_library(
    name = "factory",
    hdrs = ["factory.h"],
    tags = if_google(["avoid_dep"]),
    deps = ["//xla/backends/gpu/core/autotuner:factory"],
)

cc_library(
    name = "legacy_cache",
    hdrs = ["legacy_cache.h"],
    tags = if_google(["avoid_dep"]),
    deps = ["//xla/backends/gpu/core/autotuner:legacy_cache"],
)

cc_library(
    name = "fission_backend",
    hdrs = ["fission_backend.h"],
    tags = if_google(["avoid_dep"]),
    deps = ["//xla/backends/gpu/core/autotuner:fission_backend"],
)

cc_library(
    name = "miopen",
    hdrs = ["miopen.h"],
    tags = if_google(["avoid_dep"]),
    deps = ["//xla/backends/gpu/core/autotuner:miopen"],
)

alias(
    name = "autotuner_main",
    actual = "//xla/backends/gpu/core/autotuner:autotuner_main",
    tags = if_google(["avoid_dep"]),
)

cc_library(
    name = "autotuner_status_key",
    hdrs = ["autotuner_status_key.h"],
    compatible_with = get_compatible_with_portable(),
    tags = if_google(["avoid_dep"]),
    deps = ["//xla/backends/gpu/core/autotuner:autotuner_status_key"],
)

cc_library(
    name = "autotune_cache_key",
    hdrs = ["autotune_cache_key.h"],
    compatible_with = get_compatible_with_portable(),
    tags = if_google(["avoid_dep"]),
    deps = ["//xla/backends/gpu/core/autotuner:autotune_cache_key"],
)

cc_library(
    name = "autotuner_cache",
    hdrs = ["autotuner_cache.h"],
    compatible_with = get_compatible_with_portable(),
    tags = if_google(["avoid_dep"]),
    visibility = [":friends"],
    deps = ["//xla/backends/gpu/core/autotuner:autotuner_cache"],
)

cc_library(
    name = "redzone_buffers",
    hdrs = ["redzone_buffers.h"],
    tags = if_google(["avoid_dep"]),
    deps = ["//xla/backends/gpu/core/autotuner:redzone_buffers"],
)

cc_library(
    name = "gpu_default_backend_policy",
    hdrs = ["gpu_default_backend_policy.h"],
    tags = if_google(["avoid_dep"]),
    deps = ["//xla/backends/gpu/core/autotuner:gpu_default_backend_policy"],
)
