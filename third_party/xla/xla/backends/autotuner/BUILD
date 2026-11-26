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
    name = "codegen_backend",
    hdrs = ["codegen_backend.h"],
    tags = if_google(["avoid_dep"]),
    deps = ["//xla/backends/core/autotuner:codegen_backend"],
)

cc_library(
    name = "fake_codegen_backend",
    testonly = 1,
    hdrs = ["fake_codegen_backend.h"],
    tags = if_google(["avoid_dep"]),
    deps = ["//xla/backends/core/autotuner:fake_codegen_backend"],
)

cc_library(
    name = "mock_codegen_backend",
    testonly = 1,
    hdrs = ["mock_codegen_backend.h"],
    tags = if_google(["avoid_dep"]),
    deps = ["//xla/backends/core/autotuner:mock_codegen_backend"],
)

cc_library(
    name = "autotune_fingerprint",
    hdrs = ["autotune_fingerprint.h"],
    tags = if_google(["avoid_dep"]),
    deps = ["//xla/backends/core/autotuner:autotune_fingerprint"],
)

cc_library(
    name = "hlo_extractor",
    hdrs = ["hlo_extractor.h"],
    tags = if_google(["avoid_dep"]),
    deps = ["//xla/backends/core/autotuner:hlo_extractor"],
)

cc_library(
    name = "codegen_orchestrator",
    hdrs = ["codegen_orchestrator.h"],
    tags = if_google(["avoid_dep"]),
    deps = ["//xla/backends/core/autotuner:codegen_orchestrator"],
)

cc_library(
    name = "autotuner",
    hdrs = ["autotuner.h"],
    tags = if_google(["avoid_dep"]),
    deps = ["//xla/backends/core/autotuner"],
)

cc_library(
    name = "config_assigner",
    hdrs = ["config_assigner.h"],
    tags = if_google(["avoid_dep"]),
    deps = ["//xla/backends/core/autotuner:config_assigner"],
)

cc_library(
    name = "profiler",
    hdrs = ["profiler.h"],
    tags = if_google(["avoid_dep"]),
    deps = ["//xla/backends/core/autotuner:profiler"],
)

cc_library(
    name = "autotuner_cache_interface",
    hdrs = ["autotuner_cache_interface.h"],
    tags = if_google(["avoid_dep"]),
    deps = ["//xla/backends/core/autotuner:autotuner_cache_interface"],
)

cc_library(
    name = "autotune_cache_store",
    hdrs = ["autotune_cache_store.h"],
    tags = if_google(["avoid_dep"]),
    deps = ["//xla/backends/core/autotuner:autotune_cache_store"],
)

cc_library(
    name = "tiered_cache",
    hdrs = ["tiered_cache.h"],
    tags = if_google(["avoid_dep"]),
    deps = ["//xla/backends/core/autotuner:tiered_cache"],
)

cc_library(
    name = "in_memory_store",
    hdrs = ["in_memory_store.h"],
    tags = if_google(["avoid_dep"]),
    deps = ["//xla/backends/core/autotuner:in_memory_store"],
)

cc_library(
    name = "directory_store",
    hdrs = ["directory_store.h"],
    tags = if_google(["avoid_dep"]),
    deps = ["//xla/backends/core/autotuner:directory_store"],
)

cc_library(
    name = "config_runner",
    hdrs = ["config_runner.h"],
    tags = if_google(["avoid_dep"]),
    deps = ["//xla/backends/core/autotuner:config_runner"],
)

cc_library(
    name = "config_selector",
    hdrs = ["config_selector.h"],
    tags = if_google(["avoid_dep"]),
    deps = ["//xla/backends/core/autotuner:config_selector"],
)
