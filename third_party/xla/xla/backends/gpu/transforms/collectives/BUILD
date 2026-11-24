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
    name = "convert_async_collectives_to_sync",
    hdrs = ["convert_async_collectives_to_sync.h"],
    tags = if_google(["avoid_dep"]),
    deps = ["//xla/backends/gpu/hlo/transforms/collectives:convert_async_collectives_to_sync"],
)

cc_library(
    name = "collective_backend_assigner",
    hdrs = ["collective_backend_assigner.h"],
    tags = if_google(["avoid_dep"]),
    deps = ["//xla/backends/gpu/hlo/transforms/collectives:collective_backend_assigner"],
)

cc_library(
    name = "collective_domain",
    hdrs = ["collective_domain.h"],
    tags = if_google(["avoid_dep"]),
    deps = ["//xla/backends/gpu/hlo/transforms/collectives:collective_domain"],
)

cc_library(
    name = "collective_domain_assigner",
    hdrs = ["collective_domain_assigner.h"],
    tags = if_google(["avoid_dep"]),
    deps = ["//xla/backends/gpu/hlo/transforms/collectives:collective_domain_assigner"],
)

cc_library(
    name = "legalize_collective_domain",
    hdrs = ["legalize_collective_domain.h"],
    tags = if_google(["avoid_dep"]),
    deps = ["//xla/backends/gpu/hlo/transforms/collectives:legalize_collective_domain"],
)

cc_library(
    name = "collective_ops_utils",
    hdrs = ["collective_ops_utils.h"],
    # copybara:uncomment compatible_with = ["//buildenv/target:non_prod"],
    tags = if_google(["avoid_dep"]),
    deps = ["//xla/backends/gpu/hlo/transforms/collectives:collective_ops_utils"],
)

cc_library(
    name = "gpu_collective_combiner_utils",
    hdrs = ["gpu_collective_combiner_utils.h"],
    tags = if_google(["avoid_dep"]),
    deps = ["//xla/backends/gpu/hlo/transforms/collectives:gpu_collective_combiner_utils"],
)

cc_library(
    name = "gpu_all_gather_combiner",
    hdrs = ["all_gather_combiner.h"],
    tags = if_google(["avoid_dep"]),
    deps = ["//xla/backends/gpu/hlo/transforms/collectives:gpu_all_gather_combiner"],
)

cc_library(
    name = "gpu_reduce_scatter_combiner",
    hdrs = ["reduce_scatter_combiner.h"],
    tags = if_google(["avoid_dep"]),
    deps = ["//xla/backends/gpu/hlo/transforms/collectives:gpu_reduce_scatter_combiner"],
)

cc_library(
    name = "gpu_all_reduce_combiner",
    hdrs = ["all_reduce_combiner.h"],
    tags = if_google(["avoid_dep"]),
    deps = ["//xla/backends/gpu/hlo/transforms/collectives:gpu_all_reduce_combiner"],
)

cc_library(
    name = "collective_combiner_annotator",
    hdrs = ["collective_combiner_annotator.h"],
    tags = if_google(["avoid_dep"]),
    deps = ["//xla/backends/gpu/hlo/transforms/collectives:collective_combiner_annotator"],
)

cc_library(
    name = "all_gather_optimizer",
    hdrs = ["all_gather_optimizer.h"],
    tags = if_google(["avoid_dep"]),
    deps = ["//xla/backends/gpu/hlo/transforms/collectives:all_gather_optimizer"],
)

cc_library(
    name = "all_gather_major_dimension_rewriter",
    hdrs = ["all_gather_major_dimension_rewriter.h"],
    tags = if_google(["avoid_dep"]),
    deps = ["//xla/backends/gpu/hlo/transforms/collectives:all_gather_major_dimension_rewriter"],
)

cc_library(
    name = "all_gather_dynamic_slice_simplifier",
    hdrs = ["all_gather_dynamic_slice_simplifier.h"],
    tags = if_google(["avoid_dep"]),
    deps = ["//xla/backends/gpu/hlo/transforms/collectives:all_gather_dynamic_slice_simplifier"],
)

cc_library(
    name = "all_reduce_blueconnect",
    hdrs = ["all_reduce_blueconnect.h"],
    tags = if_google(["avoid_dep"]),
    deps = ["//xla/backends/gpu/hlo/transforms/collectives:all_reduce_blueconnect"],
)

cc_library(
    name = "all_reduce_splitter",
    hdrs = ["all_reduce_splitter.h"],
    tags = if_google(["avoid_dep"]),
    deps = ["//xla/backends/gpu/hlo/transforms/collectives:all_reduce_splitter"],
)

cc_library(
    name = "all_reduce_decomposer",
    hdrs = ["all_reduce_decomposer.h"],
    tags = if_google(["avoid_dep"]),
    deps = ["//xla/backends/gpu/hlo/transforms/collectives:all_reduce_decomposer"],
)

cc_library(
    name = "collective_select_folder",
    hdrs = ["collective_select_folder.h"],
    tags = if_google(["avoid_dep"]),
    deps = ["//xla/backends/gpu/hlo/transforms/collectives:collective_select_folder"],
)

cc_library(
    name = "collective_permute_cycle_decomposer",
    hdrs = ["collective_permute_cycle_decomposer.h"],
    tags = if_google(["avoid_dep"]),
    deps = ["//xla/backends/gpu/hlo/transforms/collectives:collective_permute_cycle_decomposer"],
)

cc_library(
    name = "async_collective_annotator",
    hdrs = ["async_collective_annotator.h"],
    tags = if_google(["avoid_dep"]),
    deps = ["//xla/backends/gpu/hlo/transforms/collectives:async_collective_annotator"],
)

cc_library(
    name = "collective_kernel_strategy_annotator",
    hdrs = ["collective_kernel_strategy_annotator.h"],
    tags = if_google(["avoid_dep"]),
    deps = ["//xla/backends/gpu/hlo/transforms/collectives:collective_kernel_strategy_annotator"],
)

cc_library(
    name = "collective_pipelining_analyzer",
    hdrs = ["collective_pipelining_analyzer.h"],
    tags = if_google(["avoid_dep"]),
    deps = ["//xla/backends/gpu/hlo/transforms/collectives:collective_pipelining_analyzer"],
)

cc_library(
    name = "collective_fusion",
    hdrs = ["collective_fusion.h"],
    tags = if_google(["avoid_dep"]),
    deps = ["//xla/backends/gpu/hlo/transforms/collectives:collective_fusion"],
)
