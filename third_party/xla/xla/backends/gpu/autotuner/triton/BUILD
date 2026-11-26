load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("//xla/tsl:tsl.bzl", "if_google")
load("//xla/tsl:tsl.default.bzl", "get_compatible_with_portable")
load("//xla/tsl/util:cc_embed_data.bzl", "cc_embed_data")

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
    name = "dot_search_space",
    hdrs = ["dot_search_space.h"],
    tags = if_google(["avoid_dep"]),
    deps = ["//xla/backends/gpu/core/autotuner/triton:dot_search_space"],
)

cc_library(
    name = "triton_configs",
    hdrs = ["triton_configs.h"],
    tags = if_google(["avoid_dep"]),
    deps = ["//xla/backends/gpu/core/autotuner/triton:triton_configs"],
)

cc_library(
    name = "cost_model_config_optimization",
    hdrs = ["cost_model_config_optimization.h"],
    tags = if_google(["avoid_dep"]),
    deps = ["//xla/backends/gpu/core/autotuner/triton:cost_model_config_optimization"],
)

filegroup(
    name = "all_default_configs",
    data = glob(["default_configs/*.txtpb"]),
)

exports_files(glob([
    "default_configs/*.txtpb",
]))

cc_embed_data(
    name = "embed_default_configs",
    srcs = glob(["default_configs/*.txtpb"]),
    outs = [
        "embed_default_configs.cc",
        "embed_default_configs.h",
    ],
    compatible_with = get_compatible_with_portable() + if_google(["//buildenv/target:libtpu"]),
    embedopts = ["--namespace=xla::gpu::configs"],
    flatten = True,
    visibility = [
        ":friends",
        "//xla/backends/gpu/core/autotuner/triton:__pkg__",
    ],
)
