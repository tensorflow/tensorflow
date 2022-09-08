load("tsl.bzl", "if_oss")
load("@bazel_skylib//lib:selects.bzl", "selects")
load("@bazel_skylib//rules:common_settings.bzl", "bool_setting")

# Config setting to use in select()s to distinguish open source build from
# google internal build on configurable attributes.
#
# For non-configurable distinction between OSS and Google builds, see
# `if_oss()` and `if_google()` macros in tsl.bzl.
config_setting(
    name = "oss",
    flag_values = {":oss_setting": "True"},
    visibility = ["//visibility:public"],
)

# Non-configurable setting to indicate open source build.
bool_setting(
    name = "oss_setting",
    build_setting_default = if_oss(True, False),
    visibility = ["//visibility:private"],
)

# Config setting that is satisfied when TSL is being built with CUDA
# support through e.g. `--config=cuda` (or `--config=cuda_clang` in OSS).
alias(
    name = "is_cuda_enabled",
    actual = if_oss(
        "@local_config_cuda//:is_cuda_enabled",
        "//third_party/gpus/cuda:using_clang",
    ),
)

selects.config_setting_group(
    name = "is_cuda_enabled_and_oss",
    match_all = [
        ":is_cuda_enabled",
        ":oss",
    ],
)
