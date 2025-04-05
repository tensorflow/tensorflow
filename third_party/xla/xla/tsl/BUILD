# Tensor Standard Libraries - common utilities for implementing XLA.

load("@bazel_skylib//:bzl_library.bzl", "bzl_library")
load("@bazel_skylib//lib:selects.bzl", "selects")
load("@bazel_skylib//rules:common_settings.bzl", "bool_flag", "bool_setting")
load("//xla/tsl:package_groups.bzl", "tsl_package_groups")
load("//xla/tsl:tsl.bzl", "if_google", "if_oss", "internal_visibility")
load(
    "//xla/tsl:tsl.default.bzl",
    "tsl_extra_config_settings",
    "tsl_extra_config_settings_targets",
    "tsl_google_bzl_deps",
)
load("//xla/tsl/platform:rules_cc.bzl", "cc_library")

# copybara:uncomment package(default_applicable_licenses = ["//tensorflow:license"])

tsl_package_groups()

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
        "@local_config_cuda//cuda:using_config_cuda",
    ),
    visibility = ["//visibility:public"],
)

selects.config_setting_group(
    name = "is_cuda_enabled_and_oss",
    match_all = [
        ":is_cuda_enabled",
        ":oss",
    ],
    visibility = ["//visibility:public"],
)

# Config setting that is satisfied when CUDA device code should be compiled
# with nvcc. It does not imply that CUDA support has been enabled.
alias(
    name = "is_cuda_compiler_nvcc",
    actual = if_oss(
        "@local_config_cuda//:is_cuda_compiler_nvcc",
        "@local_config_cuda//cuda:FALSE",
    ),
)

selects.config_setting_group(
    name = "is_cuda_nvcc",
    match_all = [
        ":is_cuda_enabled",
        ":is_cuda_compiler_nvcc",
    ],
)

# Crosses between framework_shared_object and a bunch of other configurations
# due to limitations in nested select() statements.
config_setting(
    name = "framework_shared_object",
    define_values = {"framework_shared_object": "true"},
    visibility = ["//visibility:public"],
)

config_setting(
    name = "macos_x86_64_with_framework_shared_object",
    constraint_values = [
        "@platforms//os:macos",
        "@platforms//cpu:x86_64",
    ],
    define_values = {
        "framework_shared_object": "true",
    },
    values = {
        "apple_platform_type": "macos",
    },
    visibility = ["//visibility:public"],
)

config_setting(
    name = "macos_arm64_with_framework_shared_object",
    constraint_values = [
        "@platforms//os:macos",
        "@platforms//cpu:aarch64",
    ],
    define_values = {
        "framework_shared_object": "true",
    },
    values = {
        "apple_platform_type": "macos",
    },
    visibility = ["//visibility:public"],
)

selects.config_setting_group(
    name = "macos_with_framework_shared_object",
    match_any = [
        ":macos_x86_64_with_framework_shared_object",
        ":macos_arm64_with_framework_shared_object",
    ],
    visibility = ["//visibility:public"],
)

# This should be removed after Tensorflow moves to cc_shared_library
config_setting(
    name = "tsl_protobuf_header_only",
    define_values = {"tsl_protobuf_header_only": "true"},
    visibility = ["//visibility:public"],
)

# Config setting for determining if we are building for Android.
config_setting(
    name = "android",
    constraint_values = if_google(
        ["@platforms//os:android"],
        [],
    ),
    values = if_oss(
        {"crosstool_top": "//external:android/crosstool"},
        {},
    ),
    visibility = ["//visibility:public"],
)

config_setting(
    name = "emscripten",
    constraint_values = if_google(
        ["@platforms//os:emscripten"],
        [],
    ),
    values = if_oss(
        {"crosstool_top": "//external:android/emscripten"},
        {},
    ),
    visibility = ["//visibility:public"],
)

config_setting(
    name = "macos_x86_64",
    constraint_values = [
        "@platforms//cpu:x86_64",
        "@platforms//os:macos",
    ],
    values = {
        "apple_platform_type": "macos",
    },
    visibility = ["//visibility:public"],
)

config_setting(
    name = "macos_arm64",
    constraint_values = [
        "@platforms//cpu:aarch64",
        "@platforms//os:macos",
    ],
    values = {
        "apple_platform_type": "macos",
    },
    visibility = ["//visibility:public"],
)

selects.config_setting_group(
    name = "macos",
    match_any = [
        ":macos_x86_64",
        ":macos_arm64",
    ],
    visibility = ["//visibility:public"],
)

config_setting(
    name = "windows_x86_64",
    constraint_values = [
        "@platforms//cpu:x86_64",
        "@platforms//os:windows",
    ],
    visibility = ["//visibility:public"],
)

config_setting(
    name = "windows_aarch64",
    constraint_values = [
        "@platforms//cpu:aarch64",
        "@platforms//os:windows",
    ],
    visibility = ["//visibility:public"],
)

config_setting(
    name = "windows",
    constraint_values = [
        "@platforms//os:windows",
    ],
    visibility = ["//visibility:public"],
)

# TODO(jakeharmon): Remove equivalent from tensorflow/BUILD
config_setting(
    name = "ios",
    constraint_values = if_google(
        ["@platforms//os:ios"],
        [],
    ),
    values = if_oss(
        {"apple_platform_type": "ios"},
        {},
    ),
    visibility = ["//visibility:public"],
)

# TODO(jakeharmon): Remove equivalent from tensorflow/BUILD
# Config setting used when building for products
# which requires restricted licenses to be avoided.
config_setting(
    name = "no_lgpl_deps",
    define_values = {"__TENSORFLOW_NO_LGPL_DEPS__": "1"},
    visibility = ["//visibility:public"],
)

config_setting(
    name = "with_xla_support",
    define_values = {"with_xla_support": "true"},
    visibility = ["//visibility:public"],
)

config_setting(
    name = "with_cross_compiler_support",
    define_values = {"with_cross_compiler_support": "true"},
    visibility = ["//visibility:public"],
)

config_setting(
    name = "android_arm",
    constraint_values =
        [
            "@platforms//cpu:armv7",
            "@platforms//os:android",
        ],
    values = dict(
        if_oss(
            {"crosstool_top": "//external:android/crosstool"},
        ),
    ),
    visibility = ["//visibility:public"],
)

config_setting(
    name = "linux_aarch64",
    constraint_values =
        [
            "@platforms//cpu:aarch64",
            "@platforms//os:linux",
        ],
    visibility = ["//visibility:public"],
)

config_setting(
    name = "linux_armhf",
    constraint_values =
        [
            "@platforms//cpu:armv7e-mf",
            "@platforms//os:linux",
        ],
    visibility = ["//visibility:public"],
)

config_setting(
    name = "linux_x86_64",
    constraint_values =
        [
            "@platforms//cpu:x86_64",
            "@platforms//os:linux",
        ],
    visibility = ["//visibility:public"],
)

config_setting(
    name = "linux_ppc64le",
    constraint_values =
        [
            "@platforms//cpu:ppc64le",
            "@platforms//os:linux",
        ],
    visibility = ["//visibility:public"],
)

config_setting(
    name = "linux_s390x",
    constraint_values =
        [
            "@platforms//cpu:s390x",
            "@platforms//os:linux",
        ],
    visibility = ["//visibility:public"],
)

config_setting(
    name = "ios_x86_64",
    constraint_values =
        [
            "@platforms//cpu:x86_64",
            "@platforms//os:ios",
        ],
    values = dict(
        if_oss(
            {"crosstool_top": "//tools/osx/crosstool:crosstool"},
        ),
    ),
    visibility = ["//visibility:public"],
)

selects.config_setting_group(
    name = "aarch32_or_cross",
    match_any = [
        ":linux_armhf",
        ":with_cross_compiler_support",
    ],
    visibility = ["//visibility:public"],
)

selects.config_setting_group(
    name = "aarch64_or_cross",
    match_any = [
        ":linux_aarch64",
        ":macos_arm64",
        ":windows_aarch64",
        ":with_cross_compiler_support",
    ],
    visibility = ["//visibility:public"],
)

selects.config_setting_group(
    name = "arm_or_cross",
    match_any = [
        ":linux_aarch64",
        ":macos_arm64",
        ":windows_aarch64",
        ":linux_armhf",
        ":with_cross_compiler_support",
    ],
    visibility = ["//visibility:public"],
)

selects.config_setting_group(
    name = "ppc64le_or_cross",
    match_any = [
        ":linux_ppc64le",
        ":with_cross_compiler_support",
    ],
    visibility = ["//visibility:public"],
)

selects.config_setting_group(
    name = "s390x_or_cross",
    match_any = [
        ":linux_s390x",
        ":with_cross_compiler_support",
    ],
    visibility = ["//visibility:public"],
)

selects.config_setting_group(
    name = "x86_or_cross",
    match_any = [
        ":linux_x86_64",
        ":macos_x86_64",
        ":windows_x86_64",
        ":with_cross_compiler_support",
    ],
    visibility = ["//visibility:public"],
)

# Config setting that disables the default logger, only logging
# to registered TFLogSinks
config_setting(
    name = "no_default_logger",
    define_values = {"no_default_logger": "true"},
    visibility = ["//visibility:public"],
)

config_setting(
    name = "fuchsia",
    constraint_values = ["@platforms//os:fuchsia"],
    visibility = ["//visibility:public"],
)

# TODO(jakeharmon): Remove equivalent from tensorflow/BUILD
config_setting(
    name = "android_x86",
    constraint_values =
        [
            "@platforms//cpu:x86_32",
            "@platforms//os:android",
        ],
    values = dict(
        if_oss(
            {"crosstool_top": "//external:android/crosstool"},
        ),
    ),
    visibility = ["//visibility:public"],
)

tsl_extra_config_settings()

selects.config_setting_group(
    name = "arm_any",
    match_any = [
        "@platforms//cpu:aarch32",
        "@platforms//cpu:aarch64",
        "@platforms//cpu:armv7",
        "@platforms//cpu:armv7-m",
        "@platforms//cpu:armv7e-m",
        "@platforms//cpu:armv7e-mf",
    ] + tsl_extra_config_settings_targets(),
    visibility = ["//visibility:public"],
)

selects.config_setting_group(
    name = "linux_any",
    match_any = [
        ":linux_aarch64",
        ":linux_armhf",
        ":linux_ppc64le",
        ":linux_s390x",
        ":linux_x86_64",
    ],
    visibility = ["//visibility:public"],
)

# Enable NUMA on Linux platforms; hwloc does not work on Android
selects.config_setting_group(
    name = "with_numa_support",
    match_any = [
        ":linux_any",
    ],
)

# TODO(jakeharmon): Remove equivalent from tensorflow/BUILD
config_setting(
    name = "fuchsia_x86_64",
    constraint_values =
        [
            "@platforms//cpu:x86_64",
            "@platforms//os:fuchsia",
        ],
    visibility = ["//visibility:public"],
)

# This flag enables experimental TPU support
bool_flag(
    name = "enable_tpu_support",
    build_setting_default = False,
)

config_setting(
    name = "with_tpu_support_define",
    define_values = {"with_tpu_support": "true"},
    visibility = ["//visibility:public"],
)

config_setting(
    name = "with_tpu_support_flag",
    flag_values = {":enable_tpu_support": "True"},
    visibility = ["//visibility:public"],
)

selects.config_setting_group(
    name = "with_tpu_support",
    match_any = [
        ":with_tpu_support_define",
        ":with_tpu_support_flag",
    ],
    visibility = ["//visibility:public"],
)

config_setting(
    name = "freebsd",
    constraint_values = [
        "@platforms//os:freebsd",
        "@platforms//cpu:x86_64",
    ],
    visibility = ["//visibility:public"],
)

config_setting(
    name = "using_no_nccl_support_define",
    define_values = dict(
        no_nccl_support = "true",
    ),
    visibility = internal_visibility(["//visibility:private"]),
)

selects.config_setting_group(
    name = "no_nccl_support",
    match_all = [
        ":using_no_nccl_support_define",
    ] + if_google([
        "@local_config_cuda//cuda:using_config_cuda",
    ]),
    visibility = ["//visibility:public"],
)

bzl_library(
    name = "tsl_bzl",
    srcs = ["tsl.bzl"],
    visibility = ["//visibility:public"],
    deps = [
        ":package_groups_bzl",
        "//third_party/compute_library:build_defs_bzl",
        "//xla/tsl/mkl:build_defs_bzl",
        "//xla/tsl/platform:rules_cc_bzl",
        "@bazel_skylib//lib:new_sets",
        "@local_config_cuda//cuda:build_defs_bzl",
        "@local_config_rocm//rocm:build_defs_bzl",
        "@local_config_tensorrt//:build_defs_bzl",
        "@local_xla//third_party/py/rules_pywrap:pywrap_bzl",
    ],
)

bzl_library(
    name = "tsl_google_bzl",
    srcs = if_google(["tsl.google.bzl"]),
    visibility = ["//visibility:public"],
    deps = [
        ":package_groups_bzl",
        ":tsl_bzl",
        "//xla/tsl/platform:build_config_root_bzl",
    ] + tsl_google_bzl_deps(),
)

bzl_library(
    name = "package_groups_bzl",
    srcs = ["package_groups.bzl"],
    visibility = ["//xla:__subpackages__"],
)

# copybara:comment_begin(oss-only)
cc_library(
    name = "grpc++",
    visibility = ["//visibility:public"],
    deps = select({
        "//conditions:default": ["@com_github_grpc_grpc//:grpc++"],
    }),
)
# copybara:comment_end

cc_library(
    name = "empty",
    visibility = ["//visibility:public"],
)

# Needed to workaround https://github.com/bazelbuild/bazel/issues/21519
alias(
    name = "bazel_issue_21519",
    actual = ":empty",
    visibility = ["//visibility:public"],
)
