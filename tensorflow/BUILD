# Description:
# TensorFlow is a computational framework, primarily for use in machine
# learning applications.

load("@bazel_skylib//:bzl_library.bzl", "bzl_library")
load("@bazel_skylib//lib:selects.bzl", "selects")
load("@bazel_skylib//rules:common_settings.bzl", "bool_flag", "bool_setting")
load("@local_config_cuda//cuda:build_defs.bzl", "if_cuda")
load(
    "//tensorflow:tensorflow.bzl",
    "VERSION",
    "VERSION_MAJOR",
    "check_deps",
    "if_google",
    "if_oss",
    "if_xla_available",
    "tf_cc_shared_object",
    "tf_custom_op_library_additional_deps_impl",
    "tf_monitoring_python_deps",
    "tf_native_cc_binary",
    "tsl_async_value_deps",
)
load(
    "//tensorflow:tensorflow.default.bzl",
    "ADDITIONAL_API_INDEXABLE_SETTINGS",
    "tf_cc_shared_library",
)
load(
    "//tensorflow/core/platform:build_config.bzl",
    "tf_additional_binary_deps",
)
load(
    "//tensorflow/core/platform:build_config_root.bzl",
    "if_static",
    "tf_additional_plugin_deps",
    "tf_additional_profiler_deps",
)
load(
    "//third_party/mkl:build_defs.bzl",
    "if_mkl_ml",
)

# copybara:uncomment_begin
# # buildifier: disable=out-of-order-load
# load("//devtools/build_cleaner/skylark:action_config_test.bzl", "action_config_test")
# load("//devtools/copybara/rules:copybara.bzl", "copybara_config_test")
# load("//tools/build_defs/license:license.bzl", "license")
# # buildifier: enable=out-of-order-load
# copybara:uncomment_end

# copybara:comment_begin(oss-only)
load("@bazel_skylib//rules:copy_file.bzl", "copy_file")
load(
    "//tensorflow/python/tools/api/generator:api_gen.bzl",
    "get_compat_files",
    "get_nested_compat_files",
)
load(
    "//tensorflow/python/tools/api/generator:api_init_files.bzl",
    "TENSORFLOW_API_INIT_FILES",
)
load(
    "//tensorflow/python/tools/api/generator:api_init_files_v1.bzl",
    "TENSORFLOW_API_INIT_FILES_V1",
)
load("//tensorflow/python/tools/api/generator2:generate_api.bzl", "generate_apis")
# copybara:comment_end

package(
    # copybara:uncomment default_applicable_licenses = [":license"],
    default_visibility = [":internal"],
)

# copybara:uncomment_begin
# license(
#     name = "license",
#     package_name = "tensorflow",
# )
#
# filegroup(
#     name = "copybara_library",
#     srcs = glob(
#         [
#             "*.bara.sky",
#             "*.dic",
#             "*.ftrcp",
#         ],
#         exclude = [
#             "copy.bara.sky",
#         ],
#     ),
#     visibility = ["//visibility:public"],
# )
#
# filegroup(
#     name = "copybara_config",
#     srcs = ["copy.bara.sky"],
#     data = [
#         ":copybara_library",
#         "//learning/brain/testing/copybara:all_bara_sky",
#         "//third_party/tsl:copybara_library",
#         "//third_party/xla:copybara_library",
#     ],
#     visibility = ["//visibility:public"],
# )
#
# copybara_config_test(
#     name = "copybara_config_test",
#     config = "copy.bara.sky",
#     tags = [
#         "noasan",
#         "nodfsan",
#         "nogotsan",
#         "nogpu",
#         "nomsan",
#         "nosan",
#         "notsan",
#         "noubsan",
#     ],
#     deps = [":copybara_config"],
# )
#
# action_config_test(
#     name = "build_cleaner_spec_test",
#     src = "build_cleaner_spec.textproto",
# )
# copybara:uncomment_end

licenses(["notice"])

exports_files([
    "LICENSE",
    # The leakr files are used by //third_party/cloud_tpu and
    # //third_party/tensorboard/google:copybara_config_test.
    "leakr_badwords.dic",
    "leakr_badfiles.dic",
    "leakr_file_type_recipe.ftrcp",
])

# copybara:comment_begin(oss-only)
TENSORFLOW_API_INIT_FILES_V2 = (
    TENSORFLOW_API_INIT_FILES +
    get_compat_files(TENSORFLOW_API_INIT_FILES, 2) +
    get_compat_files(TENSORFLOW_API_INIT_FILES_V1, 1) + get_nested_compat_files([
        1,
        2,
    ])
)

TENSORFLOW_API_INIT_FILES_V1 = (
    TENSORFLOW_API_INIT_FILES_V1 +
    get_compat_files(TENSORFLOW_API_INIT_FILES, 2) +
    get_compat_files(TENSORFLOW_API_INIT_FILES_V1, 1) + get_nested_compat_files([
        1,
        2,
    ])
)
# copybara:comment_end

# Config setting used when building for products
# which requires restricted licenses to be avoided.
config_setting(
    name = "no_lgpl_deps",
    define_values = {"__TENSORFLOW_NO_LGPL_DEPS__": "1"},
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
    name = "force_rtti",
    define_values = {
        "tf_force_rtti": "true",
    },
    visibility = ["//visibility:public"],
)

# Config setting for determining if we are building for Android.
config_setting(
    name = "android",
    constraint_values = if_google(
        ["//third_party/bazel_platforms/os:android"],
        [],
    ),
    values = if_oss(
        {"crosstool_top": "//external:android/crosstool"},
        {},
    ),
    visibility = ["//visibility:public"],
)

config_setting(
    name = "android_x86",
    constraint_values = if_google(
        ["//third_party/bazel_platforms/os:android"],
        [],
    ),
    values = dict(
        if_oss(
            {"crosstool_top": "//external:android/crosstool"},
        ),
        cpu = "x86",
    ),
    visibility = ["//visibility:public"],
)

config_setting(
    name = "android_x86_64",
    constraint_values = if_google(
        ["//third_party/bazel_platforms/os:android"],
        [],
    ),
    values = dict(
        if_oss(
            {"crosstool_top": "//external:android/crosstool"},
        ),
        cpu = "x86_64",
    ),
    visibility = ["//visibility:public"],
)

config_setting(
    name = "android_armeabi",
    constraint_values = if_google(
        ["//third_party/bazel_platforms/os:android"],
        [],
    ),
    values = dict(
        if_oss(
            {"crosstool_top": "//external:android/crosstool"},
        ),
        cpu = "armeabi",
    ),
    visibility = ["//visibility:public"],
)

# copybara:uncomment_begin(google-only)
# config_setting(
#     name = "chromiumos_x86_64",
#     constraint_values = ["//third_party/bazel_platforms/os:chromiumos"],
#     values = {"cpu": "k8"},
#     visibility = ["//visibility:public"],
# )
#
# config_setting(
#     name = "chromiumos_arm64",
#     constraint_values = ["//third_party/bazel_platforms/os:chromiumos"],
#     values = {"cpu": "arm"},
#     visibility = ["//visibility:public"],
# )
#
# config_setting(
#     name = "chromiumos_armv7",
#     constraint_values = ["//third_party/bazel_platforms/os:chromiumos"],
#     values = {"cpu": "armeabi-v7a"},
#     visibility = ["//visibility:public"],
# )
# copybara:uncomment_end

config_setting(
    name = "emscripten",
    constraint_values = if_google(
        ["//third_party/bazel_platforms/cpu:wasm32"],
        [],
    ),
    values = if_oss(
        {"crosstool_top": "//external:android/emscripten"},
        {},
    ),
    visibility = ["//visibility:public"],
)

config_setting(
    name = "raspberry_pi_armeabi",
    values = {
        "crosstool_top": "@local_config_arm_compiler//:toolchain",
        "cpu": "armeabi",
    },
    visibility = ["//visibility:public"],
)

config_setting(
    name = "android_arm",
    constraint_values = if_google(
        ["//third_party/bazel_platforms/os:android"],
        [],
    ),
    values = dict(
        if_oss(
            {"crosstool_top": "//external:android/crosstool"},
        ),
        cpu = "armeabi-v7a",
    ),
    visibility = ["//visibility:public"],
)

config_setting(
    name = "android_arm64",
    constraint_values = if_google(
        ["//third_party/bazel_platforms/os:android"],
        [],
    ),
    values = dict(
        if_oss(
            {"crosstool_top": "//external:android/crosstool"},
        ),
        cpu = "arm64-v8a",
    ),
    visibility = ["//visibility:public"],
)

config_setting(
    name = "android_mips",
    values = {
        "crosstool_top": "//external:android/crosstool",
        "cpu": "mips",
    },
    visibility = ["//visibility:public"],
)

config_setting(
    name = "android_mips64",
    values = {
        "crosstool_top": "//external:android/crosstool",
        "cpu": "mips64",
    },
    visibility = ["//visibility:public"],
)

# TODO(jakeharmon8): Remove in favor of TSL version
config_setting(
    name = "windows",
    # Internal builds query the target OS.
    constraint_values = if_google(
        ["//third_party/bazel_platforms/os:windows"],
        [],
    ),
    # OSS builds query the CPU type.
    values = if_oss(
        {"cpu": "x64_windows"},
        {},
    ),
    visibility = ["//visibility:public"],
)

config_setting(
    name = "no_tensorflow_py_deps",
    define_values = {"no_tensorflow_py_deps": "true"},
    visibility = ["//visibility:public"],
)

# Sometimes Bazel reports darwin_x86_64 as "darwin" and sometimes as
# "darwin_x86_64". The former shows up when building on a Mac x86_64 host for a Mac x86_64 target.
# The latter shows up when cross-compiling for Mac x86_64 from a Mac ARM machine and in internal
# Google builds.
config_setting(
    name = "macos_x86_64_default",
    constraint_values = if_google(
        ["//third_party/bazel_platforms/os:macos"],
        [],
    ),
    values = {
        "apple_platform_type": "macos",
        "cpu": "darwin",
    },
)

config_setting(
    name = "macos_x86_64_crosscompile",
    constraint_values = if_google(
        ["//third_party/bazel_platforms/os:macos"],
        [],
    ),
    values = {
        "apple_platform_type": "macos",
        "cpu": "darwin_x86_64",
    },
)

selects.config_setting_group(
    name = "macos_x86_64",
    match_any = [
        ":macos_x86_64_default",
        ":macos_x86_64_crosscompile",
    ],
    visibility = ["//visibility:public"],
)

config_setting(
    name = "macos_arm64",
    constraint_values = if_google(
        ["//third_party/bazel_platforms/os:macos"],
        [],
    ),
    values = {
        "apple_platform_type": "macos",
        "cpu": "darwin_arm64",
    },
    visibility = ["//visibility:public"],
)

# TODO(jakeharmon8): Remove in favor of TSL version
selects.config_setting_group(
    name = "macos",
    match_any = [
        ":macos_x86_64",
        ":macos_arm64",
    ],
    visibility = ["//visibility:public"],
)

config_setting(
    name = "ios",
    constraint_values = if_google(
        ["//third_party/bazel_platforms/os:ios"],
        [],
    ),
    values = if_oss(
        {"apple_platform_type": "ios"},
        {},
    ),
    visibility = ["//visibility:public"],
)

# TODO(jakeharmon8): Remove in favor of TSL version
config_setting(
    name = "fuchsia",
    constraint_values = if_google(
        ["//third_party/bazel_platforms/os:fuchsia"],
        [],
    ),
    values = if_oss(
        # TODO(b/149248802) When we have a Fuchsia Bazel SDK update to use the values it sets.
        {"cpu": "fuchsia"},
        {},
    ),
    visibility = ["//visibility:public"],
)

config_setting(
    name = "fuchsia_x86_64",
    constraint_values = if_google(
        ["//third_party/bazel_platforms/os:fuchsia"],
        [],
    ),
    values = {
        "cpu": "x86_64",
    },
    visibility = ["//visibility:public"],
)

config_setting(
    name = "ios_x86_64",
    constraint_values = if_google(
        ["//third_party/bazel_platforms/os:ios"],
        [],
    ),
    values = dict(
        if_oss(
            {"crosstool_top": "//tools/osx/crosstool:crosstool"},
        ),
        cpu = "ios_x86_64",
    ),
    visibility = ["//visibility:public"],
)

config_setting(
    name = "chromiumos",
    constraint_values = if_google(
        ["//third_party/bazel_platforms/os:chromiumos"],
        [],
    ),
    values = if_oss(
        {"crosstool_top": "//external:android/chromiumos"},
        {},
    ),
    visibility = ["//visibility:public"],
)

config_setting(
    name = "linux_aarch64",
    values = {"cpu": "aarch64"},
    visibility = ["//visibility:public"],
)

config_setting(
    name = "linux_armhf",
    values = {"cpu": "armhf"},
    visibility = ["//visibility:public"],
)

config_setting(
    name = "linux_x86_64",
    values = {"cpu": "k8"},
    visibility = ["//visibility:public"],
)

config_setting(
    name = "haswell",
    values = {"cpu": "haswell"},
    visibility = ["//visibility:public"],
)

# This condition takes precedence over :linux_x86_64
config_setting(
    name = "linux_x86_64_no_sse",
    values = {
        "cpu": "k8",
        "copt": "-mno-sse4.2",
    },
    visibility = ["//visibility:public"],
)

# This condition takes precedence over :linux_x86_64
# TODO(b/290533709): Remove this with PJRT build rule cleanup.
config_setting(
    name = "linux_x86_64_with_weightwatcher",
    define_values = {"tensorflow_weightwatcher": "true"},
    values = {"cpu": "k8"},
    visibility = ["//visibility:public"],
)

config_setting(
    name = "linux_ppc64le",
    values = {"cpu": "ppc"},
    visibility = ["//visibility:public"],
)

config_setting(
    name = "linux_s390x",
    values = {"cpu": "s390x"},
    visibility = ["//visibility:public"],
)

config_setting(
    name = "linux_mips64",
    values = {"cpu": "mips64"},
    visibility = ["//visibility:public"],
)

config_setting(
    name = "linux_riscv64",
    values = {"cpu": "riscv64"},
    visibility = ["//visibility:public"],
)

config_setting(
    name = "debug",
    values = {
        "compilation_mode": "dbg",
    },
    visibility = ["//visibility:public"],
)

config_setting(
    name = "optimized",
    values = {
        "compilation_mode": "opt",
    },
    visibility = ["//visibility:public"],
)

config_setting(
    name = "arm",
    values = {"cpu": "arm"},
    visibility = ["//visibility:public"],
)

config_setting(
    name = "armeabi",
    values = {"cpu": "armeabi"},
    visibility = ["//visibility:public"],
)

config_setting(
    name = "armeabi-v7a",
    values = {"cpu": "armeabi-v7a"},
    visibility = ["//visibility:public"],
)

config_setting(
    name = "arm64-v8a",
    values = {"cpu": "arm64-v8a"},
    visibility = ["//visibility:public"],
)

selects.config_setting_group(
    name = "arm_any",
    match_any = [
        ":arm",
        ":armeabi",
        ":armeabi-v7a",
        ":arm64-v8a",
        ":linux_aarch64",
        ":linux_armhf",
    ],
)

config_setting(
    name = "freebsd",
    values = {"cpu": "freebsd"},
    visibility = ["//visibility:public"],
)

# Features that are default ON are handled differently below.
#
config_setting(
    name = "no_gcp_support",
    define_values = {"no_gcp_support": "true"},
    visibility = ["//visibility:public"],
)

config_setting(
    name = "no_nccl_support",
    define_values = dict(
        if_google({"GOOGLE_CUDA_COMPILER": "clang"}),
        no_nccl_support = "true",
    ),
    visibility = ["//visibility:public"],
)

# Experimental features
config_setting(
    name = "stackdriver_support",
    define_values = {"stackdriver_support": "true"},
    visibility = ["//visibility:public"],
)

config_setting(
    name = "with_xla_support",
    define_values = {"with_xla_support": "true"},
    visibility = ["//visibility:public"],
)

# By default, XLA GPU is compiled into tensorflow when building with
# --config=cuda even when `with_xla_support` is false. The config setting
# here allows us to override the behavior if needed.
config_setting(
    name = "no_xla_deps_in_cuda",
    define_values = {"no_xla_deps_in_cuda": "true"},
    visibility = ["//visibility:public"],
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
    define_values = {
        "framework_shared_object": "true",
    },
    values = {
        "apple_platform_type": "macos",
        "cpu": "darwin",
    },
    visibility = ["//visibility:public"],
)

config_setting(
    name = "macos_arm64_with_framework_shared_object",
    define_values = {
        "framework_shared_object": "true",
    },
    values = {
        "apple_platform_type": "macos",
        "cpu": "darwin_arm64",
    },
    visibility = ["//visibility:public"],
)

selects.config_setting_group(
    name = "macos_with_framework_shared_object",
    match_any = [
        ":macos_x86_64_with_framework_shared_object",
        ":macos_arm64_with_framework_shared_object",
    ],
)

# Config setting that is satisfied when TensorFlow is being built with CUDA
# support through e.g. `--config=cuda` (or `--config=cuda_clang` in OSS).
alias(
    name = "is_cuda_enabled",
    actual = if_oss(
        "@local_config_cuda//:is_cuda_enabled",
        "@local_config_cuda//cuda:using_clang",
    ),
)

# Config setting that is satisfied when CUDA device code should be compiled
# with clang. It does not imply that CUDA support has been enabled.
alias(
    name = "is_cuda_compiler_clang",
    actual = if_oss(
        "@local_config_cuda//:is_cuda_compiler_clang",
        "@local_config_cuda//cuda:TRUE",
    ),
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

# Config setting that is satisfied when building with --config=cuda in OSS.
selects.config_setting_group(
    name = "is_cuda_enabled_and_oss",
    match_all = [
        ":is_cuda_enabled",
        ":oss",
    ],
)

# Config setting that is satisfied when building with --config=cuda for Windows
selects.config_setting_group(
    name = "is_cuda_enabled_and_windows",
    match_all = [
        ":is_cuda_enabled",
        ":windows",
    ],
)

# Config setting to use in select()s to distinguish open source build from
# google internal build on configurable attributes.
#
# For non-configurable distinction between OSS and Google builds, see
# `if_oss()` and `if_google()` macros in tensorflow.bzl.
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

# Setting to use when loading kernels dynamically
config_setting(
    name = "dynamic_loaded_kernels",
    define_values = {
        "dynamic_loaded_kernels": "true",
        "framework_shared_object": "true",
    },
    visibility = ["//visibility:public"],
)

config_setting(
    name = "using_rocm_hipcc",
    define_values = {"using_rocm_hipcc": "true"},
)

config_setting(
    name = "override_eigen_strong_inline",
    define_values = {"override_eigen_strong_inline": "true"},
    visibility = ["//visibility:public"],
)

# This flag specifies whether TensorFlow 2.0 API should be built instead
# of 1.* API. Note that TensorFlow 2.0 API is currently under development.
config_setting(
    name = "api_version_2",
    define_values = {"tf_api_version": "2"},
    visibility = ["//visibility:public"],
)

# This flag enables experimental MLIR support.
config_setting(
    name = "with_mlir_support",
    define_values = {"with_mlir_support": "true"},
    visibility = ["//visibility:public"],
)

# This flag forcibly enables experimental MLIR bridge support.
config_setting(
    name = "enable_mlir_bridge",
    define_values = {"enable_mlir_bridge": "true"},
    visibility = ["//visibility:public"],
)

# This flag forcibly disables experimental MLIR bridge support.
config_setting(
    name = "disable_mlir_bridge",
    define_values = {"enable_mlir_bridge": "false"},
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

# Specifies via a config setting if this is a mobile build or not, makes
# it easier to combine settings later.
selects.config_setting_group(
    name = "mobile",
    match_any = [
        ":android",
        ":chromiumos",
        # TODO(jakeharmon8): Move all platform configs to TSL with fuchsia (non-blocking)
        ":fuchsia",
        ":emscripten",
        ":ios",
    ],
    visibility = ["//visibility:public"],
)

# This flag disables all google production dependencies, intended for
# applications run with non-prod environment.
# TODO(timshen): Currently this option only disables some dependencies.
# See b/122528503.
# copybara:uncomment_begin(google-only)
# bool_flag(
#     name = "tf_no_prod_deps",
#     build_setting_default = False,
# )
#
# config_setting(
#     name = "no_prod_deps_define",
#     define_values = {"tf_no_prod_deps": "1"},
# )
#
# config_setting(
#     name = "no_prod_deps_flag",
#     flag_values = {":tf_no_prod_deps": "True"},
# )
#
# selects.config_setting_group(
#     name = "no_prod_deps",
#     match_any = [
#         ":no_prod_deps_define",
#         ":no_prod_deps_flag",
#     ],
# )
#
# config_setting(
#     name = "no_prod_deps_cuda",
#     define_values = {
#         "tf_no_prod_deps": "1",
#         "GOOGLE_CUDA_COMPILER": "clang",
#     },
# )
# copybara:uncomment_end

config_setting(
    name = "lite_protos_legacy",
    define_values = {"TENSORFLOW_PROTOS": "lite"},
    visibility = ["//visibility:public"],
)

config_setting(
    name = "full_protos",
    define_values = {"TENSORFLOW_PROTOS": "full"},
    visibility = ["//visibility:public"],
)

selects.config_setting_group(
    name = "lite_protos",
    match_any = [":lite_protos_legacy"],
    visibility = if_google(["//visibility:public"]),
)

selects.config_setting_group(
    name = "mobile_lite_protos",
    match_all = [
        ":lite_protos",
        ":mobile",
    ],
    visibility = ["//visibility:public"],
)

selects.config_setting_group(
    name = "mobile_full_protos",
    match_all = [
        ":full_protos",
        ":mobile",
    ],
    visibility = ["//visibility:public"],
)

# copybara:uncomment_begin(google-only)
# config_setting(
#     name = "portable_proto_force_third_party",
#     define_values = {"PORTABLE_PROTO_TRANSITION_MODE": "third_party"},
#     visibility = ["//visibility:public"],
# )
# copybara:uncomment_end

# 'enable_registration_v2' opts-in to a different implementation of op and
# kernel registration - REGISTER_OP, REGISTER_KERNEL_BUILDER, etc.
#
# This setting is currently experimental. The 'v2' implementation does _not_
# correspond to a particular, finalized design; rather, it relates to
# developing one.
#
# The current aim of the 'v2' implementation is to allow 'unused' ops and
# kernels to be discarded by the linker (to the benefit of binary size).
bool_flag(
    name = "enable_registration_v2",
    build_setting_default = False,
    visibility = ["//visibility:public"],
)

config_setting(
    name = "registration_v1",
    flag_values = {":enable_registration_v2": "False"},
    visibility = ["//visibility:public"],
)

config_setting(
    name = "registration_v2",
    flag_values = {":enable_registration_v2": "True"},
    visibility = ["//visibility:public"],
)

bool_flag(
    name = "enable_api_indexable",
    build_setting_default = False,
)

config_setting(
    name = "api_indexable_flag",
    flag_values = {":enable_api_indexable": "True"},
)

selects.config_setting_group(
    name = "api_indexable",
    match_any = [":api_indexable_flag"] + ADDITIONAL_API_INDEXABLE_SETTINGS,
    visibility = ["//visibility:public"],
)

# DO NOT ADD ANY NEW EXCEPTIONS TO THIS LIST!
# Instead, please use public APIs or public build rules TF provides.
# If you need functionality that is not exposed, we will work with you to expand our public APIs.
# TODO(b/173549186): Move Google-internal TF code out of learning/brain
package_group(
    name = "internal",
    packages = [
        "//devtools/python/indexer/...",
        "//learning/brain/keras/...",
        "//learning/brain/mlir/...",
        "//learning/brain/tfrt/...",
        "//learning/lib/ami/simple_ml/...",
        "//learning/pathways/...",
        "//learning/serving/contrib/tfrt/mlir/canonical_ops/...",
        "//learning/serving/experimental/remote_predict/...",
        "//perftools/accelerators/xprof/convert/...",
        "//perftools/accelerators/xprof/integration_tests/...",
        "//smartass/brain/configure/...",
        "//tensorflow/...",
        "//tensorflow_decision_forests/...",
        "//tensorflow_federated/...",
        "//tensorflow_models/google/recml/...",
        "//third_party/cloud_tpu/convergence_tools/sdc_monitoring/...",
        "//third_party/cloud_tpu/inference_converter/...",
        "//third_party/py/cloud_ml_autoflow/...",
        "//third_party/py/envlogger/...",
        "//third_party/py/gldm/...",
        "//third_party/py/guesslang/...",
        "//third_party/py/keras/...",
        "//third_party/py/tf_keras/...",
        "//third_party/yggdrasil_decision_forests/...",
        "//waymo/ml/cn/...",
    ],
)

package_group(
    name = "ndarray_tensor_allow_list",
    packages = [
        "//third_party/py/courier/...",
        "//third_party/py/tensorfn/...",
    ],
)

# Packages that use private types symbols, until they are exported.
# TODO(b/154650521) Remove.
# If this is modified, then copy.bara.sky must also be modified.
package_group(name = "types_allowlist")

# Packages that use StructuredTensors.
# TODO(b/159007891) Remove this package once StructuredTensor is exported.
# LINT.IfChange
package_group(name = "structured_tensor_allowlist")
# LINT.ThenChange(copy.bara.sky)

filegroup(
    name = "intel_binary_blob",
    data = if_mkl_ml(
        [
            "//third_party/mkl:intel_binary_blob",
        ],
    ),
)

bzl_library(
    name = "tensorflow_bzl",
    srcs = ["tensorflow.bzl"],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/core/platform:build_config_root_bzl",
        "//tensorflow/core/platform:rules_cc_bzl",
        "//third_party/compute_library:build_defs_bzl",
        "//third_party/llvm_openmp:openmp_bzl",
        "//third_party/mkl:build_defs_bzl",
        "//third_party/mkl_dnn:build_defs_bzl",
        "@bazel_skylib//lib:new_sets",
        "@bazel_skylib//rules:common_settings",
        "@local_config_cuda//cuda:build_defs_bzl",
        "@local_config_rocm//rocm:build_defs_bzl",
        "@local_config_tensorrt//:build_defs_bzl",
        "@local_tsl//tsl:tsl_bzl",
        "@local_tsl//tsl/platform/default:cuda_build_defs_bzl",
    ],
)

bzl_library(
    name = "tensorflow_default_bzl",
    srcs = ["tensorflow.default.bzl"],
    visibility = ["//visibility:public"],
    deps = [":tensorflow_bzl"],
)

# TODO(jakeharmon8): Remove these in favor of tsl:grpc
# copybara:comment_begin(oss-only)
cc_library(
    name = "grpc",
    visibility = ["//visibility:public"],
    deps = select({
        "//conditions:default": ["@com_github_grpc_grpc//:grpc"],
    }),
)

cc_library(
    name = "grpc++",
    visibility = ["//visibility:public"],
    deps = select({
        "//conditions:default": ["@com_github_grpc_grpc//:grpc++"],
    }),
)
# copybara:comment_end

# This is intended to be the same as tf_binary_additional_srcs:
# https://github.com/tensorflow/tensorflow/blob/cd67f4f3723f9165aabedd0171aaadc6290636e5/tensorflow/tensorflow.bzl#L396-L425
# And is usable in the "deps" attribute instead of the "srcs" attribute
# as a workaround for https://github.com/tensorflow/tensorflow/issues/34117
cc_import(
    name = "libtensorflow_framework_import_lib",
    shared_library = select({
        "//tensorflow:macos": ":libtensorflow_framework.dylib",
        "//conditions:default": ":libtensorflow_framework.so.%s" % VERSION_MAJOR,
    }),
)

# -------------------------------------------
# New rules should be added above this target.
# -------------------------------------------

# ** Targets for Windows build (start) **

# Build a shared library (DLL) by cc_binary from tf_custom_op_library_additional_deps_impl,
# it contains all object code from its dependencies.
# This target is only used for parsing the symbols to be exported in tensorflow.dll.
# Do NOT depend on it.
tf_native_cc_binary(
    name = "tf_custom_op_library_additional_deps.dll",
    linkshared = 1,
    linkstatic = 1,
    deps = tf_custom_op_library_additional_deps_impl(),
)

# Get a DEF file generated by parsing all object files
# of tf_custom_op_library_additional_deps.so
filegroup(
    name = "tensorflow_def_file",
    srcs = [":tf_custom_op_library_additional_deps.dll"],
    output_group = "def_file",
)

# Filter the DEF file to reduce the number of symbols to 64K or less.
# Note that we also write the name of the pyd file into DEF file so that
# the dynamic libraries of custom ops can find it at runtime.
genrule(
    name = "tensorflow_filtered_def_file",
    srcs = [":tensorflow_def_file"],
    outs = ["tensorflow_filtered_def_file.def"],
    cmd = select({
        "//tensorflow:windows": """
              $(location @local_config_def_file_filter//:def_file_filter) \\
              --input $(location :tensorflow_def_file) \\
              --output $@
          """,
        "//conditions:default": "touch $@",  # Just a placeholder for Unix platforms
    }),
    tools = ["@local_config_def_file_filter//:def_file_filter"],
    visibility = ["//visibility:public"],
)

# The interface library (tensorflow.dll.if.lib) for linking tensorflow DLL library (tensorflow.dll) on Windows.
# To learn more about import library (called interface library in Bazel):
#     https://docs.microsoft.com/en-us/cpp/build/linking-an-executable-to-a-dll?view=vs-2017#linking-implicitly
filegroup(
    name = "get_tensorflow_dll_import_lib",
    srcs = ["//tensorflow:tensorflow.dll"],
    output_group = "interface_library",
    visibility = ["//visibility:public"],
)

# Rename the import library for tensorflow.dll from tensorflow.dll.if.lib to tensorflow.lib
genrule(
    name = "tensorflow_dll_import_lib",
    srcs = [":get_tensorflow_dll_import_lib"],
    outs = ["tensorflow.lib"],
    cmd = select({
        "//tensorflow:windows": "cp -f $< $@",
        "//conditions:default": "touch $@",  # Just a placeholder for Unix platforms
    }),
    visibility = ["//visibility:public"],
)

# The interface library (tensorflow_cc.dll.if.lib) for linking tensorflow DLL library (tensorflow_cc.dll) on Windows.
# To learn more about import library (called interface library in Bazel):
#     https://docs.microsoft.com/en-us/cpp/build/linking-an-executable-to-a-dll?view=vs-2017#linking-implicitly
filegroup(
    name = "get_tensorflow_cc_dll_import_lib",
    srcs = ["//tensorflow:tensorflow_cc.dll"],
    output_group = "interface_library",
    visibility = ["//visibility:public"],
)

# Rename the import library for tensorflow.dll from tensorflow_cc.dll.if.lib to tensorflow.lib
genrule(
    name = "tensorflow_cc_dll_import_lib",
    srcs = [":get_tensorflow_cc_dll_import_lib"],
    outs = ["tensorflow_cc.lib"],
    cmd = select({
        "//tensorflow:windows": "cp -f $< $@",
        "//conditions:default": "touch $@",  # Just a placeholder for Unix platforms
    }),
    visibility = ["//visibility:public"],
)

# ** Targets for Windows build (end) **

exports_files(
    [
        "tf_version_script.lds",
        "tf_exported_symbols.lds",
        "tf_private_symbols.lds",
    ],
)

genrule(
    name = "install_headers",
    srcs = [
        "//tensorflow/c:headers",
        "//tensorflow/c/eager:headers",
        "//tensorflow/cc:headers",
        "//tensorflow/core:headers",
        "@local_xla//xla/stream_executor:stream_executor_install_hdrs",
    ],
    outs = ["include"],
    cmd = """
    mkdir $@
    for f in $(SRCS); do
      d="$${f%/*}"
      d="$${d#bazel-out/*/genfiles/}"
      d="$${d#bazel-out/*/bin/}"

      if [[ $${d} == *local_config_* ]]; then
        continue
      fi

      if [[ $${d} == external* ]]; then
        extname="$${d#*external/}"
        extname="$${extname%%/*}"
        if [[ $${TF_SYSTEM_LIBS:-} == *$${extname}* ]]; then
          continue
        fi

        d="$${d#*external/farmhash_archive/src}"
        d="$${d#*external/$${extname}/}"
      fi

      mkdir -p "$@/$${d}"
      cp "$${f}" "$@/$${d}/"
    done
    """,
    tags = ["manual"],
    visibility = ["//visibility:public"],
)

# copybara:comment_begin(oss-only)
copy_file(
    name = "root_init_gen",
    src = select({
        "api_version_2": "_api/v2/v2.py",
        "//conditions:default": "_api/v1/v1.py",
    }),
    out = "__init__.py",
)

genrule(
    name = "virtual_root_init_gen",
    srcs = select({
        "api_version_2": [":virtual_root_template_v2.__init__.py"],
        "//conditions:default": [":virtual_root_template_v1.__init__.py"],
    }),
    outs = ["virtual_root.__init__.py"],
    cmd = "cp $(SRCS) $(OUTS)",
)

generate_apis(
    name = "tf_python_api_gen_v1",
    api_version = 1,
    compat_api_versions = [
        1,
        2,
    ],
    compat_init_templates = [
        "compat_template_v1.__init__.py",
        "compat_template.__init__.py",
    ],
    output_dir = "_api/v1/",
    output_files = TENSORFLOW_API_INIT_FILES_V1,
    output_package = "tensorflow._api.v1",
    packages_to_ignore = ["tensorflow.python.framework.test_ops"],
    root_file_name = "v1.py",
    root_init_template = "api_template_v1.__init__.py",
)

generate_apis(
    name = "tf_python_api_gen_v2",
    api_packages_file_name = "api_packages.txt",
    api_version = 2,
    compat_api_versions = [
        1,
        2,
    ],
    compat_init_templates = [
        "compat_template_v1.__init__.py",
        "compat_template.__init__.py",
    ],
    output_dir = "_api/v2/",
    output_files = TENSORFLOW_API_INIT_FILES_V2,
    output_package = "tensorflow._api.v2",
    packages_to_ignore = ["tensorflow.python.framework.test_ops"],
    root_file_name = "v2.py",
    root_init_template = "api_template.__init__.py",
    visibility = ["//visibility:public"],
)

py_library(
    name = "tensorflow_py",
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [
        ":tensorflow_py_no_contrib",
        "//tensorflow/lite/python:analyzer",
        "//tensorflow/lite/python:lite",
        "//tensorflow/lite/python/authoring",
        "//tensorflow/python/estimator:estimator_py",
    ],
)

py_library(
    name = "tensorflow_py_no_contrib",
    srcs = select({
        "api_version_2": [":tf_python_api_gen_v2"],
        "//conditions:default": [":tf_python_api_gen_v1"],
    }) + [
        ":root_init_gen",
    ],
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/lite/python:analyzer",
        "//tensorflow/lite/python:lite",
        "//tensorflow/lite/python/authoring",
        "//tensorflow/python:no_contrib",
        "@pypi_keras_nightly//:pkg",
        "@pypi_tb_nightly//:pkg",
    ],
)
# copybara:comment_end
