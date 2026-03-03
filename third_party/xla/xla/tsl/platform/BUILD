load(
    "@bazel_skylib//:bzl_library.bzl",
    "bzl_library",
)
load(
    "//xla/tsl:tsl.bzl",
    "if_not_fuchsia",
    "if_oss",
    "internal_visibility",
    "tsl_copts",
)
load("//xla/tsl:tsl.default.bzl", "get_compatible_with_portable")
load(
    "//xla/tsl/platform:build_config.bzl",
    "tf_platform_alias",
    "tf_platform_deps",
    "tf_protobuf_compiler_deps",
    "tf_resource_deps",
    "tf_windows_aware_platform_deps",
    "tsl_cc_test",
    "tsl_grpc_credentials_deps",
)
load(
    "//xla/tsl/platform:build_config_root.bzl",
    "if_static",
)
load(
    "//xla/tsl/platform:rules_cc.bzl",
    "cc_library",
)

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [
        "//visibility:public",
    ],
    licenses = ["notice"],
)

exports_files(
    [
        "criticality.h",
        "subprocess.h",
        "env_time.h",
        "crash_analysis.h",
        "env.cc",
        "ram_file_system.h",
        "grpc_credentials.h",
        "resource.h",
        "rocm_rocdl_path.h",
        "resource_loader.h",
        "file_system.cc",
        "file_system.h",
        "file_system_helper.cc",
        "file_system_helper.h",
        "test.h",
        "threadpool.cc",
        "threadpool.h",
        "env.h",
        "numa_hwloc.cc",
        "numa_noop.cc",
    ],
    visibility = internal_visibility([
        "//tensorflow/core/platform:__subpackages__",
        "@tsl//tsl:__subpackages__",
        "//xla/tsl:__subpackages__",
    ]),
)

bzl_library(
    name = "rules_cc_bzl",
    srcs = ["rules_cc.bzl"],
    deps = tf_platform_alias("rules_cc_bzl") + [
        "//xla/tsl:package_groups_bzl",
    ],
)

bzl_library(
    name = "build_config_root_bzl",
    srcs = ["build_config_root.bzl"],
    deps = tf_platform_alias("build_config_root_bzl") + [
        "//xla/tsl:package_groups_bzl",
    ],
)

filegroup(
    name = "test_hdrs",
    testonly = 1,
    srcs = [
        "test.h",
        "test_benchmark.h",
    ],
    compatible_with = get_compatible_with_portable(),
    visibility = internal_visibility([
        "@tsl//tsl/platform:__pkg__",
        "//tensorflow/core:__pkg__",
        "//tensorflow/core/platform:__pkg__",
    ]),
)

filegroup(
    name = "android_test_srcs",
    testonly = 1,
    srcs = [
        "test.h",
    ],
    compatible_with = get_compatible_with_portable(),
    visibility = internal_visibility([
        "@tsl//tsl/platform:__pkg__",
        "//tensorflow/core:__pkg__",
        "//tensorflow/core/platform:__pkg__",
    ]),
)

filegroup(
    name = "ctstring",
    srcs = [
    ],
    compatible_with = get_compatible_with_portable(),
)

filegroup(
    name = "lib_hdrs",
    srcs = [
        "crash_analysis.h",
        "criticality.h",
        "dynamic_annotations.h",
        "env.h",
        "errors.h",
        "file_statistics.h",
        "file_system.h",
        "file_system_helper.h",
        "prefetch.h",
        "ram_file_system.h",
        "resource.h",
        "stack_frame.h",
        "statusor.h",
        "subprocess.h",
    ],
    compatible_with = get_compatible_with_portable(),
)

# Header files for tensorflow/core:platform_base.
filegroup(
    name = "base_hdrs",
    srcs = [
        "byte_order.h",
        "env_time.h",
        "macros.h",
        "threadpool.h",
        "threadpool_interface.h",
        "threadpool_options.h",
    ],
    compatible_with = get_compatible_with_portable(),
)

filegroup(
    name = "framework_lite_hdrs",
    srcs = [
        "byte_order.h",
        "dynamic_annotations.h",
        "macros.h",
        "prefetch.h",
    ],
    compatible_with = get_compatible_with_portable(),
)

# Export source files needed for mobile builds, which do not use granular targets.
filegroup(
    name = "mobile_srcs_no_runtime",
    srcs = [
        "byte_order.h",
        "criticality.h",
        "dynamic_annotations.h",
        "env.cc",
        "env.h",
        "env_time.h",
        "errors.cc",
        "errors.h",
        "file_statistics.h",
        "file_system.cc",
        "file_system.h",
        "file_system_helper.h",
        "intrusive_ptr.h",
        "macros.h",
        "prefetch.h",
        "ram_file_system.h",
        "resource.h",
        "status.cc",
        "status.h",
        "statusor.h",
        "threadpool.cc",
        "threadpool.h",
        "threadpool_interface.h",
    ] + select({
        "//xla/tsl:fuchsia": [],
        "//conditions:default": [
            "file_system_helper.cc",
        ],
    }),
    compatible_with = get_compatible_with_portable(),
)

filegroup(
    name = "mobile_srcs_only_runtime",
    srcs = [] + if_not_fuchsia(["subprocess.h"]),
    compatible_with = get_compatible_with_portable(),
    visibility = internal_visibility([
        "@tsl//tsl/platform:__pkg__",
        "//tensorflow/core:__pkg__",
        "//tensorflow/core/platform:__pkg__",
    ]),
)

filegroup(
    name = "legacy_lib_internal_headers",
    srcs = glob(
        [
            "*.h",
        ],
        exclude = [
            "dynamic_annotations.h",
        ],
    ) + [
        "//xla/tsl/platform/profile_utils:android_armv7a_cpu_utils_helper.h",
        "//xla/tsl/platform/profile_utils:clock_cycle_profiler.h",
        "//xla/tsl/platform/profile_utils:cpu_utils.h",
        "//xla/tsl/platform/profile_utils:i_cpu_utils_helper.h",
    ],
    compatible_with = get_compatible_with_portable(),
)

filegroup(
    name = "lib_proto_parsing_hdrs",
    srcs = [
        "macros.h",
    ],
    compatible_with = get_compatible_with_portable(),
    visibility = internal_visibility([
        "@tsl//tsl/platform:__pkg__",
        "//tensorflow/core:__pkg__",
    ]),
)

filegroup(
    name = "lib_internal_public_hdrs",
    srcs = [
        "intrusive_ptr.h",
        "status.h",
        "statusor.h",
    ],
    compatible_with = get_compatible_with_portable(),
    visibility = internal_visibility([
        "//tensorflow/core:__pkg__",
        "//tensorflow/core/platform:__pkg__",
        "@tsl//tsl/platform:__pkg__",
    ]),
)

filegroup(
    name = "tflite_portable_logging_hdrs",
    srcs = [
        "macros.h",
    ],
    compatible_with = get_compatible_with_portable(),
    visibility = internal_visibility([
        "//tensorflow/core:__pkg__",
        "//tensorflow/core/lib/jpeg:__pkg__",
        "@tsl//tsl/platform:__pkg__",
    ]),
)

filegroup(
    name = "jpeg_internal_hdrs",
    srcs = [
        "dynamic_annotations.h",
        "macros.h",
    ],
    compatible_with = get_compatible_with_portable(),
    visibility = internal_visibility([
        "//tensorflow/core:__pkg__",
        "//tensorflow/core/lib/jpeg:__pkg__",
        "//tensorflow/core/platform:__pkg__",
        "@tsl//tsl/platform:__pkg__",
    ]),
)

filegroup(
    name = "gif_internal_hdrs",
    srcs = [
        "dynamic_annotations.h",
        "macros.h",
    ],
    compatible_with = get_compatible_with_portable(),
    visibility = internal_visibility([
        "//tensorflow/core:__pkg__",
        "//tensorflow/core/lib/gif:__pkg__",
        "//tensorflow/core/platform:__pkg__",
        "@tsl//tsl/platform:__pkg__",
    ]),
)

filegroup(
    name = "xla_cpu_runtime_srcs",
    srcs = [
        "dynamic_annotations.h",
        "env_time.h",
        "macros.h",
    ],
    compatible_with = get_compatible_with_portable(),
)

filegroup(
    name = "stacktrace_handler_hdrs",
    srcs = [
    ],
    compatible_with = get_compatible_with_portable(),
)

filegroup(
    name = "lib_internal_private_hdrs",
    srcs = [
    ],
    compatible_with = get_compatible_with_portable(),
    visibility = internal_visibility([
        "//tensorflow/core:__pkg__",
        "@tsl//tsl/platform:__pkg__",
    ]),
)

filegroup(
    name = "tensor_float_32_hdr",
    srcs = [],
    compatible_with = get_compatible_with_portable(),
)

cc_library(
    name = "subprocess",
    textual_hdrs = [
        "subprocess.h",
    ],
    deps = tf_windows_aware_platform_deps("subprocess") + [
        ":types",
        "@tsl//tsl/platform",
    ],
)

tsl_cc_test(
    name = "subprocess_test",
    size = "small",
    srcs = ["subprocess_test.cc"],
    data = [
        "//xla/tsl/platform/testdata:test_crash",
        "//xla/tsl/platform/testdata:test_echo",
        "//xla/tsl/platform/testdata:test_echo_argv_1",
        "//xla/tsl/platform/testdata:test_noop",
        "//xla/tsl/platform/testdata:test_pwd",
        "//xla/tsl/platform/testdata:test_stderr",
    ],
    tags = ["no_windows"],
    deps = [
        ":env",
        ":subprocess",
        ":test",
        "//xla/tsl/lib/core:status_test_util",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
        "@com_google_googletest//:gtest_main",
        "@tsl//tsl/platform:path",
        "@tsl//tsl/platform:strcat",
    ],
)

cc_library(
    name = "env",
    # TODO: b/456605953 - Adding the deps that build_clenaner wants may break some TF builds, so keep
    # nofixdeps for now, and try to romove it in a separate CL.
    tags = ["nofixdeps"],
    textual_hdrs = [
        "env.h",
        "file_system.h",
        "file_system_helper.h",
        "threadpool.h",
    ],
    deps = tf_windows_aware_platform_deps("env") + if_static([":env_impl"]) + [
        ":env_time",
        ":errors",
        ":file_statistics",
        ":macros",
        ":status",
        ":statusor",
        ":threadpool_interface",
        ":types",
        "//xla/tsl/concurrency:executor",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@com_google_protobuf//:protobuf",
        "@com_google_protobuf//:protobuf_lite",
        "@tsl//tsl/platform",
        "@tsl//tsl/platform:cord",
        "@tsl//tsl/platform:platform_port",
        "@tsl//tsl/platform:protobuf",
        "@tsl//tsl/platform:stringpiece",
    ],
)

cc_library(
    name = "env_impl",
    deps = tf_windows_aware_platform_deps("env_impl"),
)

tsl_cc_test(
    name = "env_test",
    srcs = ["env_test.cc"],
    deps = [
        ":env",
        "//xla/tsl/testing:temporary_directory",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
        "@com_google_googletest//:gtest_main",
        "@tsl//tsl/platform:path",
    ],
)

cc_library(
    name = "env_time",
    compatible_with = get_compatible_with_portable(),
    textual_hdrs = ["env_time.h"],
    deps = tf_windows_aware_platform_deps("env_time") + [":types"],
)

cc_library(
    name = "errors",
    srcs = ["errors.cc"],
    hdrs = ["errors.h"],
    deps = [
        ":logging",
        ":macros",
        ":status",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:cord",
        "@tsl//tsl/platform:strcat",
    ],
)

tsl_cc_test(
    name = "errors_test",
    size = "small",
    srcs = ["errors_test.cc"],
    deps = [
        ":errors",
        ":test",
        "@com_google_absl//absl/status",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "file_system_helper",
    srcs = [
        "file_system.h",
        "file_system_helper.cc",
    ],
    hdrs = ["file_system_helper.h"],
    deps = [
        ":env",
        ":errors",
        ":file_statistics",
        ":macros",
        ":status",
        ":statusor",
        ":types",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:span",
        "@tsl//tsl/platform",
        "@tsl//tsl/platform:cord",
        "@tsl//tsl/platform:path",
        "@tsl//tsl/platform:platform_port",
        "@tsl//tsl/platform:str_util",
        "@tsl//tsl/platform:stringpiece",
    ],
)

tsl_cc_test(
    name = "file_system_helper_test",
    size = "small",
    srcs = ["file_system_helper_test.cc"],
    deps = [
        ":env",
        ":env_impl",  # buildcleaner: keep
        ":file_system_helper",
        "//xla/tsl/lib/core:status_test_util",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "file_statistics",
    hdrs = ["file_statistics.h"],
    deps = [
        ":types",
    ],
)

cc_library(
    name = "logging",
    srcs = ["logging.cc"],
    hdrs = ["logging.h"],
    compatible_with = get_compatible_with_portable(),
    visibility = [
        "//visibility:public",
    ],
    deps = [
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/log:globals",
        "@com_google_absl//absl/log:vlog_is_on",
        "@com_google_absl//absl/strings:string_view",
    ] + if_oss([":logging_initializer"]),
)

cc_library(
    name = "logging_initializer",
    srcs = ["logging_initializer.cc"],
    deps = [
        "@com_google_absl//absl/base:log_severity",
        "@com_google_absl//absl/log:globals",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
    ],
    alwayslink = True,
)

tsl_cc_test(
    name = "logging_test",
    size = "small",
    srcs = [
        "logging_test.cc",
    ],
    deps = [
        ":logging",
        ":statusor",
        ":test",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest",
        "@tsl//tsl/platform:path",
        "@tsl//tsl/platform:stacktrace_handler",
    ],
)

cc_library(
    name = "macros",
    hdrs = ["macros.h"],
    compatible_with = get_compatible_with_portable(),
    deps = ["@com_google_absl//absl/base:core_headers"],
)

cc_library(
    name = "status",
    srcs = ["status.cc"],
    hdrs = ["status.h"],
    deps = [
        ":logging",
        ":macros",
        ":stack_frame",
        ":types",
        "//xla/tsl/protobuf:error_codes_proto_impl_cc",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/base:log_severity",
        "@com_google_absl//absl/functional:function_ref",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/log:log_entry",
        "@com_google_absl//absl/log:log_sink",
        "@com_google_absl//absl/log:log_sink_registry",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:cord",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:optional",
        "@tsl//tsl/platform",
        "@tsl//tsl/platform:stacktrace",
        "@tsl//tsl/platform:str_util",
        "@tsl//tsl/platform:strcat",
        "@tsl//tsl/platform:thread_annotations",
    ] + tf_platform_deps("status"),
)

tsl_cc_test(
    name = "status_test",
    size = "small",
    srcs = ["status_test.cc"],
    deps = [
        ":errors",
        ":stack_frame",
        ":status",
        ":status_to_from_proto",
        ":test",
        "//xla/tsl/protobuf:error_codes_proto_impl_cc",
        "//xla/tsl/protobuf:status_proto_cc",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_absl//absl/strings:cord",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "status_matchers",
    testonly = 1,
    hdrs = ["status_matchers.h"],
    deps = [
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/status:status_matchers",
    ],
)

cc_library(
    name = "status_to_from_proto",
    srcs = [
        "status_to_from_proto.cc",
    ],
    hdrs = ["status_to_from_proto.h"],
    deps = [
        ":status",
        "//xla/tsl/protobuf:error_codes_proto_impl_cc",
        "//xla/tsl/protobuf:status_proto_cc",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:cord",
    ] + tf_platform_deps("status"),
)

cc_library(
    name = "statusor",
    hdrs = ["statusor.h"],
    deps = [
        ":errors",
        ":logging",
        ":macros",
        ":status",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        "@tsl//tsl/platform",
    ] + tf_platform_deps("statusor"),
)

tsl_cc_test(
    name = "statusor_test",
    size = "small",
    srcs = ["statusor_test.cc"],
    deps = [
        ":macros",
        ":statusor",
        ":test",
        ":test_benchmark",
        ":test_main",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
    ],
)

cc_library(
    name = "test",
    testonly = True,
    srcs = ["test.cc"],
    compatible_with = get_compatible_with_portable(),
    textual_hdrs = ["test.h"],
    deps = [
        ":logging",
        ":macros",
        ":types",
        "@com_google_googletest//:gtest_for_library",
        "@tsl//tsl/platform",
        "@tsl//tsl/platform:net",
        "@tsl//tsl/platform:path",
    ],
)

cc_library(
    name = "test_benchmark",
    testonly = True,
    hdrs = ["test_benchmark.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        "@com_google_benchmark//:benchmark",
        "@tsl//tsl/platform",
    ],
)

cc_library(
    name = "test_main",
    testonly = 1,
    srcs = ["test_main.cc"],
    copts = tsl_copts(),
    linkopts = select({
        "//xla/tsl:windows": [],
        "//conditions:default": ["-lm"],
    }),
    deps = [
        ":test",
        ":test_benchmark",
        "@com_google_absl//absl/strings",
        "@tsl//tsl/platform",
        "@tsl//tsl/platform:stacktrace_handler",
    ],
    alwayslink = 1,
)

tsl_cc_test(
    name = "threadpool_test",
    srcs = ["threadpool_test.cc"],
    deps = [
        ":env",
        ":env_impl",  # buildcleaner: keep
        ":test",
        "//xla/tsl/concurrency:executor",
        "@com_google_absl//absl/synchronization",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "threadpool_interface",
    hdrs = ["threadpool_interface.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        ":types",
        "@eigen_archive//:eigen3",
    ],
)

cc_library(
    name = "threadpool_options",
    hdrs = ["threadpool_options.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        ":threadpool_interface",
    ],
)

cc_library(
    name = "types",
    hdrs = ["types.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        "@com_google_absl//absl/base:core_headers",
        "@tsl//tsl/platform",
        "@tsl//tsl/platform:bfloat16",
        "@tsl//tsl/platform:ml_dtypes",
        "@tsl//tsl/platform:tstring",
    ] + tf_platform_deps("types"),
)

# copybara:comment_begin(oss-only)
cc_library(
    name = "status_macros",
    hdrs = ["status_macros.h"],
    compatible_with = get_compatible_with_portable(),
    visibility = internal_visibility([]),  # The header should be used only in OSS.
    deps = [":statusor"],
)
# copybara:comment_end

cc_library(
    name = "byte_order",
    hdrs = ["byte_order.h"],
    compatible_with = get_compatible_with_portable(),
)

cc_library(
    name = "crash_analysis",
    hdrs = ["crash_analysis.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        "@tsl//tsl/platform",
    ] + tf_platform_deps("crash_analysis"),
)

cc_library(
    name = "dynamic_annotations",
    hdrs = ["dynamic_annotations.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        "@com_google_absl//absl/base:dynamic_annotations",
    ],
)

cc_library(
    name = "grpc_credentials",
    compatible_with = get_compatible_with_portable(),
    tags = ["nofixdeps"],
    textual_hdrs = ["grpc_credentials.h"],
    deps = tsl_grpc_credentials_deps(),
)

cc_library(
    name = "intrusive_ptr",
    hdrs = ["intrusive_ptr.h"],
    deps = [],
)

tsl_cc_test(
    name = "intrusive_ptr_test",
    size = "small",
    srcs = [
        "intrusive_ptr_test.cc",
    ],
    deps = [
        ":intrusive_ptr",
        ":test",
        ":test_main",
        "@tsl//tsl/platform:refcount",
    ],
)

cc_library(
    name = "prefetch",
    hdrs = ["prefetch.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/base:prefetch",
    ],
)

cc_library(
    name = "protobuf_compiler",
    hdrs = ["protobuf_compiler.h"],
    deps = tf_protobuf_compiler_deps(),
)

cc_library(
    name = "resource",
    textual_hdrs = ["resource.h"],
    deps = [
        "@tsl//tsl/platform:stringpiece",
    ] + tf_resource_deps(),
)

cc_library(
    name = "resource_loader",
    testonly = 1,
    srcs = if_oss(
        ["//xla/tsl/platform/default:resource_loader.cc"],
        google_value = ["//xla/tsl/platform/google:resource_loader.cc"],
    ),
    textual_hdrs = ["resource_loader.h"],
    deps = [
        ":logging",
        ":test",
        "@tsl//tsl/platform",
        "@tsl//tsl/platform:path",
    ] + if_oss(["@bazel_tools//tools/cpp/runfiles"]),
)

cc_library(
    name = "rocm_rocdl_path",
    textual_hdrs = ["rocm_rocdl_path.h"],
    deps = tf_platform_deps("rocm_rocdl_path") + [":types"],
)

cc_library(
    name = "stack_frame",
    hdrs = ["stack_frame.h"],
    compatible_with = get_compatible_with_portable(),
)

cc_library(
    name = "criticality",
    compatible_with = get_compatible_with_portable(),
    textual_hdrs = ["criticality.h"],
    deps = tf_platform_deps("criticality") + ["@tsl//tsl/platform"],
)

tsl_cc_test(
    name = "criticality_test",
    size = "small",
    srcs = [
        "criticality_test.cc",
    ],
    deps = [
        ":criticality",
        ":test",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "debug_me_context",
    hdrs = ["debug_me_context.h"],
    deps = ["@com_google_absl//absl/container:flat_hash_map"],
)

cc_library(
    name = "ram_file_system",
    hdrs = ["ram_file_system.h"],
    deps = [
        ":env",
        ":errors",
        ":file_statistics",
        ":types",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:cord",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/synchronization",
    ],
)

tsl_cc_test(
    name = "ram_file_system_test",
    srcs = ["ram_file_system_test.cc"],
    deps = [
        ":env",
        ":ram_file_system",
        "//xla/tsl/lib/core:status_test_util",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "embedded_filesystem",
    srcs = ["embedded_filesystem.cc"],
    hdrs = ["embedded_filesystem.h"],
    deps = [
        ":env",
        ":file_statistics",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:span",
    ],
)

tsl_cc_test(
    name = "embedded_filesystem_test",
    srcs = ["embedded_filesystem_test.cc"],
    deps = [
        ":embedded_filesystem",
        ":env",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
    ],
)
