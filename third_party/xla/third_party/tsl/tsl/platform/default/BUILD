# Tensorflow default + linux implementations of tensorflow/core/platform libraries.
load("@bazel_skylib//:bzl_library.bzl", "bzl_library")
load("@local_tsl//tsl/platform:rules_cc.bzl", "cc_library")
load(
    "@local_xla//xla/tsl:tsl.bzl",
    "if_not_fuchsia",
    "if_not_windows",
    "internal_visibility",
    "tsl_copts",
)
load("@local_xla//xla/tsl:tsl.default.bzl", "filegroup", "tsl_grpc_cc_dependencies")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = internal_visibility([
        "//tensorflow/core/lib/jpeg:__pkg__",
        "//tensorflow/core/platform:__pkg__",
        "//tsl/platform:__pkg__",
    ]),
    licenses = ["notice"],
)

cc_library(
    name = "casts",
    hdrs = ["casts.h"],
    tags = [
        "manual",
        "no_oss",
        "nobuilder",
    ],
)

cc_library(
    name = "context",
    hdrs = ["//tsl/platform:context.h"],
    tags = [
        "manual",
        "no_oss",
        "nobuilder",
    ],
    textual_hdrs = ["context.h"],
    deps = [
        "//tsl/platform",
    ],
)

cc_library(
    name = "criticality",
    hdrs = ["//tsl/platform:criticality.h"],
    textual_hdrs = ["criticality.h"],
    deps = [
        "//tsl/platform",
    ],
)

cc_library(
    name = "cuda_libdevice_path",
    srcs = ["cuda_libdevice_path.cc"],
    hdrs = ["//tsl/platform:cuda_libdevice_path.h"],
    compatible_with = [],
    tags = [
        "manual",
        "no_oss",
        "nobuilder",
    ],
    deps = [
        "//tsl/platform",
        "//tsl/platform:logging",
        "//tsl/platform:path",
        "//tsl/platform:types",
        "@local_config_cuda//cuda:cuda_headers",
    ],
)

cc_library(
    name = "dso_loader",
    srcs = ["dso_loader.cc"] + select({
        # include dynamic loading checker only for open source build
        "@local_xla//xla/tsl:oss": ["dlopen_checker.cc"],
        "//conditions:default": ["dlopen_checker_stub.cc"],
    }),
    hdrs = ["dso_loader.h"],
    compatible_with = [],
    copts = tsl_copts(),
    tags = [
        "manual",
        "nobuilder",
    ],
    deps = [
        "//tsl/platform:load_library",
        "//tsl/platform:logging",
        "//tsl/platform:path",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@local_config_cuda//cuda:cuda_headers",
        "@local_config_nccl//:nccl_config",
        "@local_config_rocm//rocm:rocm_headers",
        "@local_config_tensorrt//:tensorrt_headers",
    ],
)

cc_library(
    name = "env",
    srcs = [
        "posix_file_system.cc",
        "//tsl/platform:env.cc",
        "//tsl/platform:file_system.cc",
        "//tsl/platform:file_system_helper.cc",
        "//tsl/platform:threadpool.cc",
    ],
    hdrs = [
        "posix_file_system.h",
        "//tsl/platform:env.h",
        "//tsl/platform:file_system.h",
        "//tsl/platform:file_system_helper.h",
        "//tsl/platform:ram_file_system.h",
        "//tsl/platform:threadpool.h",
    ],
    copts = tsl_copts(),
    tags = [
        "manual",
        "no_oss",
        "nobuilder",
    ],
    deps = [
        "//tsl/platform",
        "//tsl/platform:blocking_counter",
        "//tsl/platform:context",
        "//tsl/platform:cord",
        "//tsl/platform:denormal",
        "//tsl/platform:env_time",
        "//tsl/platform:errors",
        "//tsl/platform:file_statistics",
        "//tsl/platform:load_library",
        "//tsl/platform:logging",
        "//tsl/platform:macros",
        "//tsl/platform:mutex",
        "//tsl/platform:path",
        "//tsl/platform:platform_port",
        "//tsl/platform:protobuf",
        "//tsl/platform:regexp",
        "//tsl/platform:scanner",
        "//tsl/platform:setround",
        "//tsl/platform:status",
        "//tsl/platform:statusor",
        "//tsl/platform:str_util",
        "//tsl/platform:strcat",
        "//tsl/platform:stringpiece",
        "//tsl/platform:stringprintf",
        "//tsl/platform:threadpool_interface",
        "//tsl/platform:tracing",
        "//tsl/platform:types",
        "//tsl/protobuf:error_codes_proto_impl_cc",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:optional",
        "@eigen_archive//:eigen3",
    ],
)

cc_library(
    name = "env_impl",
    srcs = [
        "env.cc",
    ],
    tags = [
        "manual",
        "no_oss",
        "nobuilder",
    ],
    deps = [
        ":env",
        "//tsl/platform:load_library",
        "//tsl/platform:logging",
        "//tsl/platform:mutex",
        "//tsl/platform:strcat",
        "//tsl/protobuf:error_codes_proto_impl_cc",
    ],
)

cc_library(
    name = "env_time",
    srcs = ["env_time.cc"],
    hdrs = ["//tsl/platform:env_time.h"],
    tags = [
        "manual",
        "no_oss",
        "nobuilder",
    ],
    deps = ["//tsl/platform:types"],
)

cc_library(
    name = "error_logging",
    srcs = ["error_logging.cc"],
    hdrs = ["//tsl/platform:error_logging.h"],
    tags = [
        "manual",
        "no_oss",
        "nobuilder",
    ],
    deps = [
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "grpc_credentials",
    srcs = ["grpc_credentials.cc"],
    hdrs = ["//tsl/platform:grpc_credentials.h"],
    tags = [
        "manual",
        "no_oss",
        "nobuilder",
    ],
    deps = [
        "//tsl/platform:logging",
        "@com_google_absl//absl/log:check",
    ] + tsl_grpc_cc_dependencies(),
)

cc_library(
    name = "human_readable_json",
    srcs = ["human_readable_json.cc"],
    hdrs = ["//tsl/platform:human_readable_json.h"],
    tags = [
        "manual",
        "no_oss",
        "nobuilder",
    ],
    deps = [
        "//tsl/platform:errors",
        "//tsl/platform:protobuf",
        "//tsl/platform:status",
        "//tsl/platform:strcat",
    ],
)

cc_library(
    name = "load_library",
    srcs = ["load_library.cc"],
    hdrs = ["//tsl/platform:load_library.h"],
    tags = [
        "manual",
        "no_oss",
        "nobuilder",
    ],
    deps = [
        "@com_google_absl//absl/status",
    ],
)

cc_library(
    name = "logging",
    srcs = ["logging.cc"],
    hdrs = ["//tsl/platform:logging.h"],
    tags = [
        "manual",
        "no_oss",
        "nobuilder",
    ],
    textual_hdrs = ["logging.h"],
    deps = [
        "//tsl/platform",
        "//tsl/platform:env_time",
        "//tsl/platform:macros",
        "//tsl/platform:mutex",
        "//tsl/platform:types",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/strings",
    ],
)

filegroup(
    name = "xla_cpu_runtime_srcs",
    srcs = [
        "integral_types.h",
    ] + if_not_windows(["env_time.cc"]),
)

cc_library(
    name = "mutex",
    srcs = [
        "mutex.cc",
        "mutex_data.h",
        "//tsl/platform:mutex.h",
    ],
    tags = [
        "manual",
        "no_oss",
        "nobuilder",
    ],
    textual_hdrs = ["mutex.h"],
    deps = [
        "//tsl/platform",
        "//tsl/platform:macros",
        "//tsl/platform:thread_annotations",
        "//tsl/platform:types",
        "@nsync//:nsync_cpp",
    ],
)

cc_library(
    name = "net",
    srcs = ["net.cc"],
    hdrs = [
        "//tsl/platform:net.h",
    ],
    tags = [
        "manual",
        "no_oss",
        "nobuilder",
    ],
    deps = [
        "//tsl/platform:logging",
        "//tsl/platform:strcat",
    ],
    alwayslink = True,
)

cc_library(
    name = "platform_port",
    srcs = [
        "port.cc",
        "//tsl/platform:cpu_info.cc",
    ],
    hdrs = [
        "//tsl/platform:cpu_info.h",
        "//tsl/platform:demangle.h",
        "//tsl/platform:host_info.h",
        "//tsl/platform:init_main.h",
        "//tsl/platform:mem.h",
        "//tsl/platform:numa.h",
        "//tsl/platform:snappy.h",
        "//tsl/platform/profile_utils:cpu_utils.h",
    ],
    copts = tsl_copts(),
    defines = ["TF_USE_SNAPPY"] + select({
        # TF Additional NUMA defines
        "@local_xla//xla/tsl:with_numa_support": ["TENSORFLOW_USE_NUMA"],
        "//conditions:default": [],
    }),
    tags = [
        "manual",
        "no_oss",
        "nobuilder",
    ],
    deps = [
        "//tsl/platform",
        "//tsl/platform:byte_order",
        "//tsl/platform:dynamic_annotations",
        "//tsl/platform:logging",
        "//tsl/platform:macros",
        "//tsl/platform:types",
        "//tsl/platform/profile_utils:profile_utils_cpu_utils",
        "@com_google_absl//absl/base",
        "@snappy",
    ] + select({
        # TF Additional NUMA dependencies
        "@local_xla//xla/tsl:with_numa_support": [
            # Don't merge in a single line
            "@hwloc",
        ],
        "//conditions:default": [],
    }),
)

cc_library(
    name = "resource",
    srcs = ["resource.cc"],
    hdrs = ["//tsl/platform:resource.h"],
    tags = [
        "manual",
        "no_oss",
        "nobuilder",
    ],
    deps = [
        "//tsl/platform:stringpiece",
    ],
)

cc_library(
    name = "rocm_rocdl_path",
    srcs = ["rocm_rocdl_path.cc"],
    hdrs = ["//tsl/platform:rocm_rocdl_path.h"],
    compatible_with = [],
    tags = [
        "manual",
        "no_oss",
        "nobuilder",
    ],
    deps = [
        "//tsl/platform:logging",
        "//tsl/platform:path",
        "//tsl/platform:types",
        "@local_config_rocm//rocm:rocm_headers",
    ],
)

cc_library(
    name = "stacktrace",
    hdrs = ["stacktrace.h"],
    linkopts = ["-ldl"],
    deps = [
        "//tsl/platform",
        "//tsl/platform:abi",
    ],
)

cc_library(
    name = "stacktrace_handler",
    srcs = ["stacktrace_handler.cc"],
    hdrs = ["//tsl/platform:stacktrace_handler_hdrs"],
    linkstatic = 1,
    deps = [
        "//tsl/platform",
        "//tsl/platform:stacktrace",
    ],
    alwayslink = 1,
)

cc_library(
    name = "subprocess",
    srcs = ["subprocess.cc"],
    hdrs = ["//tsl/platform:subprocess.h"],
    tags = [
        "manual",
        "no_oss",
        "nobuilder",
    ],
    textual_hdrs = ["subprocess.h"],
    deps = [
        "//tsl/platform",
        "//tsl/platform:logging",
        "//tsl/platform:macros",
        "//tsl/platform:mutex",
        "//tsl/platform:types",
    ],
    alwayslink = True,
)

cc_library(
    name = "tracing",
    srcs = [
        "tracing.cc",
        "//tsl/platform:tracing.cc",
    ],
    hdrs = ["//tsl/platform:tracing.h"],
    tags = [
        "manual",
        "no_oss",
        "nobuilder",
    ],
    textual_hdrs = ["tracing_impl.h"],
    deps = [
        "//tsl/platform",
        "//tsl/platform:hash",
        "//tsl/platform:logging",
        "//tsl/platform:macros",
        "//tsl/platform:str_util",
        "//tsl/platform:strcat",
        "//tsl/platform:stringpiece",
        "//tsl/platform:types",
    ],
    alwayslink = True,
)

cc_library(
    name = "types",
    tags = [
        "manual",
        "no_oss",
        "nobuilder",
    ],
    textual_hdrs = ["integral_types.h"],
)

cc_library(
    name = "unbounded_work_queue",
    srcs = ["unbounded_work_queue.cc"],
    hdrs = ["unbounded_work_queue.h"],
    tags = [
        "manual",
        "no_oss",
        "nobuilder",
    ],
    deps = [
        "//tsl/platform:env",
        "//tsl/platform:mutex",
        "//tsl/platform:notification",
        "//tsl/platform:platform_port",
        "@com_google_absl//absl/memory",
    ],
)

cc_library(
    name = "crash_analysis",
    srcs = [
        "crash_analysis.cc",
    ],
    hdrs = ["//tsl/platform:crash_analysis.h"],
    tags = [
        "manual",
        "no_oss",
        "nobuilder",
    ],
    textual_hdrs = ["crash_analysis.h"],
    visibility = internal_visibility(["//tensorflow:__subpackages__"]),
    deps = [
        "//tsl/platform",
        "//tsl/platform:protobuf",
    ],
)

cc_library(
    name = "status",
    tags = [
        "manual",
        "no_oss",
        "nobuilder",
    ],
    textual_hdrs = ["status.h"],
    visibility = internal_visibility(["//tensorflow:__subpackages__"]),
)

cc_library(
    name = "statusor",
    tags = [
        "manual",
        "no_oss",
        "nobuilder",
    ],
    textual_hdrs = ["statusor.h"],
    visibility = internal_visibility(["//tensorflow:__subpackages__"]),
    deps = [
        "//tsl/platform:macros",
        "//tsl/platform:status",
        "@com_google_absl//absl/status:statusor",
    ],
)

bzl_library(
    name = "cuda_build_defs_bzl",
    srcs = ["cuda_build_defs.bzl"],
    visibility = internal_visibility(["//tensorflow:__subpackages__"]),
)

bzl_library(
    name = "rules_cc_bzl",
    srcs = ["rules_cc.bzl"],
)

# Export source files needed for mobile builds, which do not use granular targets.
filegroup(
    name = "additional_mobile_srcs_no_runtime",
    visibility = internal_visibility(["//tensorflow/core/platform:__pkg__"]),
)

filegroup(
    name = "mobile_srcs_no_runtime",
    srcs = [
        "casts.h",
        "context.h",
        "env.cc",
        "integral_types.h",
        "load_library.cc",
        "port.cc",
        "posix_file_system.cc",
        "posix_file_system.h",
        "stacktrace.h",
        "status.h",
        "statusor.h",
        "tracing_impl.h",
        "//tsl/platform/profile_utils:cpu_utils.h",
        "//tsl/platform/profile_utils:i_cpu_utils_helper.h",
    ],
    visibility = internal_visibility([
        "//tensorflow/core/platform:__pkg__",
        "//tsl/platform:__pkg__",
    ]),
)

filegroup(
    name = "mobile_srcs_only_runtime",
    srcs = [
        "casts.h",
        "error_logging.cc",
        "mutex.h",
        "mutex_data.h",
        "unbounded_work_queue.cc",
        "unbounded_work_queue.h",
    ] + if_not_fuchsia([
        "subprocess.cc",
        "subprocess.h",
    ]),
    visibility = internal_visibility(["//tensorflow/core/platform:__pkg__"]),
)

exports_files(
    srcs = glob(
        ["*"],
        exclude = [
            "integral_types.h",
            "logging.h",
            "test.cc",
        ],
    ),
    visibility = internal_visibility(["//tensorflow/core/platform:__pkg__"]),
)

exports_files(
    srcs = [
        "integral_types.h",
        "logging.h",
        "test.cc",
    ],
    visibility = internal_visibility([
        "//tensorflow/core:__pkg__",
        "//tensorflow/core/lib/gif:__pkg__",
        "//tensorflow/core/lib/jpeg:__pkg__",
        "//tensorflow/core/platform:__pkg__",
    ]),
)
