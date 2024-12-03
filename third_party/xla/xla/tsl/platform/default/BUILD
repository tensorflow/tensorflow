# Tensorflow default + linux implementations of tensorflow/core/platform libraries.
load("@bazel_skylib//:bzl_library.bzl", "bzl_library")
load(
    "//xla/tsl:tsl.bzl",
    "if_cuda_tools",
    "if_not_fuchsia",
    "if_not_windows",
    "if_oss",
    "internal_visibility",
    "tsl_copts",
)
load("//xla/tsl:tsl.default.bzl", "filegroup", "tsl_grpc_cc_dependencies")
load("//xla/tsl/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = internal_visibility([
        "//tensorflow/core/lib/jpeg:__pkg__",
        "//tensorflow/core/platform:__pkg__",
        "@local_tsl//tsl/platform:__pkg__",
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
    hdrs = ["@local_tsl//tsl/platform:context.h"],
    tags = [
        "manual",
        "no_oss",
        "nobuilder",
    ],
    textual_hdrs = ["context.h"],
    deps = [
        "@local_tsl//tsl/platform",
    ],
)

cc_library(
    name = "criticality",
    hdrs = ["@local_tsl//tsl/platform:criticality.h"],
    tags = [
        "manual",
        "nobuilder",
    ],
    textual_hdrs = ["criticality.h"],
    deps = [
        "@local_tsl//tsl/platform",
    ],
)

cc_library(
    name = "cuda_root_path",
    srcs = ["cuda_root_path.cc"],
    hdrs = ["@local_tsl//tsl/platform:cuda_root_path.h"],
    compatible_with = [],
    data = if_cuda_tools([
        "@cuda_nvcc//:nvvm",
    ]),
    tags = [
        "manual",
        "no_oss",
        "nobuilder",
    ],
    deps = [
        "@local_config_cuda//cuda:cuda_headers",
        "@local_tsl//tsl/platform",
        "@local_tsl//tsl/platform:env",
        "@local_tsl//tsl/platform:logging",
        "@local_tsl//tsl/platform:path",
        "@local_tsl//tsl/platform:types",
    ],
)

cc_library(
    name = "dso_loader",
    srcs = ["dso_loader.cc"] + select({
        # include dynamic loading checker only for open source build
        "//xla/tsl:oss": ["dlopen_checker.cc"],
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
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@local_config_cuda//cuda:cuda_headers",
        "@local_config_rocm//rocm:rocm_headers",
        "@local_config_tensorrt//:tensorrt_headers",
        "@local_tsl//tsl/platform:load_library",
        "@local_tsl//tsl/platform:logging",
        "@local_tsl//tsl/platform:path",
    ] + if_oss(["@local_config_nccl//:nccl_config"]),
)

cc_library(
    name = "env",
    srcs = [
        "posix_file_system.cc",
        "@local_tsl//tsl/platform:env.cc",
        "@local_tsl//tsl/platform:file_system.cc",
        "@local_tsl//tsl/platform:file_system_helper.cc",
        "@local_tsl//tsl/platform:threadpool.cc",
    ],
    hdrs = [
        "posix_file_system.h",
        "@local_tsl//tsl/platform:env.h",
        "@local_tsl//tsl/platform:file_system.h",
        "@local_tsl//tsl/platform:file_system_helper.h",
        "@local_tsl//tsl/platform:ram_file_system.h",
        "@local_tsl//tsl/platform:threadpool.h",
    ],
    copts = tsl_copts(),
    tags = [
        "manual",
        "no_oss",
        "nobuilder",
    ],
    deps = [
        "//xla/tsl/protobuf:error_codes_proto_impl_cc",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:optional",
        "@eigen_archive//:eigen3",
        "@local_tsl//tsl/platform",
        "@local_tsl//tsl/platform:blocking_counter",
        "@local_tsl//tsl/platform:context",
        "@local_tsl//tsl/platform:cord",
        "@local_tsl//tsl/platform:denormal",
        "@local_tsl//tsl/platform:env_time",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:file_statistics",
        "@local_tsl//tsl/platform:load_library",
        "@local_tsl//tsl/platform:logging",
        "@local_tsl//tsl/platform:macros",
        "@local_tsl//tsl/platform:mutex",
        "@local_tsl//tsl/platform:path",
        "@local_tsl//tsl/platform:platform_port",
        "@local_tsl//tsl/platform:protobuf",
        "@local_tsl//tsl/platform:regexp",
        "@local_tsl//tsl/platform:scanner",
        "@local_tsl//tsl/platform:setround",
        "@local_tsl//tsl/platform:status",
        "@local_tsl//tsl/platform:statusor",
        "@local_tsl//tsl/platform:str_util",
        "@local_tsl//tsl/platform:strcat",
        "@local_tsl//tsl/platform:stringpiece",
        "@local_tsl//tsl/platform:stringprintf",
        "@local_tsl//tsl/platform:threadpool_interface",
        "@local_tsl//tsl/platform:tracing",
        "@local_tsl//tsl/platform:types",
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
        "//xla/tsl/protobuf:error_codes_proto_impl_cc",
        "@local_tsl//tsl/platform:load_library",
        "@local_tsl//tsl/platform:logging",
        "@local_tsl//tsl/platform:mutex",
        "@local_tsl//tsl/platform:strcat",
    ],
)

cc_library(
    name = "env_time",
    srcs = ["env_time.cc"],
    hdrs = ["@local_tsl//tsl/platform:env_time.h"],
    tags = [
        "manual",
        "no_oss",
        "nobuilder",
    ],
    deps = ["@local_tsl//tsl/platform:types"],
)

cc_library(
    name = "error_logging",
    srcs = ["error_logging.cc"],
    hdrs = ["@local_tsl//tsl/platform:error_logging.h"],
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
    hdrs = ["@local_tsl//tsl/platform:grpc_credentials.h"],
    tags = [
        "manual",
        "no_oss",
        "nobuilder",
    ],
    deps = [
        "@com_google_absl//absl/log:check",
        "@local_tsl//tsl/platform:logging",
    ] + tsl_grpc_cc_dependencies(),
)

cc_library(
    name = "human_readable_json",
    srcs = ["human_readable_json.cc"],
    hdrs = ["@local_tsl//tsl/platform:human_readable_json.h"],
    tags = [
        "manual",
        "no_oss",
        "nobuilder",
    ],
    deps = [
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:protobuf",
        "@local_tsl//tsl/platform:strcat",
        "@local_tsl//tsl/platform:stringpiece",
        "@local_tsl//tsl/platform:types",
    ],
)

cc_library(
    name = "load_library",
    srcs = ["load_library.cc"],
    hdrs = ["@local_tsl//tsl/platform:load_library.h"],
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
    hdrs = ["@local_tsl//tsl/platform:logging.h"],
    tags = [
        "manual",
        "no_oss",
        "nobuilder",
    ],
    textual_hdrs = ["logging.h"],
    deps = [
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/base:log_severity",
        "@com_google_absl//absl/strings",
        "@local_tsl//tsl/platform",
        "@local_tsl//tsl/platform:env_time",
        "@local_tsl//tsl/platform:macros",
        "@local_tsl//tsl/platform:mutex",
        "@local_tsl//tsl/platform:types",
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
        "@local_tsl//tsl/platform:mutex.h",
    ],
    tags = [
        "manual",
        "no_oss",
        "nobuilder",
    ],
    textual_hdrs = ["mutex.h"],
    deps = [
        "@local_tsl//tsl/platform",
        "@local_tsl//tsl/platform:macros",
        "@local_tsl//tsl/platform:thread_annotations",
        "@nsync//:nsync_cpp",
    ],
)

cc_library(
    name = "net",
    srcs = ["net.cc"],
    hdrs = [
        "@local_tsl//tsl/platform:net.h",
    ],
    tags = [
        "manual",
        "no_oss",
        "nobuilder",
    ],
    deps = [
        "@local_tsl//tsl/platform:logging",
        "@local_tsl//tsl/platform:strcat",
    ],
    alwayslink = True,
)

cc_library(
    name = "platform_port",
    srcs = [
        "port.cc",
        "@local_tsl//tsl/platform:cpu_info.cc",
    ],
    hdrs = [
        "//xla/tsl/platform/profile_utils:cpu_utils.h",
        "@local_tsl//tsl/platform:cpu_info.h",
        "@local_tsl//tsl/platform:demangle.h",
        "@local_tsl//tsl/platform:host_info.h",
        "@local_tsl//tsl/platform:init_main.h",
        "@local_tsl//tsl/platform:mem.h",
        "@local_tsl//tsl/platform:numa.h",
        "@local_tsl//tsl/platform:snappy.h",
    ],
    copts = tsl_copts(),
    defines = ["TF_USE_SNAPPY"] + select({
        # TF Additional NUMA defines
        "//xla/tsl:with_numa_support": ["TENSORFLOW_USE_NUMA"],
        "//conditions:default": [],
    }),
    tags = [
        "manual",
        "no_oss",
        "nobuilder",
    ],
    deps = [
        "//xla/tsl/platform/profile_utils:profile_utils_cpu_utils",
        "@com_google_absl//absl/base",
        "@local_tsl//tsl/platform",
        "@local_tsl//tsl/platform:byte_order",
        "@local_tsl//tsl/platform:dynamic_annotations",
        "@local_tsl//tsl/platform:logging",
        "@local_tsl//tsl/platform:macros",
        "@local_tsl//tsl/platform:types",
        "@snappy",
    ] + select({
        # TF Additional NUMA dependencies
        "//xla/tsl:with_numa_support": [
            # Don't merge in a single line
            "@hwloc",
        ],
        "//conditions:default": [],
    }),
)

cc_library(
    name = "resource",
    srcs = ["resource.cc"],
    hdrs = ["@local_tsl//tsl/platform:resource.h"],
    tags = [
        "manual",
        "no_oss",
        "nobuilder",
    ],
    deps = [
        "@local_tsl//tsl/platform:stringpiece",
    ],
)

cc_library(
    name = "rocm_rocdl_path",
    srcs = ["rocm_rocdl_path.cc"],
    hdrs = ["@local_tsl//tsl/platform:rocm_rocdl_path.h"],
    compatible_with = [],
    tags = [
        "manual",
        "no_oss",
        "nobuilder",
    ],
    deps = [
        "@local_config_rocm//rocm:rocm_headers",
        "@local_tsl//tsl/platform:logging",
        "@local_tsl//tsl/platform:path",
        "@local_tsl//tsl/platform:types",
    ],
)

cc_library(
    name = "stacktrace",
    hdrs = ["stacktrace.h"],
    linkopts = ["-ldl"],
    deps = [
        "@local_tsl//tsl/platform",
        "@local_tsl//tsl/platform:abi",
    ],
)

cc_library(
    name = "stacktrace_handler",
    srcs = ["stacktrace_handler.cc"],
    hdrs = ["@local_tsl//tsl/platform:stacktrace_handler_hdrs"],
    linkstatic = 1,
    deps = [
        "@local_tsl//tsl/platform",
        "@local_tsl//tsl/platform:stacktrace",
    ],
    alwayslink = 1,
)

cc_library(
    name = "subprocess",
    srcs = ["subprocess.cc"],
    hdrs = ["@local_tsl//tsl/platform:subprocess.h"],
    tags = [
        "manual",
        "no_oss",
        "nobuilder",
    ],
    textual_hdrs = ["subprocess.h"],
    deps = [
        "@local_tsl//tsl/platform",
        "@local_tsl//tsl/platform:logging",
        "@local_tsl//tsl/platform:macros",
        "@local_tsl//tsl/platform:mutex",
        "@local_tsl//tsl/platform:types",
    ],
    alwayslink = True,
)

cc_library(
    name = "tracing",
    srcs = [
        "tracing.cc",
        "@local_tsl//tsl/platform:tracing.cc",
    ],
    hdrs = ["@local_tsl//tsl/platform:tracing.h"],
    tags = [
        "manual",
        "no_oss",
        "nobuilder",
    ],
    textual_hdrs = ["tracing_impl.h"],
    deps = [
        "//xla/tsl/profiler/backends/cpu:threadpool_listener_state",
        "@local_tsl//tsl/platform",
        "@local_tsl//tsl/platform:hash",
        "@local_tsl//tsl/platform:logging",
        "@local_tsl//tsl/platform:macros",
        "@local_tsl//tsl/platform:str_util",
        "@local_tsl//tsl/platform:strcat",
        "@local_tsl//tsl/platform:stringpiece",
        "@local_tsl//tsl/platform:types",
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
        "@com_google_absl//absl/memory",
        "@local_tsl//tsl/platform:env",
        "@local_tsl//tsl/platform:mutex",
        "@local_tsl//tsl/platform:notification",
        "@local_tsl//tsl/platform:platform_port",
    ],
)

cc_library(
    name = "crash_analysis",
    srcs = [
        "crash_analysis.cc",
    ],
    hdrs = ["@local_tsl//tsl/platform:crash_analysis.h"],
    tags = [
        "manual",
        "no_oss",
        "nobuilder",
    ],
    textual_hdrs = ["crash_analysis.h"],
    visibility = internal_visibility(["//tensorflow:__subpackages__"]),
    deps = [
        "@local_tsl//tsl/platform",
        "@local_tsl//tsl/platform:protobuf",
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
        "@com_google_absl//absl/status:statusor",
        "@local_tsl//tsl/platform:macros",
        "@local_tsl//tsl/platform:status",
    ],
)

bzl_library(
    name = "cuda_build_defs_bzl",
    srcs = ["cuda_build_defs.bzl"],
    # copybara:uncomment_begin
    # test_tags = [
    # "manual",
    # "notap",
    # ],
    # copybara:uncomment_end
    visibility = internal_visibility(["//tensorflow:__subpackages__"]),
)

bzl_library(
    name = "rules_cc_bzl",
    srcs = ["rules_cc.bzl"],
)

bzl_library(
    name = "build_config_root_bzl",
    srcs = ["build_config_root.bzl"],
    # copybara:uncomment parse_tests = False,
    deps = ["@local_tsl//third_party/py/rules_pywrap:pywrap_bzl"],
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
        "//xla/tsl/platform/profile_utils:cpu_utils.h",
        "//xla/tsl/platform/profile_utils:i_cpu_utils_helper.h",
    ],
    visibility = internal_visibility([
        "//tensorflow/core/platform:__pkg__",
        "@local_tsl//tsl/platform:__pkg__",
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
