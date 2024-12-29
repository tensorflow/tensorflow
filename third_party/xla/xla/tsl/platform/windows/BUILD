# Tensorflow windows-specific implementations of tensorflow/core/platform libraries.
load(
    "//xla/tsl:tsl.bzl",
    "internal_visibility",
    "tsl_copts",
)
load("//xla/tsl:tsl.default.bzl", "filegroup")
load(
    "//xla/tsl/platform:rules_cc.bzl",
    "cc_library",
)

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = internal_visibility([
        "//tensorflow/core/platform:__pkg__",
        "@local_tsl//tsl/platform:__pkg__",
        "//xla/tsl/platform:__pkg__",
    ]),
    licenses = ["notice"],
)

cc_library(
    name = "env",
    srcs = [
        "windows_file_system.cc",
        "windows_file_system.h",
        "//xla/tsl/platform:env.cc",
        "//xla/tsl/platform:file_system.cc",
        "//xla/tsl/platform:file_system_helper.cc",
        "//xla/tsl/platform:threadpool.cc",
        "@local_tsl//tsl/platform:ram_file_system.h",
    ],
    hdrs = [
        "//xla/tsl/platform:env.h",
        "//xla/tsl/platform:file_system.h",
        "//xla/tsl/platform:file_system_helper.h",
        "//xla/tsl/platform:threadpool.h",
    ],
    tags = [
        "manual",
        "no_oss",
        "nobuilder",
    ],
    deps = [
        ":error_windows",
        ":wide_char",
        "//xla/tsl/protobuf:error_codes_proto_impl_cc",
        "@com_google_absl//absl/functional:any_invocable",
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
    ],
)

cc_library(
    name = "env_time",
    srcs = ["env_time.cc"],
    hdrs = ["//xla/tsl/platform:env_time.h"],
    tags = [
        "manual",
        "no_oss",
        "nobuilder",
    ],
    deps = [
        "@local_tsl//tsl/platform:types",
    ],
)

cc_library(
    name = "error_windows",
    srcs = ["error_windows.cc"],
    hdrs = ["error_windows.h"],
    linkopts = ["-DEFAULTLIB:ws2_32.lib"],
    tags = [
        "manual",
        "no_oss",
        "nobuilder",
    ],
    # This code is highly windows specific and should only be used with care
    # from this package.
    visibility = ["//visibility:private"],
)

cc_library(
    name = "intrinsics_port",
    srcs = ["intrinsics_port.h"],
    tags = [
        "manual",
        "no_oss",
        "nobuilder",
    ],
    deps = ["@local_tsl//tsl/platform:types"],
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
        ":wide_char",
        "@com_google_absl//absl/status",
        "@local_tsl//tsl/platform:errors",
    ],
)

cc_library(
    name = "net",
    srcs = ["net.cc"],
    hdrs = [
        "@local_tsl//tsl/platform:net.h",
    ],
    linkopts = ["-DEFAULTLIB:ws2_32.lib"],
    tags = [
        "manual",
        "no_oss",
        "nobuilder",
    ],
    deps = [
        ":error_windows",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:logging",
    ],
)

cc_library(
    name = "platform_port",
    srcs = [
        "port.cc",
        "@local_tsl//tsl/platform:cpu_info.cc",
    ],
    hdrs = [
        "@local_tsl//tsl/platform:cpu_info.h",
        "@local_tsl//tsl/platform:demangle.h",
        "@local_tsl//tsl/platform:host_info.h",
        "@local_tsl//tsl/platform:init_main.h",
        "@local_tsl//tsl/platform:mem.h",
        "@local_tsl//tsl/platform:numa.h",
        "@local_tsl//tsl/platform:snappy.h",
    ],
    copts = tsl_copts(),
    defines = ["TF_USE_SNAPPY"],
    tags = [
        "manual",
        "no_oss",
        "nobuilder",
    ],
    deps = [
        "@local_tsl//tsl/platform",
        "@local_tsl//tsl/platform:byte_order",
        "@local_tsl//tsl/platform:dynamic_annotations",
        "@local_tsl//tsl/platform:logging",
        "@local_tsl//tsl/platform:types",
        "@snappy",
    ],
)

cc_library(
    name = "stacktrace",
    srcs = ["stacktrace.cc"],
    hdrs = ["stacktrace.h"],
    tags = [
        "manual",
        "no_oss",
        "nobuilder",
    ],
    deps = ["@local_tsl//tsl/platform:mutex"],
)

cc_library(
    name = "stacktrace_handler",
    srcs = ["stacktrace_handler.cc"],
    hdrs = ["@local_tsl//tsl/platform:stacktrace_handler.h"],
    tags = [
        "manual",
        "no_oss",
        "nobuilder",
    ],
    deps = [
        "@local_tsl//tsl/platform:mutex",
        "@local_tsl//tsl/platform:stacktrace",
        "@local_tsl//tsl/platform:types",
    ],
)

cc_library(
    name = "subprocess",
    srcs = ["subprocess.cc"],
    hdrs = ["//xla/tsl/platform:subprocess.h"],
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
        "@local_tsl//tsl/platform:strcat",
        "@local_tsl//tsl/platform:types",
    ],
)

cc_library(
    name = "wide_char",
    srcs = ["wide_char.cc"],
    hdrs = ["wide_char.h"],
    tags = [
        "manual",
        "no_oss",
        "nobuilder",
    ],
)

filegroup(
    name = "xla_cpu_runtime_srcs",
    srcs = ["env_time.cc"],
)

exports_files(
    srcs = ["intrinsics_port.h"],
    visibility = internal_visibility(["//tensorflow/core/platform:__pkg__"]),
)
