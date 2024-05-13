load(
    "@local_tsl//tsl/platform:rules_cc.bzl",
    "cc_library",
)

# Tensorflow windows-specific implementations of tensorflow/core/platform libraries.
load(
    "@local_xla//xla/tsl:tsl.bzl",
    "internal_visibility",
    "tsl_copts",
)
load("@local_xla//xla/tsl:tsl.default.bzl", "filegroup")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = internal_visibility([
        "//tensorflow/core/platform:__pkg__",
        "//tsl/platform:__pkg__",
    ]),
    licenses = ["notice"],
)

cc_library(
    name = "env",
    srcs = [
        "windows_file_system.cc",
        "windows_file_system.h",
        "//tsl/platform:env.cc",
        "//tsl/platform:file_system.cc",
        "//tsl/platform:file_system_helper.cc",
        "//tsl/platform:ram_file_system.h",
        "//tsl/platform:threadpool.cc",
    ],
    hdrs = [
        "//tsl/platform:env.h",
        "//tsl/platform:file_system.h",
        "//tsl/platform:file_system_helper.h",
        "//tsl/platform:threadpool.h",
    ],
    tags = [
        "manual",
        "no_oss",
        "nobuilder",
    ],
    deps = [
        ":error_windows",
        ":wide_char",
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
    deps = [
        "//tsl/platform:types",
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
    deps = ["//tsl/platform:types"],
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
        ":wide_char",
        "//tsl/platform:errors",
        "@com_google_absl//absl/status",
    ],
)

cc_library(
    name = "net",
    srcs = ["net.cc"],
    hdrs = [
        "//tsl/platform:net.h",
    ],
    linkopts = ["-DEFAULTLIB:ws2_32.lib"],
    tags = [
        "manual",
        "no_oss",
        "nobuilder",
    ],
    deps = [
        ":error_windows",
        "//tsl/platform:errors",
        "//tsl/platform:logging",
    ],
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
    ],
    copts = tsl_copts(),
    defines = ["TF_USE_SNAPPY"],
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
        "//tsl/platform:types",
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
    deps = ["//tsl/platform:mutex"],
)

cc_library(
    name = "stacktrace_handler",
    srcs = ["stacktrace_handler.cc"],
    hdrs = ["//tsl/platform:stacktrace_handler.h"],
    tags = [
        "manual",
        "no_oss",
        "nobuilder",
    ],
    deps = [
        "//tsl/platform:mutex",
        "//tsl/platform:stacktrace",
        "//tsl/platform:types",
    ],
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
        "//tsl/platform:strcat",
        "//tsl/platform:types",
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
