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
        "@tsl//tsl/platform:__pkg__",
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
        "//xla/tsl/platform:ram_file_system.h",
        "//xla/tsl/platform:threadpool.cc",
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
        "//xla/tsl/platform:env_time",
        "//xla/tsl/platform:errors",
        "//xla/tsl/platform:file_statistics",
        "//xla/tsl/platform:logging",
        "//xla/tsl/platform:macros",
        "//xla/tsl/platform:status",
        "//xla/tsl/platform:statusor",
        "//xla/tsl/platform:threadpool_interface",
        "//xla/tsl/platform:types",
        "//xla/tsl/protobuf:error_codes_proto_impl_cc",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/time",
        "@eigen_archive//:eigen3",
        "@tsl//tsl/platform",
        "@tsl//tsl/platform:blocking_counter",
        "@tsl//tsl/platform:context",
        "@tsl//tsl/platform:cord",
        "@tsl//tsl/platform:denormal",
        "@tsl//tsl/platform:load_library",
        "@tsl//tsl/platform:mutex",
        "@tsl//tsl/platform:path",
        "@tsl//tsl/platform:platform_port",
        "@tsl//tsl/platform:protobuf",
        "@tsl//tsl/platform:regexp",
        "@tsl//tsl/platform:setround",
        "@tsl//tsl/platform:str_util",
        "@tsl//tsl/platform:strcat",
        "@tsl//tsl/platform:stringpiece",
        "@tsl//tsl/platform:stringprintf",
        "@tsl//tsl/platform:tracing",
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
        "//xla/tsl/platform:types",
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
    deps = ["//xla/tsl/platform:types"],
)

cc_library(
    name = "load_library",
    srcs = ["load_library.cc"],
    hdrs = ["@tsl//tsl/platform:load_library.h"],
    tags = [
        "manual",
        "no_oss",
        "nobuilder",
    ],
    deps = [
        ":wide_char",
        "//xla/tsl/platform:errors",
        "@com_google_absl//absl/status",
    ],
)

cc_library(
    name = "net",
    srcs = ["net.cc"],
    hdrs = [
        "@tsl//tsl/platform:net.h",
    ],
    linkopts = ["-DEFAULTLIB:ws2_32.lib"],
    tags = [
        "manual",
        "no_oss",
        "nobuilder",
    ],
    deps = [
        ":error_windows",
        "//xla/tsl/platform:errors",
        "//xla/tsl/platform:logging",
    ],
)

cc_library(
    name = "platform_port",
    srcs = [
        "port.cc",
        "@tsl//tsl/platform:cpu_info.cc",
    ],
    hdrs = [
        "@tsl//tsl/platform:cpu_info.h",
        "@tsl//tsl/platform:demangle.h",
        "@tsl//tsl/platform:host_info.h",
        "@tsl//tsl/platform:init_main.h",
        "@tsl//tsl/platform:mem.h",
        "@tsl//tsl/platform:numa.h",
        "@tsl//tsl/platform:snappy.h",
    ],
    copts = tsl_copts(),
    defines = ["TF_USE_SNAPPY"],
    tags = [
        "manual",
        "no_oss",
        "nobuilder",
    ],
    deps = [
        "//xla/tsl/platform:byte_order",
        "//xla/tsl/platform:dynamic_annotations",
        "//xla/tsl/platform:logging",
        "//xla/tsl/platform:types",
        "@snappy",
        "@tsl//tsl/platform",
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
    deps = ["@tsl//tsl/platform:mutex"],
)

cc_library(
    name = "stacktrace_handler",
    srcs = ["stacktrace_handler.cc"],
    hdrs = ["@tsl//tsl/platform:stacktrace_handler.h"],
    tags = [
        "manual",
        "no_oss",
        "nobuilder",
    ],
    deps = [
        "//xla/tsl/platform:types",
        "@tsl//tsl/platform:mutex",
        "@tsl//tsl/platform:stacktrace",
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
        "//xla/tsl/platform:logging",
        "//xla/tsl/platform:macros",
        "//xla/tsl/platform:types",
        "@tsl//tsl/platform",
        "@tsl//tsl/platform:mutex",
        "@tsl//tsl/platform:strcat",
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
