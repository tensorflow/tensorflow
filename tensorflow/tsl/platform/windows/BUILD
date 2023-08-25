load("//tensorflow/tsl:tsl.default.bzl", "filegroup")

# Tensorflow windows-specific implementations of tensorflow/core/platform libraries.
load(
    "//tensorflow/tsl:tsl.bzl",
    "tsl_copts",
)
load(
    "//tensorflow/tsl/platform:rules_cc.bzl",
    "cc_library",
)

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [
        "//tensorflow/core/platform:__pkg__",
        "//tensorflow/tsl/platform:__pkg__",
    ],
    licenses = ["notice"],
)

cc_library(
    name = "env",
    srcs = [
        "windows_file_system.cc",
        "windows_file_system.h",
        "//tensorflow/tsl/platform:env.cc",
        "//tensorflow/tsl/platform:file_system.cc",
        "//tensorflow/tsl/platform:file_system_helper.cc",
        "//tensorflow/tsl/platform:ram_file_system.h",
        "//tensorflow/tsl/platform:threadpool.cc",
    ],
    hdrs = [
        "//tensorflow/tsl/platform:env.h",
        "//tensorflow/tsl/platform:file_system.h",
        "//tensorflow/tsl/platform:file_system_helper.h",
        "//tensorflow/tsl/platform:threadpool.h",
    ],
    tags = [
        "manual",
        "no_oss",
        "nobuilder",
    ],
    deps = [
        ":error_windows",
        ":wide_char",
        "//tensorflow/tsl/platform",
        "//tensorflow/tsl/platform:blocking_counter",
        "//tensorflow/tsl/platform:context",
        "//tensorflow/tsl/platform:cord",
        "//tensorflow/tsl/platform:denormal",
        "//tensorflow/tsl/platform:env_time",
        "//tensorflow/tsl/platform:errors",
        "//tensorflow/tsl/platform:file_statistics",
        "//tensorflow/tsl/platform:load_library",
        "//tensorflow/tsl/platform:logging",
        "//tensorflow/tsl/platform:macros",
        "//tensorflow/tsl/platform:mutex",
        "//tensorflow/tsl/platform:path",
        "//tensorflow/tsl/platform:platform_port",
        "//tensorflow/tsl/platform:protobuf",
        "//tensorflow/tsl/platform:regexp",
        "//tensorflow/tsl/platform:setround",
        "//tensorflow/tsl/platform:status",
        "//tensorflow/tsl/platform:statusor",
        "//tensorflow/tsl/platform:str_util",
        "//tensorflow/tsl/platform:strcat",
        "//tensorflow/tsl/platform:stringpiece",
        "//tensorflow/tsl/platform:stringprintf",
        "//tensorflow/tsl/platform:threadpool_interface",
        "//tensorflow/tsl/platform:tracing",
        "//tensorflow/tsl/platform:types",
        "//tensorflow/tsl/protobuf:error_codes_proto_impl_cc",
        "//third_party/eigen3",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:optional",
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
    hdrs = ["//tensorflow/tsl/platform:env_time.h"],
    tags = [
        "manual",
        "no_oss",
        "nobuilder",
    ],
    deps = [
        "//tensorflow/tsl/platform:types",
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
    deps = ["//tensorflow/tsl/platform:types"],
)

cc_library(
    name = "load_library",
    srcs = ["load_library.cc"],
    hdrs = ["//tensorflow/tsl/platform:load_library.h"],
    tags = [
        "manual",
        "no_oss",
        "nobuilder",
    ],
    deps = [
        ":wide_char",
        "//tensorflow/tsl/platform:errors",
        "//tensorflow/tsl/platform:status",
    ],
)

cc_library(
    name = "net",
    srcs = ["net.cc"],
    hdrs = [
        "//tensorflow/tsl/platform:net.h",
    ],
    linkopts = ["-DEFAULTLIB:ws2_32.lib"],
    tags = [
        "manual",
        "no_oss",
        "nobuilder",
    ],
    deps = [
        ":error_windows",
        "//tensorflow/tsl/platform:errors",
        "//tensorflow/tsl/platform:logging",
    ],
)

cc_library(
    name = "platform_port",
    srcs = [
        "port.cc",
        "//tensorflow/tsl/platform:cpu_info.cc",
    ],
    hdrs = [
        "//tensorflow/tsl/platform:cpu_info.h",
        "//tensorflow/tsl/platform:demangle.h",
        "//tensorflow/tsl/platform:host_info.h",
        "//tensorflow/tsl/platform:init_main.h",
        "//tensorflow/tsl/platform:mem.h",
        "//tensorflow/tsl/platform:numa.h",
        "//tensorflow/tsl/platform:snappy.h",
    ],
    copts = tsl_copts(),
    defines = ["TF_USE_SNAPPY"],
    tags = [
        "manual",
        "no_oss",
        "nobuilder",
    ],
    deps = [
        "//tensorflow/tsl/platform",
        "//tensorflow/tsl/platform:byte_order",
        "//tensorflow/tsl/platform:dynamic_annotations",
        "//tensorflow/tsl/platform:logging",
        "//tensorflow/tsl/platform:types",
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
    deps = ["//tensorflow/tsl/platform:mutex"],
)

cc_library(
    name = "stacktrace_handler",
    srcs = ["stacktrace_handler.cc"],
    hdrs = ["//tensorflow/tsl/platform:stacktrace_handler.h"],
    tags = [
        "manual",
        "no_oss",
        "nobuilder",
    ],
    deps = [
        "//tensorflow/tsl/platform:mutex",
        "//tensorflow/tsl/platform:stacktrace",
        "//tensorflow/tsl/platform:types",
    ],
)

cc_library(
    name = "subprocess",
    srcs = ["subprocess.cc"],
    hdrs = ["//tensorflow/tsl/platform:subprocess.h"],
    tags = [
        "manual",
        "no_oss",
        "nobuilder",
    ],
    textual_hdrs = ["subprocess.h"],
    deps = [
        "//tensorflow/tsl/platform",
        "//tensorflow/tsl/platform:logging",
        "//tensorflow/tsl/platform:macros",
        "//tensorflow/tsl/platform:mutex",
        "//tensorflow/tsl/platform:strcat",
        "//tensorflow/tsl/platform:types",
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
    visibility = ["//tensorflow/core/platform:__pkg__"],
)
