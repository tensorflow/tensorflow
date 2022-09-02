load("//tensorflow:tensorflow.bzl", "filegroup")

# Tensorflow windows-specific implementations of tensorflow/core/platform libraries.
load(
    "//tensorflow:tensorflow.bzl",
    "tf_copts",
)
load(
    "//tensorflow/core/platform:rules_cc.bzl",
    "cc_library",
)

package(
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
        "//tensorflow/core/platform:env.cc",
        "//tensorflow/core/platform:file_system.cc",
        "//tensorflow/core/platform:file_system_helper.cc",
        "//tensorflow/core/platform:ram_file_system.h",
        "//tensorflow/core/platform:threadpool.cc",
    ],
    hdrs = [
        "//tensorflow/core/platform:env.h",
        "//tensorflow/core/platform:file_system.h",
        "//tensorflow/core/platform:file_system_helper.h",
        "//tensorflow/core/platform:threadpool.h",
    ],
    tags = [
        "manual",
        "no_oss",
        "nobuilder",
    ],
    deps = [
        ":error_windows",
        ":wide_char",
        "//tensorflow/core/lib/core:error_codes_proto_cc",
        "//tensorflow/core/lib/core:stringpiece",
        "//tensorflow/core/platform",
        "//tensorflow/core/platform:blocking_counter",
        "//tensorflow/core/platform:context",
        "//tensorflow/core/platform:cord",
        "//tensorflow/core/platform:denormal",
        "//tensorflow/core/platform:env_time",
        "//tensorflow/core/platform:errors",
        "//tensorflow/core/platform:file_statistics",
        "//tensorflow/core/platform:load_library",
        "//tensorflow/core/platform:logging",
        "//tensorflow/core/platform:macros",
        "//tensorflow/core/platform:mutex",
        "//tensorflow/core/platform:path",
        "//tensorflow/core/platform:platform_port",
        "//tensorflow/core/platform:protobuf",
        "//tensorflow/core/platform:regexp",
        "//tensorflow/core/platform:setround",
        "//tensorflow/core/platform:status",
        "//tensorflow/core/platform:statusor",
        "//tensorflow/core/platform:str_util",
        "//tensorflow/core/platform:strcat",
        "//tensorflow/core/platform:stringpiece",
        "//tensorflow/core/platform:stringprintf",
        "//tensorflow/core/platform:threadpool_interface",
        "//tensorflow/core/platform:tracing",
        "//tensorflow/core/platform:types",
        "//third_party/eigen3",
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
    hdrs = ["//tensorflow/core/platform:env_time.h"],
    tags = [
        "manual",
        "no_oss",
        "nobuilder",
    ],
    deps = ["//tensorflow/core/platform:types"],
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
    deps = ["//tensorflow/core/platform:types"],
)

cc_library(
    name = "load_library",
    srcs = ["load_library.cc"],
    hdrs = ["//tensorflow/core/platform:load_library.h"],
    tags = [
        "manual",
        "no_oss",
        "nobuilder",
    ],
    deps = [
        ":wide_char",
        "//tensorflow/core/platform:errors",
        "//tensorflow/core/platform:status",
    ],
)

cc_library(
    name = "net",
    srcs = ["net.cc"],
    hdrs = [
        "//tensorflow/core/platform:net.h",
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
        "//tensorflow/core/platform:errors",
        "//tensorflow/core/platform:logging",
    ],
)

cc_library(
    name = "platform_port",
    srcs = [
        "port.cc",
        "//tensorflow/core/platform:cpu_info.cc",
    ],
    hdrs = [
        "//tensorflow/core/platform:cpu_info.h",
        "//tensorflow/core/platform:demangle.h",
        "//tensorflow/core/platform:host_info.h",
        "//tensorflow/core/platform:init_main.h",
        "//tensorflow/core/platform:numa.h",
        "//tensorflow/core/platform:snappy.h",
        "//tensorflow/tsl/platform:init_main.h",
        "//tensorflow/tsl/platform:mem.h",
    ],
    copts = tf_copts(),
    defines = ["TF_USE_SNAPPY"],
    tags = [
        "manual",
        "no_oss",
        "nobuilder",
    ],
    deps = [
        "//tensorflow/core/platform",
        "//tensorflow/core/platform:byte_order",
        "//tensorflow/core/platform:dynamic_annotations",
        "//tensorflow/core/platform:logging",
        "//tensorflow/core/platform:types",
        "//tensorflow/tsl/platform",
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
    deps = ["//tensorflow/core/platform:mutex"],
)

cc_library(
    name = "stacktrace_handler",
    srcs = ["stacktrace_handler.cc"],
    hdrs = ["//tensorflow/core/platform:stacktrace_handler.h"],
    tags = [
        "manual",
        "no_oss",
        "nobuilder",
    ],
    deps = [
        "//tensorflow/core/platform:mutex",
        "//tensorflow/core/platform:stacktrace",
        "//tensorflow/core/platform:types",
    ],
)

cc_library(
    name = "subprocess",
    srcs = ["subprocess.cc"],
    hdrs = ["//tensorflow/core/platform:subprocess.h"],
    tags = [
        "manual",
        "no_oss",
        "nobuilder",
    ],
    textual_hdrs = ["subprocess.h"],
    deps = [
        "//tensorflow/core/platform",
        "//tensorflow/core/platform:logging",
        "//tensorflow/core/platform:macros",
        "//tensorflow/core/platform:mutex",
        "//tensorflow/core/platform:strcat",
        "//tensorflow/core/platform:types",
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
