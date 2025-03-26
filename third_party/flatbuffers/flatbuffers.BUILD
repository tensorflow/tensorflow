load("@build_bazel_rules_android//android:rules.bzl", "android_library")
load("@rules_cc//cc:defs.bzl", "cc_binary", "cc_library")
load(":build_defs.bzl", "flatbuffer_py_strip_prefix_srcs")

package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

config_setting(
    name = "platform_freebsd",
    values = {"cpu": "freebsd"},
)

config_setting(
    name = "platform_openbsd",
    values = {"cpu": "openbsd"},
)

config_setting(
    name = "windows",
    values = {"cpu": "x64_windows"},
)

# Public flatc library to compile flatbuffer files at runtime.
cc_library(
    name = "flatbuffers",
    hdrs = ["//:public_headers"],
    linkstatic = 1,
    strip_include_prefix = "/include",
    visibility = ["//visibility:public"],
    deps = ["//src:flatbuffers"],
)

# Public C++ headers for the Flatbuffers library.
filegroup(
    name = "public_headers",
    srcs = [
        "include/flatbuffers/allocator.h",
        "include/flatbuffers/array.h",
        "include/flatbuffers/base.h",
        "include/flatbuffers/buffer.h",
        "include/flatbuffers/buffer_ref.h",
        "include/flatbuffers/code_generator.h",
        "include/flatbuffers/code_generators.h",
        "include/flatbuffers/default_allocator.h",
        "include/flatbuffers/detached_buffer.h",
        "include/flatbuffers/file_manager.h",
        "include/flatbuffers/flatbuffer_builder.h",
        "include/flatbuffers/flatbuffers.h",
        "include/flatbuffers/flex_flat_util.h",
        "include/flatbuffers/flexbuffers.h",
        "include/flatbuffers/grpc.h",
        "include/flatbuffers/hash.h",
        "include/flatbuffers/idl.h",
        "include/flatbuffers/minireflect.h",
        "include/flatbuffers/reflection.h",
        "include/flatbuffers/reflection_generated.h",
        "include/flatbuffers/registry.h",
        "include/flatbuffers/stl_emulation.h",
        "include/flatbuffers/string.h",
        "include/flatbuffers/struct.h",
        "include/flatbuffers/table.h",
        "include/flatbuffers/util.h",
        "include/flatbuffers/vector.h",
        "include/flatbuffers/vector_downward.h",
        "include/flatbuffers/verifier.h",
    ],
    visibility = ["//visibility:public"],
)

# Public flatc compiler library.
cc_library(
    name = "flatc_library",
    linkstatic = 1,
    visibility = ["//visibility:public"],
    deps = [
        "@flatbuffers//src:flatc_library",
    ],
)

# Public flatc compiler.
cc_binary(
    name = "flatc",
    linkopts = select({
        ":platform_freebsd": [
            "-lm",
        ],
        # If Visual Studio 2022 developers facing linking errors,
        # change the line below as ":windows": ["/DEFAULTLIB:msvcrt.lib"],
        ":windows": [],
        "//conditions:default": [
            "-lm",
            "-ldl",
        ],
    }),
    visibility = ["//visibility:public"],
    deps = [
        "@flatbuffers//src:flatc",
    ],
)

filegroup(
    name = "flatc_headers",
    srcs = [
        "include/flatbuffers/flatc.h",
    ],
    visibility = ["//visibility:public"],
)

# Library used by flatbuffer_cc_library rules.
cc_library(
    name = "runtime_cc",
    hdrs = [
        "include/flatbuffers/allocator.h",
        "include/flatbuffers/array.h",
        "include/flatbuffers/base.h",
        "include/flatbuffers/buffer.h",
        "include/flatbuffers/buffer_ref.h",
        "include/flatbuffers/default_allocator.h",
        "include/flatbuffers/detached_buffer.h",
        "include/flatbuffers/flatbuffer_builder.h",
        "include/flatbuffers/flatbuffers.h",
        "include/flatbuffers/flexbuffers.h",
        "include/flatbuffers/stl_emulation.h",
        "include/flatbuffers/string.h",
        "include/flatbuffers/struct.h",
        "include/flatbuffers/table.h",
        "include/flatbuffers/util.h",
        "include/flatbuffers/vector.h",
        "include/flatbuffers/vector_downward.h",
        "include/flatbuffers/verifier.h",
    ],
    linkstatic = 1,
    strip_include_prefix = "/include",
    visibility = ["//visibility:public"],
)

flatbuffer_py_strip_prefix_srcs(
    name = "flatbuffer_py_strip_prefix",
    srcs = [
        "python/flatbuffers/__init__.py",
        "python/flatbuffers/_version.py",
        "python/flatbuffers/builder.py",
        "python/flatbuffers/compat.py",
        "python/flatbuffers/encode.py",
        "python/flatbuffers/flexbuffers.py",
        "python/flatbuffers/number_types.py",
        "python/flatbuffers/packer.py",
        "python/flatbuffers/table.py",
        "python/flatbuffers/util.py",
    ],
    strip_prefix = "python/flatbuffers/",
)

filegroup(
    name = "runtime_py_srcs",
    srcs = [
        "__init__.py",
        "_version.py",
        "builder.py",
        "compat.py",
        "encode.py",
        "flexbuffers.py",
        "number_types.py",
        "packer.py",
        "table.py",
        "util.py",
    ],
)

py_library(
    name = "runtime_py",
    srcs = [":runtime_py_srcs"],
    visibility = ["//visibility:public"],
)

filegroup(
    name = "runtime_java_srcs",
    srcs = glob(["java/src/main/java/com/google/flatbuffers/**/*.java"]),
)

java_library(
    name = "runtime_java",
    srcs = [":runtime_java_srcs"],
    visibility = ["//visibility:public"],
)

android_library(
    name = "runtime_android",
    srcs = [":runtime_java_srcs"],
    visibility = ["//visibility:public"],
)
