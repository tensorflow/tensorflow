package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])  # Apache 2.0

config_setting(
    name = "freebsd",
    values = {"cpu": "freebsd"},
    visibility = ["//visibility:public"],
)

FLATBUFFERS_COPTS = [
    "-fexceptions",
] + select({
    "@bazel_tools//src:windows": [],
    "@bazel_tools//src:windows_msvc": [],
    "//conditions:default": ["-Wno-implicit-fallthrough"],
})

# Public flatc library to compile flatbuffer files at runtime.
cc_library(
    name = "flatbuffers",
    srcs = [
        "include/flatbuffers/code_generators.h",
        "include/flatbuffers/reflection_generated.h",
        "src/code_generators.cpp",
        "src/idl_gen_fbs.cpp",
        "src/idl_gen_general.cpp",
        "src/idl_gen_text.cpp",
        "src/idl_parser.cpp",
        "src/reflection.cpp",
        "src/util.cpp",
    ],
    hdrs = [
        "include/flatbuffers/base.h",
        "include/flatbuffers/flatbuffers.h",
        "include/flatbuffers/flexbuffers.h",
        "include/flatbuffers/hash.h",
        "include/flatbuffers/idl.h",
        "include/flatbuffers/reflection.h",
        "include/flatbuffers/stl_emulation.h",
        "include/flatbuffers/util.h",
    ],
    copts = FLATBUFFERS_COPTS,
    includes = ["include/"],
)

# Public flatc compiler library.
cc_library(
    name = "flatc_library",
    srcs = [
        "grpc/src/compiler/config.h",
        "grpc/src/compiler/go_generator.h",
        "grpc/src/compiler/schema_interface.h",
        "include/flatbuffers/base.h",
        "include/flatbuffers/code_generators.h",
        "include/flatbuffers/flatbuffers.h",
        "include/flatbuffers/flatc.h",
        "include/flatbuffers/flexbuffers.h",
        "include/flatbuffers/hash.h",
        "include/flatbuffers/idl.h",
        "include/flatbuffers/reflection.h",
        "include/flatbuffers/reflection_generated.h",
        "include/flatbuffers/stl_emulation.h",
        "include/flatbuffers/util.h",
        "src/code_generators.cpp",
        "src/flatc.cpp",
        "src/idl_gen_fbs.cpp",
        "src/idl_parser.cpp",
        "src/reflection.cpp",
        "src/util.cpp",
    ],
    hdrs = [
        "include/flatbuffers/base.h",
        "include/flatbuffers/code_generators.h",
        "include/flatbuffers/flatbuffers.h",
        "include/flatbuffers/flatc.h",
        "include/flatbuffers/idl.h",
        "include/flatbuffers/reflection.h",
        "include/flatbuffers/stl_emulation.h",
        "include/flatbuffers/util.h",
    ],
    copts = FLATBUFFERS_COPTS,
    includes = [
        "grpc/",
        "include/",
    ],
)

# Public flatc compiler.
cc_binary(
    name = "flatc",
    srcs = [
        "grpc/src/compiler/cpp_generator.cc",
        "grpc/src/compiler/cpp_generator.h",
        "grpc/src/compiler/go_generator.cc",
        "grpc/src/compiler/go_generator.h",
        "grpc/src/compiler/schema_interface.h",
        "src/flatc_main.cpp",
        "src/idl_gen_cpp.cpp",
        "src/idl_gen_general.cpp",
        "src/idl_gen_go.cpp",
        "src/idl_gen_grpc.cpp",
        "src/idl_gen_js.cpp",
        "src/idl_gen_json_schema.cpp",
        "src/idl_gen_php.cpp",
        "src/idl_gen_python.cpp",
        "src/idl_gen_text.cpp",
    ],
    copts = FLATBUFFERS_COPTS,
    includes = [
        "grpc/",
        "include/",
    ],
    linkopts = select({
        ":freebsd": [
            "-lm",
        ],
        "//conditions:default": [
            "-lm",
            "-ldl",
        ],
    }),
    deps = [
        ":flatc_library",
    ],
)

filegroup(
    name = "runtime_cc_srcs",
    srcs = [
        "include/flatbuffers/base.h",
        "include/flatbuffers/flatbuffers.h",
        "include/flatbuffers/stl_emulation.h",
        "include/flatbuffers/util.h",
    ],
)

cc_library(
    name = "runtime_cc",
    hdrs = ["runtime_cc_srcs"],
    includes = ["include"],
    linkstatic = 1,
)
