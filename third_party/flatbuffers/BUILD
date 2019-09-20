licenses(["notice"])

package(
    default_visibility = ["//visibility:public"],
    features = [
        "-layering_check",
        "-parse_headers",
    ],
)

exports_files([
    "LICENSE",
])

load(":build_defs.bzl", "flatbuffer_cc_library")

# Public flatc library to compile flatbuffer files at runtime.
cc_library(
    name = "flatbuffers",
    srcs = [
        "src/code_generators.cpp",
        "src/idl_gen_fbs.cpp",
        "src/idl_gen_general.cpp",
        "src/idl_gen_text.cpp",
        "src/idl_parser.cpp",
        "src/reflection.cpp",
        "src/util.cpp",
    ],
    hdrs = [":public_headers"],
    includes = ["include/"],
    linkstatic = 1,
)

# Public C++ headers for the Flatbuffers library.
filegroup(
    name = "public_headers",
    srcs = [
        "include/flatbuffers/base.h",
        "include/flatbuffers/code_generators.h",
        "include/flatbuffers/flatbuffers.h",
        "include/flatbuffers/flexbuffers.h",
        "include/flatbuffers/hash.h",
        "include/flatbuffers/idl.h",
        "include/flatbuffers/minireflect.h",
        "include/flatbuffers/reflection.h",
        "include/flatbuffers/reflection_generated.h",
        "include/flatbuffers/stl_emulation.h",
        "include/flatbuffers/util.h",
    ],
)

# Public flatc compiler library.
cc_library(
    name = "flatc_library",
    srcs = [
        "src/code_generators.cpp",
        "src/flatc.cpp",
        "src/idl_gen_fbs.cpp",
        "src/idl_parser.cpp",
        "src/reflection.cpp",
        "src/util.cpp",
    ],
    hdrs = [
        "include/flatbuffers/flatc.h",
        ":public_headers",
    ],
    includes = [
        "grpc/",
        "include/",
    ],
)

# Public flatc compiler.
cc_binary(
    name = "flatc",
    srcs = [
        "grpc/src/compiler/config.h",
        "grpc/src/compiler/cpp_generator.cc",
        "grpc/src/compiler/cpp_generator.h",
        "grpc/src/compiler/go_generator.cc",
        "grpc/src/compiler/go_generator.h",
        "grpc/src/compiler/java_generator.cc",
        "grpc/src/compiler/java_generator.h",
        "grpc/src/compiler/schema_interface.h",
        "src/flatc_main.cpp",
        "src/idl_gen_cpp.cpp",
        "src/idl_gen_dart.cpp",
        "src/idl_gen_general.cpp",
        "src/idl_gen_kotlin.cpp",
        "src/idl_gen_go.cpp",
        "src/idl_gen_grpc.cpp",
        "src/idl_gen_js_ts.cpp",
        "src/idl_gen_json_schema.cpp",
        "src/idl_gen_lobster.cpp",
        "src/idl_gen_lua.cpp",
        "src/idl_gen_php.cpp",
        "src/idl_gen_python.cpp",
        "src/idl_gen_rust.cpp",
        "src/idl_gen_text.cpp",
        "src/util.cpp",
    ],
    includes = [
        "grpc/",
        "include/",
    ],
    deps = [
        ":flatc_library",
    ],
)

cc_library(
    name = "runtime_cc",
    hdrs = [
        "include/flatbuffers/base.h",
        "include/flatbuffers/flatbuffers.h",
        "include/flatbuffers/flexbuffers.h",
        "include/flatbuffers/stl_emulation.h",
        "include/flatbuffers/util.h",
    ],
    includes = ["include/"],
    linkstatic = 1,
)

# Test binary.
cc_test(
    name = "flatbuffers_test",
    testonly = 1,
    srcs = [
        "include/flatbuffers/minireflect.h",
        "include/flatbuffers/registry.h",
        "src/code_generators.cpp",
        "src/idl_gen_fbs.cpp",
        "src/idl_gen_general.cpp",
        "src/idl_gen_text.cpp",
        "src/idl_parser.cpp",
        "src/reflection.cpp",
        "src/util.cpp",
        "tests/namespace_test/namespace_test1_generated.h",
        "tests/namespace_test/namespace_test2_generated.h",
        "tests/native_type_test_impl.h",
        "tests/native_type_test_impl.cpp",
        "tests/test.cpp",
        "tests/test_assert.cpp",
        "tests/test_assert.h",
        "tests/test_builder.cpp",
        "tests/test_builder.h",
        "tests/union_vector/union_vector_generated.h",
        ":public_headers",
    ],
    copts = [
        "-DFLATBUFFERS_TRACK_VERIFIER_BUFFER_SIZE",
        "-DBAZEL_TEST_DATA_PATH",
    ],
    data = [
        ":tests/include_test/include_test1.fbs",
        ":tests/include_test/sub/include_test2.fbs",
        ":tests/monster_test.bfbs",
        ":tests/monster_test.fbs",
        ":tests/monsterdata_test.golden",
        ":tests/monsterdata_test.json",
        ":tests/prototest/imported.proto",
        ":tests/prototest/test.golden",
        ":tests/prototest/test.proto",
        ":tests/prototest/test_union.golden",
        ":tests/unicode_test.json",
        ":tests/union_vector/union_vector.fbs",
        ":tests/union_vector/union_vector.json",
        ":tests/monster_extra.fbs",
        ":tests/monsterdata_extra.json",
        ":tests/arrays_test.bfbs",
        ":tests/arrays_test.fbs",
        ":tests/arrays_test.golden",
        ":tests/native_type_test.fbs",
    ],
    includes = [
        "include/",
        "tests/",
    ],
    deps = [
        ":monster_extra_cc_fbs",
        ":monster_test_cc_fbs",
        ":arrays_test_cc_fbs",
        ":native_type_test_cc_fbs",
    ],
)

# Test bzl rules

flatbuffer_cc_library(
    name = "monster_test_cc_fbs",
    srcs = ["tests/monster_test.fbs"],
    include_paths = ["tests/include_test"],
    includes = [
        "tests/include_test/include_test1.fbs",
        "tests/include_test/sub/include_test2.fbs",
    ],
)

flatbuffer_cc_library(
    name = "monster_extra_cc_fbs",
    srcs = ["tests/monster_extra.fbs"],
)

flatbuffer_cc_library(
    name = "arrays_test_cc_fbs",
    srcs = ["tests/arrays_test.fbs"],
    flatc_args = [
        "--gen-object-api",
        "--gen-compare",
        "--no-includes",
        "--gen-mutable",
        "--reflect-names",
        "--cpp-ptr-type flatbuffers::unique_ptr",
        "--scoped-enums" ],
)

flatbuffer_cc_library(
    name = "native_type_test_cc_fbs",
    srcs = ["tests/native_type_test.fbs"],
    flatc_args = [
        "--gen-object-api",
        "--gen-mutable",
        "--cpp-ptr-type flatbuffers::unique_ptr" ],
)

