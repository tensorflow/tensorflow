# Description:
#   Utilities for splitting and joining large protos > 2GB.

load("//tensorflow:pytype.default.bzl", "pytype_strict_library")
load("//tensorflow:strict.default.bzl", "py_strict_test")

# Placeholder: load py_proto_library
load(
    "//tensorflow:tensorflow.bzl",
    "if_oss",
    "tf_cc_test",
)
load(
    "//tensorflow/core/platform:build_config.bzl",
    "tf_proto_library",
)

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [
        "__subpackages__",
        "//tensorflow:internal",
    ],
    licenses = ["notice"],
)

tf_proto_library(
    name = "versions_proto",
    srcs = ["versions.proto"],
    make_default_target_header_only = True,
)

tf_proto_library(
    name = "chunk_proto",
    srcs = ["chunk.proto"],
    make_default_target_header_only = True,
    protodeps = [":versions_proto"],
)

# OSS-only: Required to link the impl files.
cc_library(
    name = "protos_impl",
    deps = if_oss([
        "//tensorflow/tools/proto_splitter:chunk_proto_cc_impl",
        "//tensorflow/tools/proto_splitter:versions_proto_cc_impl",
        "//tensorflow/tools/proto_splitter/testdata:test_message_proto_cc_impl",
    ]),
)

# copybara:uncomment_begin(google-only)
#
# py_proto_library(
#     name = "versions_proto_py_pb2",
#     deps = [
#         ":versions_proto",
#     ],
# )
#
# py_proto_library(
#     name = "chunk_proto_py_pb2",
#     deps = [
#         ":chunk_proto",
#     ],
# )
#
# copybara:uncomment_end

pytype_strict_library(
    name = "split",
    srcs = ["split.py"],
    deps = [
        ":chunk_proto_py",
        ":util",
        ":version",
        ":versions_proto_py",
        "//tensorflow/python/lib/io:file_io",
        "@absl_py//absl/logging",
        "@riegeli_py//python/riegeli",
    ],
)

pytype_strict_library(
    name = "version",
    srcs = ["version.py"],
)

py_strict_test(
    name = "split_test",
    srcs = ["split_test.py"],
    tags = [
        "no_mac",  # b/291933687
        "no_windows",  # b/291001524
    ],
    deps = [
        ":chunk_proto_py",
        ":split",
        ":versions_proto_py",
        #internal proto upb dep
        "//tensorflow/python/platform:client_testlib",
        "//tensorflow/tools/proto_splitter/testdata:test_message_proto_py",
        "@absl_py//absl/testing:parameterized",
        "@riegeli_py//python/riegeli",
    ],
)

pytype_strict_library(
    name = "constants",
    srcs = ["constants.py"],
    visibility = ["//visibility:public"],
)

pytype_strict_library(
    name = "split_graph_def",
    srcs = ["split_graph_def.py"],
    deps = [
        ":constants",
        ":split",
        ":util",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python/framework:tensor_util",
    ],
)

py_strict_test(
    name = "split_graph_def_test",
    srcs = ["split_graph_def_test.py"],
    tags = [
        "no_mac",  # b/291933687
        "no_windows",  # b/291001524
    ],
    deps = [
        ":chunk_proto_py",
        ":constants",
        ":split_graph_def",
        ":util",
        #internal proto upb dep
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python/platform:client_testlib",
        "//tensorflow/tools/proto_splitter/python:test_util",
    ],
)

pytype_strict_library(
    name = "util",
    srcs = ["util.py"],
    deps = [":chunk_proto_py"],
)

py_strict_test(
    name = "util_test",
    srcs = ["util_test.py"],
    deps = [
        ":util",
        #internal proto upb dep
        "//tensorflow/python/platform:client_testlib",
        "//tensorflow/tools/proto_splitter/testdata:test_message_proto_py",
    ],
)

cc_library(
    name = "merge_impl",
    srcs = [
        "merge.cc",
        "merge.h",
    ],
    deps = [
        ":chunk_proto_cc",
        "//tensorflow/core:lib",
        "//tensorflow/tools/proto_splitter/cc:util",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@local_tsl//tsl/platform:protobuf",
        "@riegeli//riegeli/base:object",
        "@riegeli//riegeli/bytes:fd_reader",
        "@riegeli//riegeli/bytes:string_reader",
        "@riegeli//riegeli/records:record_reader",
    ] + if_oss([
        "//tensorflow/tools/proto_splitter:protos_impl",
    ]),
)

cc_library(
    name = "merge",
    hdrs = ["merge.h"],
    visibility = [
        "//platforms/darwinn/contrib/waymo:__subpackages__",
        "//tensorflow:internal",
        "//tensorflow/cc/experimental/tf2:__subpackages__",
        "//tensorflow/cc/saved_model/image_format:__subpackages__",
        "//waymo/onboard/ml/chauffeur_net/utils:__subpackages__",
    ],
    deps = [
        ":chunk_proto_cc",
        ":merge_impl",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:string_view",
        "@local_tsl//tsl/platform:protobuf",
        "@riegeli//riegeli/bytes:fd_reader",
        "@riegeli//riegeli/records:record_reader",
    ] + if_oss([
        "//tensorflow/tools/proto_splitter:protos_impl",
    ]),
)

tf_cc_test(
    name = "merge_test",
    srcs = ["merge_test.cc"],
    data = [
        "//tensorflow/cc/saved_model:testdata/chunked_saved_model/chunked_model/saved_model.cpb",
        "//tensorflow/cc/saved_model:testdata/chunked_saved_model/chunked_model/saved_model.pbtxt",
        "//tensorflow/tools/proto_splitter/testdata:bf-split-tree.cpb",
        "//tensorflow/tools/proto_splitter/testdata:bf-split-tree.pbtxt",
        "//tensorflow/tools/proto_splitter/testdata:df-split-tree.cpb",
        "//tensorflow/tools/proto_splitter/testdata:df-split-tree.pbtxt",
        "//tensorflow/tools/proto_splitter/testdata:many-field.cpb",
        "//tensorflow/tools/proto_splitter/testdata:many-field.pbtxt",
        "//tensorflow/tools/proto_splitter/testdata:split-large-constant.cpb",
        "//tensorflow/tools/proto_splitter/testdata:split-large-constant.pbtxt",
        "//tensorflow/tools/proto_splitter/testdata:split-large-nodes.cpb",
        "//tensorflow/tools/proto_splitter/testdata:split-large-nodes.pbtxt",
        "//tensorflow/tools/proto_splitter/testdata:split-lots-nodes.cpb",
        "//tensorflow/tools/proto_splitter/testdata:split-lots-nodes.pbtxt",
        "//tensorflow/tools/proto_splitter/testdata:split-standard.cpb",
        "//tensorflow/tools/proto_splitter/testdata:split-standard.pbtxt",
        "//tensorflow/tools/proto_splitter/testdata:split-tree.pbtxt",
    ],
    tags = [
        "no_mac",  # b/291933687
        "no_windows",  # b/291001524
    ],
    deps = [
        ":chunk_proto_cc",
        ":merge",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core/platform:path",
        "//tensorflow/tools/proto_splitter/cc:test_util",
        "//tensorflow/tools/proto_splitter/cc:util",
        "//tensorflow/tools/proto_splitter/testdata:test_message_proto_cc",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
        "@local_tsl//tsl/platform:env",
        "@local_tsl//tsl/platform:protobuf",
        "@local_tsl//tsl/platform:statusor",
        "@local_xla//xla/tsl/platform:env",
    ] + if_oss([
        "//tensorflow/tools/proto_splitter:protos_impl",
    ]),
)
