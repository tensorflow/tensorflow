load("//tensorflow:strict.default.bzl", "py_strict_binary")

# Placeholder: load py_proto_library
load(
    "//tensorflow/core/platform:build_config.bzl",
    "tf_proto_library",
)

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//tensorflow/tools/proto_splitter:__subpackages__"],
    licenses = ["notice"],
)

exports_files([
    "df-split-tree.cpb",
    "df-split-tree.pbtxt",
    "bf-split-tree.cpb",
    "bf-split-tree.pbtxt",
    "split-tree.pbtxt",
    "split-large-nodes.cpb",
    "split-large-nodes.pb",
    "split-large-nodes.pbtxt",
    "split-standard.cpb",
    "split-standard.pb",
    "split-standard.pbtxt",
    "split-large-constant.cpb",
    "split-large-constant.pbtxt",
    "split-lots-nodes.cpb",
    "split-lots-nodes.pb",
    "split-lots-nodes.pbtxt",
    "split-large-constant.cpb",
    "split-large-constant.pb",
    "split-large-constant.pbtxt",
    "function-large-nodes.pb",
    "function-lots-of-nodes.pb",
    "graph-def-and-function.pb",
    "many-field.cpb",
    "many-field.pbtxt",
])

tf_proto_library(
    name = "test_message_proto",
    srcs = ["test_message.proto"],
    make_default_target_header_only = True,
    visibility = [
        "//tensorflow/cc/saved_model:__subpackages__",
        "//tensorflow/tools/proto_splitter:__subpackages__",
    ],
)

# copybara:uncomment_begin(google-only)
#
# py_proto_library(
#     name = "test_message_proto_py_pb2",
#     deps = [
#         ":test_message_proto",
#     ],
# )
#
# py_strict_binary(
#     name = "split_gen",
#     srcs = ["split_gen.py"],
#     deps = [
#         ":test_message_proto_py_pb2",
#         "//tensorflow/python/lib/io:file_io",
#         "//tensorflow/tools/proto_splitter:chunk_proto_py_pb2",
#         "//tensorflow/tools/proto_splitter:split",
#         "//tensorflow/tools/proto_splitter:util",
#         "@absl_py//absl:app",
#         "@absl_py//absl/flags",
#         "@absl_py//absl/logging",
#     ],
# )
#
# py_strict_binary(
#     name = "split_graph_def_gen",
#     srcs = ["split_graph_def_gen.py"],
#     deps = [
#         "//tensorflow/core:protos_all_py",
#         "//tensorflow/python/lib/io:file_io",
#         "//tensorflow/tools/proto_splitter:constants",
#         "//tensorflow/tools/proto_splitter:split_graph_def",
#         "//tensorflow/tools/proto_splitter/python:test_util",
#         "@absl_py//absl:app",
#         "@absl_py//absl/flags",
#         "@absl_py//absl/logging",
#     ],
# )
#
# py_strict_binary(
#     name = "split_saved_model_gen",
#     srcs = ["split_saved_model_gen.py"],
#     deps = [
#         "//tensorflow/core:protos_all_py",
#         "//tensorflow/python/lib/io:file_io",
#         "//tensorflow/tools/proto_splitter:constants",
#         "//tensorflow/tools/proto_splitter/python:saved_model",
#         "//tensorflow/tools/proto_splitter/python:test_util",
#         "@absl_py//absl:app",
#         "@absl_py//absl/flags",
#         "@absl_py//absl/logging",
#     ],
# )
#
# py_strict_binary(
#     name = "many_field_gen",
#     srcs = ["many_field_gen.py"],
#     deps = [
#         ":test_message_proto_py_pb2",
#         "//tensorflow/python/lib/io:file_io",
#         "//tensorflow/tools/proto_splitter:split",
#         "@absl_py//absl:app",
#         "@absl_py//absl/flags",
#     ],
# )
#
# copybara:uncomment_end
