load(
    "//tensorflow:tensorflow.bzl",
    "if_oss",
    "tf_cc_test",
)

# Description:
#   Utilities for splitting and joining large protos > 2GB.
load("//tensorflow/core/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [
        "//learning/brain/tfrt/aot:__subpackages__",
        "//learning/brain/tfrt/tf_tpu:__subpackages__",
        "//platforms/darwinn/contrib/waymo:__subpackages__",
        "//tensorflow/cc/saved_model/image_format:__subpackages__",
        "//tensorflow/tools/proto_splitter:__subpackages__",
        "//waymo/ml/compiler/frontend:__subpackages__",
    ],
    licenses = ["notice"],
)

cc_library(
    name = "split",
    hdrs = ["split.h"],
    deps = [
        ":util",
        "//tensorflow/tools/proto_splitter:chunk_proto_cc",
        "//tensorflow/tools/proto_splitter:versions_proto_cc",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@local_tsl//tsl/platform:protobuf",
    ] + if_oss([
        "//tensorflow/tools/proto_splitter:protos_impl",
    ]),
)

cc_library(
    name = "composable_splitter_base",
    srcs = ["composable_splitter_base.cc"],
    hdrs = ["composable_splitter_base.h"],
    deps = [
        ":max_size",
        ":split",
        ":util",
        "//tensorflow/core:lib",
        "//tensorflow/tools/proto_splitter:chunk_proto_cc",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:cord",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:protobuf",
        "@riegeli//riegeli/base:initializer",
        "@riegeli//riegeli/bytes:cord_writer",
        "@riegeli//riegeli/bytes:fd_writer",
        "@riegeli//riegeli/bytes:string_writer",
        "@riegeli//riegeli/records:record_writer",
    ] + if_oss([
        "//tensorflow/tools/proto_splitter:protos_impl",
    ]),
)

cc_library(
    name = "composable_splitter",
    hdrs = ["composable_splitter.h"],
    deps = [
        ":composable_splitter_base",
        ":util",
        "//tensorflow/tools/proto_splitter:chunk_proto_cc",
        "@local_tsl//tsl/platform:protobuf",
    ] + if_oss([
        "//tensorflow/tools/proto_splitter:protos_impl",
    ]),
)

tf_cc_test(
    name = "composable_splitter_test",
    srcs = ["composable_splitter_test.cc"],
    data = [],
    tags = [
        "no_mac",  # b/291933687
        "no_windows",  # b/291001524
    ],
    deps = [
        ":composable_splitter",
        ":test_util",
        ":util",
        "//tensorflow/core:lib",
        "//tensorflow/core:portable_gif_internal",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/platform:path",
        "//tensorflow/tools/proto_splitter:chunk_proto_cc",
        "//tensorflow/tools/proto_splitter/testdata:test_message_proto_cc",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:cord",
        "@com_google_googletest//:gtest_main",
        "@local_tsl//tsl/platform:protobuf",
        "@local_tsl//tsl/platform:status_matchers",
        "@local_xla//xla/tsl/lib/core:status_test_util",
        "@riegeli//riegeli/base:initializer",
        "@riegeli//riegeli/bytes:cord_reader",
        "@riegeli//riegeli/bytes:fd_reader",
        "@riegeli//riegeli/bytes:string_reader",
        "@riegeli//riegeli/records:record_reader",
    ] + if_oss([
        "//tensorflow/tools/proto_splitter:protos_impl",
    ]),
)

cc_library(
    name = "util",
    srcs = ["util.cc"],
    hdrs = ["util.h"],
    visibility = [
        "//learning/brain/tfrt/aot:__subpackages__",
        "//platforms/darwinn/contrib/waymo:__subpackages__",
        "//tensorflow/cc/saved_model:__subpackages__",
        "//tensorflow/cc/saved_model/image_format:__subpackages__",
        "//tensorflow/tools/proto_splitter:__subpackages__",
    ],
    deps = [
        "//tensorflow/core/platform",
        "//tensorflow/core/platform:env",
        "//tensorflow/tools/proto_splitter:chunk_proto_cc",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:cord",
        "@com_google_absl//absl/strings:str_format",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:protobuf",
        "@local_tsl//tsl/platform:statusor",
        "@riegeli//riegeli/base:initializer",
        "@riegeli//riegeli/base:types",
        "@riegeli//riegeli/bytes:fd_reader",
        "@riegeli//riegeli/bytes:string_reader",
        "@riegeli//riegeli/records:record_reader",
    ] + if_oss([
        "//tensorflow/tools/proto_splitter:protos_impl",
    ]),
)

tf_cc_test(
    name = "util_test",
    srcs = ["util_test.cc"],
    data = [
        "//tensorflow/tools/proto_splitter/testdata:split-standard.cpb",
    ],
    tags = [
        "no_mac",  # b/291933687
        "no_windows",  # b/291001524
    ],
    deps = [
        ":test_util",
        ":util",
        "//tensorflow/core/platform:macros",
        "//tensorflow/core/platform:path",
        "//tensorflow/core/platform:test",
        "//tensorflow/tools/proto_splitter:chunk_proto_cc",
        "//tensorflow/tools/proto_splitter/testdata:test_message_proto_cc",
        "@com_google_absl//absl/status",
        "@com_google_googletest//:gtest_main",
        "@local_tsl//tsl/platform:protobuf",
        "@local_tsl//tsl/platform:status",
        "@local_tsl//tsl/platform:status_matchers",
        "@local_tsl//tsl/platform:statusor",
        "@local_xla//xla/tsl/lib/core:status_test_util",
    ] + if_oss([
        "//tensorflow/tools/proto_splitter:protos_impl",
    ]),
)

cc_library(
    name = "graph_def_splitter",
    srcs = ["graph_def_splitter.cc"],
    hdrs = ["graph_def_splitter.h"],
    deps = [
        ":composable_splitter",
        ":composable_splitter_base",
        ":large_node_splitter",
        ":max_size",
        ":repeated_field_splitter",
        ":size_splitter",
        ":util",
        "//tensorflow/core:framework",
        "//tensorflow/core:protos_all_cc",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:protobuf",
        "@local_tsl//tsl/platform:statusor",
    ],
)

cc_library(
    name = "size_splitter",
    hdrs = ["size_splitter.h"],
    deps = [
        ":composable_splitter",
        ":util",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/types:optional",
    ],
)

cc_library(
    name = "repeated_field_splitter",
    hdrs = ["repeated_field_splitter.h"],
    deps = [
        ":composable_splitter",
        ":max_size",
        ":size_splitter",
        ":util",
        "//tensorflow/core:protos_all_cc",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@local_tsl//tsl/platform:protobuf",
        "@local_xla//xla/tsl/platform:errors",
        "@local_xla//xla/tsl/platform:statusor",
    ],
)

tf_cc_test(
    name = "graph_def_splitter_test",
    srcs = ["graph_def_splitter_test.cc"],
    data = [
        "//tensorflow/tools/proto_splitter/testdata:function-large-nodes.pb",
        "//tensorflow/tools/proto_splitter/testdata:function-lots-of-nodes.pb",
        "//tensorflow/tools/proto_splitter/testdata:graph-def-and-function.pb",
        "//tensorflow/tools/proto_splitter/testdata:split-large-constant.pb",
        "//tensorflow/tools/proto_splitter/testdata:split-large-nodes.pb",
        "//tensorflow/tools/proto_splitter/testdata:split-lots-nodes.pb",
    ],
    tags = [
        "no_mac",  # b/291933687
        "no_windows",  # b/291001524
    ],
    deps = [
        ":graph_def_splitter",
        ":max_size",
        ":test_util",
        ":util",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core/platform:path",
        "//tensorflow/tools/proto_splitter/testdata:test_message_proto_cc",
        "@com_google_absl//absl/strings:cord",
        "@com_google_googletest//:gtest_main",
        "@local_tsl//tsl/platform:protobuf",
    ] + if_oss([
        "//tensorflow/tools/proto_splitter:protos_impl",
    ]),
)

cc_library(
    name = "max_size",
    srcs = ["max_size.cc"],
    hdrs = ["max_size.h"],
    deps = ["@com_google_absl//absl/synchronization"],
)

cc_library(
    name = "saved_model_splitter",
    srcs = ["saved_model_splitter.cc"],
    hdrs = ["saved_model_splitter.h"],
    deps = [
        ":composable_splitter",
        ":graph_def_splitter",
        ":large_node_splitter",
        ":max_size",
        ":util",
        "//tensorflow/core:protos_all_cc",
        "@com_google_absl//absl/status",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:protobuf",
    ],
)

tf_cc_test(
    name = "saved_model_splitter_test",
    srcs = ["saved_model_splitter_test.cc"],
    data = [
        "//tensorflow/cc/saved_model:saved_model_test_files",
    ],
    tags = ["no_windows"],  # b/291001524
    deps = [
        ":max_size",
        ":saved_model_splitter",
        ":util",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core/platform:path",
        "//tensorflow/tools/proto_splitter/testdata:test_message_proto_cc",
        "@com_google_googletest//:gtest_main",
        "@local_tsl//tsl/platform:protobuf",
    ] + if_oss([
        "//tensorflow/tools/proto_splitter:protos_impl",
    ]),
)

cc_library(
    name = "large_node_splitter",
    hdrs = ["large_node_splitter.h"],
    deps = [
        ":composable_splitter",
        ":composable_splitter_base",
        ":max_size",
        ":size_splitter",
        ":util",
        "@com_google_absl//absl/status:statusor",
        "@local_tsl//tsl/platform:errors",
    ],
)

cc_library(
    name = "test_util",
    testonly = True,
    hdrs = ["test_util.h"],
    deps = [
        "//tensorflow/core:test",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:protobuf",
        "@local_tsl//tsl/platform:statusor",
    ],
)
