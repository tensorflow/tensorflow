load(
    "@io_bazel_rules_closure//closure:defs.bzl",
    "closure_proto_library",
)
load(
    "//tensorflow:tensorflow.bzl",
    "tf_cc_test",
    "tf_copts",
)
load(
    "//tensorflow/core/platform:build_config.bzl",
    "tf_proto_library",
    "tf_pyclif_proto_library",
)

package(
    default_visibility = [
        "//tensorflow/core:__subpackages__",
    ],
    licenses = ["notice"],  # Apache 2.0
)

# TODO(bmzhao): Refactor this target to use granular dependencies
# after stage 4 of the TF build refactor is complete:
# https://github.com/tensorflow/community/pull/179
cc_library(
    name = "example_parser_configuration",
    srcs = ["example_parser_configuration.cc"],
    hdrs = ["example_parser_configuration.h"],
    copts = tf_copts(),
    linkstatic = 1,
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:protos_all_cc",
    ],
    alwayslink = 1,
)

cc_library(
    name = "feature_util",
    srcs = ["feature_util.cc"],
    hdrs = ["feature_util.h"],
    deps = [
        ":example_protos_cc",
        "//tensorflow/core/platform:protobuf",
        "//tensorflow/core/platform:stringpiece",
        "//tensorflow/core/platform:types",
        "@com_google_absl//absl/base:core_headers",
    ],
    alwayslink = 1,
)

# TODO(bmzhao): Refactor this target to use granular dependencies
# after stage 4 of the TF build refactor is complete:
# https://github.com/tensorflow/community/pull/179
tf_cc_test(
    name = "example_parser_configuration_test",
    size = "small",
    srcs = ["example_parser_configuration_test.cc"],
    data = [":example_parser_configuration_testdata"],
    deps = [
        "//tensorflow/cc:cc_ops",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:direct_session_internal",
        "//tensorflow/core:example_parser_configuration",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:ops",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
        "//tensorflow/core/kernels:example_parsing_ops",
    ],
)

filegroup(
    name = "example_parser_configuration_testdata",
    srcs = [
        "testdata/parse_example_graph_def.pbtxt",
    ],
)

# TODO(bmzhao): Split this target into separate tf_proto_libraries.
# This target is a holdover from the target tensorflow/core:example_protos,
# but splitting it while require multiple LSCs.
tf_proto_library(
    name = "example_protos",
    srcs = [
        "example.proto",
        "feature.proto",
    ],
    cc_api_version = 2,
    make_default_target_header_only = True,
)

tf_proto_library(
    name = "example_parser_configuration_proto",
    srcs = ["example_parser_configuration.proto"],
    cc_api_version = 2,
    make_default_target_header_only = True,
    protodeps = [
        "//tensorflow/core/framework:tensor_shape_proto",
        "//tensorflow/core/framework:tensor_proto",
        "//tensorflow/core/framework:types_proto",
    ],
)

tf_proto_library(
    name = "protos_all",
    cc_api_version = 2,
    make_default_target_header_only = True,
    protodeps = [
        ":example_protos",
        ":example_parser_configuration_proto",
    ],
)

tf_pyclif_proto_library(
    name = "example_pyclif",
    proto_lib = "//tensorflow/core:protos_all",
    proto_srcfile = "example.proto",
)

tf_pyclif_proto_library(
    name = "feature_pyclif",
    proto_lib = "//tensorflow/core:protos_all",
    proto_srcfile = "feature.proto",
)

closure_proto_library(
    name = "example_protos_closure",
    visibility = ["//visibility:public"],
    deps = [":example_protos"],
)

filegroup(
    name = "mobile_srcs_no_runtime",
    srcs = [
        "feature_util.cc",
        "feature_util.h",
    ],
)

# TODO(bmzhao): These files are exported since they are directly referenced
# from targets in tensorflow/core/BUILD. Remove these after phase 4 of the
# TF refactor: https://github.com/tensorflow/community/pull/179
exports_files(
    srcs = [
        "example.proto",
        "example_parser_configuration_test.cc",
        "feature.proto",
        "feature_util.cc",
        "feature_util.h",
        "feature_util_test.cc",
    ],
    visibility = [
        "//tensorflow/core:__pkg__",
    ],
)
