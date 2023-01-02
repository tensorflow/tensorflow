# Description:
# TensorFlow SavedModel.

load("//tensorflow:tensorflow.default.bzl", "filegroup")
load("//tensorflow/core/platform:rules_cc.bzl", "cc_library")
load(
    "//tensorflow:tensorflow.bzl",
    "if_android",
    "if_mobile",
    "if_not_mobile",
    "tf_cc_test",
)
load(
    "//tensorflow/core/platform:build_config_root.bzl",
    "if_static",
    "if_static_and_not_mobile",
)

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

exports_files(["loader.h"])

cc_library(
    name = "constants",
    hdrs = ["constants.h"],
    deps = ["@com_google_absl//absl/strings"],
)

cc_library(
    name = "signature_constants",
    hdrs = ["signature_constants.h"],
)

cc_library(
    name = "tag_constants",
    hdrs = ["tag_constants.h"],
)

# copybara:uncomment_begin(google-only)
# cc_library(
#     name = "mobile_only_deps",
#     visibility = ["//visibility:private"],
#     deps = if_mobile(["//tensorflow/core:portable_tensorflow_lib"]),
# )
# copybara:uncomment_end

cc_library(
    name = "reader",
    srcs = ["reader.cc"],
    hdrs = ["reader.h"],
    deps = [
        ":constants",
        "//tensorflow/core:protos_all_cc",
        ":metrics",
        ":util",
    ] + if_not_mobile([
        # TODO(b/111634734): :lib and :protos_all contain dependencies that
        # cannot be built on mobile platforms. Instead, include the appropriate
        # tf_lib depending on the build platform.
        "@com_google_absl//absl/memory:memory",
        "//tensorflow/core:lib",
        "//tensorflow/core/util/tensor_bundle:byteswaptensor",
    ]),
)

tf_cc_test(
    name = "reader_test",
    srcs = ["reader_test.cc"],
    data = [
        ":saved_model_test_files",
    ],
    linkstatic = 1,
    deps = [
        ":constants",
        ":metrics",
        ":reader",
        ":tag_constants",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
        "//tensorflow/core/platform:resource_loader",
    ],
)

cc_library(
    name = "loader",
    hdrs = ["loader.h"],
    deps = [
        ":loader_lite",
    ] + if_static_and_not_mobile([
        "//tensorflow/core:tensorflow",
    ]) + if_not_mobile([
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:lib",
        "//tensorflow/core:ops",
        "//tensorflow/core:protos_all_cc",
    ]) + if_android([
        "//tensorflow/core:portable_tensorflow_lib",
    ]),
)

cc_library(
    name = "loader_lite",
    hdrs = ["loader.h"],
    deps = if_static([
        ":loader_lite_impl",
    ]) + if_not_mobile([
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
    ]),
)

cc_library(
    name = "loader_lite_impl",
    srcs = ["loader.cc"],
    hdrs = ["loader.h"],
    deps = [
        ":constants",
        ":fingerprinting",
        ":loader_util",
        ":reader",
    ] + if_not_mobile([
        ":metrics",
        ":util",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/util/tensor_bundle:naming",
    ]),
    alwayslink = 1,
)

cc_library(
    name = "bundle_v2",
    srcs = ["bundle_v2.cc"],
    hdrs = ["bundle_v2.h"],
    deps = [
        ":constants",
        ":fingerprinting",
        ":metrics",
        ":reader",
        ":util",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/platform:strcat",
        "//tensorflow/core/util/tensor_bundle",
        "//tensorflow/core/util/tensor_bundle:byteswaptensor",
        "@com_google_absl//absl/container:flat_hash_set",
    ],
)

cc_library(
    name = "loader_util",
    srcs = ["loader_util.cc"],
    hdrs = ["loader_util.h"],
    deps = [":constants"] + if_not_mobile([
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:protos_all_cc",
    ]),
)

tf_cc_test(
    name = "bundle_v2_test",
    srcs = ["bundle_v2_test.cc"],
    data = [
        ":saved_model_test_files",
    ],
    linkstatic = 1,
    deps = [
        ":bundle_v2",
        ":metrics",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core/platform:test",
    ],
)

tf_cc_test(
    name = "saved_model_bundle_test",
    srcs = ["saved_model_bundle_test.cc"],
    data = [
        ":saved_model_test_files",
    ],
    linkstatic = 1,
    deps = [
        ":constants",
        ":loader",
        ":metrics",
        ":reader",
        ":signature_constants",
        ":tag_constants",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:tensorflow",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
    ],
)

tf_cc_test(
    name = "saved_model_bundle_lite_test",
    srcs = ["saved_model_bundle_lite_test.cc"],
    data = [
        ":saved_model_test_files",
    ],
    linkstatic = 1,
    deps = [
        ":constants",
        ":loader",
        ":signature_constants",
        ":tag_constants",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:tensorflow",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
    ],
)

# A subset of the TF2 saved models can be generated with this tool.
py_binary(
    name = "testdata/generate_saved_models",
    srcs = ["testdata/generate_saved_models.py"],
    data = [
        ":saved_model_asset_data",
        ":saved_model_static_hashtable_asset_data",
    ],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:lookup_ops",
        "//tensorflow/python:tensor_spec",
        "//tensorflow/python:variables",
        "//tensorflow/python/compat:v2_compat",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/module",
        "//tensorflow/python/saved_model",
        "//tensorflow/python/saved_model:save_options",
        "//tensorflow/python/trackable:asset",
        "@absl_py//absl:app",
    ],
)

# TODO(b/32673259): add a test to continuously validate these files.
filegroup(
    name = "saved_model_test_files",
    srcs = glob([
        "testdata/AssetModule/**",
        "testdata/half_plus_two_pbtxt/**",
        "testdata/half_plus_two_main_op/**",
        "testdata/half_plus_two/**",
        "testdata/half_plus_two_v2/**",
        "testdata/x_plus_y_v2_debuginfo/**",
        "testdata/CyclicModule/**",
        "testdata/StaticHashTableModule/**",
        "testdata/VarsAndArithmeticObjectGraph/**",
        "testdata/fuzz_generated/**",
        "testdata/SimpleV1Model/**",
        "testdata/OptimizerSlotVariableModule/**",
    ]),
)

filegroup(
    name = "saved_model_fingerprinting_test_files",
    srcs = glob([
        "testdata/bert2/**",
        "testdata/bert1/**",
    ]),
)

alias(
    name = "saved_model_half_plus_two",
    actual = ":saved_model_test_files",
)

filegroup(
    name = "saved_model_asset_data",
    srcs = [
        "testdata/test_asset.txt",
    ],
)

filegroup(
    name = "saved_model_static_hashtable_asset_data",
    srcs = [
        "testdata/static_hashtable_asset.txt",
    ],
)

exports_files(
    glob([
        "testdata/half_plus_two_pbtxt/**",
        "testdata/half_plus_two_main_op/**",
        "testdata/half_plus_two/**",
        "testdata/half_plus_two_v2/**",
        "testdata/x_plus_y_v2_debuginfo/**",
        "testdata/CyclicModule/**",
        "testdata/VarsAndArithmeticObjectGraph/**",
        "testdata/fuzz_generated/**",
    ]),
)

# Linked directly into ":tensorflow_framework".
cc_library(
    name = "metrics_impl",
    srcs = [
        "metrics.cc",
        "metrics.h",
    ],
    visibility = [
        "//tensorflow:__pkg__",
        "//tensorflow/python:__pkg__",
        "//tensorflow/security/fuzzing/cc/ops:__pkg__",  # TODO(b/261455394): Remove.
    ],
    deps = if_not_mobile(["//tensorflow/core:lib"]) + if_android(["//tensorflow/core:portable_tensorflow_lib_lite"]),
    alwayslink = True,
)

cc_library(
    name = "metrics",
    hdrs = ["metrics.h"],
    visibility = ["//tensorflow/python/saved_model:__subpackages__"],
    deps = if_static([
        ":metrics_impl",
    ]) + if_not_mobile(["//tensorflow/core:lib"]) + if_android(["//tensorflow/core:portable_tensorflow_lib_lite"]),
)

tf_cc_test(
    name = "metrics_test",
    size = "small",
    srcs = ["metrics_test.cc"],
    deps = [
        ":metrics",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
    ],
)

cc_library(
    name = "util",
    srcs = ["util.cc"],
    hdrs = ["util.h"],
    deps = [
        "//tensorflow/core/protobuf:for_core_protos_cc",
    ] + if_not_mobile(["//tensorflow/core:lib"]) + if_android(["//tensorflow/core:portable_tensorflow_lib_lite"]),
)

tf_cc_test(
    name = "util_test",
    size = "small",
    srcs = ["util_test.cc"],
    deps = [
        ":util",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core/protobuf:for_core_protos_cc",
    ],
)

# Linked directly into ":tensorflow_framework".
cc_library(
    name = "fingerprinting_impl",
    srcs = [
        "fingerprinting.cc",
        "fingerprinting.h",
    ],
    visibility = [
        "//tensorflow:__pkg__",
        "//tensorflow/python:__pkg__",
        "//tensorflow/security/fuzzing/cc/ops:__pkg__",  # TODO(b/261455394): Remove.
    ],
    deps = [
        ":constants",
        "//tensorflow/core/graph/regularization:simple_delete",
        "//tensorflow/core/graph/regularization:util",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/platform:types",
        "//tensorflow/core/util/tensor_bundle:naming",
        "@com_google_protobuf//:protobuf_headers",
        "@com_google_absl//absl/container:btree",
        "@com_google_absl//absl/strings",
    ] + if_not_mobile(["//tensorflow/core:lib"]) + if_android(["//tensorflow/core:portable_tensorflow_lib_lite"]),
    alwayslink = True,
)

cc_library(
    name = "fingerprinting",
    hdrs = ["fingerprinting.h"],
    visibility = ["//tensorflow/python/saved_model:__subpackages__"],
    deps = if_static([
        ":fingerprinting_impl",
        "@com_google_absl//absl/strings",
        "//tensorflow/core:protos_all_cc",
    ]) + if_not_mobile(["//tensorflow/core:lib"]) + if_android(["//tensorflow/core:portable_tensorflow_lib_lite"]),
)

tf_cc_test(
    name = "fingerprinting_test",
    size = "small",
    srcs = ["fingerprinting_test.cc"],
    data = [
        ":saved_model_fingerprinting_test_files",
        ":saved_model_test_files",
    ],
    deps = [
        ":fingerprinting",
        "//tensorflow/cc:cc_ops",
        "//tensorflow/cc:scope",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core/protobuf:for_core_protos_cc",
    ],
)
