# Description:
# TensorFlow SavedModel.

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
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],  # Apache 2.0
)

exports_files([
    "LICENSE",
    "loader.h",
])

cc_library(
    name = "constants",
    hdrs = ["constants.h"],
)

cc_library(
    name = "signature_constants",
    hdrs = ["signature_constants.h"],
)

cc_library(
    name = "tag_constants",
    hdrs = ["tag_constants.h"],
)

cc_library(
    name = "reader",
    srcs = ["reader.cc"],
    hdrs = ["reader.h"],
    deps = [":constants"] + if_not_mobile([
        # TODO(b/111634734): :lib and :protos_all contain dependencies that
        # cannot be built on mobile platforms. Instead, include the appropriate
        # tf_lib depending on the build platform.
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
    ]),
)

tf_cc_test(
    name = "reader_test",
    srcs = ["reader_test.cc"],
    data = [
        ":saved_model_half_plus_two",
    ],
    linkstatic = 1,
    deps = [
        ":constants",
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
        "//tensorflow/core:android_tensorflow_lib",
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
        ":reader",
    ] + if_not_mobile([
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
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/platform:strcat",
        "//tensorflow/core/util/tensor_bundle",
        "@com_google_absl//absl/container:flat_hash_set",
    ],
)

tf_cc_test(
    name = "bundle_v2_test",
    srcs = ["bundle_v2_test.cc"],
    data = [
        ":saved_model_half_plus_two",
    ],
    linkstatic = 1,
    deps = [
        ":bundle_v2",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
        "//tensorflow/core/platform:test",
    ],
)

tf_cc_test(
    name = "saved_model_bundle_test",
    srcs = ["saved_model_bundle_test.cc"],
    data = [
        ":saved_model_half_plus_two",
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

tf_cc_test(
    name = "saved_model_bundle_lite_test",
    srcs = ["saved_model_bundle_lite_test.cc"],
    data = [
        ":saved_model_half_plus_two",
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
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        "//tensorflow/python:dtypes",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:tensor_spec",
        "//tensorflow/python:variables",
        "//tensorflow/python/compat:v2_compat",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/module",
        "//tensorflow/python/saved_model",
        "//tensorflow/python/saved_model:save_options",
        "@absl_py//absl:app",
    ],
)

# TODO(b/32673259): add a test to continuously validate these files.
filegroup(
    name = "saved_model_half_plus_two",
    srcs = glob([
        "testdata/half_plus_two_pbtxt/**",
        "testdata/half_plus_two_main_op/**",
        "testdata/half_plus_two/**",
        "testdata/half_plus_two_v2/**",
        "testdata/x_plus_y_v2_debuginfo/**",
        "testdata/CyclicModule/**",
        "testdata/VarsAndArithmeticObjectGraph/**",
    ]),
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
    ]),
)
