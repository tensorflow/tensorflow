#include "third_party/absl/strings/str_cat.h"
#TODO(aselle) : describe this package.

load("//tensorflow:strict.default.bzl", "py_strict_binary")
load(
    "//tensorflow:tensorflow.bzl",
    "tf_cc_test",
)
load("//tensorflow:tensorflow.default.bzl", "filegroup")
load(
    "//tensorflow/core/platform:rules_cc.bzl",
    "cc_library",
)

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [
        "//tensorflow/cc/experimental/libtf:__subpackages__",
    ],
    licenses = ["notice"],
)

cc_library(
    name = "libtf",
    srcs = [
        "object.cc",
    ],
    hdrs = [
        "object.h",
        "value.h",
        "value_iostream.h",
    ],
    deps = [
        "//tensorflow/c/eager:abstract_tensor_handle",
        "//tensorflow/c/eager:immediate_execution_tensor_handle",
        "//tensorflow/cc/experimental/libtf/impl:iostream",
        "//tensorflow/cc/experimental/libtf/impl:none",
        "//tensorflow/cc/experimental/libtf/impl:scalars",
        "//tensorflow/cc/experimental/libtf/impl:string",
        "//tensorflow/cc/experimental/libtf/impl:tensor_spec",
        "//tensorflow/core/platform:errors",
        "//tensorflow/core/platform:intrusive_ptr",
        "//tensorflow/core/platform:status",
        "//tensorflow/core/platform:statusor",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "module",
    srcs = ["module.cc"],
    hdrs = ["module.h"],
    deps = [
        "//tensorflow/cc/experimental/libexport:load",
        "//tensorflow/cc/experimental/libtf/runtime",
        "//tensorflow/core/platform:errors",
        "//tensorflow/core/platform:statusor",
        "//tensorflow/core/protobuf:for_core_protos_cc",
    ],
)

cc_library(
    name = "function",
    srcs = [
        "function.cc",
    ],
    hdrs = [
        "function.h",
    ],
    deps = [
        ":libtf",
        "//tensorflow/c/eager:abstract_context",
        "//tensorflow/c/eager:abstract_function",
        "//tensorflow/c/eager:abstract_tensor_handle",
        "//tensorflow/core:framework",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/platform:errors",
        "//tensorflow/core/platform:statusor",
        "@com_google_absl//absl/cleanup",
    ],
)

py_strict_binary(
    name = "generate_testdata",
    srcs = ["tests/generate_testdata.py"],
    python_version = "PY3",
    deps = [
        "//tensorflow/python/compat:v2_compat",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:tensor_spec",
        "//tensorflow/python/module",
        "//tensorflow/python/ops:variables",
        "//tensorflow/python/saved_model",
        "@absl_py//absl:app",
        "@absl_py//absl/flags",
    ],
)

filegroup(
    name = "testdata",
    srcs = glob([
        "tests/testdata/**",
    ]),
)

tf_cc_test(
    name = "libtf_object_test",
    size = "medium",
    srcs = ["tests/object_test.cc"],
    deps = [
        ":libtf",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core/platform:statusor",
    ],
)

tf_cc_test(
    name = "libtf_perf_test",
    size = "medium",
    srcs = ["tests/perf_test.cc"],
    deps = [
        ":libtf",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
    ],
)

tf_cc_test(
    name = "libtf_value_test",
    size = "medium",
    srcs = ["tests/value_test.cc"],
    deps = [
        ":libtf",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
    ],
)

tf_cc_test(
    name = "libtf_visit_test",
    size = "medium",
    srcs = ["tests/visit_test.cc"],
    deps = [
        ":libtf",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
    ],
)

cc_library(
    name = "runtime_test",
    testonly = 1,
    srcs = [
        "tests/runtime_test.cc",
    ],
    hdrs = [
        "tests/runtime_test.h",
    ],
    data = [":testdata"],
    deps = [
        ":libtf",
        "//tensorflow/c:tf_datatype",
        "//tensorflow/c:tf_status_helper",
        "//tensorflow/c/eager:unified_api_testutil",
        "//tensorflow/cc/experimental/libtf/runtime",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core/platform:resource_loader",
        "//tensorflow/core/platform:status_matchers",
        "//tensorflow/core/platform:statusor",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
    ],
)

tf_cc_test(
    name = "libtf_runtime_test",
    size = "medium",
    srcs = [
        "tests/runtime_test_core.cc",
    ],
    deps = [
        ":runtime_test",
        "//tensorflow/cc/experimental/libtf/runtime",
        "//tensorflow/cc/experimental/libtf/runtime/core",
    ],
)

tf_cc_test(
    name = "libtf_module_test",
    size = "medium",
    srcs = ["tests/module_test.cc"],
    data = [":testdata"],
    deps = [
        ":libtf",
        ":module",
        "//tensorflow/cc/experimental/libexport:load",
        "//tensorflow/cc/experimental/libtf/runtime/core",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core/platform:path",
        "//tensorflow/core/platform:protobuf",
        "//tensorflow/core/platform:resource_loader",
        "//tensorflow/core/platform:status_matchers",
        "//tensorflow/core/platform:statusor",
    ],
)

tf_cc_test(
    name = "libtf_tensor_test",
    size = "medium",
    srcs = ["tests/tensor_test.cc"],
    tags = ["no_oss"],  # TODO(b/193268458): Need to disable TFRT.
    deps = [
        ":libtf",
        "//tensorflow/c:tf_status_helper",
        "//tensorflow/c/eager:abstract_context",
        "//tensorflow/c/eager:abstract_tensor_handle",
        "//tensorflow/c/eager:c_api_experimental",
        "//tensorflow/c/eager:c_api_unified_internal",
        "//tensorflow/c/eager:unified_api_testutil",
        "//tensorflow/compiler/mlir/tensorflow/c:mlir_c_api_registration",
        "//tensorflow/core:framework",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core/lib/llvm_rtti",
        "//tensorflow/core/platform:errors",
    ],
)

tf_cc_test(
    name = "function_test",
    size = "medium",
    srcs = ["tests/function_test.cc"],
    tags = ["no_oss"],  # TODO(b/193268458): Need to disable TFRT.
    deps = [
        ":function",
        ":libtf",
        "//tensorflow/c:tf_status_helper",
        "//tensorflow/c/eager:abstract_context",
        "//tensorflow/c/eager:abstract_function",
        "//tensorflow/c/eager:abstract_tensor_handle",
        "//tensorflow/c/eager:graph_function",
        "//tensorflow/c/eager:unified_api_testutil",
        "//tensorflow/compiler/mlir/tensorflow/c:mlir_c_api_registration",
        "//tensorflow/core:framework",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core/platform:errors",
        "//tensorflow/core/platform:refcount",
        "//tensorflow/core/platform:statusor",
    ],
)

tf_cc_test(
    name = "variable_test",
    size = "small",
    srcs = ["tests/variable_test.cc"],
    tags = ["no_oss"],  # TODO(b/193268458): Need to disable TFRT.
    deps = [
        ":function",
        ":libtf",
        "//tensorflow/c:tf_status_helper",
        "//tensorflow/c/eager:abstract_context",
        "//tensorflow/c/eager:abstract_function",
        "//tensorflow/c/eager:abstract_tensor_handle",
        "//tensorflow/c/eager:graph_function",
        "//tensorflow/c/eager:unified_api_testutil",
        "//tensorflow/c/experimental/ops:resource_variable_ops",
        "//tensorflow/compiler/mlir/tensorflow/c:mlir_c_api_registration",
        "//tensorflow/core:framework",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core/platform:errors",
        "//tensorflow/core/platform:refcount",
        "//tensorflow/core/platform:statusor",
    ],
)

tf_cc_test(
    name = "libtf_transform_test",
    size = "medium",
    srcs = ["tests/mlir_transform_test.cc"],
    data = [":testdata"],
    deps = [
        ":libtf",
        "//tensorflow/c/eager:c_api_experimental",
        "//tensorflow/cc/experimental/libtf/mlir:transform",
        "//tensorflow/compiler/mlir/tensorflow/c:mlir_c_api_registration",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core/platform:path",
        "//tensorflow/core/platform:resource_loader",
        "//tensorflow/core/platform:statusor",
    ],
)
