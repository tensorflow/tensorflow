# Experimental SavedModel C APIs for TensorFlow. See RFC
# https://github.com/tensorflow/community/pull/207
# Targets in this directory are pure C++ "Classes" underlying the C API types
# under tf/c/experimental/saved_model/public/. They are subject to change and
# have visibility limited to Tensorflow's implementation only.
load(
    "//tensorflow:tensorflow.bzl",
    "tf_cc_test",
)
load("//tensorflow/core/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [
        "//tensorflow/c:__subpackages__",
        "//tensorflow/c/experimental/saved_model/internal:__pkg__",
    ],
    licenses = ["notice"],
)

cc_library(
    name = "concrete_function",
    hdrs = [
        "concrete_function.h",
    ],
    deps = [
        ":function_metadata",
        "//tensorflow/c/eager:immediate_execution_operation",
        "//tensorflow/c/eager:immediate_execution_tensor_handle",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "function_metadata",
    hdrs = [
        "function_metadata.h",
    ],
)

cc_library(
    name = "saved_model_api",
    hdrs = [
        "saved_model_api.h",
    ],
    deps = [
        ":concrete_function",
        ":signature_def_function",
        "//tensorflow/cc/saved_model:bundle_v2",
        "//tensorflow/core:lib",
        "@com_google_absl//absl/container:flat_hash_map",
    ],
)

cc_library(
    name = "saved_model_utils",
    srcs = [
        "saved_model_utils.cc",
    ],
    hdrs = [
        "saved_model_utils.h",
    ],
    deps = [
        ":function_metadata",
        "//tensorflow/c:tf_tensor_internal",
        "//tensorflow/c/eager:immediate_execution_context",
        "//tensorflow/c/experimental/saved_model/core/revived_types:asset",
        "//tensorflow/c/experimental/saved_model/core/revived_types:constant",
        "//tensorflow/c/experimental/saved_model/core/revived_types:partially_revived_objects",
        "//tensorflow/c/experimental/saved_model/core/revived_types:restored_resource_revival_state",
        "//tensorflow/c/experimental/saved_model/core/revived_types:tf_concrete_function",
        "//tensorflow/c/experimental/saved_model/core/revived_types:tf_concrete_function_revival_state",
        "//tensorflow/c/experimental/saved_model/core/revived_types:tf_signature_def_function_revival_state",
        "//tensorflow/c/experimental/saved_model/core/revived_types:variable",
        "//tensorflow/cc/saved_model:loader_util",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:optional",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "signature_def_function",
    hdrs = [
        "signature_def_function.h",
    ],
    deps = [
        ":signature_def_function_metadata",
        "//tensorflow/c/eager:immediate_execution_operation",
        "//tensorflow/c/eager:immediate_execution_tensor_handle",
        "//tensorflow/core:protos_all_cc",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "signature_def_function_metadata",
    srcs = [
        "signature_def_function_metadata.cc",
    ],
    hdrs = [
        "signature_def_function_metadata.h",
    ],
    deps = [
        ":tensor_spec",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
    ],
)

cc_library(
    name = "test_utils",
    testonly = True,
    srcs = [
        "test_utils.cc",
    ],
    hdrs = [
        "test_utils.h",
    ],
    deps = [
        "//tensorflow/c:tensor_interface",
        "//tensorflow/c/eager:immediate_execution_context",
        "//tensorflow/c/eager:immediate_execution_tensor_handle",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core/common_runtime:core_cpu_lib",
        "//tensorflow/core/common_runtime/eager:context",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "tf_concrete_function_test_protos",
    testonly = True,
    srcs = ["tf_concrete_function_test_protos.cc"],
    hdrs = ["tf_concrete_function_test_protos.h"],
    deps = [
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "tf_saved_model_api",
    srcs = [
        "tf_saved_model_api.cc",
    ],
    hdrs = ["tf_saved_model_api.h"],
    deps = [
        ":concrete_function",
        ":saved_model_api",
        ":saved_model_utils",
        ":signature_def_function",
        "//tensorflow/c/eager:immediate_execution_context",
        "//tensorflow/c/eager:immediate_execution_tensor_handle",
        "//tensorflow/c/experimental/saved_model/core/ops:restore_ops",
        "//tensorflow/c/experimental/saved_model/core/revived_types:constant",
        "//tensorflow/c/experimental/saved_model/core/revived_types:flat_tensor_function",
        "//tensorflow/c/experimental/saved_model/core/revived_types:partially_revived_objects",
        "//tensorflow/c/experimental/saved_model/core/revived_types:revived_objects",
        "//tensorflow/c/experimental/saved_model/core/revived_types:tensorhandle_convertible",
        "//tensorflow/c/experimental/saved_model/core/revived_types:tf_concrete_function",
        "//tensorflow/c/experimental/saved_model/core/revived_types:variable",
        "//tensorflow/cc/saved_model:bundle_v2",
        "//tensorflow/cc/saved_model:constants",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:optional",
    ],
)

tf_cc_test(
    name = "constant_loading_test",
    srcs = [
        "constant_loading_test.cc",
    ],
    deps = [
        ":saved_model_utils",
        ":test_utils",
        "//tensorflow/c/eager:immediate_execution_tensor_handle",
        "//tensorflow/c/experimental/saved_model/core/revived_types:constant",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core/common_runtime:core_cpu_lib",
        "//tensorflow/core/common_runtime/eager:context",
        "//tensorflow/core/common_runtime/eager:core",
        "@com_google_absl//absl/status",
    ],
)

tf_cc_test(
    name = "object_graph_traversal_test",
    srcs = [
        "object_graph_traversal_test.cc",
    ],
    deps = [
        ":saved_model_utils",
        ":test_utils",
        ":tf_concrete_function_test_protos",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core/common_runtime:core_cpu_lib",
        "//tensorflow/core/common_runtime/eager:context",
        "//tensorflow/core/common_runtime/eager:core",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:optional",
    ],
)

tf_cc_test(
    name = "saved_variable_loading_test",
    srcs = [
        "saved_variable_loading_test.cc",
    ],
    deps = [
        ":saved_model_utils",
        ":test_utils",
        "//tensorflow/c:tensor_interface",
        "//tensorflow/c/eager:immediate_execution_tensor_handle",
        "//tensorflow/c/experimental/saved_model/core/revived_types:constant",
        "//tensorflow/core:all_kernels",
        "//tensorflow/core:framework",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core/common_runtime:core_cpu_lib",
        "//tensorflow/core/common_runtime/eager:context",
        "//tensorflow/core/common_runtime/eager:core",
        "//tensorflow/core/common_runtime/eager:tensor_handle",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/types:optional",
    ],
)

tf_cc_test(
    name = "signature_flattening_test",
    srcs = [
        "signature_flattening_test.cc",
    ],
    deps = [
        ":saved_model_utils",
        "//tensorflow/c/experimental/saved_model/core:tf_concrete_function_test_protos",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core/common_runtime/eager:core",
    ],
)

cc_library(
    name = "tensor_spec",
    srcs = [
        "tensor_spec.cc",
    ],
    hdrs = [
        "tensor_spec.h",
    ],
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:protos_all_cc",
    ],
)

tf_cc_test(
    name = "tf_concrete_function_loading_test",
    srcs = [
        "tf_concrete_function_loading_test.cc",
    ],
    deps = [
        ":saved_model_utils",
        ":test_utils",
        ":tf_concrete_function_test_protos",
        "//tensorflow/c:tensor_interface",
        "//tensorflow/c/eager:immediate_execution_tensor_handle",
        "//tensorflow/c/experimental/saved_model/core/revived_types:constant",
        "//tensorflow/c/experimental/saved_model/core/revived_types:tensorhandle_convertible",
        "//tensorflow/c/experimental/saved_model/core/revived_types:tf_concrete_function",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core/common_runtime:core_cpu_lib",
        "//tensorflow/core/common_runtime/eager:context",
        "//tensorflow/core/common_runtime/eager:core",
        "@com_google_absl//absl/status",
        "@local_xla//xla/tsl/protobuf:error_codes_proto_impl_cc",
    ],
)
