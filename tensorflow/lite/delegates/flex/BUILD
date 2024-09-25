load(
    "//tensorflow:tensorflow.bzl",
    "if_mobile",
    "if_not_mobile",
    "tf_cc_test",
    "tf_features_nolayering_check_if_ios",
    "tf_opts_nortti_if_lite_protos",
    "tf_opts_nortti_if_mobile",
)
load("//tensorflow:tensorflow.default.bzl", "get_compatible_with_portable")
load("//tensorflow/lite:build_def.bzl", "tflite_copts")
load("//tensorflow/lite:special_rules.bzl", "internal_visibility_allowlist")
load("//tensorflow/lite/delegates/flex:build_def.bzl", "tflite_flex_cc_library", "tflite_flex_shared_library")

default_visibility = [
    "//tensorflow/compiler/mlir/lite:__subpackages__",
    "//tensorflow/lite/android:__subpackages__",
    "//tensorflow/lite/toco/tflite:__subpackages__",
]

#
# This is a TF Lite delegate that is powered by TensorFlow's Eager.
#
package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = default_visibility,
    licenses = ["notice"],
)

exports_files([
    "delegate.h",
    "exported_symbols.lds",
    "version_script.lds",
])

cc_library(
    name = "buffer_map",
    srcs = ["buffer_map.cc"],
    hdrs = ["buffer_map.h"],
    compatible_with = get_compatible_with_portable(),
    copts = tf_opts_nortti_if_lite_protos(),
    features = tf_features_nolayering_check_if_ios(),
    deps = [
        ":buffer_map_util",
        ":util",
        "//tensorflow/lite:string",
        "//tensorflow/lite/core/c:common",
        "//tensorflow/lite/kernels/internal:compatibility",
    ] + if_mobile([
        "//tensorflow/core:portable_tensorflow_lib_lite",
    ]) + if_not_mobile([
        "//tensorflow/c:c_api_internal",
        "//tensorflow/core:framework",
    ]),
)

cc_library(
    name = "buffer_map_util",
    srcs = ["buffer_map_util.cc"],
    hdrs = ["buffer_map_util.h"],
    compatible_with = get_compatible_with_portable(),
    copts = tf_opts_nortti_if_lite_protos(),
    features = tf_features_nolayering_check_if_ios(),
    deps = [
        ":util",
        "//tensorflow/lite:string_util",
        "//tensorflow/lite/core/c:common",
        "//tensorflow/lite/experimental/resource",
    ] + if_mobile([
        "//tensorflow/core:portable_tensorflow_lib_lite",
    ]) + if_not_mobile([
        "//tensorflow/c:c_api_internal",
        "//tensorflow/core:framework",
        "//tensorflow/c:tf_tensor_internal",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/platform:status",
    ]),
)

tf_cc_test(
    name = "buffer_map_test",
    size = "small",
    srcs = ["buffer_map_test.cc"],
    deps = [
        ":buffer_map",
        ":buffer_map_util",
        "//tensorflow/core:framework",
        "//tensorflow/lite:framework",
        "//tensorflow/lite:string_util",
        "//tensorflow/lite:util",
        "//tensorflow/lite/core/c:c_api_types",
        "//tensorflow/lite/testing:util",
        "@com_google_googletest//:gtest_main",
    ],
)

# Define the standard flex delegate library, that pulls in the standard set
# of TensorFlow ops and kernels, using tflite_flex_cc_library with no
# models parameter. Custom flex delegate can be defined with
# tflite_flex_cc_library if the parameter models is provided. Tensorflow
# user-provided ops could also be supported by passing to additional_deps.
# Ex:
# tflite_flex_cc_library(
#   name = "sample_delegate",
#   models = ["model1.tflite", "model2.tflite"],
#   additional_deps = ["your_custom_ops_lib"],
# )
tflite_flex_cc_library(
    name = "delegate",
    compatible_with = get_compatible_with_portable(),
    visibility = ["//visibility:public"],
)

# Compared to the library above, this one doesn't define a strong symbol for
# AcquireFlexDelegate(). This is useful if one doesn't want the default flex
# delegate to be automatically applied when building the interpreter.
tflite_flex_cc_library(
    name = "delegate_without_symbol",
    link_symbol = False,
    visibility = ["//visibility:public"],
)

# Shared lib target for convenience, pulls in the standard set of TensorFlow
# ops and kernels. The output library name is platform dependent:
#   - Linux/Android: `libtensorflowlite_flex.so`
#   - Mac: `libtensorflowlite_flex.dylib`
#   - Windows: `tensorflowlite_flex.dll`
tflite_flex_shared_library(
    name = "tensorflowlite_flex",
)

cc_library(
    name = "delegate_symbol",
    srcs = [
        "delegate_symbol.cc",
    ],
    compatible_with = get_compatible_with_portable(),
    copts = tflite_copts(),
    visibility = ["//visibility:public"],
    deps = [
        ":delegate_only_runtime",
        "//tensorflow/lite/core/c:c_api_types",
    ],
    alwayslink = 1,
)

# Delegate implementation that does *not* pull in the standard set of TensorFlow
# ops and kernels.
cc_library(
    name = "delegate_only_runtime",
    srcs = [
        "delegate.cc",
        "kernel.cc",
        "kernel.h",
    ],
    hdrs = [
        "delegate.h",
    ],
    compatible_with = get_compatible_with_portable(),
    copts = tflite_copts() + tf_opts_nortti_if_mobile(),
    features = tf_features_nolayering_check_if_ios(),
    visibility = ["//visibility:public"],
    deps = [
        ":buffer_map",
        ":delegate_data",
        ":tflite_subgraph_execute",
        ":util",
        "//tensorflow/core/tfrt/fallback:op_kernel_runner",
        "//tensorflow/lite:kernel_api",
        "//tensorflow/lite:macros",
        "//tensorflow/lite:minimal_logging",
        "//tensorflow/lite:string",
        "//tensorflow/lite:string_util",
        "//tensorflow/lite:util",
        "//tensorflow/lite/core/api",
        "//tensorflow/lite/core/c:common",
        "//tensorflow/lite/delegates/utils:simple_delegate",
        "//tensorflow/lite/kernels:kernel_util",
        "@com_google_absl//absl/strings",
        "@flatbuffers",
    ] + if_mobile([
        "//tensorflow/core:portable_tensorflow_lib_lite",
    ]) + if_not_mobile([
        "//tensorflow/core/common_runtime/eager:context",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:framework",
    ]),
    alwayslink = 1,
)

tf_cc_test(
    name = "delegate_test",
    size = "small",
    srcs = ["delegate_test.cc"],
    tags = [
        "no_gpu",  # GPU + flex is not officially supported.
    ],
    deps = [
        ":delegate",
        ":test_util",
        "//tensorflow/lite:shared_library",
        "//tensorflow/lite/kernels:test_util",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "delegate_data",
    srcs = ["delegate_data.cc"],
    hdrs = ["delegate_data.h"],
    compatible_with = get_compatible_with_portable(),
    copts = tf_opts_nortti_if_mobile(),
    features = tf_features_nolayering_check_if_ios(),
    visibility = ["//visibility:public"],
    deps = [
        ":buffer_map",
        ":subgraph_resource",
        ":util",
        "//tensorflow/lite:cc_api_experimental",
        "//tensorflow/lite:util",
        "//tensorflow/lite/core/c:common",
        "//tensorflow/lite/schema:schema_fbs",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@flatbuffers",
    ] + if_mobile([
        "//tensorflow/core:portable_tensorflow_lib_lite",
    ]) + if_not_mobile([
        "//tensorflow/core/common_runtime/eager:context",
        "//tensorflow/core/common_runtime/eager:core_no_xla",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
    ]),
)

tf_cc_test(
    name = "delegate_data_test",
    size = "small",
    srcs = ["delegate_data_test.cc"],
    deps = [
        ":delegate_data",
        "//tensorflow/core:test",
        "//tensorflow/core/common_runtime/eager:context",
        "//tensorflow/core/platform:mutex",
        "//tensorflow/core/platform:protobuf",
        "//tensorflow/core/platform:status",
        "//tensorflow/lite:framework",
        "//tensorflow/lite/core/api:error_reporter",
        "//tensorflow/lite/core/c:common",
        "//tensorflow/lite/kernels:subgraph_test_util",
        "//tensorflow/lite/testing:util",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "subgraph_resource",
    hdrs = ["subgraph_resource.h"],
    compatible_with = get_compatible_with_portable(),
    features = tf_features_nolayering_check_if_ios(),
    deps = [
        "//tensorflow/lite:cc_api_experimental",
        "//tensorflow/lite/core/c:common",
    ] + if_mobile([
        "//tensorflow/core:portable_tensorflow_lib_lite",
    ]) + if_not_mobile([
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
    ]),
)

tf_cc_test(
    name = "kernel_test",
    size = "small",
    srcs = [
        "kernel.h",
        "kernel_test.cc",
    ],
    tags = ["no_gpu"],  # GPU + flex is not officially supported.
    deps = [
        ":delegate",
        ":delegate_data",
        ":test_util",
        "//tensorflow/core/platform:status",
        "//tensorflow/core/tfrt/fallback:op_kernel_runner",
        "//tensorflow/lite/core/c:common",
        "//tensorflow/lite/delegates/utils:simple_delegate",
        "//tensorflow/lite/kernels:kernel_util",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "test_util",
    testonly = True,
    srcs = ["test_util.cc"],
    hdrs = ["test_util.h"],
    visibility = internal_visibility_allowlist(),
    deps = [
        "//tensorflow/c:c_api_internal",
        "//tensorflow/lite:string",
        "//tensorflow/lite/kernels:test_util",
        "@com_google_absl//absl/memory",
        "@flatbuffers",
    ],
)

cc_library(
    name = "util",
    srcs = ["util.cc"],
    hdrs = ["util.h"],
    compatible_with = get_compatible_with_portable(),
    features = tf_features_nolayering_check_if_ios(),
    #TODO(b/206038955): Consider restrict the visibility to '//third_party/fcp/client:__subpackages__'.
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/lite:kernel_api",
        "//tensorflow/lite:string_util",
        "//tensorflow/lite:util",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/core/c:c_api_types",
        "//tensorflow/lite/core/c:common",
        "//tensorflow/lite/kernels/internal:tensor",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
    ] + if_mobile([
        "//tensorflow/core:portable_tensorflow_lib_lite",
    ]) + if_not_mobile([
        "//tensorflow/c:c_api_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core:framework",
        "//tensorflow/core/protobuf:error_codes_proto_impl_cc",
    ]),
)

tf_cc_test(
    name = "util_test",
    size = "small",
    srcs = ["util_test.cc"],
    deps = [
        ":util",
        "//tensorflow/c:tf_datatype",
        "//tensorflow/core:framework",
        "//tensorflow/core:portable_gif_internal",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/platform:errors",
        "//tensorflow/core/platform:status",
        "//tensorflow/core/protobuf:error_codes_proto_impl_cc",
        "//tensorflow/lite:string",
        "//tensorflow/lite:string_util",
        "//tensorflow/lite:util",
        "//tensorflow/lite/core/c:c_api_types",
        "//tensorflow/lite/core/c:common",
        "@com_google_absl//absl/status",
        "@com_google_googletest//:gtest_main",
    ],
)

tf_cc_test(
    name = "allowlisted_flex_ops_test",
    size = "small",
    srcs = [
        "allowlisted_flex_ops_test.cc",
    ],
    features = tf_features_nolayering_check_if_ios(),
    deps = [
        ":delegate",
        "//tensorflow/compiler/mlir/lite/delegates/flex:allowlisted_flex_ops_lib",
        "@com_google_googletest//:gtest_main",
    ] + if_mobile([
        "//tensorflow/core:portable_tensorflow_lib_lite",
    ]) + if_not_mobile([
        "//tensorflow/core:framework",
    ]),
)

# Alias to support selective build of image ops.
# TODO(b/163285312): Remove after tensorflow/core refactoring completed.
cc_library(
    name = "portable_images_lib",
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/core:portable_gif_internal",
        "//tensorflow/core:portable_jpeg_internal",
        "//tensorflow/core/lib/png:png_io",
    ],
)

cc_library(
    name = "tflite_subgraph_execute",
    srcs = ["tflite_subgraph_execute.cc"],
    compatible_with = get_compatible_with_portable(),
    copts = tf_opts_nortti_if_mobile(),
    features = tf_features_nolayering_check_if_ios(),
    deps = [
        ":buffer_map_util",
        ":subgraph_resource",
        ":util",
        "//tensorflow/lite:cc_api_experimental",
        "//tensorflow/lite:string_util",
        "//tensorflow/lite/core/c:c_api_types",
        "//tensorflow/lite/core/c:common",
        "//tensorflow/lite/kernels:builtin_ops",
        "//tensorflow/lite/kernels:kernel_util",
        "//tensorflow/lite/kernels/internal:tensor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
    ] + if_mobile([
        "//tensorflow/core:portable_tensorflow_lib_lite",
    ]) + if_not_mobile([
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/c:tf_tensor_internal",
    ]),
    alwayslink = 1,
)
