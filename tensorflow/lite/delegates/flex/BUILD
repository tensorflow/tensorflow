load("//tensorflow:tensorflow.bzl", "tf_cc_test", "tf_opts_nortti_if_android", "tf_opts_nortti_if_lite_protos")
load("//tensorflow/lite:build_def.bzl", "tflite_copts")
load("//tensorflow/lite:special_rules.bzl", "internal_visibility_allowlist")
load("//tensorflow/lite/delegates/flex:build_def.bzl", "tflite_flex_cc_library", "tflite_flex_shared_library")
load("//tensorflow:tensorflow.bzl", "get_compatible_with_cloud")

#
# This is a TF Lite delegate that is powered by TensorFlow's Eager.
#
package(
    default_visibility = [
        "//tensorflow/compiler/mlir/lite:__subpackages__",
        "//tensorflow/lite/android:__subpackages__",
        "//tensorflow/lite/toco/tflite:__subpackages__",
    ],
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
    copts = tf_opts_nortti_if_lite_protos(),
    deps = [
        ":buffer_map_util",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite:string",
        "//tensorflow/lite/kernels/internal:compatibility",
    ] + select({
        "//tensorflow:android": [
            "//tensorflow/core:portable_tensorflow_lib_lite",
        ],
        "//tensorflow:ios": [
            "//tensorflow/core:portable_tensorflow_lib_lite",
        ],
        "//conditions:default": [
            "//tensorflow/c:c_api_internal",
            "//tensorflow/core:framework",
        ],
    }),
)

cc_library(
    name = "buffer_map_util",
    srcs = ["buffer_map_util.cc"],
    hdrs = ["buffer_map_util.h"],
    copts = tf_opts_nortti_if_lite_protos(),
    deps = [
        ":util",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite:string_util",
    ] + select({
        "//tensorflow:android": [
            "//tensorflow/core:portable_tensorflow_lib_lite",
        ],
        "//tensorflow:ios": [
            "//tensorflow/core:portable_tensorflow_lib_lite",
        ],
        "//conditions:default": [
            "//tensorflow/c:c_api_internal",
            "//tensorflow/core:framework",
            "//tensorflow/core:protos_all_cc",
            "//tensorflow/core/platform:status",
        ],
    }),
)

tf_cc_test(
    name = "buffer_map_test",
    size = "small",
    srcs = ["buffer_map_test.cc"],
    deps = [
        ":buffer_map",
        "//tensorflow/lite:framework",
        "//tensorflow/lite:string_util",
        "//tensorflow/lite:util",
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
    visibility = ["//visibility:public"],
)

# Shared lib target for convenience, pulls in the standard set of TensorFlow
# ops and kernels. The output library name is platform dependent:
#   - Linux/Android: `libtensorflowlite_flex.so`
#   - Mac: `libtensorflowlite_flex.dylib`
#   - Windows: `libtensorflowlite_flex.dll`
tflite_flex_shared_library(
    name = "tensorflowlite_flex",
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
    copts = tflite_copts(),
    visibility = ["//visibility:public"],
    deps = [
        ":buffer_map",
        ":delegate_data",
        ":util",
        ":tflite_subgraph_execute",
        "@flatbuffers",
        "@com_google_absl//absl/strings:strings",
        "//tensorflow/lite/core/api",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite:kernel_api",
        "//tensorflow/lite:macros",
        "//tensorflow/lite:minimal_logging",
        "//tensorflow/lite:string",
        "//tensorflow/lite:string_util",
        "//tensorflow/lite:util",
        "//tensorflow/lite/delegates/utils:simple_delegate",
        "//tensorflow/lite/kernels:kernel_util",
    ] + select({
        "//tensorflow:android": [
            "//tensorflow/core:portable_tensorflow_lib_lite",
        ],
        "//tensorflow:ios": [
            "//tensorflow/core:portable_tensorflow_lib_lite",
        ],
        "//conditions:default": [
            "//tensorflow/core/common_runtime/eager:context",
            "//tensorflow/core/common_runtime/eager:execute",
            "//tensorflow/core/common_runtime/eager:tensor_handle",
            "//tensorflow/core:lib",
            "//tensorflow/core:protos_all_cc",
            "//tensorflow/core:framework",
        ],
    }),
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
    visibility = ["//visibility:public"],
    deps = [
        ":buffer_map",
        ":subgraph_resource",
        ":util",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@flatbuffers",
        "//tensorflow/lite/schema:schema_fbs",
        "//tensorflow/lite:cc_api",
        "//tensorflow/lite:util",
    ] + select({
        "//tensorflow:android": [
            "//tensorflow/core:portable_tensorflow_lib_lite",
        ],
        "//tensorflow:ios": [
            "//tensorflow/core:portable_tensorflow_lib_lite",
        ],
        "//conditions:default": [
            "//tensorflow/core/common_runtime/eager:context",
            "//tensorflow/core/common_runtime/eager:core",
            "//tensorflow/core:core_cpu",
            "//tensorflow/core:framework",
            "//tensorflow/core:lib",
            "//tensorflow/core:protos_all_cc",
        ],
    }),
)

tf_cc_test(
    name = "delegate_data_test",
    size = "small",
    srcs = ["delegate_data_test.cc"],
    deps = [
        ":delegate_data",
        "//tensorflow/core:test",
        "//tensorflow/core/common_runtime/eager:context",
        "//tensorflow/core/platform:protobuf",
        "//tensorflow/core/platform:status",
        "//tensorflow/lite:framework",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/core/api:error_reporter",
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
    deps = [
        "//tensorflow/lite:cc_api",
    ] + select({
        "//tensorflow:android": [
            "//tensorflow/core:portable_tensorflow_lib_lite",
        ],
        "//tensorflow:ios": [
            "//tensorflow/core:portable_tensorflow_lib_lite",
        ],
        "//conditions:default": [
            "//tensorflow/core:framework",
            "//tensorflow/core:lib",
        ],
    }),
)

tf_cc_test(
    name = "kernel_test",
    size = "small",
    srcs = ["kernel_test.cc"],
    tags = ["no_gpu"],  # GPU + flex is not officially supported.
    deps = [
        ":delegate",
        ":delegate_data",
        ":test_util",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "test_util",
    testonly = True,
    srcs = ["test_util.cc"],
    hdrs = ["test_util.h"],
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
    deps = [
        "//tensorflow/lite/c:common",
        "//tensorflow/lite:kernel_api",
    ] + select({
        "//tensorflow:android": [
            "//tensorflow/core:portable_tensorflow_lib_lite",
        ],
        "//tensorflow:ios": [
            "//tensorflow/core:portable_tensorflow_lib_lite",
        ],
        "//conditions:default": [
            "//tensorflow/c:c_api_internal",
            "//tensorflow/core:lib",
            "//tensorflow/core:framework",
        ],
    }),
)

tf_cc_test(
    name = "util_test",
    size = "small",
    srcs = ["util_test.cc"],
    deps = [
        ":util",
        "//tensorflow/lite:string",
        "//tensorflow/lite/testing:util",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "allowlisted_flex_ops_lib",
    srcs = [
        "allowlisted_flex_ops.cc",
    ],
    hdrs = [
        "allowlisted_flex_ops.h",
        "allowlisted_flex_ops_internal.h",
    ],
    compatible_with = get_compatible_with_cloud(),
    visibility = internal_visibility_allowlist(),
    deps = select({
        "//tensorflow:android": [
            "//tensorflow/core:portable_tensorflow_lib_lite",
        ],
        "//tensorflow:ios": [
            "//tensorflow/core:portable_tensorflow_lib_lite",
        ],
        "//conditions:default": [
            "//tensorflow/core:framework",
        ],
    }),
)

tf_cc_test(
    name = "allowlisted_flex_ops_test",
    size = "small",
    srcs = [
        "allowlisted_flex_ops_test.cc",
    ],
    deps = [
        ":delegate",
        ":allowlisted_flex_ops_lib",
        "@com_google_googletest//:gtest_main",
    ] + select({
        "//tensorflow:android": [
            "//tensorflow/core:portable_tensorflow_lib_lite",
        ],
        "//tensorflow:ios": [
            "//tensorflow/core:portable_tensorflow_lib_lite",
        ],
        "//conditions:default": [
            "//tensorflow/core:framework",
        ],
    }),
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
    copts = tf_opts_nortti_if_android(),
    deps = [
        ":buffer_map_util",
        ":subgraph_resource",
        "//tensorflow/lite:cc_api",
        "//tensorflow/lite:string_util",
        "//tensorflow/lite/c:c_api_types",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/kernels:builtin_ops",
        "//tensorflow/lite/kernels:kernel_util",
    ] + select({
        "//tensorflow:android": [
            "//tensorflow/core:portable_tensorflow_lib_lite",
        ],
        "//tensorflow:ios": [
            "//tensorflow/core:portable_tensorflow_lib_lite",
        ],
        "//conditions:default": [
            "//tensorflow/core:framework",
            "//tensorflow/core:lib",
        ],
    }),
    alwayslink = 1,
)
