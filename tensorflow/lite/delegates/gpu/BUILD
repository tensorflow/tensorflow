load("//tensorflow/lite:special_rules.bzl", "tflite_extra_gles_deps")
load("@build_bazel_rules_apple//apple:ios.bzl", "ios_static_framework")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],  # Apache 2.0
)

# Primary purpose of this config is to replace ::util::Status with our custom
# light implementation ::tflite::gpu::StatusLite to reduce binary size.  Besides
# that, certain features that were hard to communicate without full open source
# were hidden away too such as compiled models, serialization, and metadata.
# While the latter will be fully available with the open source release, the
# former will have to stay until absl::Status is released.
config_setting(
    name = "tflite_gpu_binary_release",
    values = {"copt": "-DTFLITE_GPU_BINARY_RELEASE"},
)

config_setting(
    name = "tflite_gpu_extra_gles_deps",
    values = {
        "copt": "-DTFLITE_GPU_EXTRA_GLES_DEPS",
        "cpu": "k8",
    },
)

cc_library(
    name = "gl_delegate",
    srcs = ["gl_delegate.cc"],
    hdrs = ["gl_delegate.h"],
    linkopts = select({
        "//tensorflow:android": [
            "-lEGL",
            "-lGLESv3",
        ],
        "//conditions:default": [],
    }),
    deps = [
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/types:span",
        "//tensorflow/lite:kernel_api",
        "//tensorflow/lite:minimal_logging",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/delegates/gpu/common:convert",
        "//tensorflow/lite/delegates/gpu/common:model",
        "//tensorflow/lite/delegates/gpu/common:model_builder",
        "//tensorflow/lite/delegates/gpu/common:model_transformer",
        "//tensorflow/lite/delegates/gpu/common:shape",
        "//tensorflow/lite/delegates/gpu/common:status",
        "//tensorflow/lite/delegates/gpu/common:tensor",
        "//tensorflow/lite/delegates/gpu/common/transformations:general_transformations",
        "//tensorflow/lite/delegates/gpu/gl:api",
        "//tensorflow/lite/delegates/gpu/gl:command_queue",
        "//tensorflow/lite/delegates/gpu/gl:compiler",
        "//tensorflow/lite/delegates/gpu/gl:egl_environment",
        "//tensorflow/lite/delegates/gpu/gl:request_gpu_info",
        "//tensorflow/lite/delegates/gpu/gl:gl_call",
        "//tensorflow/lite/delegates/gpu/gl/converters:bhwc_to_phwc4",
        "//tensorflow/lite/delegates/gpu/gl/converters:phwc4_to_bhwc",
        "//tensorflow/lite/delegates/gpu/gl/kernels:registry",
        "//tensorflow/lite/delegates/gpu/gl/workgroups:best_effort_calculator",
    ] + select({
        "//conditions:default": [
            "//tensorflow/lite/delegates/gpu/gl:common_cc_fbs",
            "//tensorflow/lite/delegates/gpu/gl:metadata_cc_fbs",
            "//tensorflow/lite/delegates/gpu/gl:workgroups_cc_fbs",
            "@flatbuffers",
            "//tensorflow/lite/schema:schema_fbs",
        ],
        ":tflite_gpu_binary_release": [],
    }) + tflite_extra_gles_deps(),
)

objc_library(
    name = "metal_delegate",
    srcs = ["metal_delegate.mm"],
    hdrs = ["metal_delegate.h"],
    sdk_frameworks = ["Metal"],
    deps = [
        "//tensorflow/lite:kernel_api",
        "//tensorflow/lite:minimal_logging",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/delegates/gpu/common:convert",
        "//tensorflow/lite/delegates/gpu/common:model",
        "//tensorflow/lite/delegates/gpu/common:model_builder",
        "//tensorflow/lite/delegates/gpu/common:model_transformer",
        "//tensorflow/lite/delegates/gpu/common:shape",
        "//tensorflow/lite/delegates/gpu/common:status",
        "//tensorflow/lite/delegates/gpu/common:tensor",
        "//tensorflow/lite/delegates/gpu/common:types",
        "//tensorflow/lite/delegates/gpu/common/transformations:general_transformations",
        "//tensorflow/lite/delegates/gpu/metal:api",
        "//tensorflow/lite/delegates/gpu/metal:buffer_convert",
        "//tensorflow/lite/delegates/gpu/metal:compiled_model",
        "//tensorflow/lite/delegates/gpu/metal:inference_context",
        "@com_google_absl//absl/types:span",
    ],
)

objc_library(
    name = "metal_delegate_internal",
    hdrs = ["metal_delegate_internal.h"],
    sdk_frameworks = ["Metal"],
    deps = [
        "//tensorflow/lite/delegates/gpu:metal_delegate",
    ],
)

# build -c opt --config android_arm64 --copt -Os --copt -DTFLITE_GPU_BINARY_RELEASE --copt --linkopt -s --strip always :libtensorflowlite_gpu_gl.so
cc_binary(
    name = "libtensorflowlite_gpu_gl.so",
    linkopts = [
        "-Wl,-soname=libtensorflowlite_gpu_gl.so",
    ] + select({
        "//tensorflow:android": [
            "-lEGL",
            "-lGLESv3",
            "-fvisibility=hidden",
        ],
        "//tensorflow:windows": [],
        "//conditions:default": [
            "-fvisibility=hidden",
        ],
    }),
    linkshared = 1,
    linkstatic = 1,
    tags = [
        "nobuilder",
        "notap",
    ],
    deps = [":gl_delegate"],
)

# build -c opt --config android_arm64 --copt -Os --copt -DTFLITE_GPU_BINARY_RELEASE --copt --linkopt -s --strip always :libtensorflowlite_gpu_delegate.so
cc_binary(
    name = "libtensorflowlite_gpu_delegate.so",
    linkopts = [
        "-Wl,-soname=libtensorflowlite_gpu_delegate.so",
    ] + select({
        "//tensorflow:android": [
            "-lEGL",
            "-lGLESv3",
            "-fvisibility=hidden",
        ],
        "//tensorflow:windows": [],
        "//conditions:default": [
            "-fvisibility=hidden",
        ],
    }),
    linkshared = 1,
    linkstatic = 1,
    tags = [
        "nobuilder",
        "notap",
    ],
    deps = [":delegate"],
)

# bazel build -c opt --cpu ios_arm64 --copt -Os --copt -DTFLITE_GPU_BINARY_RELEASE --copt -fvisibility=hidden --linkopt -s --strip always --cxxopt=-std=c++14 :libtensorflowlite_gpu_metal --apple_platform_type=ios
ios_static_framework(
    name = "tensorflow_lite_gpu_framework",
    hdrs = [
        "metal_delegate.h",
        "metal_delegate_internal.h",
    ],
    minimum_os_version = "10.0",
    deps = [":metal_delegate"],
)

cc_library(
    name = "api",
    srcs = ["api.cc"],
    hdrs = ["api.h"],
    deps = [
        "//tensorflow/lite/delegates/gpu/common:data_type",
        "//tensorflow/lite/delegates/gpu/common:status",
        "//tensorflow/lite/delegates/gpu/common:util",
        "//tensorflow/lite/delegates/gpu/gl:portable",
        "@com_google_absl//absl/types:span",
        "@com_google_absl//absl/types:variant",
        "@opencl_headers",
    ],
)

cc_library(
    name = "spi",
    hdrs = ["spi.h"],
    deps = [
        ":api",
        "//tensorflow/lite/delegates/gpu/common:access_type",
        "//tensorflow/lite/delegates/gpu/common:status",
    ],
)

cc_library(
    name = "delegate",
    srcs = ["delegate.cc"],
    hdrs = ["delegate.h"],
    linkopts = select({
        "//tensorflow:android": [
            "-lEGL",
            "-lGLESv3",
        ],
        "//conditions:default": [],
    }),
    deps = [
        "//tensorflow/lite:kernel_api",
        "//tensorflow/lite:minimal_logging",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/delegates/gpu:api",
        "//tensorflow/lite/delegates/gpu/cl:api",
        "//tensorflow/lite/delegates/gpu/cl:opencl_wrapper",
        "//tensorflow/lite/delegates/gpu/cl:tensor_type_util",
        "//tensorflow/lite/delegates/gpu/common:model",
        "//tensorflow/lite/delegates/gpu/common:model_builder",
        "//tensorflow/lite/delegates/gpu/common:model_transformer",
        "//tensorflow/lite/delegates/gpu/common:status",
        "//tensorflow/lite/delegates/gpu/common/transformations:general_transformations",
        "//tensorflow/lite/delegates/gpu/gl:api2",
        "@com_google_absl//absl/types:span",
    ],
)
