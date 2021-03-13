load("@bazel_skylib//lib:selects.bzl", "selects")
load("//tensorflow/lite:special_rules.bzl", "tflite_extra_gles_deps")
load("@build_bazel_rules_apple//apple:ios.bzl", "ios_static_framework")
load("@build_bazel_rules_apple//apple:macos.bzl", "macos_dylib")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],  # Apache 2.0
)

exports_files([
    "delegate.h",
    "metal_delegate.h",
])

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
            # We don't need to link libGLESv3, because if it exists,
            # it is a symlink to libGLESv2.
            # See Compatibility Definition Document:
            # https://source.android.com/compatibility/10/android-10-cdd#7_1_4_1_opengl_es
            "-lGLESv2",
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
        "//tensorflow/lite/delegates/gpu/common/transformations:model_transformations",
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
    module_name = "TensorFlowLiteCMetal",
    sdk_frameworks = ["Metal"],
    deps = [
        "//tensorflow/lite:kernel_api",
        "//tensorflow/lite:minimal_logging",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/delegates/gpu/common:convert",
        "//tensorflow/lite/delegates/gpu/common:gpu_info",
        "//tensorflow/lite/delegates/gpu/common:model",
        "//tensorflow/lite/delegates/gpu/common:model_builder",
        "//tensorflow/lite/delegates/gpu/common:model_transformer",
        "//tensorflow/lite/delegates/gpu/common:precision",
        "//tensorflow/lite/delegates/gpu/common:quantization_util",
        "//tensorflow/lite/delegates/gpu/common:shape",
        "//tensorflow/lite/delegates/gpu/common:status",
        "//tensorflow/lite/delegates/gpu/common:tensor",
        "//tensorflow/lite/delegates/gpu/common:types",
        "//tensorflow/lite/delegates/gpu/metal:buffer_convert",
        "//tensorflow/lite/delegates/gpu/metal:inference_context",
        "@com_google_absl//absl/types:span",
    ],
    alwayslink = 1,
)

objc_library(
    name = "metal_delegate_internal",
    hdrs = ["metal_delegate_internal.h"],
    sdk_frameworks = ["Metal"],
    deps = [
        "//tensorflow/lite/delegates/gpu:metal_delegate",
    ],
    alwayslink = 1,
)

# build -c opt --config android_arm64 --copt -Os --copt -DTFLITE_GPU_BINARY_RELEASE --linkopt -s --strip always :libtensorflowlite_gpu_gl.so
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

# build -c opt --config android_arm64 --copt -Os --copt -DTFLITE_GPU_BINARY_RELEASE --linkopt -s --strip always :libtensorflowlite_gpu_delegate.so
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
    minimum_os_version = "11.4",
    deps = [":metal_delegate"],
)

# Note: Support for MacOS is best-effort at the moment.
# bazel build -c opt --copt -Os --copt -DTFLITE_GPU_BINARY_RELEASE --copt -fvisibility=hidden --linkopt -s --strip always --cxxopt=-std=c++14 :tensorflow_lite_gpu_dylib --apple_platform_type=macos
macos_dylib(
    name = "tensorflow_lite_gpu_dylib",
    minimum_os_version = "10.13",
    tags = [
        "manual",
        "nobuilder",
        "notap",
    ],
    deps = [
        ":metal_delegate",
        ":metal_delegate_internal",
    ],
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
        "@vulkan_headers//:vulkan_headers_no_prototypes",
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

# Currently the GPU delegate needs to be built on Android (due to EGL dependency),
# or built with -DCL_DELEGATE_NO_GL (disabling OpenGL backend fallback), or both.
selects.config_setting_group(
    name = "supports_gpu_delegate",
    match_any = [
        "//tensorflow:android",
        "//tensorflow/lite/delegates/gpu/cl:opencl_delegate_no_gl",
    ],
)

cc_library(
    name = "delegate",
    srcs = ["delegate.cc"],
    hdrs = ["delegate.h"],
    linkopts = select({
        "//tensorflow:android": [
            "-lEGL",
            # We don't need to link libGLESv3, because if it exists,
            # it is a symlink to libGLESv2.
            # See Compatibility Definition Document:
            # https://source.android.com/compatibility/10/android-10-cdd#7_1_4_1_opengl_es
            "-lGLESv2",
        ],
        "//conditions:default": [],
    }),
    deps = select({
        "//tensorflow/lite/delegates/gpu/cl:opencl_delegate_no_gl": [],
        "//conditions:default": [
            "//tensorflow/lite/delegates/gpu/gl:api2",
        ],
    }) + [
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/types:span",
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
        "//tensorflow/lite/delegates/gpu/common:quantization_util",
        "//tensorflow/lite/delegates/gpu/common:status",
        "//tensorflow/lite/kernels/internal:optimized_base",
    ],
)
