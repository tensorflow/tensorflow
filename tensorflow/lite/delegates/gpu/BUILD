package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # Apache 2.0

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
        "@com_google_absl//absl/types:span",
        "//tensorflow/lite:kernel_api",
        "//tensorflow/lite:minimal_logging",
        "//tensorflow/lite/c:c_api_internal",
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
    }),
)

objc_library(
    name = "metal_delegate",
    srcs = ["metal_delegate.mm"],
    hdrs = ["metal_delegate.h"],
    copts = ["-std=c++11"],
    sdk_frameworks = ["Metal"],
    deps = [
        "//tensorflow/lite:kernel_api",
        "//tensorflow/lite:minimal_logging",
        "//tensorflow/lite/c:c_api_internal",
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

# build -c opt --config android_arm64 --copt -Os --copt -DTFLITE_GPU_BINARY_RELEASE --copt -fvisibility=hidden --linkopt -s --strip always :libtflite_gpu_gl.so
cc_binary(
    name = "libtflite_gpu_gl.so",
    linkopts = select({
        "//tensorflow:android": [
            "-lEGL",
            "-lGLESv3",
        ],
        "//conditions:default": [],
    }),
    linkshared = 1,
    linkstatic = 1,
    tags = [
        "nobuilder",
        "notap",
    ],
    deps = [":gl_delegate"],
)

# build -c opt --config ios_arm64 --copt -Os --copt -DTFLITE_GPU_BINARY_RELEASE --copt -fvisibility=hidden --linkopt -s --strip always :libtflite_gpu_metal.so
cc_binary(
    name = "libtflite_gpu_metal.so",
    linkshared = 1,
    linkstatic = 1,
    tags = [
        "nobuilder",
        "notap",
    ],
    deps = [":metal_delegate"],
)
