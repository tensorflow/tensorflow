load("@bazel_skylib//rules:build_test.bzl", "build_test")
load(
    "//tensorflow:tensorflow.bzl",
    "if_android",
    "if_cuda_or_rocm",
    "tf_cc_test",
    "tf_cc_tests",
    "tf_copts",
)
load("//tensorflow:tensorflow.default.bzl", "filegroup", "tf_cuda_cc_test", "tf_disable_ptxas_warning_flags", "tf_kernel_library")
load(
    "//tensorflow/core/platform:build_config_root.bzl",
    "tf_cuda_tests_tags",
)
load("//tensorflow/core/platform:rules_cc.bzl", "cc_library")

# TODO(rmlarsen): Remove ASAP.
package_group(
    name = "friends",
    packages = [
        "//tensorflow/...",
        "//tensorflow_text/...",
        "//waymo/onboard/ml/...",
    ],
)

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [
        ":friends",
        "//tensorflow:__subpackages__",
        "//tensorflow:internal",
    ],
    licenses = ["notice"],
)

# Export a few files for use on Android.
exports_files([
    "adjust_contrast_op.cc",
    "adjust_contrast_op.h",
    "adjust_hue_op.cc",
    "adjust_hue_op.h",
    "adjust_saturation_op.cc",
    "adjust_saturation_op.h",
    "crop_and_resize_op.cc",
    "crop_and_resize_op.h",
    "colorspace_op.cc",
    "colorspace_op.h",
    "extract_image_patches_op.cc",
    "extract_image_patches_op.h",
    "image_ops.h",
    "image_ops.cc",
    "mirror_pad_op.cc",
    "mirror_pad_op.h",
    "mirror_pad_op_cpu_impl.h",
    "mirror_pad_op_cpu_impl_1.cc",
    "mirror_pad_op_cpu_impl_2.cc",
    "mirror_pad_op_cpu_impl_3.cc",
    "mirror_pad_op_cpu_impl_4.cc",
    "mirror_pad_op_cpu_impl_5.cc",
    "non_max_suppression_op.cc",
    "non_max_suppression_op.h",
    "resize_bicubic_op.cc",
    "resize_bilinear_op.cc",
    "resize_bilinear_op.h",
    "resize_nearest_neighbor_op.cc",
    "resize_nearest_neighbor_op.h",
    "sample_distorted_bounding_box_op.cc",
    "decode_image_op.cc",
    "encode_jpeg_op.cc",
    "encode_png_op.cc",
])

# Private support libraries ---------------------------------------------------
cc_library(
    name = "sampling_kernels",
    srcs = ["sampling_kernels.cc"],
    hdrs = ["sampling_kernels.h"],
    visibility = ["//visibility:private"],
    deps = ["//tensorflow/core:lib"],
)

tf_cc_test(
    name = "sampling_kernels_test",
    srcs = ["sampling_kernels_test.cc"],
    deps = [
        ":sampling_kernels",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
        "@com_google_absl//absl/strings",
    ],
)

# Public support libraries ----------------------------------------------------<
cc_library(
    name = "image",
    visibility = ["//visibility:public"],
    deps = [
        ":adjust_contrast_op",
        ":adjust_hue_op",
        ":adjust_saturation_op",
        ":attention_ops",
        ":colorspace_op",
        ":crop_and_resize_op",
        ":decode_image_op",
        ":draw_bounding_box_op",
        ":encode_jpeg_op",
        ":encode_png_op",
        ":extract_image_patches_op",
        ":extract_jpeg_shape_op",
        ":extract_volume_patches_op",
        ":generate_box_proposals_op",
        ":image_ops",
        ":mirror_pad_op",
        ":non_max_suppression_op",
        ":random_crop_op",
        ":resize_area_op",
        ":resize_bicubic_op",
        ":resize_bilinear_op",
        ":resize_nearest_neighbor_op",
        ":sample_distorted_bounding_box_op",
        ":scale_and_translate_op",
    ],
)

IMAGE_DEPS = [
    "@eigen_archive//:eigen3",
    "//tensorflow/core:framework",
    "//tensorflow/core:gif_internal",
    "//tensorflow/core:jpeg_internal",
    "//tensorflow/core:lib",
    "//tensorflow/core:lib_internal",
    "//tensorflow/core/lib/png:png_io",
    "//tensorflow/core/lib/webp:webp_io",
    "//tensorflow/core:protos_all_cc",
    "//tensorflow/core/framework:bounds_check",
    "//tensorflow/core/kernels:eigen_helpers",
    "//tensorflow/core/util/tensor_bundle",
    "//tensorflow/core/util:image_resizer_state",
]

IMAGE_TEST_DEPS = [
    "//tensorflow/core/kernels:ops_testutil",
    "//tensorflow/core/kernels:ops_util",
    "//tensorflow/core:core_cpu",
    "//tensorflow/core:framework",
    "//tensorflow/core:lib",
    "//tensorflow/core:lib_internal",
    "//tensorflow/core:protos_all_cc",
    "//tensorflow/core:test",
    "//tensorflow/core:test_main",
    "//tensorflow/core:testlib",
]

tf_kernel_library(
    name = "adjust_contrast_op",
    prefix = "adjust_contrast_op",
    deps = IMAGE_DEPS + ["//tensorflow/core/util:determinism_for_kernels"],
)

cc_library(
    name = "adjust_hsv_gpu_lib",
    hdrs = ["adjust_hsv_gpu.cu.h"],
    deps = ["//tensorflow/core:framework"],
)

tf_kernel_library(
    name = "adjust_hue_op",
    prefix = "adjust_hue_op",
    deps = IMAGE_DEPS + [":adjust_hsv_gpu_lib"],
)

tf_kernel_library(
    name = "adjust_saturation_op",
    prefix = "adjust_saturation_op",
    deps = IMAGE_DEPS + [":adjust_hsv_gpu_lib"],
)

tf_kernel_library(
    name = "attention_ops",
    prefix = "attention_ops",
    deps = IMAGE_DEPS,
)

tf_kernel_library(
    name = "colorspace_op",
    gpu_copts = tf_disable_ptxas_warning_flags(),
    prefix = "colorspace_op",
    deps = IMAGE_DEPS,
)

tf_kernel_library(
    name = "crop_and_resize_op",
    prefix = "crop_and_resize_op",
    deps = IMAGE_DEPS + [
        "//tensorflow/core:framework_internal",
        "//tensorflow/core/util:determinism_for_kernels",
    ] + if_cuda_or_rocm([
        "//tensorflow/core/platform:stream_executor",
    ]),
)

tf_kernel_library(
    name = "decode_image_op",
    prefix = "decode_image_op",
    deps = IMAGE_DEPS + [
        "@com_google_absl//absl/strings",
        "@local_xla//xla/tsl/util:byte_swap_array",
    ],
)

tf_kernel_library(
    name = "draw_bounding_box_op",
    prefix = "draw_bounding_box_op",
    deps = IMAGE_DEPS,
)

tf_kernel_library(
    name = "encode_jpeg_op",
    prefix = "encode_jpeg_op",
    deps = IMAGE_DEPS,
)

tf_kernel_library(
    name = "encode_png_op",
    prefix = "encode_png_op",
    deps = IMAGE_DEPS + [
        "//tensorflow/core/util:overflow",
        "@com_google_absl//absl/strings",
        "@local_tsl//tsl/platform:mutex",
    ],
)

tf_kernel_library(
    name = "extract_jpeg_shape_op",
    prefix = "extract_jpeg_shape_op",
    deps = IMAGE_DEPS,
)

tf_kernel_library(
    name = "extract_image_patches_op",
    prefix = "extract_image_patches_op",
    deps = [
        "//tensorflow/core/kernels:ops_util",
    ] + IMAGE_DEPS,
)

tf_kernel_library(
    name = "extract_volume_patches_op",
    prefix = "extract_volume_patches_op",
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core/kernels:eigen_helpers",
        "//tensorflow/core/kernels:ops_util",
        "@eigen_archive//:eigen3",
    ],
)

tf_kernel_library(
    name = "generate_box_proposals_op",
    features = ["-layering_check"],
    gpu_srcs = ["generate_box_proposals_op.cu.cc"],
    deps = ["//tensorflow/core/kernels:gpu_prim_hdrs"] + if_cuda_or_rocm([
        ":non_max_suppression_op_gpu",
    ]),
)

tf_kernel_library(
    name = "non_max_suppression_op",
    features = ["-layering_check"],
    prefix = "non_max_suppression_op",
    deps = IMAGE_DEPS + ["//tensorflow/core/kernels:gpu_prim_hdrs"] + if_cuda_or_rocm([
        "@com_google_absl//absl/strings",
    ]),
)

tf_kernel_library(
    name = "scale_and_translate_op",
    prefix = "scale_and_translate_op",
    deps = IMAGE_DEPS + [":sampling_kernels"],
)

tf_kernel_library(
    name = "random_crop_op",
    prefix = "random_crop_op",
    deps = IMAGE_DEPS,
)

tf_kernel_library(
    name = "resize_area_op",
    prefix = "resize_area_op",
    deps = IMAGE_DEPS,
)

tf_kernel_library(
    name = "resize_bicubic_op",
    prefix = "resize_bicubic_op",
    deps = IMAGE_DEPS,
)

tf_kernel_library(
    name = "resize_bilinear_op",
    prefix = "resize_bilinear_op",
    deps = IMAGE_DEPS + [
        "//tensorflow/core/kernels:cast_op",
        "//tensorflow/core/util:determinism_for_kernels",
    ],
)

tf_kernel_library(
    name = "resize_nearest_neighbor_op",
    prefix = "resize_nearest_neighbor_op",
    deps = IMAGE_DEPS + ["//tensorflow/core/util:determinism_for_kernels"],
)

tf_kernel_library(
    name = "sample_distorted_bounding_box_op",
    prefix = "sample_distorted_bounding_box_op",
    deps = IMAGE_DEPS + [
        "//tensorflow/core/kernels:stateless_random_ops",
        "@com_google_absl//absl/log:check",
    ],
)

tf_kernel_library(
    name = "image_ops",
    prefix = "image_ops",
    deps = IMAGE_DEPS,
)

tf_kernel_library(
    name = "mirror_pad_op",
    prefix = "mirror_pad_op",
    deps = IMAGE_DEPS,
)

# Tests ------------------------

tf_cc_tests(
    name = "bonus_tests",
    srcs = [
        "adjust_contrast_op_test.cc",
        "colorspace_op_test.cc",
        "crop_and_resize_op_test.cc",
        "mirror_pad_op_test.cc",
        "non_max_suppression_op_test.cc",
        "resize_area_op_test.cc",
        "resize_bicubic_op_test.cc",
        "scale_and_translate_op_test.cc",
    ],
    linkopts = select({
        "//tensorflow:macos": ["-headerpad_max_install_names"],
        "//conditions:default": [],
    }),
    deps = [
        ":image",
        ":mirror_pad_op",
        ":sampling_kernels",
    ] + IMAGE_TEST_DEPS,
)

tf_cc_test(
    name = "non_max_suppression_op_benchmark_test",
    srcs = ["non_max_suppression_op_benchmark_test.cc"],
    deps = [
        ":image",
    ] + IMAGE_TEST_DEPS,
)

tf_cuda_cc_test(
    name = "resize_ops_test",
    srcs = [
        "resize_bilinear_op_test.cc",
        "resize_nearest_neighbor_op_test.cc",
    ],
    tags = [
        "no_cuda_asan",  # TODO(b/171334997): re-enable
        "no_cuda_on_cpu_tap",
    ],
    deps = [
        ":image",
        ":sampling_kernels",
    ] + IMAGE_TEST_DEPS,
)

tf_cuda_cc_test(
    name = "adjust_contrast_op_benchmark_test",
    srcs = ["adjust_contrast_op_benchmark_test.cc"],
    deps = [
        ":image",
    ] + IMAGE_TEST_DEPS,
)

tf_cuda_cc_test(
    name = "crop_and_resize_op_benchmark_test",
    srcs = ["crop_and_resize_op_benchmark_test.cc"],
    deps = [
        ":image",
    ] + IMAGE_TEST_DEPS,
)

tf_cuda_cc_test(
    name = "mirror_pad_op_benchmark_test",
    srcs = ["mirror_pad_op_benchmark_test.cc"],
    deps = [
        ":mirror_pad_op",
    ] + IMAGE_TEST_DEPS,
)

tf_cc_test(
    name = "scale_and_translate_op_benchmark_test",
    srcs = ["scale_and_translate_op_benchmark_test.cc"],
    deps = [
        ":scale_and_translate_op",
        "@com_google_benchmark//:benchmark",
        "@local_tsl//tsl/platform:test_benchmark",
        "@local_xla//xla/tsl/platform:test_main",
    ] + IMAGE_TEST_DEPS,
)

tf_cuda_cc_test(
    name = "non_max_suppression_op_gpu_test",
    srcs = ["non_max_suppression_op_gpu_test.cc"],
    tags = tf_cuda_tests_tags() + [
        "no_cuda_asan",  # TODO(b/171263349): re-enable.
        "no_cuda_on_cpu_tap",
    ],
    deps = [
        ":image",
        "@com_google_absl//absl/strings",
    ] + IMAGE_TEST_DEPS,
)

tf_cuda_cc_test(
    name = "resize_benchmark_test",
    srcs = ["resize_op_benchmark_test.cc"],
    deps = [
        ":image",
    ] + IMAGE_TEST_DEPS,
)

tf_cc_test(
    name = "encode_jpeg_op_test",
    size = "small",
    srcs = ["encode_jpeg_op_test.cc"],
    deps = [
        ":encode_jpeg_op",
    ] + IMAGE_TEST_DEPS,
)

cc_library(
    name = "android_tensorflow_image_op",
    srcs = if_android(["decode_image_op.cc"]),
    copts = tf_copts(),
    linkopts = ["-ldl"],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/core:portable_gif_internal",
        "//tensorflow/core:portable_jpeg_internal",
        "//tensorflow/core:portable_tensorflow_lib_lite",
        "//tensorflow/core/framework:bounds_check",
        "//tensorflow/core/framework:op_requires",
        "//tensorflow/core/framework:types_proto_cc",
        "//tensorflow/core/lib/core:status",
        "//tensorflow/core/lib/png:png_io",
        "//tensorflow/core/lib/webp:webp_io",
        "//tensorflow/core/platform:byte_order",
        "//tensorflow/core/platform:errors",
        "@com_google_absl//absl/strings",
        "@local_xla//xla/tsl/util:byte_swap_array",
    ],
    alwayslink = 1,
)

build_test(
    name = "android_tensorflow_image_op_build_test",
    targets = [":android_tensorflow_image_op"],
)

# A file group which contains all operators which are known to work on mobile.
filegroup(
    name = "portable_all_op_kernels",
    srcs = glob(
        [
            "*.cc",
            "*.h",
        ],
        exclude = [
            "*test.cc",
            "*test.h",
            "*_test_*",
            "decode_image_op.*",
            "encode_png_op.*",
            "encode_jpeg_op.*",
            "extract_jpeg_shape_op.*",
            "decode_jpeg_op.*",
            "decode_and_crop_jpeg_op.*",
            "decode_gif_op.*",
        ],
    ),
    visibility = ["//tensorflow:__subpackages__"],
)

filegroup(
    name = "portable_all_ops_textual_hdrs",
    srcs = ["mirror_pad_op.h"],
    visibility = ["//visibility:public"],
)
