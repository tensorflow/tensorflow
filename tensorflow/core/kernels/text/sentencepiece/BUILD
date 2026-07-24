# Memorymappable, WASM compilable, implementation of the encoder.
#

load("@bazel_skylib//:bzl_library.bzl", "bzl_library")
load("@flatbuffers//:build_defs.bzl", "flatbuffer_cc_library")

# [internal] proxy flatbuffers_library load stripped
load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@tensorflow_text//:tftext.bzl", "tf_cc_library", "tflite_cc_library")
load("//tensorflow:tensorflow.bzl", "tf_cc_test")

licenses(["notice"])

# If you update visibility, you must update the copybara visibility rules too.
package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    # copybara:uncomment default_compatible_with = ["//buildenv/target:non_prod"],
    default_visibility = [
        "//tensorflow/core/kernels/text:__subpackages__",
        "@tensorflow_text//:__subpackages__",
    ],
)

filegroup(
    name = "testdata",
    srcs = [
        "//tensorflow/core/kernels/text:test_data/fast_sentencepiece.model",
    ],
)

filegroup(
    name = "config_fbs",
    srcs = ["config.fbs"],
)

filegroup(
    name = "sp_headers",
    srcs = [
        "py_tflite_registerer.h",
    ],
    visibility = ["//visibility:public"],
)

flatbuffer_cc_library(
    name = "config",
    srcs = [
        "config.fbs",
    ],
    # copybara:uncomment compatible_with = ["//buildenv/target:non_prod"],
)

flatbuffer_cc_library(
    name = "encoder_config",
    srcs = [
        "encoder_config.fbs",
    ],
    # copybara:uncomment compatible_with = ["//buildenv/target:non_prod"],
    include_paths = ["."],
    includes = [":config_fbs"],
)

flatbuffer_cc_library(
    name = "decoder_config",
    srcs = [
        "decoder_config.fbs",
    ],
    # copybara:uncomment compatible_with = ["//buildenv/target:non_prod"],
    include_paths = ["."],
    includes = [":config_fbs"],
)

cc_library(
    name = "utils",
    srcs = [
    ],
    hdrs = [
        "utils.h",
    ],
)

cc_library(
    name = "double_array_trie",
    srcs = [
    ],
    hdrs = [
        "double_array_trie.h",
    ],
    deps = [
        ":config",
        ":utils",
    ],
)

cc_library(
    name = "double_array_trie_builder",
    srcs = [
        "double_array_trie_builder.cc",
    ],
    hdrs = [
        "double_array_trie_builder.h",
    ],
    visibility = [
        "//tensorflow/core/kernels/text:__subpackages__",
        "@tensorflow_text//:__subpackages__",
    ],
    deps = [
        ":config",
        ":utils",
        "@darts_clone",
    ],
)

tf_cc_test(
    name = "double_array_trie_test",
    srcs = [
        "double_array_trie_test.cc",
    ],
    deps = [
        ":double_array_trie",
        ":double_array_trie_builder",
        ":encoder_config",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "sentencepiece_constants",
    srcs = [],
    hdrs = ["sentencepiece_constants.h"],
)

cc_library(
    name = "model_converter",
    srcs = [
        "model_converter.cc",
    ],
    hdrs = [
        "model_converter.h",
    ],
    deps = [
        ":decoder_config",
        ":double_array_trie_builder",
        ":encoder_config",
        ":sentencepiece_constants",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_sentencepiece//:sentencepiece_model_cc_proto",
    ],
)

cc_library(
    name = "optimized_encoder",
    srcs = [
        "optimized_encoder.cc",
    ],
    hdrs = [
        "optimized_encoder.h",
    ],
    deps = [
        ":double_array_trie",
        ":encoder_config",
    ],
)

cc_library(
    name = "optimized_decoder",
    srcs = [
        "optimized_decoder.cc",
    ],
    hdrs = [
        "optimized_decoder.h",
    ],
    deps = [
        ":decoder_config",
        ":double_array_trie",
    ],
)

cc_library(
    name = "sentencepiece_tokenizer_h",
    hdrs = [
        "sentencepiece_tokenizer.h",
    ],
)

cc_library(
    name = "sentencepiece_detokenizer_h",
    hdrs = [
        "sentencepiece_detokenizer.h",
    ],
)

tf_cc_library(
    name = "sentencepiece_tokenizer_kernel",
    srcs = ["sentencepiece_tokenizer_kernel.cc"],
    tf_deps = [
        "//tensorflow/core:lib",
        "//tensorflow/core:framework",
    ],
    deps = [
        ":optimized_encoder",
        ":sentencepiece_tokenizer_h",
        "@com_google_absl//absl/status",
    ],
)

tf_cc_library(
    name = "sentencepiece_detokenizer_kernel",
    srcs = ["sentencepiece_detokenizer_kernel.cc"],
    tf_deps = [
        "//tensorflow/core:lib",
        "//tensorflow/core:framework",
    ],
    deps = [
        ":optimized_decoder",
        ":sentencepiece_detokenizer_h",
        "@com_google_absl//absl/status",
    ],
)

cc_library(
    name = "sentencepiece_tokenizer_tflite",
    srcs = ["sentencepiece_tokenizer_tflite.cc"],
    deps =
        [
            ":optimized_encoder",
            ":sentencepiece_tokenizer_h",
            "//tensorflow/lite:framework",
            "//tensorflow/lite:string_util",
            "//tensorflow/lite/c:common",
            "//tensorflow/lite/kernels:kernel_util",
            "//tensorflow/lite/kernels/internal:tensor",
            "@flatbuffers",
        ],
)

cc_library(
    name = "sentencepiece_detokenizer_tflite",
    srcs = ["sentencepiece_detokenizer_tflite.cc"],
    deps =
        [
            ":optimized_decoder",
            ":sentencepiece_detokenizer_h",
            "//tensorflow/lite:framework",
            "//tensorflow/lite:string_util",
            "//tensorflow/lite/c:common",
            "//tensorflow/lite/kernels:kernel_util",
            "//tensorflow/lite/kernels/internal:tensor",
            "@flatbuffers",
        ],
)

tf_cc_test(
    name = "optimized_encoder_test",
    srcs = [
        "optimized_encoder_test.cc",
    ],
    # copybara:uncomment compatible_with = ["//buildenv/target:prod"],
    data = [
        ":testdata",
    ],
    deps = [
        ":double_array_trie_builder",
        ":encoder_config",
        ":model_converter",
        ":optimized_encoder",
        "//tensorflow/core:lib",
        "//tensorflow/core/platform:resource_loader",
        "//tensorflow/lite/kernels:test_util",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_googletest//:gtest_main",
        "@com_google_sentencepiece//:sentencepiece_cc_proto",
        "@com_google_sentencepiece//:sentencepiece_processor",
    ],
)

tf_cc_test(
    name = "optimized_decoder_test",
    srcs = [
        "optimized_decoder_test.cc",
    ],
    # copybara:uncomment compatible_with = ["//buildenv/target:prod"],
    data = [
        ":testdata",
    ],
    deps = [
        ":model_converter",
        ":optimized_decoder",
        "//tensorflow/core:lib",
        "//tensorflow/core/platform:resource_loader",
        "//tensorflow/lite/kernels:test_util",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_googletest//:gtest_main",
        "@com_google_sentencepiece//:sentencepiece_cc_proto",
        "@com_google_sentencepiece//:sentencepiece_processor",
    ],
)

cc_library(
    name = "py_tflite_registerer",
    srcs = ["py_tflite_registerer.cc"],
    hdrs = ["py_tflite_registerer.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":sentencepiece_detokenizer_tflite",
        ":sentencepiece_tokenizer_tflite",
        "//tensorflow/lite:framework",
        "//tensorflow/lite/kernels:builtin_ops",
    ],
    alwayslink = 1,
)

config_setting(
    name = "armeabi_v7a_and_fastbuild",
    constraint_values = ["@platforms//cpu:armv7"],
    values = {
        "compilation_mode": "fastbuild",
    },
    visibility = ["//visibility:public"],
)

config_setting(
    name = "armeabi_v7a_and_dbg",
    constraint_values = ["@platforms//cpu:armv7"],
    values = {
        "compilation_mode": "dbg",
    },
    visibility = ["//visibility:public"],
)

config_setting(
    name = "android",
    values = {"crosstool_top": "//external:android/crosstool"},
    visibility = ["//visibility:public"],
)

config_setting(
    name = "macos_i386",
    constraint_values = [
        "@platforms//cpu:x86_32",
        "@platforms//os:macos",
    ],
    values = {
        "apple_platform_type": "macos",
    },
    visibility = ["//visibility:public"],
)

config_setting(
    name = "macos_x86_64",
    constraint_values = [
        "@platforms//cpu:x86_64",
        "@platforms//os:macos",
    ],
    values = {
        "apple_platform_type": "macos",
    },
    visibility = ["//visibility:public"],
)

alias(
    name = "macos",
    actual = select({
        ":macos_i386": ":macos_i386",
        ":macos_x86_64": ":macos_x86_64",
        "//conditions:default": ":macos_i386",  # Arbitrarily chosen from above.
    }),
    visibility = ["//visibility:public"],
)

config_setting(
    name = "ios",
    constraint_values = [
        "@platforms//os:ios",
    ],
    values = {
        "apple_platform_type": "ios",
    },
    visibility = ["//visibility:public"],
)

alias(
    name = "apple",
    actual = select({
        ":macos": ":macos",
        ":ios": ":ios",
        "//conditions:default": ":ios",  # Arbitrarily chosen from above.
    }),
    visibility = ["//visibility:public"],
)

bzl_library(
    name = "native_bzl",
    srcs = ["native.bzl"],
    # copybara:uncomment parse_tests = False,
    visibility = ["//visibility:private"],
    deps = [
        "//tensorflow:tensorflow_bzl",
        "@bazel_skylib//lib:selects",
    ],
)

# [internal] flatbuffers_library target stripped

# [internal] flatbuffers_library target stripped

# [internal] flatbuffers_library target stripped
