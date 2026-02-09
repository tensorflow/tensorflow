# Memorymappable, WASM compilable, implementation of the encoder.
#

load("@flatbuffers//:build_defs.bzl", "flatbuffer_cc_library")
load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")
load("//tensorflow_text:tftext.bzl", "tf_cc_library", "tflite_cc_library")

licenses(["notice"])

# Visibility rules
package(default_visibility = ["//visibility:public"])

filegroup(
    name = "testdata",
    srcs = [
        "//tensorflow_text:python/ops/test_data/fast_sentencepiece.model",
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
)

flatbuffer_cc_library(
    name = "encoder_config",
    srcs = [
        "encoder_config.fbs",
    ],
    includes = [":config_fbs"],
)

flatbuffer_cc_library(
    name = "decoder_config",
    srcs = [
        "decoder_config.fbs",
    ],
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
    deps = [
        ":config",
        ":utils",
        "@darts_clone",
    ],
)

cc_test(
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
        ":config",
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
        ":config",
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
        "config",
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
        # tf:lib tensorflow dep,
        # tf:framework tensorflow dep,
    ],
    deps = [
        ":optimized_encoder",
        ":sentencepiece_tokenizer_h",
    ],
)

tf_cc_library(
    name = "sentencepiece_detokenizer_kernel",
    srcs = ["sentencepiece_detokenizer_kernel.cc"],
    tf_deps = [
        # tf:lib tensorflow dep,
        # tf:framework tensorflow dep,
    ],
    deps = [
        ":optimized_decoder",
        ":sentencepiece_detokenizer_h",
        # tf/protobuf:error_codes_proto_impl_cc tensorflow dep,
    ],
)

tflite_cc_library(
    name = "sentencepiece_tokenizer_tflite",
    srcs = ["sentencepiece_tokenizer_tflite.cc"],
    deps =
        [
            ":optimized_encoder",
            ":sentencepiece_tokenizer_h",
            "@flatbuffers",
            # lite:framework tensorflow dep,
            # lite:string_util tensorflow dep,
            # lite/c:common tensorflow dep,
            # lite/kernels:kernel_util tensorflow dep,
            # lite/kernels/internal:tensor tensorflow dep,
        ],
)

tflite_cc_library(
    name = "sentencepiece_detokenizer_tflite",
    srcs = ["sentencepiece_detokenizer_tflite.cc"],
    deps =
        [
            ":optimized_decoder",
            ":sentencepiece_detokenizer_h",
            "@flatbuffers",
            # lite:framework tensorflow dep,
            # lite:string_util tensorflow dep,
            # lite/c:common tensorflow dep,
            # lite/kernels:kernel_util tensorflow dep,
            # lite/kernels/internal:tensor tensorflow dep,
        ],
)

cc_test(
    name = "optimized_encoder_test",
    srcs = [
        "optimized_encoder_test.cc",
    ],
    data = [
        ":testdata",
    ],
    deps = [
        ":double_array_trie_builder",
        ":encoder_config",
        ":model_converter",
        ":optimized_encoder",
        "//file/base:path",
        "//file/localfile",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_sentencepiece//:sentencepiece_cc_proto",
        "@com_google_sentencepiece//:sentencepiece_processor",
        # tf:lib tensorflow dep,
        # lite/kernels:test_util tensorflow dep,
    ],
)

cc_test(
    name = "optimized_decoder_test",
    srcs = [
        "optimized_decoder_test.cc",
    ],
    data = [
        ":testdata",
    ],
    deps = [
        ":model_converter",
        ":optimized_decoder",
        "//file/base:path",
        "//file/localfile",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_sentencepiece//:sentencepiece_cc_proto",
        "@com_google_sentencepiece//:sentencepiece_processor",
        # tf:lib tensorflow dep,
        # lite/kernels:test_util tensorflow dep,
    ],
)

tflite_cc_library(
    name = "py_tflite_registerer",
    srcs = ["py_tflite_registerer.cc"],
    hdrs = ["py_tflite_registerer.h"],
    deps = [
        ":sentencepiece_detokenizer_tflite",
        ":sentencepiece_tokenizer_tflite",
        # lite:framework tensorflow dep,
        # lite/kernels:builtin_ops tensorflow dep,
    ],
    alwayslink = 1,
)

config_setting(
    name = "armeabi_v7a_and_fastbuild",
    constraint_values = ["//third_party/bazel_platforms/cpu:armv7"],
    values = {
        "compilation_mode": "fastbuild",
    },
    visibility = ["//visibility:public"],
)

config_setting(
    name = "armeabi_v7a_and_dbg",
    constraint_values = ["//third_party/bazel_platforms/cpu:armv7"],
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
    values = {
        "apple_platform_type": "macos",
        "cpu": "darwin",
    },
    visibility = ["//visibility:public"],
)

config_setting(
    name = "macos_x86_64",
    values = {
        "apple_platform_type": "macos",
        "cpu": "darwin_x86_64",
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
    values = {
        "crosstool_top": "@bazel_tools//tools/cpp:toolchain",
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
