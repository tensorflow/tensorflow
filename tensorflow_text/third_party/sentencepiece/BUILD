# [internal] load cc_proto_library.bzl

package(
    default_visibility = ["//visibility:public"],
    features = [
        "layering_check",
        "parse_headers",
    ],
)

licenses(["notice"])  # Apache 2, BSD, MIT

proto_library(
    name = "sentencepiece_proto",
    srcs = ["src/sentencepiece.proto"],
)

cc_proto_library(
    name = "sentencepiece_cc_proto",
    deps = [":sentencepiece_proto"],
)

proto_library(
    name = "sentencepiece_model_proto",
    srcs = ["src/sentencepiece_model.proto"],
)

cc_proto_library(
    name = "sentencepiece_model_cc_proto",
    deps = [":sentencepiece_model_proto"],
)

genrule(
    name = "config_h",
    srcs = ["config.h.in"],
    outs = ["config.h"],
    cmd = "cp $< $@",
)

cc_library(
    name = "common",
    hdrs = [
        "config.h",
        "src/common.h",
    ],
    deps = [
        "@com_google_absl//absl/base",
    ],
)

cc_library(
    name = "sentencepiece_processor",
    srcs = [
        "src/bpe_model.cc",
        "src/char_model.cc",
        "src/error.cc",
        "src/filesystem.cc",
        "src/model_factory.cc",
        "src/model_interface.cc",
        "src/normalizer.cc",
        "src/sentencepiece_processor.cc",
        "src/unigram_model.cc",
        "src/util.cc",
        "src/word_model.cc",
    ],
    hdrs = [
        "src/bpe_model.h",
        "src/char_model.h",
        "src/filesystem.h",
        "src/freelist.h",
        "src/model_factory.h",
        "src/model_interface.h",
        "src/normalizer.h",
        "src/sentencepiece_processor.h",
        "src/trainer_interface.h",
        "src/unigram_model.h",
        "src/util.h",
        "src/word_model.h",
    ],
    defines = ["_USE_TF_STRING_VIEW"],
    includes = [
        ".",
        "src",
    ],
    linkstatic = 1,
    deps =
        [
            ":common",
            ":sentencepiece_cc_proto",
            ":sentencepiece_model_cc_proto",
            "@com_google_absl//absl/container:flat_hash_map",
            "@com_google_absl//absl/container:flat_hash_set",
            "@com_google_absl//absl/memory",
            "@com_google_absl//absl/strings",
            "@com_google_absl//absl/strings:str_format",
            "@darts_clone",
        ],
)
