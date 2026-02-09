# Code that exposes C++ libraries to Python via pybind11.

# Placeholder: load py_test
load("//tensorflow_text:tftext.bzl", "if_pywrap", "pybind_extension", "pywrap_binaries", "pywrap_library")

licenses(["notice"])

package(default_visibility = [
    "//nlp/sage/nlu/features/python:__pkg__",
    "//nlp/semantic_parsing/learning/neural/portable/tools/release:__pkg__",
    "//tensorflow_text:__subpackages__",
])

pybind_extension(
    name = "tflite_registrar",
    srcs = [
        "tflite_registrar.cc",
    ],
    deps = [
        # lite:framework tensorflow dep,
        # lite/c:common tensorflow dep,
        # lite/kernels:builtin_ops tensorflow dep,
        "//tensorflow_text/core/kernels:tflite_ops",
    ],
)

pybind_extension(
    name = "pywrap_fast_bert_normalizer_model_builder",
    srcs = ["pywrap_fast_bert_normalizer_model_builder.cc"],
    copts = ["-fexceptions"],
    features = ["-use_header_modules"],
    deps = [
        "//tensorflow_text/core/kernels:fast_bert_normalizer_model_builder",
    ],
)

py_test(
    name = "pywrap_fast_bert_normalizer_model_builder_test",
    srcs = ["pywrap_fast_bert_normalizer_model_builder_test.py"],
    data = [
        "//tensorflow_text:python/ops/test_data/fast_bert_normalizer_model.fb",
        "//tensorflow_text:python/ops/test_data/fast_bert_normalizer_model_lower_case_nfd_strip_accents.fb",
    ],
    tags = [
        "-sickbay",  # Depends on ICU version.
    ],
    deps = [
        ":pywrap_fast_bert_normalizer_model_builder",
        "@absl_py//absl/flags",
        "@release_or_nightly//:tensorflow_pkg",  # tensorflow package dep
    ] + if_pywrap([":pybinds_library"]),
)

pybind_extension(
    name = "pywrap_fast_wordpiece_tokenizer_model_builder",
    srcs = ["pywrap_fast_wordpiece_tokenizer_model_builder.cc"],
    copts = ["-fexceptions"],
    data = [
        "pywrap_fast_wordpiece_tokenizer_model_builder.pyi",
    ],
    features = ["-use_header_modules"],
    deps = [
        "//tensorflow_text/core/kernels:fast_wordpiece_tokenizer_model_builder",
    ],
)

py_test(
    name = "pywrap_fast_wordpiece_tokenizer_model_builder_test",
    srcs = ["pywrap_fast_wordpiece_tokenizer_model_builder_test.py"],
    data = [
        "//tensorflow_text:python/ops/test_data/fast_wordpiece_tokenizer_model.fb",
    ],
    deps = [
        ":pywrap_fast_wordpiece_tokenizer_model_builder",
        "@release_or_nightly//:tensorflow_pkg",  # tensorflow package dep
    ] + if_pywrap([":pybinds_library"]),
)

pybind_extension(
    name = "pywrap_phrase_tokenizer_model_builder",
    srcs = ["pywrap_phrase_tokenizer_model_builder.cc"],
    copts = ["-fexceptions"],
    features = ["-use_header_modules"],
    visibility = [
        "//knowledge/cerebra/nlu/models/smartv2/input:__pkg__",
        "//tensorflow_text:__subpackages__",
    ],
    deps = [
        "//tensorflow_text/core/kernels:phrase_tokenizer_model_builder",
    ],
)

py_test(
    name = "pywrap_phrase_tokenizer_model_builder_test",
    srcs = ["pywrap_phrase_tokenizer_model_builder_test.py"],
    data = [
        "//tensorflow_text:python/ops/test_data/phrase_tokenizer_model_test.fb",
    ],
    deps = [
        ":pywrap_phrase_tokenizer_model_builder",
        "@release_or_nightly//:tensorflow_pkg",  # tensorflow package dep
    ] + if_pywrap([":pybinds_library"]),
)

pybind_extension(
    name = "pywrap_model_converter",
    srcs = ["pywrap_model_converter.cc"],
    copts = ["-fexceptions"],
    features = ["-use_header_modules"],
    deps = [
        "//tensorflow_text/core/kernels/sentencepiece:model_converter",
    ],
)

pybind_extension(
    name = "pywrap_whitespace_tokenizer_config_builder",
    srcs = ["pywrap_whitespace_tokenizer_config_builder.cc"],
    copts = ["-fexceptions"],
    features = ["-use_header_modules"],
    deps = [
        "//tensorflow_text/core/kernels:whitespace_tokenizer_config_builder",
    ],
)

py_test(
    name = "pywrap_whitespace_tokenizer_config_builder_test",
    srcs = ["pywrap_whitespace_tokenizer_config_builder_test.py"],
    deps = [
        ":pywrap_whitespace_tokenizer_config_builder",
        "@release_or_nightly//:tensorflow_pkg",  # tensorflow package dep
    ] + if_pywrap([":pybinds_library"]),
)

pywrap_library(
    name = "pybinds_library",
    deps = [
        ":pywrap_fast_bert_normalizer_model_builder",
        ":pywrap_fast_wordpiece_tokenizer_model_builder",
        ":pywrap_model_converter",
        ":pywrap_phrase_tokenizer_model_builder",
        ":pywrap_whitespace_tokenizer_config_builder",
        ":tflite_registrar",
    ],
)

pywrap_binaries(
    name = "pybinds_binaries",
    dep = ":pybinds_library",
    visibility = ["//visibility:public"],
)
