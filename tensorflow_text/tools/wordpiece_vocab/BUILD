# Placeholder: load py_library
# Placeholder: load py_test
load("//tensorflow_text:tftext.bzl", "if_pywrap")

licenses(["notice"])

package(
    default_visibility = [
        "//tensorflow_text:__subpackages__",
    ],
)

py_library(
    name = "wordpiece_vocab",
    srcs = ["__init__.py"],
    deps = [
        ":bert_vocab_from_dataset",
        ":wordpiece_tokenizer_learner_lib",
    ],
)

py_library(
    name = "wordpiece_tokenizer_learner_lib",
    srcs = ["wordpiece_tokenizer_learner_lib.py"],
    deps = [
        "@pypi//numpy:pkg",
    ],
)

py_test(
    name = "wordpiece_tokenizer_learner_test",
    srcs = ["wordpiece_tokenizer_learner_test.py"],
    deps = [
        ":wordpiece_tokenizer_learner_lib",
        "@pypi//numpy:pkg",
        # python/data/ops:dataset_ops tensorflow dep,
        "//tensorflow_text:ops",
    ] + if_pywrap([
        "//tensorflow_text/core/pybinds:pybinds_library",
    ]),
)

py_library(
    name = "bert_vocab_from_dataset",
    srcs = ["bert_vocab_from_dataset.py"],
    deps = [
        ":wordpiece_tokenizer_learner_lib",
        "//tensorflow_text:bert_tokenizer",
    ],
)

py_test(
    name = "bert_vocab_from_dataset_test",
    srcs = ["bert_vocab_from_dataset_test.py"],
    deps = [
        ":bert_vocab_from_dataset",
        ":wordpiece_tokenizer_learner_lib",
        "@absl_py//absl/testing:absltest",
        "@pypi//numpy:pkg",
        "@release_or_nightly//:tensorflow_pkg",  # tensorflow package dep
    ],
)
