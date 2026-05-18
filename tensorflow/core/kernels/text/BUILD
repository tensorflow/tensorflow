"""Kernels for tf.text ops."""

load("@com_google_protobuf//bazel:proto_library.bzl", "proto_library")
load("@flatbuffers//:build_defs.bzl", "flatbuffer_cc_library")

# [internal] proxy flatbuffers_library load stripped
load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:defs.bzl", "cc_proto_library")
load("@tensorflow_text//:tftext.bzl", "tf_cc_library", "tflite_cc_library")
load("//tensorflow:tensorflow.bzl", "tf_cc_test")

licenses(["notice"])

# If you update visibility, you must update the copybara visibility rules too.
package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "boise_offset_converter",
    srcs = ["boise_offset_converter.cc"],
    hdrs = ["boise_offset_converter.h"],
    deps = [
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
    ],
)

tf_cc_test(
    name = "boise_offset_converter_test",
    size = "small",
    srcs = ["boise_offset_converter_test.cc"],
    deps = [
        ":boise_offset_converter",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest_main",
    ],
)

tf_cc_library(
    name = "boise_offset_converter_kernel",
    srcs = ["boise_offset_converter_kernel.cc"],
    hdrs = ["boise_offset_converter_kernel.h"],
    tf_deps = [
        "//tensorflow/core:framework",
    ],
    deps = [
        ":boise_offset_converter_kernel_template",
        "//tensorflow/lite/kernels/shim:tf_op_shim",
    ],
)

tf_cc_library(
    name = "boise_offset_converter_kernel_template",
    hdrs = ["boise_offset_converter_kernel_template.h"],
    tf_deps = [
        "//tensorflow/core/platform:tstring",
    ],
    deps = [
        ":boise_offset_converter",
        "//tensorflow/lite/kernels/shim:op_kernel",
        "//tensorflow/lite/kernels/shim:shape",
        "//tensorflow/lite/kernels/shim:status_macros",
        "@com_google_absl//absl/status",
    ],
)

cc_library(
    name = "byte_splitter",
    srcs = ["byte_splitter.cc"],
    hdrs = ["byte_splitter.h"],
    deps = [
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
    ],
)

tf_cc_test(
    name = "byte_splitter_test",
    size = "small",
    srcs = ["byte_splitter_test.cc"],
    deps = [
        ":byte_splitter",
        "@com_google_googletest//:gtest_main",
    ],
)

tf_cc_library(
    name = "byte_splitter_kernel",
    srcs = ["byte_splitter_kernel.cc"],
    hdrs = ["byte_splitter_kernel.h"],
    tf_deps = [
        "//tensorflow/core:framework",
    ],
    deps = [
        ":byte_splitter_kernel_template",
        "//tensorflow/lite/kernels/shim:tf_op_shim",
    ],
)

tf_cc_library(
    name = "byte_splitter_kernel_template",
    hdrs = ["byte_splitter_kernel_template.h"],
    tf_deps = [
        "//tensorflow/core/platform:tstring",
    ],
    deps = [
        ":byte_splitter",
        "//tensorflow/lite/kernels/shim:op_kernel",
        "//tensorflow/lite/kernels/shim:shape",
        "//tensorflow/lite/kernels/shim:status_macros",
        "@com_google_absl//absl/status",
    ],
)

cc_library(
    name = "byte_splitter_tflite",
    srcs = ["byte_splitter_tflite.cc"],
    hdrs = ["byte_splitter_tflite.h"],
    deps = [
        ":byte_splitter_kernel_template",
        "//tensorflow/lite:mutable_op_resolver",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/kernels/shim:tflite_op_shim",
    ],
)

tf_cc_library(
    name = "constrained_sequence",
    srcs = ["constrained_sequence.cc"],
    hdrs = ["constrained_sequence.h"],
    tf_deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
    ],
)

tf_cc_library(
    name = "constrained_sequence_kernel",
    srcs = ["constrained_sequence_kernel.cc"],
    tf_deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
    ],
    deps = [
        ":constrained_sequence",
    ],
    alwayslink = 1,
)

cc_library(
    name = "ngrams_op_cc",
    srcs = ["ops/ngrams_op.cc"],
    deps = [
        ":ngrams_kernel",
        "//tensorflow/core:framework",
    ],
    alwayslink = True,
)

cc_library(
    name = "constrained_sequence_op_cc",
    srcs = ["ops/constrained_sequence_op.cc"],
    deps = ["//tensorflow/core:framework"],
    alwayslink = True,
)

cc_library(
    name = "text_similarity_metric_ops_cc",
    srcs = ["ops/rouge_l_op.cc"],
    deps = ["//tensorflow/core:framework"],
    alwayslink = True,
)

cc_library(
    name = "unicode_script_tokenizer_cc",
    srcs = ["ops/unicode_script_tokenize_op.cc"],
    deps = ["//tensorflow/core:framework"],
    alwayslink = True,
)

cc_library(
    name = "whitespace_tokenizer_cc",
    srcs = ["ops/whitespace_tokenize_op.cc"],
    deps = ["//tensorflow/core:framework"],
    alwayslink = True,
)

exports_files(["test_data/fast_sentencepiece.model"])

tf_cc_test(
    name = "constrained_sequence_kernel_input_validation_test",
    srcs = ["constrained_sequence_kernel_input_validation_test.cc"],
    linkstatic = 1,
    deps = [
        ":constrained_sequence_kernel",
        ":constrained_sequence_op_cc",
        ":text_kernels_test_util",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:testlib",
        "//tensorflow/core/kernels:ops_testutil",
        "@com_google_googletest//:gtest_main",
    ],
)

tf_cc_test(
    name = "exp_greedy_constrained_sequence_kernel_test",
    srcs = ["exp_greedy_constrained_sequence_kernel_test.cc"],
    linkstatic = 1,
    deps = [
        ":constrained_sequence_kernel",
        ":constrained_sequence_op_cc",
        ":text_kernels_test_util",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:testlib",
        "//tensorflow/core/kernels:ops_testutil",
        "@com_google_googletest//:gtest_main",
    ],
)

tf_cc_test(
    name = "exp_viterbi_constrained_sequence_kernel_test",
    srcs = ["exp_viterbi_constrained_sequence_kernel_test.cc"],
    linkstatic = 1,
    deps = [
        ":constrained_sequence_kernel",
        ":constrained_sequence_op_cc",
        ":text_kernels_test_util",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:testlib",
        "//tensorflow/core/kernels:ops_testutil",
        "@com_google_googletest//:gtest_main",
    ],
)

tf_cc_library(
    name = "fast_bert_normalizer",
    hdrs = ["fast_bert_normalizer.h"],
    deps = [
        ":darts_clone_trie_wrapper",
        ":fast_bert_normalizer_model",
        "//tensorflow/lite/kernels/shim:status_macros",
        "@com_google_absl//absl/strings",
        "@icu//:common",
        "@icu//:nfkc",  # buildcleaner: keep
    ],
)

flatbuffer_cc_library(
    name = "fast_bert_normalizer_model",
    srcs = [
        "fast_bert_normalizer_model.fbs",
    ],
    # copybara:uncomment compatible_with = ["//buildenv/target:non_prod"],
)

tf_cc_library(
    name = "fast_bert_normalizer_model_builder",
    srcs = ["fast_bert_normalizer_model_builder.cc"],
    hdrs = ["fast_bert_normalizer_model_builder.h"],
    tf_deps = [
        "//tensorflow/core:lib",
    ],
    deps = [
        ":darts_clone_trie_builder",
        ":fast_bert_normalizer",
        ":fast_bert_normalizer_model",
        "//tensorflow/lite/kernels/shim:status_macros",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_googlesource_code_re2//:re2",
        "@icu//:common",
        "@icu//:nfkc",  # buildcleaner: keep
    ],
)

tf_cc_test(
    name = "fast_bert_normalizer_test",
    size = "small",
    srcs = ["fast_bert_normalizer_test.cc"],
    linkstatic = 1,
    deps = [
        ":fast_bert_normalizer",
        ":fast_bert_normalizer_model_builder",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "fast_bert_normalizer_kernel_template",
    hdrs = ["fast_bert_normalizer_kernel_template.h"],
    deps = [
        ":fast_bert_normalizer",
        "//tensorflow/lite/kernels/shim:op_kernel",
        "//tensorflow/lite/kernels/shim:status_macros",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
    ],
)

tf_cc_library(
    name = "fast_bert_normalizer_tf_kernel",
    srcs = ["fast_bert_normalizer_tf_kernel.cc"],
    hdrs = ["fast_bert_normalizer_tf_kernel.h"],
    tf_deps = [
        "//tensorflow/core:framework",
    ],
    deps = [
        ":fast_bert_normalizer_kernel_template",
        "//tensorflow/lite/kernels/shim:tf_op_shim",
    ],
)

cc_library(
    name = "fast_bert_normalizer_tflite",
    srcs = ["fast_bert_normalizer_tflite.cc"],
    hdrs = ["fast_bert_normalizer_tflite.h"],
    deps = [
        ":fast_bert_normalizer_kernel_template",
        "//tensorflow/lite:mutable_op_resolver",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/kernels/shim:tflite_op_shim",
    ],
)

tf_cc_test(
    name = "log_greedy_constrained_sequence_kernel_test",
    srcs = ["log_greedy_constrained_sequence_kernel_test.cc"],
    linkstatic = 1,
    deps = [
        ":constrained_sequence_kernel",
        ":constrained_sequence_op_cc",
        ":text_kernels_test_util",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:testlib",
        "//tensorflow/core/kernels:ops_testutil",
        "@com_google_googletest//:gtest_main",
    ],
)

tf_cc_test(
    name = "log_viterbi_constrained_sequence_kernel_test",
    srcs = ["log_viterbi_constrained_sequence_kernel_test.cc"],
    linkstatic = 1,
    deps = [
        ":constrained_sequence_kernel",
        ":constrained_sequence_op_cc",
        ":text_kernels_test_util",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:testlib",
        "//tensorflow/core/kernels:ops_testutil",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "darts_clone_trie_builder",
    srcs = [
        "darts_clone_trie_builder.cc",
    ],
    hdrs = [
        "darts_clone_trie_builder.h",
    ],
    deps = [
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@darts_clone",
    ],
)

cc_library(
    name = "darts_clone_trie_wrapper",
    hdrs = [
        "darts_clone_trie_wrapper.h",
    ],
    deps = [
        "@com_google_absl//absl/status:statusor",
    ],
)

tf_cc_test(
    name = "darts_clone_trie_test",
    size = "small",
    srcs = ["darts_clone_trie_test.cc"],
    linkstatic = 1,
    deps = [
        ":darts_clone_trie_builder",
        ":darts_clone_trie_wrapper",
        "@com_google_absl//absl/status",
        "@com_google_googletest//:gtest_main",
        "@xla//xla/tsl/platform:statusor",
    ],
)

tf_cc_library(
    name = "disjoint_set_forest",
    hdrs = ["disjoint_set_forest.h"],
    tf_deps = [
        "//tensorflow/core:lib",
    ],
)

tf_cc_test(
    name = "disjoint_set_forest_test",
    size = "small",
    srcs = ["disjoint_set_forest_test.cc"],
    linkstatic = 1,
    deps = [
        ":disjoint_set_forest",
        "@com_google_googletest//:gtest_main",
    ],
)

tf_cc_library(
    name = "fast_wordpiece_tokenizer",
    srcs = ["fast_wordpiece_tokenizer.cc"],
    hdrs = [
        "fast_wordpiece_tokenizer.h",
    ],
    deps = [
        ":darts_clone_trie_wrapper",
        ":fast_wordpiece_tokenizer_model",
        ":fast_wordpiece_tokenizer_utils",
        "//tensorflow/lite/kernels/shim:status_macros",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@icu//:nfkc",
    ],
)

tf_cc_test(
    name = "fast_wordpiece_tokenizer_test",
    srcs = ["fast_wordpiece_tokenizer_test.cc"],
    data = [
        "test_data/fast_wordpiece_tokenizer_model.fb",
        "test_data/fast_wordpiece_tokenizer_model_ver_15_1.fb",
        "test_data/fast_wordpiece_tokenizer_model_ver_16_0.fb",
    ],
    linkstatic = 1,
    deps = [
        ":fast_wordpiece_tokenizer",
        ":fast_wordpiece_tokenizer_model_builder",
        "//tensorflow/core:lib",
        "//tensorflow/core/platform:resource_loader",
        "@com_google_googletest//:gtest_main",
        "@icu//:headers",
        "@xla//xla/tsl/platform:statusor",
    ],
)

flatbuffer_cc_library(
    name = "fast_wordpiece_tokenizer_model",
    srcs = [
        "fast_wordpiece_tokenizer_model.fbs",
    ],
    # copybara:uncomment compatible_with = ["//buildenv/target:non_prod"],
)

flatbuffer_cc_library(
    name = "phrase_tokenizer_model",
    srcs = [
        "phrase_tokenizer_model.fbs",
    ],
    # copybara:uncomment compatible_with = ["//buildenv/target:non_prod"],
)

tf_cc_library(
    name = "fast_wordpiece_tokenizer_model_builder",
    srcs = ["fast_wordpiece_tokenizer_model_builder.cc"],
    hdrs = [
        "fast_wordpiece_tokenizer_model_builder.h",
    ],
    deps = [
        ":darts_clone_trie_builder",
        ":darts_clone_trie_wrapper",
        ":fast_wordpiece_tokenizer_model",
        ":fast_wordpiece_tokenizer_utils",
        ":sentence_fragmenter_v2",
        ":string_vocab",
        "//tensorflow/lite/kernels/shim:status_macros",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@cppitertools",
        "@icu//:nfkc",
    ],
)

tf_cc_library(
    name = "phrase_tokenizer_model_builder",
    srcs = ["phrase_tokenizer_model_builder.cc"],
    hdrs = [
        "phrase_tokenizer_model_builder.h",
    ],
    deps = [
        ":phrase_tokenizer_model",
        ":string_vocab",
        ":whitespace_tokenizer_config_builder",
        "//tensorflow/core/kernels/text/sentencepiece:double_array_trie_builder",
        "//tensorflow/lite/kernels/shim:status_macros",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
    ],
)

tf_cc_library(
    name = "fast_wordpiece_tokenizer_kernel",
    srcs = ["fast_wordpiece_tokenizer_kernel.cc"],
    hdrs = ["fast_wordpiece_tokenizer_kernel.h"],
    tf_deps = [
        "//tensorflow/core:framework",
    ],
    deps = [
        ":fast_wordpiece_tokenizer_kernel_template",
        "//tensorflow/lite/kernels/shim:tf_op_shim",
    ],
)

cc_library(
    name = "fast_wordpiece_tokenizer_kernel_template",
    hdrs = ["fast_wordpiece_tokenizer_kernel_template.h"],
    deps = [
        ":fast_wordpiece_tokenizer",
        "//tensorflow/lite/kernels/shim:op_kernel",
        "//tensorflow/lite/kernels/shim:status_macros",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "fast_wordpiece_tokenizer_tflite",
    srcs = ["fast_wordpiece_tokenizer_tflite.cc"],
    hdrs = ["fast_wordpiece_tokenizer_tflite.h"],
    deps = [
        ":fast_wordpiece_tokenizer_kernel_template",
        "//tensorflow/lite:mutable_op_resolver",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/kernels/shim:tflite_op_shim",
    ],
)

cc_library(
    name = "fast_wordpiece_tokenizer_utils",
    hdrs = [
        "fast_wordpiece_tokenizer_utils.h",
    ],
    deps = [
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@icu//:nfkc",
    ],
)

tf_cc_test(
    name = "fast_wordpiece_tokenizer_utils_test",
    srcs = ["fast_wordpiece_tokenizer_utils_test.cc"],
    linkstatic = 1,
    deps = [
        ":fast_wordpiece_tokenizer_utils",
        "@com_google_googletest//:gtest_main",
    ],
)

tf_cc_library(
    name = "mst_op_kernels",
    srcs = ["mst_op_kernels.cc"],
    tf_deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
    ],
    deps = [
        ":mst_solver",
        "@com_google_absl//absl/strings",
    ],
)

tf_cc_library(
    name = "mst_solver",
    hdrs = ["mst_solver.h"],
    tf_deps = [
        "//tensorflow/core:lib",
    ],
    deps = [
        ":disjoint_set_forest",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
)

tf_cc_test(
    name = "mst_solver_test",
    size = "small",
    srcs = ["mst_solver_test.cc"],
    linkstatic = 1,
    deps = [
        ":mst_solver",
        "//tensorflow/core:test",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ],
)

tf_cc_test(
    name = "mst_solver_random_comparison_test",
    size = "small",
    timeout = "long",
    srcs = ["mst_solver_random_comparison_test.cc"],
    linkstatic = 1,
    tags = [
        "nofastbuild",  # exclude from non-opt TAP projects
        "optonly",  # exclude from non-opt TAP projects
    ],
    deps = [
        ":mst_solver",
        ":spanning_tree_iterator",
        "//tensorflow/core:lib",
        "@com_google_googletest//:gtest",  # google-only
        "@com_google_googletest//:gtest_main",
    ],
)

proto_library(
    name = "edit_changes_proto",
    srcs = ["edit_changes.proto"],
)

cc_proto_library(
    name = "edit_changes_cc_proto",
    deps = [":edit_changes_proto"],
)

tf_cc_library(
    name = "ngrams_kernel_template",
    hdrs = ["ngrams_kernel_template.h"],
    tf_deps = [
        "//tensorflow/core/platform:tstring",
    ],
    deps = [
        "//tensorflow/lite/kernels/shim:op_kernel",
        "//tensorflow/lite/kernels/shim:status_macros",
        "//tensorflow/lite/kernels/shim:tensor_view",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
    ],
)

tf_cc_library(
    name = "ngrams_kernel",
    srcs = ["ngrams_kernel.cc"],
    hdrs = ["ngrams_kernel.h"],
    tf_deps = [
        "//tensorflow/core:framework",
    ],
    deps = [
        ":ngrams_kernel_template",
        "//tensorflow/lite/kernels/shim:tf_op_shim",
    ],
)

tf_cc_test(
    name = "ngrams_kernel_test",
    srcs = ["ngrams_kernel_test.cc"],
    linkstatic = 1,
    deps = [
        ":ngrams_op_cc",
        "//tensorflow/core:framework",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core/framework:shape_inference_testutil",
        "//tensorflow/core/framework:tensor_testutil",
    ],
)

cc_library(
    name = "ngrams_tflite",
    srcs = ["ngrams_tflite.cc"],
    hdrs = ["ngrams_tflite.h"],
    deps = [
        ":ngrams_kernel_template",
        "//tensorflow/lite:mutable_op_resolver",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/kernels/shim:tflite_op_shim",
    ],
)

tf_cc_test(
    name = "ngrams_tflite_test",
    srcs = ["ngrams_tflite_test.cc"],
    # copybara:uncomment compatible_with = ["//buildenv/target:prod"],
    linkstatic = 1,
    deps = [
        ":ngrams_tflite",
        "//tensorflow/lite:string_util",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/kernels:test_util",
        "//tensorflow/lite/schema:schema_fbs",
        "@com_google_googletest//:gtest_main",
        "@flatbuffers",
    ],
)

tf_cc_library(
    name = "normalize_kernels",
    srcs = ["normalize_kernels.cc"],
    tf_deps = [
        "//tensorflow/core:framework",
    ],
    deps = [
        ":edit_changes_cc_proto",
        "@com_google_absl//absl/strings",
        "@icu//:nfkc",
    ],
)

tf_cc_test(
    name = "normalize_kernels_test",
    srcs = ["normalize_kernels_test.cc"],
    linkstatic = 1,
    deps = [
        ":normalize_kernels",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "ragged_tensor_to_tensor_tflite",
    srcs = ["ragged_tensor_to_tensor_tflite.cc"],
    hdrs = ["ragged_tensor_to_tensor_tflite.h"],
    deps = [
        "//tensorflow/core/util:ragged_to_dense_util_common",
        "//tensorflow/lite:framework",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/kernels:kernel_util",
        "//tensorflow/lite/kernels/internal:tensor",
        "//tensorflow/lite/kernels/internal:types",
        "@flatbuffers",
    ],
)

tf_cc_test(
    name = "ragged_tensor_to_tensor_tflite_test",
    srcs = ["ragged_tensor_to_tensor_tflite_test.cc"],
    # copybara:uncomment compatible_with = ["//buildenv/target:prod"],
    linkstatic = 1,
    deps = [
        ":ragged_tensor_to_tensor_tflite",  # buildcleaner: keep
        "//tensorflow/lite:framework",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/kernels:test_util",
        "//tensorflow/lite/kernels/internal:tensor",
        "//tensorflow/lite/schema:schema_fbs",
        "@com_google_googletest//:gtest_main",
        "@flatbuffers",
    ],
)

tf_cc_library(
    name = "regex_split",
    srcs = ["regex_split.cc"],
    hdrs = ["regex_split.h"],
    # copybara:uncomment compatible_with = ["//buildenv/target:non_prod"],
    deps = [
        "@com_google_absl//absl/strings",
        "@com_googlesource_code_re2//:re2",
    ],
)

tf_cc_library(
    name = "regex_split_kernels",
    srcs = ["regex_split_kernels.cc"],
    tf_deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
    ],
    deps = [
        ":regex_split",
        "@com_google_absl//absl/strings",
    ],
)

tf_cc_test(
    name = "regex_split_test",
    srcs = ["regex_split_test.cc"],
    linkstatic = 1,
    deps = [
        ":regex_split",
        "//tensorflow/core:lib",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
        "@com_googlesource_code_re2//:re2",
    ],
)

cc_library(
    name = "round_robin_trimmer",
    hdrs = ["round_robin_trimmer.h"],
    deps = [
        ":trimmer",
    ],
)

tf_cc_test(
    name = "round_robin_trimmer_test",
    size = "small",
    srcs = ["round_robin_trimmer_test.cc"],
    linkstatic = 1,
    deps = [
        ":round_robin_trimmer",
        "//tensorflow/core:test_main",
        "@com_google_googletest//:gtest",
    ],
)

tf_cc_library(
    name = "round_robin_trimmer_kernel",
    srcs = ["round_robin_trimmer_kernel.cc"],
    hdrs = ["round_robin_trimmer_kernel.h"],
    tf_deps = [
        "//tensorflow/core:framework",
    ],
    deps = [
        ":round_robin_trimmer_kernel_template",
        "//tensorflow/lite/kernels/shim:tf_op_shim",
    ],
)

tf_cc_library(
    name = "round_robin_trimmer_kernel_template",
    hdrs = ["round_robin_trimmer_kernel_template.h"],
    tf_deps = [
        "//tensorflow/core:framework",
    ],
    deps = [
        ":round_robin_trimmer",
        "//tensorflow/lite/kernels/shim:op_kernel",
        "//tensorflow/lite/kernels/shim:shape",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "round_robin_trimmer_tflite",
    srcs = ["round_robin_trimmer_tflite.cc"],
    hdrs = ["round_robin_trimmer_tflite.h"],
    deps = [
        ":round_robin_trimmer_kernel_template",
        "//tensorflow/lite:mutable_op_resolver",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/kernels/shim:op_kernel",
        "//tensorflow/lite/kernels/shim:tflite_op_shim",
        "//tensorflow/lite/kernels/shim:tflite_op_wrapper",
    ],
)

tf_cc_library(
    name = "rouge_l_kernel",
    srcs = ["rouge_l_kernel.cc"],
    tf_deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
    ],
    deps = ["@com_google_absl//absl/strings"],
    alwayslink = 1,
)

tf_cc_test(
    name = "rouge_l_kernel_test",
    size = "small",
    srcs = ["rouge_l_kernel_test.cc"],
    linkstatic = 1,
    deps = [
        ":rouge_l_kernel",
        ":text_similarity_metric_ops_cc",
        "//tensorflow/core:framework",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
        "//tensorflow/core/kernels:ops_testutil",
    ],
)

tf_cc_library(
    name = "sentence_breaking_kernels",
    srcs = ["sentence_breaking_kernels.cc"],
    tf_deps = [
        "//tensorflow/core:framework",
    ],
    deps = [
        ":sentence_breaking_utils",
        ":sentence_fragmenter",
        "@com_google_absl//absl/strings",
        "@icu//:common",
    ],
)

tf_cc_library(
    name = "sentence_breaking_utils",
    srcs = ["sentence_breaking_utils.cc"],
    hdrs = ["sentence_breaking_utils.h"],
    tf_deps = [
        "//tensorflow/core:lib",
    ],
    deps = [
        "@com_google_absl//absl/strings",
        "@icu//:common",
    ],
)

tf_cc_test(
    name = "sentence_breaking_utils_test",
    size = "small",
    srcs = ["sentence_breaking_utils_test.cc"],
    linkstatic = 1,
    deps = [
        ":sentence_breaking_utils",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
        "@icu//:common",
    ],
)

tf_cc_library(
    name = "sentence_fragmenter",
    srcs = ["sentence_fragmenter.cc"],
    hdrs = ["sentence_fragmenter.h"],
    tf_deps = [
        "//tensorflow/core:lib",
    ],
    deps = [
        ":sentence_breaking_utils",
    ],
)

tf_cc_library(
    name = "sentence_fragmenter_v2",
    srcs = ["sentence_fragmenter_v2.cc"],
    hdrs = ["sentence_fragmenter_v2.h"],
    tf_deps = [
        "//tensorflow/core:lib",
    ],
    deps = [
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@icu//:common",
    ],
)

tf_cc_test(
    name = "sentence_fragmenter_v2_test",
    srcs = ["sentence_fragmenter_v2_test.cc"],
    linkstatic = 1,
    deps = [
        ":sentence_fragmenter_v2",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
        "@icu//:common",
    ],
)

tf_cc_library(
    name = "sentence_fragmenter_v2_kernel",
    srcs = ["sentence_fragmenter_v2_kernel.cc"],
    hdrs = ["sentence_fragmenter_v2_kernel.h"],
    tf_deps = [
        "//tensorflow/core:framework",
    ],
    deps = [
        ":sentence_fragmenter_v2_kernel_template",
        "//tensorflow/lite/kernels/shim:tf_op_shim",
    ],
)

tf_cc_library(
    name = "sentence_fragmenter_v2_kernel_template",
    hdrs = ["sentence_fragmenter_v2_kernel_template.h"],
    tf_deps = [
        "//tensorflow/core:framework",
    ],
    deps = [
        ":sentence_fragmenter_v2",
        "//tensorflow/lite/kernels/shim:op_kernel",
        "//tensorflow/lite/kernels/shim:shape",
        "//tensorflow/lite/kernels/shim:status_macros",
        "@com_google_absl//absl/status",
    ],
)

cc_library(
    name = "sentence_fragmenter_v2_tflite",
    srcs = ["sentence_fragmenter_v2_tflite.cc"],
    hdrs = ["sentence_fragmenter_v2_tflite.h"],
    deps = [
        ":sentence_fragmenter_v2_kernel_template",
        "//tensorflow/lite:mutable_op_resolver",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/kernels/shim:tflite_op_shim",
    ],
)

tf_cc_library(
    name = "sentencepiece_kernels",
    srcs = ["sentencepiece_kernels.cc"],
    tf_deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_headers_lib",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:core_cpu_base",
    ],
    deps = [
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/meta:type_traits",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:span",
        "@com_google_sentencepiece//:sentencepiece_cc_proto",
        "@com_google_sentencepiece//:sentencepiece_model_cc_proto",
        "@com_google_sentencepiece//:sentencepiece_processor",
    ],
)

tf_cc_library(
    name = "spanning_tree_iterator",
    testonly = 1,
    srcs = ["spanning_tree_iterator.cc"],
    hdrs = ["spanning_tree_iterator.h"],
    tf_deps = [
        "//tensorflow/core:lib",
    ],
)

tf_cc_test(
    name = "spanning_tree_iterator_test",
    size = "small",
    srcs = ["spanning_tree_iterator_test.cc"],
    linkstatic = 1,
    deps = [
        ":spanning_tree_iterator",
        "//tensorflow/core:lib",
        "@com_google_googletest//:gtest_main",
    ],
)

tf_cc_library(
    name = "split_merge_tokenize_kernel",
    srcs = ["split_merge_tokenize_kernel.cc"],
    tf_deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
    ],
    deps = [
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@icu//:common",
    ],
)

cc_library(
    name = "text_kernels_test_util",
    testonly = 1,
    srcs = ["text_kernels_test_util.cc"],
    hdrs = ["text_kernels_test_util.h"],
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:testlib",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "tflite_ops",
    hdrs = [
        "byte_splitter_tflite.h",
        "fast_bert_normalizer_tflite.h",
        "fast_wordpiece_tokenizer_tflite.h",
        "ngrams_tflite.h",
        "ragged_tensor_to_tensor_tflite.h",
        "round_robin_trimmer_tflite.h",
        "sentence_fragmenter_v2_tflite.h",
        "utf8_binarize_tflite.h",
        "whitespace_tokenizer_tflite.h",
    ],
    deps = [
        ":byte_splitter_tflite",
        ":fast_bert_normalizer_tflite",
        ":fast_wordpiece_tokenizer_tflite",
        ":ngrams_tflite",
        ":ragged_tensor_to_tensor_tflite",
        ":round_robin_trimmer_tflite",
        ":sentence_fragmenter_v2_tflite",
        ":utf8_binarize_tflite",
        ":whitespace_tokenizer_tflite",
        "//tensorflow/lite:mutable_op_resolver",
        "//tensorflow/lite/c:common",
    ],
)

tf_cc_library(
    name = "tokenizer_from_logits_kernel",
    srcs = ["tokenizer_from_logits_kernel.cc"],
    tf_deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
    ],
    deps = [
        "@com_google_absl//absl/strings",
        "@icu//:common",
    ],
)

cc_library(
    name = "trimmer",
    hdrs = ["trimmer.h"],
    deps = [
        "@com_google_absl//absl/types:span",
    ],
)

tf_cc_library(
    name = "unicode_script_tokenize_kernel",
    srcs = ["unicode_script_tokenize_kernel.cc"],
    tf_deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
    ],
    deps = [
        "@icu//:common",
    ],
    alwayslink = 1,
)

tf_cc_test(
    name = "unicode_script_tokenize_kernel_test",
    srcs = ["unicode_script_tokenize_kernel_test.cc"],
    linkstatic = 1,
    deps = [
        ":text_kernels_test_util",
        ":unicode_script_tokenize_kernel",
        ":unicode_script_tokenizer_cc",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "//tensorflow/core:testlib",
        "//tensorflow/core/kernels:ops_testutil",
        "@com_google_googletest//:gtest_main",
    ],
)

tf_cc_library(
    name = "whitespace_tokenize_kernel",
    srcs = ["whitespace_tokenize_kernel.cc"],
    tf_deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
    ],
    deps = [
        "@icu//:common",
    ],
    alwayslink = 1,
)

tf_cc_test(
    name = "whitespace_tokenize_kernel_test",
    srcs = ["whitespace_tokenize_kernel_test.cc"],
    linkstatic = 1,
    deps = [
        ":text_kernels_test_util",
        ":whitespace_tokenize_kernel",
        ":whitespace_tokenizer_cc",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "//tensorflow/core:testlib",
        "//tensorflow/core/kernels:ops_testutil",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "whitespace_tokenizer",
    srcs = ["whitespace_tokenizer.cc"],
    hdrs = ["whitespace_tokenizer.h"],
    deps = [
        "@com_google_absl//absl/strings",
        "@icu//:common",
    ],
)

tf_cc_test(
    name = "whitespace_tokenizer_test",
    size = "small",
    srcs = ["whitespace_tokenizer_test.cc"],
    linkstatic = 1,
    deps = [
        ":whitespace_tokenizer",
        ":whitespace_tokenizer_config_builder",
        "//tensorflow/core:lib",
        "//tensorflow/core:test_main",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_googletest//:gtest",
    ],
)

tf_cc_library(
    name = "whitespace_tokenizer_kernel",
    srcs = ["whitespace_tokenizer_kernel.cc"],
    hdrs = ["whitespace_tokenizer_kernel.h"],
    tf_deps = [
        "//tensorflow/core:framework",
    ],
    deps = [
        ":whitespace_tokenizer_kernel_template",
        "//tensorflow/lite/kernels/shim:tf_op_shim",
    ],
)

tf_cc_library(
    name = "whitespace_tokenizer_kernel_template",
    hdrs = ["whitespace_tokenizer_kernel_template.h"],
    tf_deps = [
        "//tensorflow/core:framework",
    ],
    deps = [
        ":whitespace_tokenizer",
        "//tensorflow/lite/kernels/shim:op_kernel",
        "//tensorflow/lite/kernels/shim:shape",
        "//tensorflow/lite/kernels/shim:tensor_view",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
    ],
)

cc_library(
    name = "whitespace_tokenizer_tflite",
    srcs = ["whitespace_tokenizer_tflite.cc"],
    hdrs = ["whitespace_tokenizer_tflite.h"],
    deps = [
        ":whitespace_tokenizer_kernel_template",
        "//tensorflow/lite:mutable_op_resolver",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/kernels/shim:tflite_op_shim",
    ],
)

cc_library(
    name = "whitespace_tokenizer_config_builder",
    srcs = ["whitespace_tokenizer_config_builder.cc"],
    hdrs = ["whitespace_tokenizer_config_builder.h"],
    deps = [
        "@icu//:common",
    ],
)

tf_cc_test(
    name = "whitespace_tokenizer_config_builder_test",
    size = "small",
    srcs = ["whitespace_tokenizer_config_builder_test.cc"],
    linkstatic = 1,
    deps = [
        ":whitespace_tokenizer",
        ":whitespace_tokenizer_config_builder",
        "//tensorflow/core:lib",
        "//tensorflow/core:test_main",
        "@com_google_googletest//:gtest",
        "@icu//:common",
    ],
)

cc_library(
    name = "string_vocab",
    srcs = ["string_vocab.cc"],
    hdrs = ["string_vocab.h"],
    deps = [
        ":wordpiece_tokenizer",
        "@com_google_absl//absl/container:flat_hash_map",
    ],
)

cc_library(
    name = "phrase_tokenizer",
    srcs = ["phrase_tokenizer.cc"],
    hdrs = ["phrase_tokenizer.h"],
    deps = [
        ":phrase_tokenizer_model",
        ":string_vocab",
        ":whitespace_tokenizer",
        ":whitespace_tokenizer_config_builder",
        "//tensorflow/core/kernels/text/sentencepiece:double_array_trie",
        "//tensorflow/lite/kernels/shim:status_macros",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/random",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
    ],
)

tf_cc_test(
    name = "phrase_tokenizer_test",
    size = "small",
    srcs = ["phrase_tokenizer_test.cc"],
    data = [
        "test_data/phrase_tokenizer_model.fb",
    ],
    linkstatic = 1,
    deps = [
        ":phrase_tokenizer",
        "//tensorflow/core:lib",
        "//tensorflow/core:test_main",
        "//tensorflow/core/platform:resource_loader",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "phrase_tokenizer_kernel_template",
    hdrs = ["phrase_tokenizer_kernel_template.h"],
    deps = [
        ":phrase_tokenizer",
        "//tensorflow/lite/kernels/shim:op_kernel",
        "//tensorflow/lite/kernels/shim:status_macros",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
    ],
)

tf_cc_library(
    name = "phrase_tokenizer_kernel",
    srcs = ["phrase_tokenizer_kernel.cc"],
    hdrs = ["phrase_tokenizer_kernel.h"],
    tf_deps = [
        "//tensorflow/core:framework",
    ],
    deps = [
        ":phrase_tokenizer_kernel_template",
        "//tensorflow/lite/kernels/shim:tf_op_shim",
    ],
)

cc_library(
    name = "utf8_binarize",
    srcs = ["utf8_binarize.cc"],
    hdrs = ["utf8_binarize.h"],
    deps = [
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@icu//:common",
    ],
)

tf_cc_test(
    name = "utf8_binarize_test",
    srcs = ["utf8_binarize_test.cc"],
    linkopts = ["-ldl"],
    deps = [
        ":utf8_binarize",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
    ],
)

tf_cc_test(
    name = "wordpiece_kernel_test",
    srcs = [
        "ops/wordpiece_op.cc",
        "wordpiece_kernel_test.cc",
    ],
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core/framework:fake_input",
        "//tensorflow/core/framework:shape_inference",
        "//tensorflow/core/framework:shape_inference_testutil",
        "//tensorflow/core/framework:tensor",
        "//tensorflow/core/framework:tensor_shape",
        "//tensorflow/core/framework:tensor_testutil",
        "//tensorflow/core/kernels:ops_testutil",
    ],
)

cc_library(
    name = "utf8_binarize_kernel_template",
    hdrs = ["utf8_binarize_kernel_template.h"],
    deps = [
        ":utf8_binarize",
        "//tensorflow/core:portable_gif_internal",
        "//tensorflow/lite/kernels/shim:op_kernel",
        "//tensorflow/lite/kernels/shim:shape",
        "//tensorflow/lite/kernels/shim:status_macros",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
    ],
)

tf_cc_library(
    name = "wordpiece_kernel",
    srcs = ["wordpiece_kernel.cc"],
    tf_deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
    ],
    deps = [
        ":wordpiece_tokenizer",
        "@com_google_absl//absl/strings",
    ],
    alwayslink = 1,
)

cc_library(
    name = "utf8_binarize_kernel",
    srcs = ["utf8_binarize_kernel.cc"],
    hdrs = ["utf8_binarize_kernel.h"],
    deps = [
        ":utf8_binarize_kernel_template",
        "//tensorflow/core:framework",
        "//tensorflow/lite/kernels/shim:tf_op_shim",
    ],
)

cc_library(
    name = "wordpiece_tokenizer",
    srcs = ["wordpiece_tokenizer.cc"],
    hdrs = ["wordpiece_tokenizer.h"],
    deps = [
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@icu//:headers",
    ],
)

cc_library(
    name = "utf8_binarize_tflite",
    srcs = ["utf8_binarize_tflite.cc"],
    hdrs = ["utf8_binarize_tflite.h"],
    deps = [
        ":utf8_binarize_kernel_template",
        "//tensorflow/lite:mutable_op_resolver",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/kernels/shim:tflite_op_shim",
    ],
)

# [internal] flatbuffers_library target stripped

# [internal] flatbuffers_library target stripped

# [internal] flatbuffers_library target stripped
