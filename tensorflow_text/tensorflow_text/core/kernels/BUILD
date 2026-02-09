"""Kernels for tf.text ops."""

load("@flatbuffers//:build_defs.bzl", "flatbuffer_cc_library")
load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")

# Placeholder: load proto_library
load("//tensorflow_text:tftext.bzl", "tf_cc_library", "tflite_cc_library")
# [internal] load cc_proto_library.bzl

licenses(["notice"])

# Visibility rules
package(default_visibility = ["//visibility:public"])

exports_files(["LICENSE"])

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

cc_test(
    name = "boise_offset_converter_test",
    size = "small",
    srcs = ["boise_offset_converter_test.cc"],
    deps = [
        ":boise_offset_converter",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest_main",
    ],
)

tf_cc_library(
    name = "boise_offset_converter_kernel",
    srcs = ["boise_offset_converter_kernel.cc"],
    hdrs = ["boise_offset_converter_kernel.h"],
    tf_deps = [
        # tf:framework tensorflow dep,
    ],
    deps = [
        ":boise_offset_converter_kernel_template",
        # lite/kernels/shim:tf_op_shim tensorflow dep,
    ],
)

tf_cc_library(
    name = "boise_offset_converter_kernel_template",
    hdrs = ["boise_offset_converter_kernel_template.h"],
    tf_deps = [
        # tf/platform:tstring tensorflow dep,
    ],
    deps = [
        ":boise_offset_converter",
        "@com_google_absl//absl/status",
        # lite/kernels/shim:op_kernel tensorflow dep,
        # lite/kernels/shim:shape tensorflow dep,
        # lite/kernels/shim:status_macros tensorflow dep,
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

cc_test(
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
        # tf:framework tensorflow dep,
    ],
    deps = [
        ":byte_splitter_kernel_template",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        # lite/kernels/shim:tf_op_shim tensorflow dep,
    ],
)

tf_cc_library(
    name = "byte_splitter_kernel_template",
    hdrs = ["byte_splitter_kernel_template.h"],
    tf_deps = [
        # tf/platform:tstring tensorflow dep,
    ],
    deps = [
        ":byte_splitter",
        "@com_google_absl//absl/status",
        # lite/kernels/shim:op_kernel tensorflow dep,
        # lite/kernels/shim:shape tensorflow dep,
        # lite/kernels/shim:status_macros tensorflow dep,
    ],
)

tflite_cc_library(
    name = "byte_splitter_tflite",
    srcs = ["byte_splitter_tflite.cc"],
    hdrs = ["byte_splitter_tflite.h"],
    deps = [
        ":byte_splitter_kernel_template",
        # lite:mutable_op_resolver tensorflow dep,
        # lite/c:common tensorflow dep,
        # lite/kernels/shim:tflite_op_shim tensorflow dep,
    ],
)

tf_cc_library(
    name = "constrained_sequence",
    srcs = ["constrained_sequence.cc"],
    hdrs = ["constrained_sequence.h"],
    tf_deps = [
        # tf:framework tensorflow dep,
        # tf:lib tensorflow dep,
        # tf:protos_all_cc tensorflow dep,
    ],
)

tf_cc_library(
    name = "constrained_sequence_kernel",
    srcs = ["constrained_sequence_kernel.cc"],
    tf_deps = [
        # tf:framework tensorflow dep,
        # tf:lib tensorflow dep,
        # tf:protos_all_cc tensorflow dep,
    ],
    deps = [
        ":constrained_sequence",
        "@com_google_absl//absl/base:core_headers",
    ],
)

cc_test(
    name = "constrained_sequence_kernel_input_validation_test",
    srcs = ["constrained_sequence_kernel_input_validation_test.cc"],
    deps = [
        ":text_kernels_test_util",
        "@com_google_googletest//:gtest_main",
        # tf:framework tensorflow dep,
        # tf:lib tensorflow dep,
        # tf:protos_all_cc tensorflow dep,
        # tf:test tensorflow dep,
        # tf:testlib tensorflow dep,
        # tf/kernels:ops_testutil tensorflow dep,
        "//tensorflow_text:constrained_sequence_op_cc",
    ],
)

cc_test(
    name = "exp_greedy_constrained_sequence_kernel_test",
    srcs = ["exp_greedy_constrained_sequence_kernel_test.cc"],
    deps = [
        ":text_kernels_test_util",
        "@com_google_googletest//:gtest_main",
        # tf:framework tensorflow dep,
        # tf:lib tensorflow dep,
        # tf:protos_all_cc tensorflow dep,
        # tf:test tensorflow dep,
        # tf:testlib tensorflow dep,
        # tf/kernels:ops_testutil tensorflow dep,
        "//tensorflow_text:constrained_sequence_op_cc",
    ],
)

cc_test(
    name = "exp_viterbi_constrained_sequence_kernel_test",
    srcs = ["exp_viterbi_constrained_sequence_kernel_test.cc"],
    deps = [
        ":text_kernels_test_util",
        "@com_google_googletest//:gtest_main",
        # tf:framework tensorflow dep,
        # tf:lib tensorflow dep,
        # tf:protos_all_cc tensorflow dep,
        # tf:test tensorflow dep,
        # tf:testlib tensorflow dep,
        # tf/kernels:ops_testutil tensorflow dep,
        "//tensorflow_text:constrained_sequence_op_cc",
    ],
)

tf_cc_library(
    name = "fast_bert_normalizer",
    hdrs = ["fast_bert_normalizer.h"],
    deps = [
        ":darts_clone_trie_builder",
        ":darts_clone_trie_wrapper",
        ":fast_bert_normalizer_model",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@icu//:common",
        # lite/kernels/shim:status_macros tensorflow dep,
    ],
)

flatbuffer_cc_library(
    name = "fast_bert_normalizer_model",
    srcs = [
        "fast_bert_normalizer_model.fbs",
    ],
)

tf_cc_library(
    name = "fast_bert_normalizer_model_builder",
    srcs = ["fast_bert_normalizer_model_builder.cc"],
    hdrs = ["fast_bert_normalizer_model_builder.h"],
    tf_deps = [
        # tf:lib tensorflow dep,
    ],
    deps = [
        ":darts_clone_trie_builder",
        ":fast_bert_normalizer",
        ":fast_bert_normalizer_model",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@icu//:common",
        "@icu//:nfkc_cf",  # Needed for NFKC_Casefold Unicode Normalization form.
        "@com_googlesource_code_re2//:re2",
        # lite/kernels/shim:status_macros tensorflow dep,
    ],
)

cc_test(
    name = "fast_bert_normalizer_test",
    size = "small",
    srcs = ["fast_bert_normalizer_test.cc"],
    deps = [
        ":fast_bert_normalizer",
        ":fast_bert_normalizer_model_builder",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "fast_bert_normalizer_kernel_template",
    hdrs = ["fast_bert_normalizer_kernel_template.h"],
    deps = [
        ":fast_bert_normalizer",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        # lite/kernels/shim:op_kernel tensorflow dep,
        # lite/kernels/shim:status_macros tensorflow dep,
    ],
)

tf_cc_library(
    name = "fast_bert_normalizer_tf_kernel",
    srcs = ["fast_bert_normalizer_tf_kernel.cc"],
    hdrs = ["fast_bert_normalizer_tf_kernel.h"],
    tf_deps = [
        # tf:framework tensorflow dep,
    ],
    deps = [
        ":fast_bert_normalizer_kernel_template",
        # lite/kernels/shim:tf_op_shim tensorflow dep,
    ],
)

tflite_cc_library(
    name = "fast_bert_normalizer_tflite",
    srcs = ["fast_bert_normalizer_tflite.cc"],
    hdrs = ["fast_bert_normalizer_tflite.h"],
    deps = [
        ":fast_bert_normalizer_kernel_template",
        # lite:mutable_op_resolver tensorflow dep,
        # lite/c:common tensorflow dep,
        # lite/kernels/shim:tflite_op_shim tensorflow dep,
    ],
)

cc_test(
    name = "log_greedy_constrained_sequence_kernel_test",
    srcs = ["log_greedy_constrained_sequence_kernel_test.cc"],
    deps = [
        ":text_kernels_test_util",
        "@com_google_googletest//:gtest_main",
        # tf:framework tensorflow dep,
        # tf:lib tensorflow dep,
        # tf:protos_all_cc tensorflow dep,
        # tf:test tensorflow dep,
        # tf:testlib tensorflow dep,
        # tf/kernels:ops_testutil tensorflow dep,
        "//tensorflow_text:constrained_sequence_op_cc",
    ],
)

cc_test(
    name = "log_viterbi_constrained_sequence_kernel_test",
    srcs = ["log_viterbi_constrained_sequence_kernel_test.cc"],
    deps = [
        ":text_kernels_test_util",
        "@com_google_googletest//:gtest_main",
        # tf:framework tensorflow dep,
        # tf:lib tensorflow dep,
        # tf:protos_all_cc tensorflow dep,
        # tf:test tensorflow dep,
        # tf:testlib tensorflow dep,
        # tf/kernels:ops_testutil tensorflow dep,
        "//tensorflow_text:constrained_sequence_op_cc",
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

cc_test(
    name = "darts_clone_trie_test",
    size = "small",
    srcs = ["darts_clone_trie_test.cc"],
    deps = [
        ":darts_clone_trie_builder",
        ":darts_clone_trie_wrapper",
        "@com_google_absl//absl/status",
        "@com_google_googletest//:gtest_main",
    ],
)

tf_cc_library(
    name = "disjoint_set_forest",
    hdrs = ["disjoint_set_forest.h"],
    tf_deps = [
        # tf:lib tensorflow dep,
    ],
)

cc_test(
    name = "disjoint_set_forest_test",
    size = "small",
    srcs = ["disjoint_set_forest_test.cc"],
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
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@icu//:nfkc",
        # lite/kernels/shim:status_macros tensorflow dep,
    ],
)

cc_test(
    name = "fast_wordpiece_tokenizer_test",
    srcs = ["fast_wordpiece_tokenizer_test.cc"],
    data = [
        "//tensorflow_text:python/ops/test_data/fast_wordpiece_tokenizer_model.fb",
        "//tensorflow_text:python/ops/test_data/fast_wordpiece_tokenizer_model_ver_15_1.fb",
        "//tensorflow_text:python/ops/test_data/fast_wordpiece_tokenizer_model_ver_16_0.fb",
    ],
    deps = [
        ":fast_wordpiece_tokenizer",
        ":fast_wordpiece_tokenizer_model_builder",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/flags:flag",
        "@icu//:headers",
        # tf:lib tensorflow dep,
    ],
)

flatbuffer_cc_library(
    name = "fast_wordpiece_tokenizer_model",
    srcs = [
        "fast_wordpiece_tokenizer_model.fbs",
    ],
)

flatbuffer_cc_library(
    name = "phrase_tokenizer_model",
    srcs = [
        "phrase_tokenizer_model.fbs",
    ],
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
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@cppitertools",
        "@icu//:nfkc",
        # lite/kernels/shim:status_macros tensorflow dep,
    ],
)

tf_cc_library(
    name = "phrase_tokenizer_model_builder",
    srcs = ["phrase_tokenizer_model_builder.cc"],
    hdrs = [
        "phrase_tokenizer_model_builder.h",
    ],
    deps = [
        ":darts_clone_trie_builder",
        ":darts_clone_trie_wrapper",
        ":fast_wordpiece_tokenizer_utils",
        ":phrase_tokenizer_model",
        ":sentence_fragmenter_v2",
        ":string_vocab",
        ":whitespace_tokenizer_config_builder",
        ":wordpiece_tokenizer",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@icu//:nfkc",
        # lite/kernels/shim:status_macros tensorflow dep,
        "//tensorflow_text/core/kernels/sentencepiece:double_array_trie_builder",
    ],
)

tf_cc_library(
    name = "fast_wordpiece_tokenizer_kernel",
    srcs = ["fast_wordpiece_tokenizer_kernel.cc"],
    hdrs = ["fast_wordpiece_tokenizer_kernel.h"],
    tf_deps = [
        # tf:framework tensorflow dep,
    ],
    deps = [
        ":fast_wordpiece_tokenizer_kernel_template",
        # lite/kernels/shim:tf_op_shim tensorflow dep,
    ],
)

cc_library(
    name = "fast_wordpiece_tokenizer_kernel_template",
    hdrs = ["fast_wordpiece_tokenizer_kernel_template.h"],
    deps = [
        ":fast_wordpiece_tokenizer",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        # lite/kernels/shim:op_kernel tensorflow dep,
        # lite/kernels/shim:status_macros tensorflow dep,
    ],
)

tflite_cc_library(
    name = "fast_wordpiece_tokenizer_tflite",
    srcs = ["fast_wordpiece_tokenizer_tflite.cc"],
    hdrs = ["fast_wordpiece_tokenizer_tflite.h"],
    deps = [
        ":fast_wordpiece_tokenizer_kernel_template",
        # lite:mutable_op_resolver tensorflow dep,
        # lite/c:common tensorflow dep,
        # lite/kernels/shim:tflite_op_shim tensorflow dep,
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

cc_test(
    name = "fast_wordpiece_tokenizer_utils_test",
    srcs = ["fast_wordpiece_tokenizer_utils_test.cc"],
    deps = [
        ":fast_wordpiece_tokenizer_utils",
        "@com_google_googletest//:gtest_main",
    ],
)

tf_cc_library(
    name = "mst_op_kernels",
    srcs = ["mst_op_kernels.cc"],
    tf_deps = [
        # tf:framework tensorflow dep,
        # tf:lib tensorflow dep,
    ],
    deps = [
        ":mst_solver",
    ],
)

tf_cc_library(
    name = "mst_solver",
    hdrs = ["mst_solver.h"],
    tf_deps = [
        # tf:lib tensorflow dep,
    ],
    deps = [
        ":disjoint_set_forest",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
)

cc_test(
    name = "mst_solver_test",
    size = "small",
    srcs = ["mst_solver_test.cc"],
    deps = [
        ":mst_solver",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
        # tf:test tensorflow dep,
    ],
)

cc_test(
    name = "mst_solver_random_comparison_test",
    size = "small",
    timeout = "long",
    srcs = ["mst_solver_random_comparison_test.cc"],
    tags = [
        "nofastbuild",  # exclude from non-opt TAP projects
        "optonly",  # exclude from non-opt TAP projects
    ],
    deps = [
        ":mst_solver",
        ":spanning_tree_iterator",
        "@com_google_googletest//:gtest",  # google-only
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/flags:flag",
        # tf:lib tensorflow dep,
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
        # tf/platform:tstring tensorflow dep,
    ],
    deps = [
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        # lite/kernels/shim:op_kernel tensorflow dep,
        # lite/kernels/shim:status_macros tensorflow dep,
        # lite/kernels/shim:tensor_view tensorflow dep,
    ],
)

tf_cc_library(
    name = "ngrams_kernel",
    srcs = ["ngrams_kernel.cc"],
    hdrs = ["ngrams_kernel.h"],
    tf_deps = [
        # tf:framework tensorflow dep,
    ],
    deps = [
        ":ngrams_kernel_template",
        # lite/kernels/shim:tf_op_shim tensorflow dep,
    ],
)

cc_test(
    name = "ngrams_kernel_test",
    srcs = ["ngrams_kernel_test.cc"],
    deps = [
        # tf:framework tensorflow dep,
        # tf:test tensorflow dep,
        # tf:test_main tensorflow dep,
        # tf/framework:shape_inference_testutil tensorflow dep,
        # tf/framework:tensor_testutil tensorflow dep,
        "//tensorflow_text:ngrams_op_cc",
    ],
)

tflite_cc_library(
    name = "ngrams_tflite",
    srcs = ["ngrams_tflite.cc"],
    hdrs = ["ngrams_tflite.h"],
    deps = [
        ":ngrams_kernel_template",
        # lite:mutable_op_resolver tensorflow dep,
        # lite/c:common tensorflow dep,
        # lite/kernels/shim:tflite_op_shim tensorflow dep,
    ],
)

cc_test(
    name = "ngrams_tflite_test",
    srcs = ["ngrams_tflite_test.cc"],
    deps = [
        ":ngrams_tflite",
        "@com_google_googletest//:gtest_main",
        "@flatbuffers",
        # lite:string_util tensorflow dep,
        # lite/c:common tensorflow dep,
        # lite/kernels:test_util tensorflow dep,
        # lite/schema:schema_fbs tensorflow dep,
    ],
)

tf_cc_library(
    name = "normalize_kernels",
    srcs = ["normalize_kernels.cc"],
    tf_deps = [
        # tf:framework tensorflow dep,
    ],
    deps = [
        ":edit_changes_cc_proto",
        "@com_google_absl//absl/strings",
        "@icu//:nfkc",
        "@icu//:nfkc_cf",
    ],
)

tflite_cc_library(
    name = "ragged_tensor_to_tensor_tflite",
    srcs = ["ragged_tensor_to_tensor_tflite.cc"],
    hdrs = ["ragged_tensor_to_tensor_tflite.h"],
    deps = [
        "@flatbuffers",
        # tf/util:ragged_to_dense_util_common tensorflow dep,
        # lite:framework tensorflow dep,
        # lite/c:common tensorflow dep,
        # lite/kernels:kernel_util tensorflow dep,
        # lite/kernels/internal:tensor tensorflow dep,
        # lite/kernels/internal:types tensorflow dep,
    ],
)

cc_test(
    name = "ragged_tensor_to_tensor_tflite_test",
    srcs = ["ragged_tensor_to_tensor_tflite_test.cc"],
    deps = [
        ":ragged_tensor_to_tensor_tflite",
        "@com_google_googletest//:gtest_main",
        "@flatbuffers",
        # lite:framework tensorflow dep,
        # lite/c:common tensorflow dep,
        # lite/kernels:test_util tensorflow dep,
        # lite/kernels/internal:tensor tensorflow dep,
        # lite/schema:schema_fbs tensorflow dep,
    ],
)

tf_cc_library(
    name = "regex_split",
    srcs = ["regex_split.cc"],
    hdrs = ["regex_split.h"],
    deps = [
        "@com_google_absl//absl/strings",
        "@com_googlesource_code_re2//:re2",
    ],
)

tf_cc_library(
    name = "regex_split_kernels",
    srcs = ["regex_split_kernels.cc"],
    tf_deps = [
        # tf:framework tensorflow dep,
        # tf:lib tensorflow dep,
    ],
    deps = [
        ":regex_split",
        "@com_google_absl//absl/memory",
    ],
)

cc_test(
    name = "regex_split_test",
    srcs = ["regex_split_test.cc"],
    deps = [
        ":regex_split",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/strings",
        "@com_googlesource_code_re2//:re2",
        # tf:lib tensorflow dep,
    ],
)

cc_library(
    name = "round_robin_trimmer",
    hdrs = ["round_robin_trimmer.h"],
    deps = [
        ":trimmer",
        "@com_google_absl//absl/types:span",
    ],
)

cc_test(
    name = "round_robin_trimmer_test",
    size = "small",
    srcs = ["round_robin_trimmer_test.cc"],
    deps = [
        ":round_robin_trimmer",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        # tf:lib tensorflow dep,
        # tf:test_main tensorflow dep,
    ],
)

tf_cc_library(
    name = "round_robin_trimmer_kernel",
    srcs = ["round_robin_trimmer_kernel.cc"],
    hdrs = ["round_robin_trimmer_kernel.h"],
    tf_deps = [
        # tf:framework tensorflow dep,
    ],
    deps = [
        ":round_robin_trimmer_kernel_template",
        # lite/kernels/shim:op_kernel tensorflow dep,
        # lite/kernels/shim:tf_op_shim tensorflow dep,
    ],
)

tf_cc_library(
    name = "round_robin_trimmer_kernel_template",
    hdrs = ["round_robin_trimmer_kernel_template.h"],
    tf_deps = [
        # tf:framework tensorflow dep,
    ],
    deps = [
        ":round_robin_trimmer",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/types:span",
        # lite/kernels/shim:op_kernel tensorflow dep,
        # lite/kernels/shim:shape tensorflow dep,
        # lite/kernels/shim:tensor_view tensorflow dep,
    ],
)

tflite_cc_library(
    name = "round_robin_trimmer_tflite",
    srcs = ["round_robin_trimmer_tflite.cc"],
    hdrs = ["round_robin_trimmer_tflite.h"],
    deps = [
        ":round_robin_trimmer_kernel_template",
        # lite:mutable_op_resolver tensorflow dep,
        # lite/c:common tensorflow dep,
        # lite/kernels/shim:op_kernel tensorflow dep,
        # lite/kernels/shim:tflite_op_shim tensorflow dep,
        # lite/kernels/shim:tflite_op_wrapper tensorflow dep,
    ],
)

tf_cc_library(
    name = "rouge_l_kernel",
    srcs = ["rouge_l_kernel.cc"],
    tf_deps = [
        # tf:framework tensorflow dep,
        # tf:lib tensorflow dep,
    ],
    deps = [
        "@com_google_absl//absl/strings",
    ],
)

cc_test(
    name = "rouge_l_kernel_test",
    size = "small",
    srcs = ["rouge_l_kernel_test.cc"],
    deps = [
        ":rouge_l_kernel",
        # tf:framework tensorflow dep,
        # tf:test tensorflow dep,
        # tf:test_main tensorflow dep,
        # tf:testlib tensorflow dep,
        # tf/kernels:ops_testutil tensorflow dep,
        "//tensorflow_text:text_similarity_metric_ops_cc",
    ],
)

tf_cc_library(
    name = "sentence_breaking_kernels",
    srcs = ["sentence_breaking_kernels.cc"],
    tf_deps = [
        # tf:framework tensorflow dep,
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
        # tf:lib tensorflow dep,
    ],
    deps = [
        "@com_google_absl//absl/strings",
        "@icu//:common",
    ],
)

cc_test(
    name = "sentence_breaking_utils_test",
    size = "small",
    srcs = ["sentence_breaking_utils_test.cc"],
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
        # tf:lib tensorflow dep,
    ],
    deps = [
        ":sentence_breaking_utils",
        "@com_google_absl//absl/status",
    ],
)

tf_cc_library(
    name = "sentence_fragmenter_v2",
    srcs = ["sentence_fragmenter_v2.cc"],
    hdrs = ["sentence_fragmenter_v2.h"],
    tf_deps = [
        # tf:lib tensorflow dep,
    ],
    deps = [
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@icu//:common",
    ],
)

cc_test(
    name = "sentence_fragmenter_v2_test",
    srcs = ["sentence_fragmenter_v2_test.cc"],
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
        # tf:framework tensorflow dep,
    ],
    deps = [
        ":sentence_fragmenter_v2_kernel_template",
        # lite/kernels/shim:tf_op_shim tensorflow dep,
    ],
)

tf_cc_library(
    name = "sentence_fragmenter_v2_kernel_template",
    hdrs = ["sentence_fragmenter_v2_kernel_template.h"],
    tf_deps = [
        # tf:framework tensorflow dep,
    ],
    deps = [
        ":sentence_fragmenter_v2",
        "@com_google_absl//absl/status",
        # lite/kernels/shim:op_kernel tensorflow dep,
        # lite/kernels/shim:shape tensorflow dep,
        # lite/kernels/shim:status_macros tensorflow dep,
        # lite/kernels/shim:tensor_view tensorflow dep,
    ],
)

tflite_cc_library(
    name = "sentence_fragmenter_v2_tflite",
    srcs = ["sentence_fragmenter_v2_tflite.cc"],
    hdrs = ["sentence_fragmenter_v2_tflite.h"],
    deps = [
        ":sentence_fragmenter_v2_kernel_template",
        # lite:mutable_op_resolver tensorflow dep,
        # lite/c:common tensorflow dep,
        # lite/kernels/shim:tflite_op_shim tensorflow dep,
    ],
)

tf_cc_library(
    name = "sentencepiece_kernels",
    srcs = ["sentencepiece_kernels.cc"],
    tf_deps = [
        # tf:framework tensorflow dep,
        # tf:framework_headers_lib tensorflow dep,
        # tf:lib tensorflow dep,
        # tf:protos_all_cc tensorflow dep,
    ],
    deps = [
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/meta:type_traits",
        "@com_google_absl//absl/strings",
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
        # tf:lib tensorflow dep,
    ],
)

cc_test(
    name = "spanning_tree_iterator_test",
    size = "small",
    srcs = ["spanning_tree_iterator_test.cc"],
    deps = [
        ":spanning_tree_iterator",
        "@com_google_googletest//:gtest_main",
        # tf:lib tensorflow dep,
    ],
)

tf_cc_library(
    name = "split_merge_tokenize_kernel",
    srcs = ["split_merge_tokenize_kernel.cc"],
    tf_deps = [
        # tf:framework tensorflow dep,
        # tf:lib tensorflow dep,
    ],
    deps = [
        "@com_google_absl//absl/base:core_headers",
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
        "@com_google_googletest//:gtest",
        # tf:framework tensorflow dep,
        # tf:testlib tensorflow dep,
    ],
)

tflite_cc_library(
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
        "//tensorflow_text/core/kernels/sentencepiece:sp_headers",
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
        # lite:mutable_op_resolver tensorflow dep,
        # lite/c:common tensorflow dep,
        "//tensorflow_text/core/kernels/sentencepiece:py_tflite_registerer",
    ],
)

tf_cc_library(
    name = "tokenizer_from_logits_kernel",
    srcs = ["tokenizer_from_logits_kernel.cc"],
    tf_deps = [
        # tf:framework tensorflow dep,
        # tf:lib tensorflow dep,
    ],
    deps = [
        "@com_google_absl//absl/base:core_headers",
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
        # tf:framework tensorflow dep,
        # tf:lib tensorflow dep,
    ],
    deps = [
        "@icu//:common",
    ],
)

cc_test(
    name = "unicode_script_tokenize_kernel_test",
    srcs = ["unicode_script_tokenize_kernel_test.cc"],
    deps = [
        ":text_kernels_test_util",
        "@com_google_googletest//:gtest_main",
        # tf:framework tensorflow dep,
        # tf:lib tensorflow dep,
        # tf:test tensorflow dep,
        # tf:testlib tensorflow dep,
        # tf/kernels:ops_testutil tensorflow dep,
        "//tensorflow_text:unicode_script_tokenizer_cc",
    ],
)

tf_cc_library(
    name = "whitespace_tokenize_kernel",
    srcs = ["whitespace_tokenize_kernel.cc"],
    tf_deps = [
        # tf:framework tensorflow dep,
        # tf:lib tensorflow dep,
    ],
    deps = [
        "@icu//:common",
    ],
)

cc_test(
    name = "whitespace_tokenize_kernel_test",
    srcs = ["whitespace_tokenize_kernel_test.cc"],
    deps = [
        ":text_kernels_test_util",
        "@com_google_googletest//:gtest_main",
        # tf:framework tensorflow dep,
        # tf:lib tensorflow dep,
        # tf:test tensorflow dep,
        # tf:testlib tensorflow dep,
        # tf/kernels:ops_testutil tensorflow dep,
        "//tensorflow_text:whitespace_tokenizer_cc",
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

cc_test(
    name = "whitespace_tokenizer_test",
    size = "small",
    srcs = ["whitespace_tokenizer_test.cc"],
    deps = [
        ":whitespace_tokenizer",
        ":whitespace_tokenizer_config_builder",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        # tf:lib tensorflow dep,
        # tf:test_main tensorflow dep,
    ],
)

tf_cc_library(
    name = "whitespace_tokenizer_kernel",
    srcs = ["whitespace_tokenizer_kernel.cc"],
    hdrs = ["whitespace_tokenizer_kernel.h"],
    tf_deps = [
        # tf:framework tensorflow dep,
    ],
    deps = [
        ":whitespace_tokenizer_kernel_template",
        # lite/kernels/shim:op_kernel tensorflow dep,
        # lite/kernels/shim:tf_op_shim tensorflow dep,
    ],
)

tf_cc_library(
    name = "whitespace_tokenizer_kernel_template",
    hdrs = ["whitespace_tokenizer_kernel_template.h"],
    tf_deps = [
        # tf:framework tensorflow dep,
    ],
    deps = [
        ":whitespace_tokenizer",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        # lite/kernels/shim:op_kernel tensorflow dep,
        # lite/kernels/shim:shape tensorflow dep,
        # lite/kernels/shim:tensor_view tensorflow dep,
    ],
)

tflite_cc_library(
    name = "whitespace_tokenizer_tflite",
    srcs = ["whitespace_tokenizer_tflite.cc"],
    hdrs = ["whitespace_tokenizer_tflite.h"],
    deps = [
        ":whitespace_tokenizer_kernel_template",
        # lite:mutable_op_resolver tensorflow dep,
        # lite/c:common tensorflow dep,
        # lite/kernels/shim:tflite_op_shim tensorflow dep,
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

cc_test(
    name = "whitespace_tokenizer_config_builder_test",
    size = "small",
    srcs = ["whitespace_tokenizer_config_builder_test.cc"],
    deps = [
        ":whitespace_tokenizer",
        ":whitespace_tokenizer_config_builder",
        "@com_google_googletest//:gtest_main",
        "@icu//:common",
        # tf:lib tensorflow dep,
        # tf:test_main tensorflow dep,
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
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/random",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        # lite/kernels/shim:status_macros tensorflow dep,
        "//tensorflow_text/core/kernels/sentencepiece:double_array_trie",
    ],
)

cc_test(
    name = "phrase_tokenizer_test",
    size = "small",
    srcs = ["phrase_tokenizer_test.cc"],
    data = [
        "//tensorflow_text:python/ops/test_data/phrase_tokenizer_model.fb",
    ],
    deps = [
        ":phrase_tokenizer",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        # tf:lib tensorflow dep,
        # tf:test_main tensorflow dep,
    ],
)

cc_library(
    name = "phrase_tokenizer_kernel_template",
    hdrs = ["phrase_tokenizer_kernel_template.h"],
    deps = [
        ":phrase_tokenizer",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        # lite/kernels/shim:op_kernel tensorflow dep,
        # lite/kernels/shim:status_macros tensorflow dep,
    ],
)

tf_cc_library(
    name = "phrase_tokenizer_kernel",
    srcs = ["phrase_tokenizer_kernel.cc"],
    hdrs = ["phrase_tokenizer_kernel.h"],
    tf_deps = [
        # tf:framework tensorflow dep,
    ],
    deps = [
        ":phrase_tokenizer_kernel_template",
        # lite/kernels/shim:op_kernel tensorflow dep,
        # lite/kernels/shim:tf_op_shim tensorflow dep,
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

cc_test(
    name = "utf8_binarize_test",
    size = "small",
    srcs = ["utf8_binarize_test.cc"],
    deps = [
        ":utf8_binarize",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/types:span",
        # tf:lib tensorflow dep,
        # tf:test_main tensorflow dep,
    ],
)

cc_library(
    name = "utf8_binarize_kernel_template",
    hdrs = ["utf8_binarize_kernel_template.h"],
    deps = [
        ":utf8_binarize",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        # tf/platform:tstring tensorflow dep,
        # lite/kernels/shim:op_kernel tensorflow dep,
        # lite/kernels/shim:shape tensorflow dep,
        # lite/kernels/shim:status_macros tensorflow dep,
    ],
)

tf_cc_library(
    name = "utf8_binarize_kernel",
    srcs = ["utf8_binarize_kernel.cc"],
    hdrs = ["utf8_binarize_kernel.h"],
    tf_deps = [
        # tf:framework tensorflow dep,
    ],
    deps = [
        ":utf8_binarize_kernel_template",
        # lite/kernels/shim:op_kernel tensorflow dep,
        # lite/kernels/shim:tf_op_shim tensorflow dep,
    ],
)

tflite_cc_library(
    name = "utf8_binarize_tflite",
    srcs = ["utf8_binarize_tflite.cc"],
    hdrs = ["utf8_binarize_tflite.h"],
    deps = [
        ":utf8_binarize_kernel_template",
        # lite:mutable_op_resolver tensorflow dep,
        # lite/c:common tensorflow dep,
        # lite/kernels/shim:tflite_op_shim tensorflow dep,
    ],
)

tf_cc_library(
    name = "wordpiece_kernel",
    srcs = ["wordpiece_kernel.cc"],
    tf_deps = [
        # tf:framework tensorflow dep,
        # tf:lib tensorflow dep,
    ],
    deps = [
        ":wordpiece_tokenizer",
        "@com_google_absl//absl/base:core_headers",
    ],
)

tf_cc_library(
    name = "wordpiece_tokenizer",
    srcs = ["wordpiece_tokenizer.cc"],
    hdrs = ["wordpiece_tokenizer.h"],
    deps = [
        "@com_google_absl//absl/strings",
        "@icu//:common",
    ],
)

cc_test(
    name = "wordpiece_kernel_test",
    size = "small",
    srcs = ["wordpiece_kernel_test.cc"],
    deps = [
        ":wordpiece_kernel",
        # tf:framework tensorflow dep,
        # tf:test tensorflow dep,
        # tf:test_main tensorflow dep,
        # tf:testlib tensorflow dep,
        # tf/kernels:ops_testutil tensorflow dep,
        "//tensorflow_text:wordpiece_tokenizer_cc",
    ],
)
