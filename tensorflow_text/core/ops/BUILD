# Tests for op kernel shape functions

load("@rules_cc//cc:cc_test.bzl", "cc_test")

licenses(["notice"])

package(default_visibility = ["//visibility:private"])

exports_files(glob(["*.cc"]))

cc_test(
    name = "byte_splitter_op_test",
    size = "small",
    srcs = [
        "byte_splitter_op.cc",
        "byte_splitter_op_test.cc",
    ],
    deps = [
        # tf:framework tensorflow dep,
        # tf:test tensorflow dep,
        # tf:test_main tensorflow dep,
        # tf/framework:shape_inference_testutil tensorflow dep,
        "//tensorflow_text/core/kernels:byte_splitter_kernel",
    ],
)

cc_test(
    name = "boise_offset_converter_op_test",
    size = "small",
    srcs = [
        "boise_offset_converter_op.cc",
        "boise_offset_converter_op_test.cc",
    ],
    deps = [
        # tf:framework tensorflow dep,
        # tf:test tensorflow dep,
        # tf:test_main tensorflow dep,
        # tf/framework:shape_inference_testutil tensorflow dep,
        "//tensorflow_text/core/kernels:boise_offset_converter_kernel",
    ],
)

cc_test(
    name = "fast_wordpiece_tokenizer_op_test",
    size = "small",
    srcs = [
        "fast_wordpiece_tokenizer_op.cc",
        "fast_wordpiece_tokenizer_op_test.cc",
    ],
    deps = [
        # tf:framework tensorflow dep,
        # tf:test tensorflow dep,
        # tf:test_main tensorflow dep,
        # tf/framework:shape_inference_testutil tensorflow dep,
        "//tensorflow_text/core/kernels:fast_wordpiece_tokenizer_kernel",
    ],
)

cc_test(
    name = "sentence_fragmenter_v2_op_test",
    size = "small",
    srcs = [
        "sentence_fragmenter_v2_op.cc",
        "sentence_fragmenter_v2_op_test.cc",
    ],
    deps = [
        # tf:framework tensorflow dep,
        # tf:test tensorflow dep,
        # tf:test_main tensorflow dep,
        # tf/framework:shape_inference_testutil tensorflow dep,
        "//tensorflow_text/core/kernels:sentence_fragmenter_v2_kernel",
    ],
)

cc_test(
    name = "trimmer_ops_test",
    size = "small",
    srcs = [
        "trimmer_ops.cc",
        "trimmer_ops_test.cc",
    ],
    deps = [
        # tf:framework tensorflow dep,
        # tf:test tensorflow dep,
        # tf:test_main tensorflow dep,
        # tf/framework:shape_inference_testutil tensorflow dep,
        # tf/framework:tensor_testutil tensorflow dep,
        "//tensorflow_text/core/kernels:round_robin_trimmer_kernel",
    ],
)

cc_test(
    name = "utf8_binarize_op_test",
    size = "small",
    srcs = [
        "utf8_binarize_op.cc",
        "utf8_binarize_op_test.cc",
    ],
    deps = [
        # tf:framework tensorflow dep,
        # tf:protos_all_cc tensorflow dep,
        # tf:test tensorflow dep,
        # tf:test_main tensorflow dep,
        # tf/framework:shape_inference_testutil tensorflow dep,
        "//tensorflow_text/core/kernels:utf8_binarize_kernel",
    ],
)
