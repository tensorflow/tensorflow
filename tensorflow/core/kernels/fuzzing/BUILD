load("//tensorflow/core/kernels/fuzzing:tf_ops_fuzz_target_lib.bzl", "tf_ops_fuzz_target_lib", "tf_oss_fuzz_corpus", "tf_oss_fuzz_dict")
load("//tensorflow/core/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

cc_library(
    name = "fuzz_session",
    hdrs = ["fuzz_session.h"],
    deps = [
        "//tensorflow/cc:scope",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:tensorflow",
    ],
)

tf_ops_fuzz_target_lib("string_to_number")

tf_oss_fuzz_corpus("string_to_number")

tf_ops_fuzz_target_lib("encode_base64")

tf_ops_fuzz_target_lib("decode_base64")

tf_ops_fuzz_target_lib("encode_jpeg")

tf_ops_fuzz_target_lib("decode_bmp")

tf_oss_fuzz_corpus("decode_bmp")

tf_ops_fuzz_target_lib("decode_png")

tf_oss_fuzz_corpus("decode_png")

tf_oss_fuzz_dict("decode_png")

tf_ops_fuzz_target_lib("decode_wav")

tf_oss_fuzz_corpus("decode_wav")

tf_oss_fuzz_dict("decode_wav")

tf_ops_fuzz_target_lib("example_proto_fast_parsing")

tf_ops_fuzz_target_lib("parse_tensor_op")

tf_ops_fuzz_target_lib("decode_compressed")

tf_ops_fuzz_target_lib("decode_json_example")

tf_oss_fuzz_corpus("decode_json_example")

tf_oss_fuzz_dict("decode_json_example")

tf_ops_fuzz_target_lib("check_numerics")

tf_ops_fuzz_target_lib("one_hot")

tf_ops_fuzz_target_lib("scatter_nd")

tf_oss_fuzz_corpus("scatter_nd")

tf_ops_fuzz_target_lib("decode_csv")

tf_oss_fuzz_corpus("decode_csv")
