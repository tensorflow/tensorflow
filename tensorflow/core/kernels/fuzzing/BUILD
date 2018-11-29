licenses(["notice"])  # Apache 2.0

package(
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "fuzz_session",
    hdrs = ["fuzz_session.h"],
    deps = [
        "//tensorflow/cc:cc_ops",
        "//tensorflow/cc:scope",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:direct_session",
        "//tensorflow/core:ops",
        "//tensorflow/core:tensorflow",
    ],
)

load("//tensorflow/core/kernels/fuzzing:tf_ops_fuzz_target_lib.bzl", "tf_ops_fuzz_target_lib")
load("//tensorflow/core/kernels/fuzzing:tf_ops_fuzz_target_lib.bzl", "tf_oss_fuzz_corpus")

tf_ops_fuzz_target_lib("identity")

tf_ops_fuzz_target_lib("string_to_number")

tf_oss_fuzz_corpus("string_to_number")

tf_ops_fuzz_target_lib("string_split")

tf_oss_fuzz_corpus("string_split")

tf_ops_fuzz_target_lib("string_split_v2")

tf_oss_fuzz_corpus("string_split_v2")

tf_ops_fuzz_target_lib("encode_base64")

tf_ops_fuzz_target_lib("decode_base64")

tf_ops_fuzz_target_lib("encode_jpeg")

tf_ops_fuzz_target_lib("decode_bmp")

tf_ops_fuzz_target_lib("decode_png")

tf_ops_fuzz_target_lib("decode_wav")

tf_ops_fuzz_target_lib("example_proto_fast_parsing")

tf_ops_fuzz_target_lib("parse_tensor_op")

tf_ops_fuzz_target_lib("decode_compressed")

tf_ops_fuzz_target_lib("decode_json_example")
