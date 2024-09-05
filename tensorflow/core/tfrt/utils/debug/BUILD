load("//tensorflow:tensorflow.bzl", "tf_cc_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    licenses = ["notice"],
)

cc_library(
    name = "node_io_dump_rewriter",
    srcs = ["node_io_dump_rewriter.cc"],
    hdrs = ["node_io_dump_rewriter.h"],
    deps = [
        "//tensorflow/core:core_cpu_base",
        "//tensorflow/core:framework",
        "//tensorflow/core/lib/core:errors",
        "//tensorflow/core/platform:status",
        "//tensorflow/core/protobuf:for_core_protos_cc",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:statusor",
    ],
)

# copybara:uncomment_begin(Google Internal Test)
# tf_cc_test(
#     name = "node_io_dump_rewriter_test",
#     srcs = ["node_io_dump_rewriter_test.cc"],
#     data = [
#         "//tensorflow/core/tfrt/saved_model/tests:toy_v1/1/saved_model.pb",
#         "//tensorflow/core/tfrt/saved_model/tests:toy_v1/1/variables/variables.data-00000-of-00001",
#         "//tensorflow/core/tfrt/saved_model/tests:toy_v1/1/variables/variables.index",
#         "//tensorflow/core/tfrt/saved_model/tests:toy_v2/saved_model.pb",
#         "//tensorflow/core/tfrt/saved_model/tests:toy_v2/variables/variables.data-00000-of-00001",
#         "//tensorflow/core/tfrt/saved_model/tests:toy_v2/variables/variables.index",
#     ],
#     deps = [
#         ":node_io_dump_rewriter",
#         "@com_google_googletest//:gtest_main",
#         "@com_google_absl//absl/log:check",
#         "@com_google_absl//absl/status",
#         "@com_google_absl//absl/strings",
#         "//tensorflow/cc:array_ops",
#         "//tensorflow/cc:cc_ops",
#         "//tensorflow/cc:scope",
#         "//tensorflow/cc:sendrecv_ops",
#         "//tensorflow/cc/saved_model:reader",
#         "//tensorflow/core:core_cpu_base",
#         "//tensorflow/core:portable_gif_internal",
#         "//tensorflow/core:test",
#         "//tensorflow/core/framework:op",
#         "//tensorflow/core/framework:tensor",
#         "//tensorflow/core/framework:types_proto_cc",
#         "//tensorflow/core/platform:env",
#         "//tensorflow/core/platform:resource_loader",
#         "//tensorflow/core/protobuf:for_core_protos_cc",
#         "//tensorflow/core/tfrt/saved_model:saved_model_cpu",
#         "//tensorflow/core/tfrt/saved_model:saved_model_testutil",
#         "//tensorflow/python/framework:test_ops_kernels",
#         "@local_tsl//tsl/platform:path",
#         "@local_tsl//tsl/platform:statusor",
#         "@local_tsl//tsl/platform:test",
#     ],
# )
# copybara:uncomment_end
