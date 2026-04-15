# Raw-bytes fuzzers for TensorFlow GraphDef-consuming APIs.

load(
    "//tensorflow/security/fuzzing:tf_fuzzing.bzl",
    "tf_cc_fuzz_test",
)

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    licenses = ["notice"],
)

tf_cc_fuzz_test(
    name = "import_graph_def_fuzz",
    srcs = ["import_graph_def_fuzz.cc"],
    tags = ["no_oss"],
    deps = [
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/common_runtime:graph_constructor",
    ],
)
