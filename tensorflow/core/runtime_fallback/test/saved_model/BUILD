load("//tensorflow:pytype.default.bzl", "pytype_strict_binary")
load("//tensorflow:tensorflow.bzl", "if_google", "tf_cc_shared_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [":internal"],
    licenses = ["notice"],
)

package_group(
    name = "internal",
    packages = [
        "//tensorflow/core/tfrt/saved_model/tests/...",
    ],
)

pytype_strict_binary(
    name = "gen_saved_model_v1",
    srcs = ["gen_saved_model_v1.py"],
    deps = [
        "//tensorflow/python/client:session",
        "//tensorflow/python/distribute:input_lib",  # build_cleaner: keep;
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/ops:parsing_ops",
        "//tensorflow/python/ops:variable_scope",
        "//tensorflow/python/ops:variables",
        "//tensorflow/python/saved_model:builder",
        "//tensorflow/python/saved_model:signature_constants",
        "//tensorflow/python/saved_model:signature_def_utils",
        "//tensorflow/python/saved_model:tag_constants",
        "//tensorflow/python/saved_model:utils",
        "@absl_py//absl:app",
        "@absl_py//absl/flags",
    ] + if_google([
        "//learning/brain/public:disable_tf2",  # build_cleaner: keep; go/disable_tf2
    ]),
)

genrule(
    name = "saved_model_gen_saved_model_v1",
    srcs = [],
    outs = [
        "basic_v1/saved_model.pb",
    ],
    cmd = "$(location gen_saved_model_v1) --saved_model_path=$(RULEDIR)/basic_v1",
    tools = ["gen_saved_model_v1"],
)

tf_cc_shared_test(
    name = "saved_model_test",
    srcs = ["saved_model_test.cc"],
    data = [
        "basic_v1/saved_model.pb",
    ],
    tags = ["no_oss"],
    deps = [
        "//tensorflow/core/tfrt/saved_model:saved_model_testutil",
        "@com_google_googletest//:gtest_main",
    ] + select({
        "//tensorflow:android": [
            "//tensorflow/core:portable_tensorflow_lib_lite",  # TODO(annarev): exclude runtime srcs
        ],
        "//conditions:default": [
            "//tensorflow/cc/saved_model:loader",
            "//tensorflow/core:test",
            "//tensorflow/core/platform:resource_loader",
        ],
    }),
)
