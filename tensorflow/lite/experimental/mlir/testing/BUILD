load(
    ":build_def.bzl",
    "gen_zip_test",
    "generated_test_models_all",
)

package(
    default_visibility = [
        "//visibility:public",
    ],
    licenses = ["notice"],
)

py_library(
    name = "mlir_convert",
    srcs = ["mlir_convert.py"],
    data = [
        "//tensorflow/compiler/mlir/lite:tf_tfl_translate",
    ],
    srcs_version = "PY3",
    deps = [
        "//tensorflow:tensorflow_py",
        "//tensorflow/lite/python:test_util",
        "//tensorflow/lite/testing:_pywrap_string_util",
        "//tensorflow/lite/testing:generate_examples_lib",
        "//tensorflow/lite/testing:generate_examples_report",
        "//tensorflow/python:graph_util",
        "//third_party/py/numpy",
        "@six_archive//:six",
    ],
)

[gen_zip_test(
    name = "zip_test_%s" % test_name,
    size = "medium",
    srcs = ["generated_examples_zip_test.cc"],
    args = args + select({
        "//tensorflow:android": [],
        "//conditions:default": [
            "--zip_file_path=$(location :zip_%s)" % test_name,
            # TODO(angerson) We may be able to add an external unzip binary instead
            # of relying on an existing one for OSS builds.
            #"--unzip_binary_path=$(location //third_party/unzip)",
        ],
    }),
    conversion_mode = conversion_mode,
    # copybara:uncomment_begin(no special handling for Android in OSS)
    # data = select({
    # "//tensorflow:android": [],
    # "//conditions:default": [
    # ":zip_%s" % test_name,
    # "//third_party/unzip",
    # ],
    # }),
    # copybara:uncomment_end_and_comment_begin
    data = [":zip_%s" % test_name],
    # copybara:comment_end
    shard_count = 20,
    tags = tags + [
        "gen_zip_test",
        "tflite_not_portable_intentional",
    ],
    test_name = test_name,
    deps = [
        "//tensorflow/lite/testing:parse_testdata_lib",
        "//tensorflow/lite/testing:tflite_driver",
        "//tensorflow/lite/testing:util",
        "@com_google_googletest//:gtest",
        "@com_googlesource_code_re2//:re2",
        "//tensorflow/lite:builtin_op_data",
        "//tensorflow/lite:framework",
        "//tensorflow/lite/kernels:builtin_ops",
    ] + select({
        "//conditions:default": [
            "//tensorflow/core:framework_internal",
            "//tensorflow/core:lib",
            "//tensorflow/core:test",
        ],
        "//tensorflow:android": [
            "//tensorflow/core:portable_tensorflow_lib",
            "//tensorflow/core:portable_tensorflow_test_lib",
        ],
    }),
) for conversion_mode, test_name, tags, args in generated_test_models_all()]

py_library(
    name = "op_tests",
    srcs = glob(["op_tests/*.py"]),
    srcs_version = "PY3",
    deps = [
        "//third_party/py/numpy",
        "//tensorflow:tensorflow_py",
        # copybara:uncomment_begin(b/186563810)
        # "//third_party/py/tensorflow_addons",
        # copybara:uncomment_end
        "//tensorflow/lite/testing:zip_test_utils",
    ],
)

py_binary(
    name = "generate_examples",
    srcs = ["generate_examples.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":mlir_convert",
        ":op_tests",
        "//tensorflow:tensorflow_py",
        "//tensorflow/lite/testing:generate_examples_lib",
        "//tensorflow/lite/testing:zip_test_utils",
        "//third_party/py/numpy",
        "@six_archive//:six",
    ],
)
