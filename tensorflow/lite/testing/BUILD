package(default_visibility = [
    "//visibility:public",
])

licenses(["notice"])  # Apache 2.0

load(
    "//tensorflow/lite:build_def.bzl",
    "gen_zip_test",
    "generated_test_models_all",
)
load("//tensorflow/lite:special_rules.bzl", "tflite_portable_test_suite")
load(
    "//tensorflow:tensorflow.bzl",
    "tf_cc_test",
    "py_test",
)

[gen_zip_test(
    name = "zip_test_%s" % test_name,
    size = "large",
    srcs = ["generated_examples_zip_test.cc"],
    args = args + select({
        "//tensorflow:android": [],
        "//conditions:default": [
            "--zip_file_path=$(location :zip_%s)" % test_name,
            # TODO(angerson) We may be able to add an external unzip binary instead
            # of relying on an existing one for OSS builds.
            "--unzip_binary_path=/usr/bin/unzip",
        ],
    }),
    conversion_mode = conversion_mode,
    data = [
        ":zip_%s" % test_name,
    ],
    shard_count = 20,
    tags = tags + [
        "gen_zip_test",
        "tflite_not_portable_intentional",
    ],
    test_name = test_name,
    deps = [
        ":parse_testdata_lib",
        ":tflite_driver",
        ":util",
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
            "//tensorflow/core:android_tensorflow_lib",
            "//tensorflow/core:android_tensorflow_test_lib",
        ],
    }),
) for conversion_mode, test_name, tags, args in generated_test_models_all()]

test_suite(
    name = "generated_zip_tests",
    tags = [
        "gen_zip_test",
    ],
)

py_binary(
    name = "generate_examples",
    srcs = ["generate_examples.py"],
    data = [
        "//tensorflow/lite/toco",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":generate_examples_report",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:graph_util",
        "//third_party/py/numpy",
        "@six_archive//:six",
    ],
)

py_library(
    name = "generate_examples_report",
    srcs = ["generate_examples_report.py"],
    srcs_version = "PY2AND3",
)

cc_library(
    name = "parse_testdata_lib",
    srcs = ["parse_testdata.cc"],
    hdrs = ["parse_testdata.h"],
    deps = [
        ":message",
        ":split",
        ":test_runner",
        "//tensorflow/lite:framework",
    ],
)

cc_library(
    name = "message",
    srcs = ["message.cc"],
    hdrs = ["message.h"],
    deps = [":tokenize"],
)

cc_test(
    name = "message_test",
    srcs = ["message_test.cc"],
    deps = [
        ":message",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "split",
    srcs = ["split.cc"],
    hdrs = ["split.h"],
    deps = [
        "//tensorflow/lite:string",
    ],
)

cc_test(
    name = "split_test",
    size = "small",
    srcs = ["split_test.cc"],
    deps = [
        ":split",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "join",
    hdrs = ["join.h"],
    deps = ["//tensorflow/lite:string"],
)

cc_test(
    name = "join_test",
    size = "small",
    srcs = ["join_test.cc"],
    deps = [
        ":join",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "tflite_driver",
    srcs = ["tflite_driver.cc"],
    hdrs = ["tflite_driver.h"],
    deps = [
        ":split",
        ":test_runner",
        "//tensorflow/lite:builtin_op_data",
        "//tensorflow/lite:framework",
        "//tensorflow/lite:string_util",
        "//tensorflow/lite/delegates/flex:delegate",
        "//tensorflow/lite/kernels:builtin_ops",
        "@com_google_absl//absl/strings",
    ],
)

tf_cc_test(
    name = "tflite_driver_test",
    size = "small",
    srcs = ["tflite_driver_test.cc"],
    data = ["//tensorflow/lite:testdata/multi_add.bin"],
    tags = [
        "tflite_not_portable_android",
        "tflite_not_portable_ios",
    ],
    deps = [
        ":tflite_driver",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "tokenize",
    srcs = ["tokenize.cc"],
    hdrs = ["tokenize.h"],
    deps = [
        "//tensorflow/lite:string",
    ],
)

cc_test(
    name = "tokenize_test",
    srcs = ["tokenize_test.cc"],
    deps = [
        ":tokenize",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "test_runner",
    hdrs = ["test_runner.h"],
    deps = [
        "//tensorflow/lite:string",
    ],
)

cc_library(
    name = "util",
    hdrs = ["util.h"],
    deps = [
        "//tensorflow/lite:framework",
        "//tensorflow/lite:string",
        "//tensorflow/lite/core/api",
    ],
)

cc_test(
    name = "test_runner_test",
    srcs = ["test_runner_test.cc"],
    deps = [
        ":test_runner",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_binary(
    name = "nnapi_example",
    srcs = ["nnapi_example.cc"],
    deps = [
        ":parse_testdata_lib",
        ":tflite_driver",
        "//tensorflow/lite/nnapi:nnapi_lib",
    ],
)

cc_library(
    name = "tf_driver",
    srcs = ["tf_driver.cc"],
    hdrs = ["tf_driver.h"],
    deps = [
        ":join",
        ":split",
        ":test_runner",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:tensorflow",
        "//tensorflow/lite:string_util",
        "@com_google_absl//absl/strings",
    ],
)

cc_test(
    name = "tf_driver_test",
    size = "small",
    srcs = ["tf_driver_test.cc"],
    data = ["//tensorflow/lite:testdata/multi_add.pb"],
    tags = [
        "tflite_not_portable",
    ],
    deps = [
        ":tf_driver",
        "//tensorflow/lite:string_util",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "generate_testspec",
    srcs = ["generate_testspec.cc"],
    hdrs = ["generate_testspec.h"],
    deps = [
        ":join",
        ":split",
        ":tf_driver",
        "//tensorflow/core:framework",
        "//tensorflow/lite:string",
    ],
)

cc_test(
    name = "generate_testspec_test",
    size = "small",
    srcs = ["generate_testspec_test.cc"],
    tags = [
        "tflite_not_portable",
    ],
    deps = [
        ":generate_testspec",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "init_tensorflow",
    srcs = [
        "init_tensorflow.cc",
    ],
    hdrs = [
        "init_tensorflow.h",
    ],
    visibility = [
        "//tensorflow/lite/java/src/main/native:__subpackages__",
        "//tensorflow/lite/testing:__subpackages__",
        "//tensorflow/lite/tools/benchmark:__subpackages__",
    ],
    deps = select({
        "//conditions:default": [
            "//tensorflow/core:lib",
        ],
        "//tensorflow:android": [
            "//tensorflow/core:android_tensorflow_lib",
        ],
    }),
)

cc_library(
    name = "tflite_diff_util",
    srcs = ["tflite_diff_util.cc"],
    hdrs = ["tflite_diff_util.h"],
    deps = [
        ":generate_testspec",
        ":parse_testdata_lib",
        ":tflite_driver",
        "//tensorflow/lite:framework",
        "//tensorflow/lite:string",
    ],
)

cc_library(
    name = "tflite_diff_flags",
    hdrs = ["tflite_diff_flags.h"],
    deps = [
        ":split",
        ":tflite_diff_util",
    ] + select({
        "//conditions:default": [
            "//tensorflow/core:framework_internal",
            "//tensorflow/core:lib",
        ],
        "//tensorflow:android": [
            "//tensorflow/core:android_tensorflow_lib",
        ],
    }),
)

tf_cc_test(
    name = "tflite_diff_example_test",
    size = "medium",
    srcs = ["tflite_diff_example_test.cc"],
    args = [
        "--tensorflow_model=third_party/tensorflow/lite/testdata/multi_add.pb",
        "--tflite_model=third_party/tensorflow/lite/testdata/multi_add.bin",
        "--input_layer=a,b,c,d",
        "--input_layer_type=float,float,float,float",
        "--input_layer_shape=1,3,4,3:1,3,4,3:1,3,4,3:1,3,4,3",
        "--output_layer=x,y",
    ],
    data = [
        "//tensorflow/lite:testdata/multi_add.bin",
        "//tensorflow/lite:testdata/multi_add.pb",
    ],
    tags = [
        "no_cuda_on_cpu_tap",
        "no_oss",  # needs test data
        "tflite_not_portable",
    ],
    deps = [
        ":init_tensorflow",
        ":tflite_diff_flags",
        ":tflite_diff_util",
    ],
)

cc_binary(
    name = "tflite_diff",
    srcs = ["tflite_diff_example_test.cc"],
    deps = [
        ":init_tensorflow",
        ":tflite_diff_flags",
        ":tflite_diff_util",
    ],
)

tflite_portable_test_suite()
