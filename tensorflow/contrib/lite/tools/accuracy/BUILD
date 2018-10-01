package(default_visibility = [
    "//visibility:public",
])

licenses(["notice"])  # Apache 2.0

load("//tensorflow:tensorflow.bzl", "tf_cc_binary", "tf_cc_test")
load("//tensorflow/contrib/lite:build_def.bzl", "tflite_copts", "tflite_linkopts")
load("//tensorflow/contrib/lite:special_rules.bzl", "tflite_portable_test_suite")

common_linkopts = tflite_linkopts() + select({
    "//conditions:default": [],
    "//tensorflow:android": [
        "-pie",
        "-llog",
    ],
})

cc_library(
    name = "utils",
    srcs = ["utils.cc"],
    hdrs = ["utils.h"],
    copts = tflite_copts(),
    deps = [
        "//tensorflow/contrib/lite:framework",
        "//tensorflow/contrib/lite/kernels:builtin_ops",
    ] + select(
        {
            "//tensorflow:android": [
                "//tensorflow/core:android_tensorflow_lib",
            ],
            "//conditions:default": [
                "//tensorflow/core:framework",
            ],
        },
    ),
)

tf_cc_test(
    name = "utils_test",
    srcs = ["utils_test.cc"],
    args = [
        "--test_model_file=$(location //tensorflow/contrib/lite:testdata/multi_add.bin)",
    ],
    data = ["//tensorflow/contrib/lite:testdata/multi_add.bin"],
    linkopts = common_linkopts,
    linkstatic = 1,
    tags = [
        "tflite_not_portable_android",
        "tflite_not_portable_ios",
    ],
    deps = [
        ":utils",
        "@com_google_googletest//:gtest",
    ] + select(
        {
            "//tensorflow:android": [
                "//tensorflow/core:android_tensorflow_lib",
                "//tensorflow/core:android_tensorflow_test_lib",
            ],
            "//conditions:default": [
                "//tensorflow/core:framework_internal",
                "//tensorflow/core:lib",
            ],
        },
    ),
)

cc_library(
    name = "run_tflite_model_op",
    srcs = ["run_tflite_model_op.cc"],
    copts = tflite_copts(),
    deps = [
        ":utils",
        "//tensorflow/contrib/lite:framework",
        "//tensorflow/contrib/lite/kernels:builtin_ops",
    ] + select(
        {
            "//tensorflow:android": [
                "//tensorflow/core:android_tensorflow_lib",
            ],
            "//conditions:default": [
                "//tensorflow/core:tensorflow",
                "//tensorflow/core:protos_all_cc",
                "//tensorflow/core:core_cpu",
                "//tensorflow/core:framework",
                "//tensorflow/core:lib",
                "//tensorflow/core:ops",
            ],
        },
    ),
    alwayslink = 1,
)

cc_library(
    name = "android_required_build_flags",
    srcs = ["android_required_build_flags.cc"],
    copts = tflite_copts(),
)

tf_cc_test(
    name = "run_tflite_model_op_test",
    srcs = ["run_tflite_model_op_test.cc"],
    args = [
        "--test_model_file=$(location //tensorflow/contrib/lite:testdata/multi_add.bin)",
    ],
    data = ["//tensorflow/contrib/lite:testdata/multi_add.bin"],
    linkopts = common_linkopts,
    linkstatic = 1,
    tags = [
        "tflite_not_portable_android",
        "tflite_not_portable_ios",
    ],
    deps = [
        "//tensorflow/cc:cc_ops",
        "//tensorflow/cc:scope",
        ":run_tflite_model_op",
        ":android_required_build_flags",
        "@com_google_googletest//:gtest",
    ] + select(
        {
            "//tensorflow:android": [
                "//tensorflow/core:android_tensorflow_lib",
                "//tensorflow/core:android_tensorflow_test_lib",
            ],
            "//conditions:default": [
                "//tensorflow/core:core_cpu",
                "//tensorflow/core:framework",
                "//tensorflow/core:framework_internal",
                "//tensorflow/core:lib",
                "//tensorflow/core:ops",
                "//tensorflow/core:protos_all_cc",
                "//tensorflow/core:tensorflow",
            ],
        },
    ),
)

cc_library(
    name = "stage",
    hdrs = ["stage.h"],
    copts = tflite_copts(),
    deps = [
        "//tensorflow/cc:scope",
    ],
)

cc_library(
    name = "file_reader_stage",
    srcs = ["file_reader_stage.cc"],
    hdrs = ["file_reader_stage.h"],
    deps = [
        ":stage",
        "//tensorflow/cc:cc_ops",
        "//tensorflow/cc:scope",
    ],
)

tf_cc_test(
    name = "file_reader_stage_test",
    srcs = ["file_reader_stage_test.cc"],
    linkopts = common_linkopts,
    linkstatic = 1,
    tags = ["tflite_not_portable_ios"],
    deps = [
        ":file_reader_stage",
        "@com_google_googletest//:gtest",
    ] + select(
        {
            "//tensorflow:android": [
                "//tensorflow/core:android_tensorflow_lib",
                "//tensorflow/core/kernels:android_whole_file_read_ops",
                "//tensorflow/core:android_tensorflow_test_lib",
            ],
            "//conditions:default": [
                "//tensorflow/core:core_cpu",
                "//tensorflow/core:tensorflow",
            ],
        },
    ),
)

cc_library(
    name = "run_tflite_model_stage",
    srcs = ["run_tflite_model_stage.cc"],
    hdrs = ["run_tflite_model_stage.h"],
    copts = tflite_copts(),
    deps = [
        ":run_tflite_model_op",
        ":stage",
        "//tensorflow/cc:cc_ops",
        "//tensorflow/cc:scope",
    ],
)

cc_library(
    name = "accuracy_eval_stage",
    hdrs = ["accuracy_eval_stage.h"],
    copts = tflite_copts(),
    deps = [
    ] + select(
        {
            "//tensorflow:android": [
                "//tensorflow/core:android_tensorflow_lib",
            ],
            "//conditions:default": [
                "//tensorflow/core:framework",
            ],
        },
    ),
)

cc_library(
    name = "eval_pipeline",
    srcs = ["eval_pipeline.cc"],
    hdrs = ["eval_pipeline.h"],
    copts = tflite_copts(),
    deps = [
        ":accuracy_eval_stage",
        ":stage",
    ] + select(
        {
            "//tensorflow:android": [
                "//tensorflow/core:android_tensorflow_lib",
            ],
            "//conditions:default": [
                "//tensorflow/core:framework",
                "//tensorflow/core:core_cpu",
            ],
        },
    ),
)

tf_cc_test(
    name = "eval_pipeline_test",
    srcs = ["eval_pipeline_test.cc"],
    linkopts = common_linkopts,
    linkstatic = 1,
    tags = ["tflite_not_portable_ios"],
    deps = [
        ":eval_pipeline",
        "//tensorflow/cc:cc_ops",
        "@com_google_googletest//:gtest",
    ] + select(
        {
            "//tensorflow:android": [
                "//tensorflow/core:android_tensorflow_lib",
                "//tensorflow/core:android_tensorflow_test_lib",
            ],
            "//conditions:default": [
                "//tensorflow/core:framework",
                "//tensorflow/core:core_cpu",
                "//tensorflow/core:ops",
                "//tensorflow/core:tensorflow",
            ],
        },
    ),
)

cc_library(
    name = "eval_pipeline_builder",
    srcs = ["eval_pipeline_builder.cc"],
    hdrs = ["eval_pipeline_builder.h"],
    copts = tflite_copts(),
    deps = [
        ":eval_pipeline",
        ":accuracy_eval_stage",
        ":stage",
        "@com_google_absl//absl/memory",
        "//tensorflow/cc:cc_ops",
    ] + select(
        {
            "//tensorflow:android": [
                "//tensorflow/core:android_tensorflow_lib",
            ],
            "//conditions:default": [
                "//tensorflow/core:framework",
                "//tensorflow/core:core_cpu",
                "//tensorflow/core:ops",
                "//tensorflow/core:tensorflow",
            ],
        },
    ),
)

tf_cc_test(
    name = "eval_pipeline_builder_test",
    srcs = ["eval_pipeline_builder_test.cc"],
    linkopts = common_linkopts,
    linkstatic = 1,
    tags = ["tflite_not_portable_ios"],
    deps = [
        ":eval_pipeline_builder",
        "//tensorflow/cc:cc_ops",
        "@com_google_googletest//:gtest",
    ] + select(
        {
            "//tensorflow:android": [
                "//tensorflow/core:android_tensorflow_lib",
                "//tensorflow/core:android_tensorflow_test_lib",
            ],
            "//conditions:default": [
                "//tensorflow/core:framework",
                "//tensorflow/core:core_cpu",
                "//tensorflow/core:ops",
                "//tensorflow/core:tensorflow",
            ],
        },
    ),
)

cc_library(
    name = "csv_writer",
    hdrs = ["csv_writer.h"],
    copts = tflite_copts(),
    deps = select(
        {
            "//tensorflow:android": [
                "//tensorflow/core:android_tensorflow_lib",
            ],
            "//conditions:default": [
                "//tensorflow/core:lib",
            ],
        },
    ),
)

tflite_portable_test_suite()
