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
    name = "inception_preprocessing",
    srcs = ["inception_preprocessing.cc"],
    hdrs = ["inception_preprocessing.h"],
    copts = tflite_copts(),
    deps = [
        "//tensorflow/contrib/lite/tools/accuracy:android_required_build_flags",
        "//tensorflow/contrib/lite/tools/accuracy:stage",
        "//tensorflow/cc:cc_ops",
        "//tensorflow/cc:scope",
    ] + select(
        {
            "//tensorflow:android": [
                "//tensorflow/core:android_tensorflow_lib",
                "//tensorflow/core/kernels:android_tensorflow_image_op",
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
)

tf_cc_test(
    name = "inception_preprocessing_test",
    srcs = ["inception_preprocessing_test.cc"],
    args = [
        "--test_image=$(location :testdata/grace_hopper.jpg)",
    ],
    data = [":testdata/grace_hopper.jpg"],
    linkopts = common_linkopts,
    linkstatic = 1,
    tags = [
        "no_oss",  # b/114307765
        "tflite_not_portable_android",
        "tflite_not_portable_ios",
    ],
    deps = [
        ":inception_preprocessing",
        "//tensorflow/contrib/lite/tools/accuracy:android_required_build_flags",
        "@com_google_googletest//:gtest",
    ] + select(
        {
            "//tensorflow:android": [
                "//tensorflow/core:android_tensorflow_lib",
                "//tensorflow/core:android_tensorflow_test_lib",
            ],
            "//conditions:default": [
                "//tensorflow/core:core_cpu",
                "//tensorflow/core:framework_internal",
                "//tensorflow/core:lib",
            ],
        },
    ),
)

cc_library(
    name = "imagenet_topk_eval",
    srcs = ["imagenet_topk_eval.cc"],
    hdrs = ["imagenet_topk_eval.h"],
    copts = tflite_copts(),
    deps = [
        "//tensorflow/contrib/lite/tools/accuracy:accuracy_eval_stage",
    ] + select(
        {
            "//tensorflow:android": [
                "//tensorflow/core:android_tensorflow_lib",
            ],
            "//conditions:default": [
                "//tensorflow/core:framework",
                "//tensorflow/core:lib",
            ],
        },
    ),
)

tf_cc_test(
    name = "imagenet_topk_eval_test",
    srcs = ["imagenet_topk_eval_test.cc"],
    linkopts = common_linkopts,
    linkstatic = 1,
    tags = ["tflite_not_portable_ios"],
    deps = [
        ":imagenet_topk_eval",
        "@com_google_googletest//:gtest",
    ] + select(
        {
            "//tensorflow:android": [
                "//tensorflow/core:android_tensorflow_lib",
                "//tensorflow/core:android_tensorflow_test_lib",
            ],
            "//conditions:default": [
                "//tensorflow/core:framework",
            ],
        },
    ),
)

cc_library(
    name = "imagenet_model_evaluator",
    srcs = ["imagenet_model_evaluator.cc"],
    hdrs = ["imagenet_model_evaluator.h"],
    copts = tflite_copts(),
    deps = [
        ":imagenet_topk_eval",
        ":inception_preprocessing",
        "//tensorflow/contrib/lite/tools/accuracy:android_required_build_flags",
        "//tensorflow/contrib/lite/tools/accuracy:eval_pipeline",
        "//tensorflow/contrib/lite/tools/accuracy:eval_pipeline_builder",
        "//tensorflow/contrib/lite/tools/accuracy:file_reader_stage",
        "//tensorflow/contrib/lite/tools/accuracy:run_tflite_model_stage",
        "//tensorflow/contrib/lite/tools/accuracy:utils",
        "@com_google_absl//absl/memory",
        "//tensorflow/cc:cc_ops",
        "//tensorflow/cc:scope",
    ] + select(
        {
            "//tensorflow:android": [
                "//tensorflow/core:android_tensorflow_lib",
                "//tensorflow/core/kernels:android_whole_file_read_ops",
                "//tensorflow/core/kernels:android_tensorflow_image_op",
            ],
            "//conditions:default": [
                "//tensorflow/core:tensorflow",
                "//tensorflow/core:lib_internal",
                "//tensorflow/core:framework_internal",
                "//tensorflow/core:framework",
                "//tensorflow/core:lib",
                "//tensorflow/core:core_cpu",
            ],
        },
    ),
)

tf_cc_binary(
    name = "imagenet_accuracy_eval",
    srcs = ["imagenet_accuracy_eval.cc"],
    copts = tflite_copts(),
    linkopts = common_linkopts,
    deps = [
        ":imagenet_model_evaluator",
        ":imagenet_topk_eval",
        "@com_google_absl//absl/memory",
        "//tensorflow/contrib/lite/tools/accuracy:android_required_build_flags",
        "//tensorflow/contrib/lite/tools/accuracy:csv_writer",
    ] + select(
        {
            "//tensorflow:android": [
                "//tensorflow/core:android_tensorflow_lib",
            ],
            "//conditions:default": [
                "//tensorflow/core:lib",
                "//tensorflow/core:framework_internal",
            ],
        },
    ),
)

tflite_portable_test_suite()
