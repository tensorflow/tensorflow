package(default_visibility = [
    "//visibility:public",
])

licenses(["notice"])  # Apache 2.0

load("//tensorflow:tensorflow.bzl", "tf_cc_binary", "tf_cc_test")

cc_library(
    name = "inception_preprocessing",
    srcs = ["inception_preprocessing.cc"],
    hdrs = ["inception_preprocessing.h"],
    copts = [
        "-D__ANDROID_TYPES_FULL__",
        "-DSUPPORT_SELECTIVE_REGISTRATION",
    ],
    deps = [
        ":stage",
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
    deps = [
        ":inception_preprocessing",
        ":android_required_build_flags",
        "@com_google_googletest//:gtest",
    ] + select(
        {
            "//tensorflow:android": [
                "//tensorflow/core:android_tensorflow_lib",
                "//tensorflow/core:android_test_lib",
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
    name = "utils",
    srcs = ["utils.cc"],
    hdrs = ["utils.h"],
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
    deps = [
        ":utils",
        "@com_google_googletest//:gtest",
    ] + select(
        {
            "//tensorflow:android": [
                "//tensorflow/core:android_tensorflow_lib",
                "//tensorflow/core:android_test_lib",
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
)

tf_cc_test(
    name = "run_tflite_model_op_test",
    srcs = ["run_tflite_model_op_test.cc"],
    args = [
        "--test_model_file=$(location //tensorflow/contrib/lite:testdata/multi_add.bin)",
    ],
    data = ["//tensorflow/contrib/lite:testdata/multi_add.bin"],
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
                "//tensorflow/core:android_test_lib",
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
    deps = [
        ":file_reader_stage",
        "@com_google_googletest//:gtest",
    ] + select(
        {
            "//tensorflow:android": [
                "//tensorflow/core:android_tensorflow_lib",
                "//tensorflow/core/kernels:android_whole_file_read_ops",
                "//tensorflow/core:android_test_lib",
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
    name = "imagenet_topk_eval",
    srcs = ["imagenet_topk_eval.cc"],
    hdrs = ["imagenet_topk_eval.h"],
    deps = [
        ":accuracy_eval_stage",
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
    name = "imagenet_topk_eval_test",
    srcs = ["imagenet_topk_eval_test.cc"],
    deps = [
        ":imagenet_topk_eval",
        "@com_google_googletest//:gtest",
    ] + select(
        {
            "//tensorflow:android": [
                "//tensorflow/core:android_tensorflow_lib",
                "//tensorflow/core:android_test_lib",
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
    deps = [
        ":eval_pipeline",
        "//tensorflow/cc:cc_ops",
        "@com_google_googletest//:gtest",
    ] + select(
        {
            "//tensorflow:android": [
                "//tensorflow/core:android_tensorflow_lib",
                "//tensorflow/core:android_test_lib",
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
    deps = [
        ":eval_pipeline_builder",
        "//tensorflow/cc:cc_ops",
        "@com_google_googletest//:gtest",
    ] + select(
        {
            "//tensorflow:android": [
                "//tensorflow/core:android_tensorflow_lib",
                "//tensorflow/core:android_test_lib",
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

cc_library(
    name = "imagenet_model_evaluator",
    srcs = ["imagenet_model_evaluator.cc"],
    hdrs = ["imagenet_model_evaluator.h"],
    deps = [
        ":android_required_build_flags",
        ":eval_pipeline",
        ":eval_pipeline_builder",
        ":file_reader_stage",
        ":imagenet_topk_eval",
        ":inception_preprocessing",
        ":run_tflite_model_stage",
        ":utils",
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
    deps = [
        ":android_required_build_flags",
        ":csv_writer",
        ":imagenet_model_evaluator",
        ":imagenet_topk_eval",
        "@com_google_absl//absl/memory",
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
