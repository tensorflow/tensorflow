load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("//tensorflow:strict.default.bzl", "py_strict_binary", "py_strict_library")
load("//tensorflow:tensorflow.bzl", "tf_cc_binary", "tf_cc_test")
load("//tensorflow:tensorflow.default.bzl", "tf_py_strict_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [
        "//visibility:public",
    ],
    licenses = ["notice"],
)

py_strict_binary(
    name = "accuracy_utils_py",
    srcs = ["accuracy_utils.py"],
    main = "accuracy_utils.py",
    deps = [
        "//tensorflow:tensorflow_py",
        "//third_party/py/numpy",
    ],
)

py_strict_library(
    name = "accuracy_utils_lib",
    srcs = ["accuracy_utils.py"],
    deps = [
        "//tensorflow:tensorflow_py",
        "//third_party/py/numpy",
    ],
)

py_strict_binary(
    name = "recognize_commands_py",
    srcs = ["recognize_commands.py"],
    main = "recognize_commands.py",
    deps = ["//third_party/py/numpy"],
)

py_strict_library(
    name = "recognize_commands_lib",
    srcs = ["recognize_commands.py"],
    deps = ["//third_party/py/numpy"],
)

py_strict_binary(
    name = "test_streaming_accuracy_py",
    srcs = ["test_streaming_accuracy.py"],
    main = "test_streaming_accuracy.py",
    deps = [
        ":accuracy_utils_lib",
        ":recognize_commands_lib",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python/ops:io_ops",
        "//third_party/py/numpy",
    ],
)

py_strict_library(
    name = "models",
    srcs = [
        "models.py",
    ],
    deps = ["//tensorflow:tensorflow_py"],
)

tf_py_strict_test(
    name = "models_test",
    size = "medium",
    srcs = ["models_test.py"],
    deps = [
        ":models",
        "//tensorflow:tensorflow_py_no_contrib",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/platform:client_testlib",
    ],
)

py_strict_library(
    name = "input_data",
    srcs = [
        "input_data.py",
    ],
    deps = [
        "//tensorflow:tensorflow_py",
        "//tensorflow/python/ops:audio_ops_gen",
        "//tensorflow/python/ops:io_ops",
        "//tensorflow/python/platform:gfile",
        "//tensorflow/python/util:compat",
        "//third_party/py/numpy",
    ],
)

tf_py_strict_test(
    name = "input_data_test",
    size = "small",
    srcs = ["input_data_test.py"],
    tags = [
        "v1only",  # uses contrib
    ],
    deps = [
        ":input_data",
        ":models",
        "//tensorflow:tensorflow_py_no_contrib",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/platform:client_testlib",
    ],
)

py_strict_binary(
    name = "train",
    srcs = ["train.py"],
    deps = [":train_main_lib"],
)

py_strict_library(
    name = "train_main_lib",
    srcs = [
        "train.py",
    ],
    deps = [
        ":input_data",
        ":models",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python/platform:gfile",
        "//third_party/py/numpy",
    ],
)

tf_py_strict_test(
    name = "train_test",
    size = "small",
    srcs = ["train_test.py"],
    tags = [
        "v1only",  # uses contrib
    ],
    deps = [
        ":train_main_lib",
        "//tensorflow:tensorflow_py_no_contrib",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/platform:client_testlib",
        "//tensorflow/python/platform:gfile",
    ],
)

py_strict_binary(
    name = "freeze",
    srcs = ["freeze.py"],
    deps = [":freeze_main_lib"],
)

py_strict_library(
    name = "freeze_main_lib",
    srcs = ["freeze.py"],
    deps = [
        ":input_data",
        ":models",
        "//tensorflow:tensorflow_py_no_contrib",
        "//tensorflow/python/ops:audio_ops_gen",
    ],
)

py_strict_library(
    name = "freeze_lib",
    srcs = [
        "freeze.py",
    ],
    tags = [
        "no_pip",  # b/131330719
    ],
    deps = [
        ":input_data",
        ":models",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python/ops:audio_ops_gen",
    ],
)

tf_py_strict_test(
    name = "freeze_test",
    size = "small",
    srcs = ["freeze_test.py"],
    tags = [
        "v1only",  # uses contrib
    ],
    deps = [
        ":freeze_main_lib",
        "//tensorflow/python/framework:convert_to_constants",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/ops:variables",
        "//tensorflow/python/platform:client_testlib",
    ],
)

py_strict_binary(
    name = "wav_to_features",
    srcs = ["wav_to_features.py"],
    deps = [":wav_to_features_main_lib"],
)

py_strict_library(
    name = "wav_to_features_main_lib",
    srcs = ["wav_to_features.py"],
    deps = [
        ":input_data",
        ":models",
        "//tensorflow:tensorflow_py_no_contrib",
        "//tensorflow/python/platform:gfile",
    ],
)

py_strict_library(
    name = "wav_to_features_lib",
    srcs = [
        "wav_to_features.py",
    ],
    deps = [
        ":input_data",
        ":models",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python/platform:gfile",
    ],
)

tf_py_strict_test(
    name = "wav_to_features_test",
    size = "small",
    srcs = ["wav_to_features_test.py"],
    tags = [
        "v1only",  # uses contrib
    ],
    deps = [
        ":wav_to_features_main_lib",
        "//tensorflow:tensorflow_py_no_contrib",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/platform:client_testlib",
    ],
)

py_strict_binary(
    name = "generate_streaming_test_wav",
    srcs = ["generate_streaming_test_wav.py"],
    deps = [":generate_streaming_test_wav_main_lib"],
)

py_strict_library(
    name = "generate_streaming_test_wav_main_lib",
    srcs = ["generate_streaming_test_wav.py"],
    deps = [
        ":input_data",
        ":models",
        "//tensorflow:tensorflow_py_no_contrib",
        "//third_party/py/numpy",
    ],
)

py_strict_library(
    name = "generate_streaming_test_wav_lib",
    srcs = [
        "generate_streaming_test_wav.py",
    ],
    deps = [
        ":input_data",
        ":models",
        "//tensorflow:tensorflow_py",
        "//third_party/py/numpy",
    ],
)

tf_py_strict_test(
    name = "generate_streaming_test_wav_test",
    size = "small",
    srcs = ["generate_streaming_test_wav_test.py"],
    tags = [
        "v1only",  # uses contrib
    ],
    deps = [
        ":generate_streaming_test_wav_main_lib",
        "//tensorflow/python/platform:client_testlib",
    ],
)

tf_cc_binary(
    name = "label_wav_cc",
    srcs = [
        "label_wav.cc",
    ],
    deps = [
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:tensorflow",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@local_xla//xla/tsl/platform:status",
        "@local_xla//xla/tsl/platform:types",
        "@local_xla//xla/tsl/util:command_line_flags",
    ],
)

py_strict_binary(
    name = "label_wav",
    srcs = ["label_wav.py"],
    deps = [":label_wav_main_lib"],
)

py_strict_library(
    name = "label_wav_main_lib",
    srcs = ["label_wav.py"],
    deps = ["//tensorflow:tensorflow_py_no_contrib"],
)

py_strict_library(
    name = "label_wav_lib",
    srcs = [
        "label_wav.py",
    ],
    deps = [
        "//tensorflow:tensorflow_py",
    ],
)

tf_py_strict_test(
    name = "label_wav_test",
    size = "medium",
    srcs = ["label_wav_test.py"],
    tags = [
        "v1only",  # uses contrib
    ],
    deps = [
        ":label_wav_main_lib",
        "//tensorflow:tensorflow_py_no_contrib",
        "//tensorflow/python/platform:client_testlib",
    ],
)

cc_library(
    name = "recognize_commands",
    srcs = [
        "recognize_commands.cc",
    ],
    hdrs = [
        "recognize_commands.h",
    ],
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core:tensorflow",
        "@com_google_absl//absl/status",
    ],
)

tf_cc_test(
    name = "recognize_commands_test",
    size = "medium",
    srcs = [
        "recognize_commands_test.cc",
    ],
    deps = [
        ":recognize_commands",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
    ],
)

cc_library(
    name = "accuracy_utils",
    srcs = [
        "accuracy_utils.cc",
    ],
    hdrs = [
        "accuracy_utils.h",
    ],
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core:tensorflow",
        "//tensorflow/core/platform:numbers",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
    ],
)

tf_cc_test(
    name = "accuracy_utils_test",
    size = "medium",
    srcs = [
        "accuracy_utils_test.cc",
    ],
    deps = [
        ":accuracy_utils",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
    ],
)

tf_cc_binary(
    name = "test_streaming_accuracy",
    srcs = [
        "test_streaming_accuracy.cc",
    ],
    deps = [
        ":accuracy_utils",
        ":recognize_commands",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:protos_all_cc",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@local_xla//xla/tsl/platform:types",
    ],
)

py_strict_library(
    name = "test_lib",
    deps = [
        ":freeze_main_lib",
        ":generate_streaming_test_wav_main_lib",
        ":input_data",
        ":label_wav_main_lib",
        ":models",
        ":train_main_lib",
        ":wav_to_features_main_lib",
    ],
)
