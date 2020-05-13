load("@flatbuffers//:build_defs.bzl", "flatbuffer_py_library")

package(
    default_visibility = ["//tensorflow:internal"],
    licenses = ["notice"],  # Apache 2.0
)

exports_files(["tflite_convert.py"])

flatbuffer_py_library(
    name = "schema_py",
    srcs = ["//tensorflow/lite/schema:schema.fbs"],
)

py_library(
    name = "interpreter",
    srcs = [
        "interpreter.py",
    ],
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/lite/python/interpreter_wrapper:_pywrap_tensorflow_interpreter_wrapper",
        "//tensorflow/python:util",
        "//third_party/py/numpy",
    ],
)

py_test(
    name = "interpreter_test",
    srcs = ["interpreter_test.py"],
    data = [
        "//tensorflow/lite:testdata/sparse_tensor.bin",
        "//tensorflow/lite/python/testdata:interpreter_test_data",
        "//tensorflow/lite/python/testdata:test_delegate.so",
    ],
    python_version = "PY3",
    srcs_version = "PY2AND3",
    tags = [
        "no_windows",
        "noasan",  # TODO(b/137568139): enable after this is fixed.
        "nomsan",  # TODO(b/137568139): enable after this is fixed.
        "notsan",  # TODO(b/149882556): enable after this is fixed.
    ],
    deps = [
        ":interpreter",
        "//tensorflow/lite/python/testdata:_pywrap_test_registerer",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform",
        "//third_party/py/numpy",
        "@six_archive//:six",
    ],
)

py_binary(
    name = "tflite_convert",
    srcs = ["tflite_convert.py"],
    python_version = "PY3",
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
    deps = [
        ":tflite_convert_main_lib",
        "@six_archive//:six",
    ],
)

py_library(
    name = "tflite_convert_main_lib",
    srcs = ["tflite_convert.py"],
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
    deps = [
        ":tflite_convert_lib",
        "@six_archive//:six",
    ],
)

py_library(
    name = "tflite_convert_lib",
    srcs = ["tflite_convert.py"],
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
    deps = [
        ":lite",
        "//tensorflow/lite/toco/logging:gen_html",
        "//tensorflow/lite/toco/logging:toco_conversion_log_proto_py",
        "@six_archive//:six",
    ],
)

py_test(
    name = "tflite_convert_test",
    srcs = ["tflite_convert_test.py"],
    data = [
        ":tflite_convert.par",
        "@tflite_mobilenet_ssd_quant_protobuf//:tflite_graph.pb",
    ],
    python_version = "PY3",
    srcs_version = "PY2AND3",
    tags = [
        "no_oss",
        "no_pip",
        "no_windows",
        "noasan",  # b/144707533
    ],
    deps = [
        ":tflite_convert",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_test_lib",
    ],
)

py_library(
    name = "lite",
    srcs = ["lite.py"],
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
    deps = [
        ":convert",
        ":convert_saved_model",
        ":interpreter",
        ":lite_constants",
        ":op_hint",
        ":util",
        "//tensorflow/lite/experimental/examples/lstm:tflite_lstm_ops",
        "//tensorflow/lite/experimental/microfrontend:audio_microfrontend_py",
        "//tensorflow/lite/experimental/tensorboard:ops_util",
        "//tensorflow/lite/python/optimize:calibrator",
        "//tensorflow/python:graph_util",
        "//tensorflow/python/keras",
        "//tensorflow/python/saved_model:constants",
        "//tensorflow/python/saved_model:loader",
        "@six_archive//:six",
    ],
)

py_test(
    name = "lite_test",
    srcs = ["lite_test.py"],
    data = ["@tflite_mobilenet_ssd_quant_protobuf//:tflite_graph.pb"],
    python_version = "PY3",
    shard_count = 4,
    srcs_version = "PY2AND3",
    tags = [
        "no_windows",
    ],
    deps = [
        ":lite",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_test_lib",
        "@six_archive//:six",
    ],
)

py_test(
    name = "lite_v2_test",
    srcs = ["lite_v2_test.py"],
    python_version = "PY3",
    shard_count = 4,
    srcs_version = "PY2AND3",
    tags = [
        "no_windows",
    ],
    deps = [
        ":lite",
        ":lite_v2_test_util",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_test_lib",
        "@six_archive//:six",
    ],
)

py_library(
    name = "lite_v2_test_util",
    testonly = 1,
    srcs = ["lite_v2_test_util.py"],
    srcs_version = "PY2AND3",
    tags = [
        "no_windows",
    ],
    deps = [
        ":lite",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_test_lib",
        "@six_archive//:six",
    ],
)

py_test(
    name = "lite_flex_test",
    srcs = ["lite_flex_test.py"],
    python_version = "PY3",
    srcs_version = "PY2AND3",
    tags = [
        # TODO(b/111881877): Enable in oss after resolving op registry issues.
        "no_oss",
        "no_windows",
    ],
    deps = [
        ":lite",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_test_lib",
    ],
)

py_library(
    name = "util",
    srcs = ["util.py"],
    srcs_version = "PY2AND3",
    visibility = [
        "//tensorflow/lite:__subpackages__",
    ],
    deps = [
        ":lite_constants",
        ":op_hint",
        "//tensorflow/python:tf_optimizer",
        "//tensorflow/python/eager:wrap_function",
        "@six_archive//:six",
    ],
)

py_test(
    name = "util_test",
    srcs = ["util_test.py"],
    python_version = "PY3",
    srcs_version = "PY2AND3",
    tags = [
        "no_windows",
    ],
    deps = [
        ":util",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_test_lib",
        "@six_archive//:six",
    ],
)

py_library(
    name = "wrap_toco",
    srcs = [
        "wrap_toco.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/python:_pywrap_toco_api",
        "//tensorflow/python:pywrap_tensorflow",
        "//tensorflow/python:util",
    ],
)

py_library(
    name = "lite_constants",
    srcs = ["lite_constants.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/lite/toco:toco_flags_proto_py",
        "//tensorflow/python:dtypes",
    ],
)

py_library(
    name = "convert",
    srcs = ["convert.py"],
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
    deps = [
        ":lite_constants",
        ":util",
        ":wrap_toco",
        "//tensorflow/lite/toco:model_flags_proto_py",
        "//tensorflow/lite/toco:toco_flags_proto_py",
        "//tensorflow/lite/toco/python:toco_from_protos",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:platform",
        "@six_archive//:six",
    ],
)

py_library(
    name = "op_hint",
    srcs = ["op_hint.py"],
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:graph_util",
        "//tensorflow/python:platform",
        "//tensorflow/python:util",
    ],
)

py_test(
    name = "convert_test",
    srcs = ["convert_test.py"],
    python_version = "PY3",
    srcs_version = "PY2AND3",
    deps = [
        ":convert",
        ":interpreter",
        ":op_hint",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:platform_test",
        "//tensorflow/python:session",
    ],
)

py_library(
    name = "convert_saved_model",
    srcs = ["convert_saved_model.py"],
    srcs_version = "PY2AND3",
    visibility = [
        "//tensorflow/lite:__subpackages__",
    ],
    deps = [
        ":util",
        "//tensorflow/python:graph_util",
        "//tensorflow/python:platform",
        "//tensorflow/python/saved_model",
    ],
)

py_test(
    name = "convert_saved_model_test",
    srcs = ["convert_saved_model_test.py"],
    python_version = "PY3",
    srcs_version = "PY2AND3",
    tags = [
        "no_windows",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":convert_saved_model",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:layers",
        "//tensorflow/python:nn",
        "//tensorflow/python:platform_test",
        "//tensorflow/python:session",
        "//tensorflow/python/keras",
        "//tensorflow/python/ops/losses",
        "//tensorflow/python/saved_model",
    ],
)

py_binary(
    name = "convert_file_to_c_source",
    srcs = ["convert_file_to_c_source.py"],
    python_version = "PY3",
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
    deps = [
        ":lite",
        ":util",
        "@six_archive//:six",
    ],
)

sh_test(
    name = "convert_file_to_c_source_test",
    srcs = ["convert_file_to_c_source_test.sh"],
    data = [":convert_file_to_c_source"],
)
