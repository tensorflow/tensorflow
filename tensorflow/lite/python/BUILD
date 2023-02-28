load("@flatbuffers//:build_defs.bzl", "flatbuffer_py_library")
load("//tensorflow:tensorflow.default.bzl", "get_compatible_with_portable")
load("//tensorflow/lite:special_rules.bzl", "internal_visibility_allowlist")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [
        "//tensorflow:__subpackages__",
        "//tensorflow:internal",
        "//tensorflow_estimator:__subpackages__",
        "//third_party/py/tensorflow_federated:__subpackages__",
        "//third_party/tflite_micro:__subpackages__",
    ],
    licenses = ["notice"],
)

exports_files(["tflite_convert.py"])

flatbuffer_py_library(
    name = "schema_py",
    srcs = ["//tensorflow/lite/schema:schema.fbs"],
)

flatbuffer_py_library(
    name = "conversion_metadata_schema_py",
    srcs = ["//tensorflow/lite/schema:conversion_metadata.fbs"],
)

py_library(
    name = "interpreter",
    srcs = [
        "interpreter.py",
    ],
    compatible_with = get_compatible_with_portable(),
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/lite/python/interpreter_wrapper:_pywrap_tensorflow_interpreter_wrapper",
        "//tensorflow/lite/python/metrics",
        "//tensorflow/python:util",
        "//tensorflow/python/util:tf_export",
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
    srcs_version = "PY3",
    tags = [
        "no_oss",  # TODO(b/190842754): Enable test in OSS.
        "no_pip",  # TODO(b/187847053): Enable test in pip.
    ],
    deps = [
        ":interpreter",
        "//tensorflow:tensorflow_py",
        "//tensorflow/lite/python/metrics",
        "//tensorflow/lite/python/testdata:_pywrap_test_registerer",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform",
        "//third_party/py/numpy",
    ],
)

py_binary(
    name = "tflite_convert",
    srcs = ["tflite_convert.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [
        ":tflite_convert_main_lib",
        "//tensorflow:tensorflow_py",
        "@absl_py//absl:app",
    ],
)

py_library(
    name = "tflite_convert_main_lib",
    srcs = ["tflite_convert.py"],
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [
        ":tflite_convert_lib",
        "//tensorflow:tensorflow_py",
        "@absl_py//absl:app",
    ],
)

py_library(
    name = "tflite_convert_lib",
    srcs = ["tflite_convert.py"],
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [
        ":lite",
        "//tensorflow/lite/toco/logging:gen_html",
        "//tensorflow/lite/toco/logging:toco_conversion_log_proto_py",
        "//tensorflow/python:util",
        "//tensorflow/python/platform",
        "@absl_py//absl:app",
    ],
)

py_library(
    name = "test_util",
    srcs = ["test_util.py"],
    srcs_version = "PY3",
    deps = [
        ":lite",
        ":schema_util",
        "//tensorflow/lite/tools:visualize",
        "//tensorflow/python:framework",
    ],
)

py_test(
    name = "test_util_test",
    srcs = ["test_util_test.py"],
    data = [
        "//tensorflow/lite:testdata/add.bin",
        "//tensorflow/lite:testdata/softplus_flex.bin",
    ],
    python_version = "PY3",
    deps = [
        ":test_util",
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
    # Increased thread count for reducing timeout failures.
    shard_count = 10,
    srcs_version = "PY3",
    tags = [
        "no_oss",
        "no_pip",
        "no_windows",
        "noasan",  # b/144707533
        "notsan",  # b/160824139
    ],
    deps = [
        ":convert",
        ":test_util",
        ":tflite_convert",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:framework",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform",
        "//tensorflow/python:random_ops",
        "//tensorflow/python:session",
        "//tensorflow/python:tf2",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/saved_model",
        "//tensorflow/python/saved_model:save",
        "//tensorflow/python/trackable:autotrackable",
        "//tensorflow/python/training:training_util",
        "//third_party/py/numpy",
    ],
)

py_library(
    name = "lite",
    srcs = ["lite.py"],
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [
        ":conversion_metadata_schema_py",
        ":convert",
        ":convert_phase",
        ":convert_saved_model",
        ":interpreter",
        ":lite_constants",
        ":op_hint",
        ":util",
        "//tensorflow/lite/experimental/microfrontend:audio_microfrontend_py",
        "//tensorflow/lite/python/metrics",
        "//tensorflow/lite/python/optimize:calibrator",
        "//tensorflow/lite/tools:flatbuffer_utils",
        "//tensorflow/lite/tools/optimize/debugging/python:debugger",
        "//tensorflow/python/client:session",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/eager:function",
        "//tensorflow/python/framework",
        "//tensorflow/python/framework:convert_to_constants",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:errors",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/platform",
        "//tensorflow/python/saved_model:load",
        "//tensorflow/python/saved_model:loader",
        "//tensorflow/python/saved_model:signature_constants",
        "//tensorflow/python/saved_model:tag_constants",
        "//tensorflow/python/util",
        "//tensorflow/python/util:tf_export",
        "@absl_py//absl/logging",
    ],
)

py_test(
    name = "lite_test",
    srcs = ["lite_test.py"],
    data = [
        "//tensorflow/lite/python/testdata:control_flow_v1.pbtxt",
        "@tflite_mobilenet_ssd_quant_protobuf//:tflite_graph.pb",
    ],
    python_version = "PY3",
    shard_count = 4,
    srcs_version = "PY3",
    tags = [
        "no_windows",
    ],
    deps = [
        ":lite",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_test_lib",
    ],
)

py_test(
    name = "lite_v2_test",
    srcs = ["lite_v2_test.py"],
    data = [
        "//tensorflow/lite/python/testdata:test_delegate.so",
        "//tensorflow/lite/python/testdata/control_flow_v1_saved_model:saved_model.pb",
    ],
    python_version = "PY3",
    shard_count = 18,
    srcs_version = "PY3",
    tags = [
        "no_windows",
    ],
    deps = [
        ":lite",
        ":lite_v2_test_util",
        ":test_util",
        "//tensorflow:tensorflow_py",
        "//tensorflow/lite/python/testdata:_pywrap_test_registerer",
        "//tensorflow/lite/python/testdata:double_op",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_test_lib",
    ],
)

py_library(
    name = "lite_v2_test_util",
    testonly = 1,
    srcs = ["lite_v2_test_util.py"],
    srcs_version = "PY3",
    tags = [
        "no_windows",
    ],
    deps = [
        ":lite",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_test_lib",
    ],
)

py_test(
    name = "lite_flex_test",
    srcs = ["lite_flex_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":lite",
        ":test_util",
        "//tensorflow:tensorflow_py",
        "//tensorflow/lite/python/testdata:double_op",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_test_lib",
    ],
)

py_library(
    name = "util",
    srcs = ["util.py"],
    srcs_version = "PY3",
    visibility = internal_visibility_allowlist(),
    deps = [
        ":conversion_metadata_schema_py",
        ":op_hint",
        ":schema_py",
        ":schema_util",
        "//tensorflow/lite/python:tflite_keras_util",
        "//tensorflow/lite/toco:toco_flags_proto_py",
        "//tensorflow/lite/tools:flatbuffer_utils",
        "//tensorflow/python:convert_to_constants",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:error_interpolation",
        "//tensorflow/python:graph_util",
        "//tensorflow/python:tf_optimizer",
        "//tensorflow/python/eager:function",
        "//tensorflow/python/training:saver",
        "@absl_py//absl/logging",
        "@flatbuffers//:runtime_py",
    ],
)

py_test(
    name = "util_test",
    srcs = ["util_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    tags = [
        "no_windows",
    ],
    deps = [
        ":util",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:convert_to_constants",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:session",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_library(
    name = "tflite_keras_util",
    srcs = [
        "tflite_keras_util.py",
    ],
    srcs_version = "PY3",
    deps = [
        "//tensorflow/python:util",
        "//tensorflow/python/eager:def_function",
    ],
)

py_library(
    name = "wrap_toco",
    srcs = [
        "wrap_toco.py",
    ],
    srcs_version = "PY3",
    deps = [
        "//tensorflow/python:_pywrap_toco_api",
        "//tensorflow/python:pywrap_tensorflow",
        "//tensorflow/python:util",
    ],
)

py_library(
    name = "lite_constants",
    srcs = ["lite_constants.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow/lite/toco:toco_flags_proto_py",
        "//tensorflow/python:dtypes",
    ],
)

py_library(
    name = "convert",
    srcs = ["convert.py"],
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [
        ":convert_phase",
        ":lite_constants",
        ":util",
        ":wrap_toco",
        "//tensorflow/lite/python/metrics:metrics_wrapper",
        "//tensorflow/lite/toco:model_flags_proto_py",
        "//tensorflow/lite/toco:toco_flags_proto_py",
        "//tensorflow/lite/toco/python:toco_from_protos",
        "//tensorflow/lite/tools:flatbuffer_utils",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:platform",
        "//tensorflow/python:tensor_shape",
        "//tensorflow/python/util",
        "//tensorflow/python/util:tf_export",
    ],
)

py_library(
    name = "op_hint",
    srcs = ["op_hint.py"],
    srcs_version = "PY3",
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
    srcs_version = "PY3",
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
    srcs_version = "PY3",
    visibility = [
        "//tensorflow/lite:__subpackages__",
    ],
    deps = [
        ":convert_phase",
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
    srcs_version = "PY3",
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
        "//tensorflow/python/ops/losses",
        "//tensorflow/python/saved_model",
    ],
)

py_binary(
    name = "convert_file_to_c_source",
    srcs = ["convert_file_to_c_source.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [
        ":util",
        "@absl_py//absl:app",
        "@absl_py//absl/flags",
    ],
)

sh_test(
    name = "convert_file_to_c_source_test",
    srcs = ["convert_file_to_c_source_test.sh"],
    data = [":convert_file_to_c_source"],
)

py_library(
    name = "schema_util",
    srcs = ["schema_util.py"],
    srcs_version = "PY3",
    visibility = ["//tensorflow/lite/schema:utils_friends"],
    deps = [
        "//tensorflow/python:util",
    ],
)

# Use py_library since the metrics module is imported in a try-except block,
# which doesn't work with the pytype_strict_library.
py_library(
    name = "convert_phase",
    srcs = ["convert_phase.py"],
    srcs_version = "PY3",
    visibility = ["//tensorflow/lite:__subpackages__"],
    deps = [
        "//tensorflow/lite/python/metrics",
        "//tensorflow/lite/python/metrics:converter_error_data_proto_py",
    ],
)

py_library(
    name = "analyzer",
    srcs = [
        "analyzer.py",
    ],
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/lite/python:wrap_toco",
        "//tensorflow/lite/python/analyzer_wrapper:_pywrap_analyzer_wrapper",
        "//tensorflow/python/util:tf_export",
    ],
)

py_test(
    name = "analyzer_test",
    srcs = ["analyzer_test.py"],
    data = [
        "//tensorflow/lite:testdata/add.bin",
        "//tensorflow/lite:testdata/conv_huge_im2col.bin",
        "//tensorflow/lite:testdata/multi_add_flex.bin",
    ],
    python_version = "PY3",
    deps = [
        ":analyzer",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_test_lib",
    ],
)
