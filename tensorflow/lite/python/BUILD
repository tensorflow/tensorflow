load("@flatbuffers//:build_defs.bzl", "flatbuffer_py_library")
load("@rules_shell//shell:sh_test.bzl", "sh_test")
load("//tensorflow:pytype.default.bzl", "pytype_strict_contrib_test", "pytype_strict_library")
load("//tensorflow:strict.default.bzl", "py_strict_binary", "py_strict_library", "py_strict_test")
load("//tensorflow:tensorflow.default.bzl", "get_compatible_with_portable", "pywrap_binaries", "pywrap_library")
load("//tensorflow/core/platform:build_config_root.bzl", "if_pywrap")
load("//tensorflow/lite:special_rules.bzl", "internal_visibility_allowlist")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [
        "//tensorflow:__subpackages__",
        "//tensorflow:internal",
        "//third_party/odml/infra/genai/conversion:__subpackages__",
        "//third_party/odml/model_customization/quantization:__subpackages__",
        "//third_party/py/ai_edge_torch:__subpackages__",
        "//third_party/py/tensorflow_federated:__subpackages__",
        "//third_party/tflite_micro:__subpackages__",
    ],
    licenses = ["notice"],
)

exports_files([
    "tflite_convert.py",
    "pywrap_tflite_common.json",
    "pywrap_tflite_common.lds",
    "pywrap_tflite_common_darwin.lds",
])

flatbuffer_py_library(
    name = "schema_py",
    srcs = ["//tensorflow/compiler/mlir/lite/schema:schema.fbs"],
)

flatbuffer_py_library(
    name = "conversion_metadata_schema_py",
    srcs = ["//tensorflow/compiler/mlir/lite/schema:conversion_metadata.fbs"],
)

py_strict_library(
    name = "interpreter",
    srcs = [
        "interpreter.py",
    ],
    compatible_with = get_compatible_with_portable(),
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/lite/python/interpreter_wrapper:_pywrap_tensorflow_interpreter_wrapper",
        "//tensorflow/lite/python/metrics",
        "//tensorflow/python/util:tf_export",
        "//third_party/py/numpy",
    ],
)

py_strict_test(
    name = "interpreter_test",
    srcs = ["interpreter_test.py"],
    data = [
        "//tensorflow/lite:testdata/sparse_tensor.bin",
        "//tensorflow/lite/python/testdata:interpreter_test_data",
        "//tensorflow/lite/python/testdata:test_delegate.so",
    ],
    # Static linking is required because this loads a cc_binary as a shared
    # library, which would otherwise create ODR violations.
    # copybara:uncomment linking_mode = "static",
    tags = [
        "no_oss",  # TODO(b/190842754): Enable test in OSS.
    ],
    deps = [
        ":interpreter",
        #internal proto upb dep
        "//third_party/py/numpy",
        "//tensorflow:tensorflow_py",
        "//tensorflow/lite/python:lite",
        "//tensorflow/lite/python/metrics",
        "//tensorflow/lite/python/testdata:_pywrap_test_registerer",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/platform:client_testlib",
        "//tensorflow/python/platform:resource_loader",
    ],
)

py_strict_binary(
    name = "tflite_convert",
    srcs = ["tflite_convert.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":tflite_convert_main_lib",
        "//tensorflow:tensorflow_py",
        "//tensorflow/lite/python:lite",
        "@absl_py//absl:app",
    ],
)

py_strict_library(
    name = "tflite_convert_main_lib",
    srcs = ["tflite_convert.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":convert",
        "//tensorflow:tensorflow_py",
        "//tensorflow/lite/toco:toco_flags_proto_py",
        "//tensorflow/lite/toco/logging:gen_html",
        "//tensorflow/python:tf2",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/platform:gfile",
        "//tensorflow/python/util:keras_deps",
        "@absl_py//absl:app",
    ],
)

py_strict_library(
    name = "test_util",
    srcs = ["test_util.py"],
    deps = [
        ":lite",
        ":schema_py",
        ":schema_util",
        "//tensorflow/lite/tools:visualize_lib",
    ],
)

py_strict_test(
    name = "test_util_test",
    srcs = ["test_util_test.py"],
    data = [
        "//tensorflow/lite:testdata/add.bin",
        "//tensorflow/lite:testdata/softplus_flex.bin",
    ],
    deps = [
        ":test_util",
        #internal proto upb dep
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/platform:client_testlib",
        "//tensorflow/python/platform:gfile",
        "//tensorflow/python/platform:resource_loader",
    ],
)

py_strict_test(
    name = "tflite_convert_test",
    srcs = ["tflite_convert_test.py"],
    data = [
        ":tflite_convert.par",
        "@tflite_mobilenet_ssd_quant_protobuf//:tflite_graph.pb",
    ],
    # Increased thread count for reducing timeout failures.
    shard_count = 10,
    tags = [
        "no_oss",
        "no_windows",
    ],
    deps = [
        ":convert",
        ":test_util",
        ":tflite_convert_main_lib",
        #internal proto upb dep
        "//third_party/py/numpy",
        "//tensorflow:tensorflow_py",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:tf2",
        "//tensorflow/python/client:session",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:importer",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:random_ops",
        "//tensorflow/python/platform:client_testlib",
        "//tensorflow/python/platform:gfile",
        "//tensorflow/python/platform:resource_loader",
        "//tensorflow/python/saved_model",
        "//tensorflow/python/saved_model:save",
        "//tensorflow/python/trackable:autotrackable",
        "//tensorflow/python/training:training_util",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_strict_library(
    name = "lite",
    srcs = ["lite.py"],
    tags = [
        "ignore_for_dep=third_party.py.keras",
    ],
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
        "//tensorflow/compiler/mlir/quantization/stablehlo:quantization_config_proto_py",
        "//tensorflow/compiler/mlir/quantization/tensorflow/python:representative_dataset",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/lite/experimental/microfrontend:audio_microfrontend_py",
        "//tensorflow/lite/profiling/proto:model_runtime_info_py",
        "//tensorflow/lite/profiling/proto:profiling_info_py",
        "//tensorflow/lite/python/metrics",
        "//tensorflow/lite/python/optimize:calibrator",
        "//tensorflow/lite/tools:flatbuffer_utils",
        "//tensorflow/lite/tools/optimize/debugging/python:debugger",
        "//tensorflow/python/client:session",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/eager:function",
        "//tensorflow/python/framework:byte_swap_tensor",
        "//tensorflow/python/framework:convert_to_constants",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:errors",
        "//tensorflow/python/framework:importer",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:tensor_spec",
        "//tensorflow/python/framework:versions",
        "//tensorflow/python/platform:gfile",
        "//tensorflow/python/saved_model:load",
        "//tensorflow/python/saved_model:loader",
        "//tensorflow/python/saved_model:save",
        "//tensorflow/python/saved_model:save_options",
        "//tensorflow/python/saved_model:signature_constants",
        "//tensorflow/python/saved_model:tag_constants",
        "//tensorflow/python/util:deprecation",
        "//tensorflow/python/util:keras_deps",
        "//tensorflow/python/util:nest",
        "//tensorflow/python/util:tf_export",
        "@absl_py//absl/logging",
    ],
)

py_strict_test(
    name = "lite_test",
    srcs = ["lite_test.py"],
    data = [
        "//tensorflow/lite/python/testdata:control_flow_v1.pbtxt",
        "@tflite_mobilenet_ssd_quant_protobuf//:tflite_graph.pb",
    ],
    shard_count = 4,
    tags = [
        "no_windows",
    ],
    deps = [
        ":conversion_metadata_schema_py",
        ":convert",
        ":interpreter",
        ":lite",
        ":lite_constants",
        ":schema_py",
        ":util",
        #internal proto upb dep
        "//third_party/py/numpy",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python/client:session",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:convert_to_constants",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/framework:versions",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:logging_ops",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/ops:nn_ops",
        "//tensorflow/python/ops:random_ops",
        "//tensorflow/python/ops:variable_scope",
        "//tensorflow/python/ops:variables",
        "//tensorflow/python/platform:client_testlib",
        "//tensorflow/python/platform:gfile",
        "//tensorflow/python/platform:resource_loader",
        "//tensorflow/python/saved_model",
        "//tensorflow/python/training:training_util",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_strict_test(
    name = "lite_v2_test",
    srcs = ["lite_v2_test.py"],
    data = [
        "//tensorflow/lite/python/testdata:test_delegate.so",
        "//tensorflow/lite/python/testdata/control_flow_v1_saved_model:saved_model.pb",
    ],
    shard_count = 18,
    tags = [
        "no_windows",
    ],
    deps = [
        ":conversion_metadata_schema_py",
        ":convert",
        ":interpreter",
        ":lite",
        ":lite_v2_test_util",
        ":schema_py",
        ":test_util",
        ":util",
        #internal proto upb dep
        "//third_party/py/numpy",
        "//tensorflow:tensorflow_py",
        "//tensorflow/compiler/mlir/quantization/stablehlo:quantization_options_proto_py",
        "//tensorflow/lite/python/testdata:_pywrap_test_registerer",
        "//tensorflow/lite/python/testdata:double_op",
        "//tensorflow/lite/tools:flatbuffer_utils",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/framework:versions",
        "//tensorflow/python/lib/io:file_io",
        "//tensorflow/python/ops:map_ops",
        "//tensorflow/python/ops:rnn",
        "//tensorflow/python/platform:client_testlib",
        "//tensorflow/python/platform:resource_loader",
        "//tensorflow/python/saved_model",
        "//tensorflow/python/saved_model:loader",
        "//tensorflow/python/saved_model:save",
        "//tensorflow/python/saved_model:save_options",
        "//tensorflow/python/trackable:autotrackable",
        "@absl_py//absl/testing:parameterized",
        "@pypi_jax//:pkg",
    ],
)

py_strict_library(
    name = "lite_v2_test_util",
    testonly = 1,
    srcs = ["lite_v2_test_util.py"],
    tags = [
        "no_windows",
    ],
    deps = [
        ":interpreter",
        ":lite",
        "//tensorflow:tensorflow_py_no_contrib",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:tensor_spec",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/ops:variables",
        "//tensorflow/python/platform:client_testlib",
        "//tensorflow/python/trackable:autotrackable",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_strict_test(
    name = "lite_flex_test",
    srcs = ["lite_flex_test.py"],
    deps = [
        ":convert",
        ":interpreter",
        ":lite",
        ":test_util",
        #internal proto upb dep
        "//third_party/py/numpy",
        "//tensorflow:tensorflow_py",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/lite/python/testdata:double_op",
        "//tensorflow/python/client:session",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:importer",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:list_ops",
        "//tensorflow/python/ops:nn_ops",
        "//tensorflow/python/ops:variables",
        "//tensorflow/python/platform:client_testlib",
        "//tensorflow/python/saved_model",
        "//tensorflow/python/trackable:autotrackable",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_strict_library(
    name = "util",
    srcs = ["util.py"],
    visibility = internal_visibility_allowlist(),
    deps = [
        ":conversion_metadata_schema_py",
        ":op_hint",
        ":schema_py",
        ":schema_util",
        ":tflite_keras_util",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/lite/tools:flatbuffer_utils",
        "//tensorflow/python/eager:function",
        "//tensorflow/python/framework:convert_to_constants",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:error_interpolation",
        "//tensorflow/python/grappler:tf_optimizer",
        "//tensorflow/python/training:saver",
        "//third_party/py/numpy",
        "@absl_py//absl/logging",
        "@flatbuffers//:runtime_py",
    ],
)

py_strict_test(
    name = "util_test",
    srcs = ["util_test.py"],
    tags = [
        "no_windows",
    ],
    deps = [
        ":util",
        #internal proto upb dep
        "//third_party/py/numpy",
        "//tensorflow:tensorflow_py",
        "//tensorflow/lite/python:lite",
        "//tensorflow/lite/tools:flatbuffer_utils",
        "//tensorflow/python/client:session",
        "//tensorflow/python/framework:convert_to_constants",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/ops:while_loop",
        "//tensorflow/python/platform:client_testlib",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_strict_library(
    name = "tflite_keras_util",
    srcs = [
        "tflite_keras_util.py",
    ],
    deps = [
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/framework:tensor_spec",
        "//tensorflow/python/util:compat",
        "//tensorflow/python/util:keras_deps",
        "//tensorflow/python/util:nest",
    ],
)

py_strict_library(
    name = "lite_constants",
    srcs = ["lite_constants.py"],
    deps = [
        "//tensorflow/compiler/mlir/lite:converter_flags_proto_py",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/util:all_util",
        "//tensorflow/python/util:tf_export",
    ],
)

pytype_strict_library(
    name = "convert",
    srcs = ["convert.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":convert_phase",
        ":lite_constants",
        ":util",
        "//tensorflow/compiler/mlir/lite:converter_flags_proto_py",
        "//tensorflow/compiler/mlir/lite:model_flags_proto_py",
        "//tensorflow/compiler/mlir/lite:types_proto_py",
        "//tensorflow/compiler/mlir/lite/metrics:converter_error_data_proto_py",
        "//tensorflow/compiler/mlir/lite/python:wrap_converter",
        "//tensorflow/compiler/mlir/quantization/stablehlo:quantization_config_proto_py",
        "//tensorflow/compiler/mlir/quantization/stablehlo:quantization_options_proto_py",
        "//tensorflow/lite/python/metrics:metrics_wrapper",
        "//tensorflow/lite/tools:flatbuffer_utils",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:tensor_shape",
        "//tensorflow/python/util:deprecation",
        "//tensorflow/python/util:tf_export",
    ] + if_pywrap(
        if_true = [":pywrap_tflite"],
    ),
)

py_strict_library(
    name = "op_hint",
    srcs = ["op_hint.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:graph_util",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:tensor_util",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/util:all_util",
        "//tensorflow/python/util:compat",
        "//tensorflow/python/util:deprecation",
        "//tensorflow/python/util:tf_export",
    ],
)

pytype_strict_contrib_test(
    name = "convert_test",
    srcs = ["convert_test.py"],
    deps = [
        ":convert",
        ":interpreter",
        ":op_hint",
        "//tensorflow/compiler/mlir/lite:converter_flags_proto_py",
        "//tensorflow/compiler/mlir/lite/metrics:converter_error_data_proto_py",
        "//tensorflow/lite/python/metrics:metrics_wrapper",
        "//tensorflow/python/client:session",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:graph_util",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:array_ops_stack",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/platform:client_testlib",
        "//third_party/py/numpy",
    ] + if_pywrap(
        if_true = [":pywrap_tflite"],
    ),
)

py_strict_library(
    name = "convert_saved_model",
    srcs = ["convert_saved_model.py"],
    visibility = [
        "//tensorflow/lite:__subpackages__",
    ],
    deps = [
        ":convert_phase",
        ":util",
        "//tensorflow/python/client:session",
        "//tensorflow/python/framework:graph_util",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/platform:tf_logging",
        "//tensorflow/python/saved_model",
        "//tensorflow/python/saved_model:constants",
        "//tensorflow/python/saved_model:loader",
    ],
)

py_strict_test(
    name = "convert_saved_model_test",
    srcs = ["convert_saved_model_test.py"],
    tags = [
        "no_windows",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":convert_saved_model",
        #internal proto upb dep
        "//tensorflow/python/client:session",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:tensor_shape",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/layers",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops/losses",
        "//tensorflow/python/platform:client_testlib",
        "//tensorflow/python/platform:test",
        "//tensorflow/python/saved_model",
        "//tensorflow/python/saved_model:signature_constants",
        "//tensorflow/python/saved_model:tag_constants",
    ],
)

py_strict_binary(
    name = "convert_file_to_c_source",
    srcs = ["convert_file_to_c_source.py"],
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

py_strict_library(
    name = "schema_util",
    srcs = ["schema_util.py"],
    visibility = ["//tensorflow/lite/schema:utils_friends"],
    deps = [
        "//tensorflow/python/util:all_util",
    ],
)

# Use py_library since the metrics module is imported in a try-except block,
# which doesn't work with the pytype_strict_library.
py_strict_library(
    name = "convert_phase",
    srcs = ["convert_phase.py"],
    visibility = ["//tensorflow/lite:__subpackages__"],
    deps = [
        "//tensorflow/compiler/mlir/lite/metrics:converter_error_data_proto_py",
        "//tensorflow/lite/python/metrics",
    ],
)

py_strict_library(
    name = "analyzer",
    srcs = [
        "analyzer.py",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/compiler/mlir/lite/python:wrap_converter",
        "//tensorflow/python/util:tf_export",
    ] + if_pywrap(
        if_false = [
            "//tensorflow/lite/python/analyzer_wrapper:_pywrap_analyzer_wrapper",
        ],
        if_true = [
            ":pywrap_tflite",
        ],
    ),
)

py_strict_test(
    name = "analyzer_test",
    srcs = ["analyzer_test.py"],
    data = [
        "//tensorflow/lite:testdata/add.bin",
        "//tensorflow/lite:testdata/conv_huge_im2col.bin",
        "//tensorflow/lite:testdata/multi_add_flex.bin",
    ],
    deps = [
        ":analyzer",
        #internal proto upb dep
        "//tensorflow:tensorflow_py",
        "//tensorflow/lite/python:lite",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/platform:client_testlib",
        "//tensorflow/python/platform:resource_loader",
        "//tensorflow/python/trackable:autotrackable",
    ],
)

# Use pywrap_library to avoid duplicate registration of pybind11 modules.
# A great example on how to use pywrap_library is
# https://github.com/vam-google/symbol-locations/blob/main/pybind/BUILD
# The following pywrap_library is used by LiteRT repo to avoid shared links provided
# by Tensorflow under tensorflow/python:_pywrap_tensorflow
# This isolate LiteRT's pybind11 dependencies. To use, add pybind deps under pywrap_tflite
# and refer pywrap_tflite to any target that needsd to selected shared objects.
py_strict_library(
    name = "tflite_pywrap_deps",
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/lite/experimental/genai:pywrap_genai_ops",
        "//tensorflow/lite/python/analyzer_wrapper:_pywrap_analyzer_wrapper",
        "//tensorflow/lite/python/interpreter_wrapper:_pywrap_tensorflow_interpreter_wrapper",
        "//tensorflow/lite/python/metrics:_pywrap_tensorflow_lite_metrics_wrapper",
        "//tensorflow/lite/python/optimize:_pywrap_tensorflow_lite_calibration_wrapper",
        "//tensorflow/lite/testing:_pywrap_string_util",
        "//tensorflow/lite/tools/optimize/python:_pywrap_modify_model_interface",
        "//tensorflow/lite/tools/optimize/sparsity:format_converter_wrapper_pybind11",
    ],
)

pywrap_library(
    name = "pywrap_tflite",
    common_lib_def_files_or_filters = {
        "third_party/tensorflow/lite/python/pywrap_tflite_common": "pywrap_tflite_common.json",
    },
    common_lib_version_scripts = {
        "third_party/tensorflow/lite/python/pywrap_tflite_common": select({
            "@bazel_tools//src/conditions:windows": None,
            "@bazel_tools//src/conditions:darwin": "pywrap_tflite_common_darwin.lds",
            "//conditions:default": "pywrap_tflite_common.lds",
        }),
    },
    pywrap_count = 8,
    visibility = ["//visibility:public"],
    deps = [
        ":tflite_pywrap_deps",
    ],
)

pywrap_binaries(
    name = "pywrap_tflite_binaries",
    dep = ":pywrap_tflite",
)
