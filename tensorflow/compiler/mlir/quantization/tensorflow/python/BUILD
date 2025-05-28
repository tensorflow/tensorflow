load("@local_xla//xla/tsl/platform:build_config_root.bzl", "if_pywrap")
load("//tensorflow:pytype.default.bzl", "pytype_strict_library")
load(
    "//tensorflow:tensorflow.default.bzl",
    "get_compatible_with_portable",
    "tf_py_strict_test",
    "tf_python_pybind_extension",
)

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [
        "//tensorflow/compiler/mlir/quantization/stablehlo:__subpackages__",
        "//tensorflow/compiler/mlir/quantization/tensorflow:internal_visibility_allowlist_package",
        "//tensorflow/lite:__subpackages__",
        "//tensorflow/python:__subpackages__",
        "//tensorflow/tools/pip_package:__subpackages__",
    ],
    licenses = ["notice"],
)

# Do NOT directly depend on `quantize_model_cc_impl` unless it is necessary
# (i.e. undefined symbol). See the comments in `quantize_model_cc`.
cc_library(
    name = "quantize_model_cc_impl",
    srcs = ["quantize_model.cc"],
    hdrs = ["quantize_model.h"],
    compatible_with = get_compatible_with_portable(),
    visibility = [
        # Directly linked to `libtensorflow_cc.so` or
        # `_pywrap_tensorflow_internal.so` if static build.
        "//tensorflow:__pkg__",
        "//tensorflow/python:__pkg__",
    ],
    deps = [
        ":py_function_lib",
        "//tensorflow/cc/saved_model:loader",
        "//tensorflow/compiler/mlir/quantization/stablehlo:quantization_config_proto_cc",
        "//tensorflow/compiler/mlir/quantization/stablehlo/cc:config",
        "//tensorflow/compiler/mlir/quantization/stablehlo/cc:context",
        "//tensorflow/compiler/mlir/quantization/stablehlo/cc:debugger",
        "//tensorflow/compiler/mlir/quantization/stablehlo/cc:io",
        "//tensorflow/compiler/mlir/quantization/stablehlo/cc:saved_model_export",
        "//tensorflow/compiler/mlir/quantization/stablehlo/cc:saved_model_import",
        "//tensorflow/compiler/mlir/quantization/stablehlo/cc:tf_post_calibration",
        "//tensorflow/compiler/mlir/quantization/stablehlo/cc:tf_pre_calibration",
        "//tensorflow/compiler/mlir/quantization/stablehlo/cc:types",
        "//tensorflow/compiler/mlir/quantization/stablehlo/cc:weight_only_ptq",
        "//tensorflow/compiler/mlir/quantization/stablehlo/cc/calibration:component",
        "//tensorflow/compiler/mlir/quantization/stablehlo/cc/calibration:statistics",
        "//tensorflow/compiler/mlir/quantization/tensorflow:exported_model_proto_cc",
        "//tensorflow/compiler/mlir/quantization/tensorflow:quantization_options_proto_cc",
        "//tensorflow/compiler/mlir/quantization/tensorflow:tf_passes",
        "//tensorflow/compiler/mlir/quantization/tensorflow:tf_quantize_passes",
        "//tensorflow/compiler/mlir/quantization/tensorflow:tf_quantize_preprocess",
        "//tensorflow/compiler/mlir/quantization/tensorflow/calibrator:calibration_statistics_saver_op",  # Required for CalibrationStatisticsSaver op registration.
        "//tensorflow/compiler/mlir/quantization/tensorflow/calibrator:custom_aggregator_op",  # Required for CustomAggregator op registration.
        "//tensorflow/compiler/mlir/quantization/tensorflow/cc:run_passes",
        "//tensorflow/compiler/mlir/quantization/tensorflow/debugging:dump_tensor_op",  # Required for DumpTensor op registration.
        "//tensorflow/compiler/mlir/tensorflow:mlir_import_options",
        "//tensorflow/compiler/mlir/tensorflow:translate_lib",
        "//tensorflow/compiler/mlir/tensorflow/transforms:tf_dialect_passes",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/platform:path",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@local_tsl//tsl/platform:path",
        "@local_xla//xla/tsl/platform:errors",
        "@local_xla//xla/tsl/platform:statusor",
    ],
)

# OSS: This is a header-only target. The implementation target `quantize_model_cc_impl` is
# directly linked to `lib_pywrap_tensorflow_internal.so`, so in most use cases of python-
# exported symbols depending directly on `quantize_model_cc_impl` should be unnecessary.
# Using the header-only target will help avoid the ODR violation.
cc_library(
    name = "quantize_model_cc",
    hdrs = ["quantize_model.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        ":py_function_lib",
        "//tensorflow/compiler/mlir/quantization/tensorflow:exported_model_proto_cc",
        "//tensorflow/compiler/mlir/quantization/tensorflow:quantization_options_proto_cc",
        "//tensorflow/core/protobuf:for_core_protos_cc",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
    ],
)

pytype_strict_library(
    name = "py_function_lib_py",
    srcs = ["py_function_lib.py"],
    deps = [
        ":pywrap_function_lib",
        ":representative_dataset",
        ":save_model",
        "//tensorflow/compiler/mlir/quantization/stablehlo:quantization_config_proto_py",
        "//tensorflow/compiler/mlir/quantization/tensorflow:exported_model_proto_py",
        "//tensorflow/compiler/mlir/quantization/tensorflow:quantization_options_proto_py",
        "//tensorflow/compiler/mlir/quantization/tensorflow/calibrator:calibration_algorithm",
        "//tensorflow/compiler/mlir/quantization/tensorflow/calibrator:calibration_statistics_proto_py",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python/client:session",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:wrap_function",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:tensor_conversion",
        "//tensorflow/python/lib/io:file_io",
        "//tensorflow/python/saved_model:load",
        "//tensorflow/python/saved_model:loader",
        "//tensorflow/python/trackable:autotrackable",
        "//tensorflow/python/types:core",
        "@absl_py//absl/logging",
    ],
)

tf_py_strict_test(
    name = "py_function_lib_py_test",
    srcs = ["py_function_lib_test.py"],
    main = "py_function_lib_test.py",
    deps = ["//tensorflow/python/platform:client_testlib"],
)

cc_library(
    name = "type_casters",
    hdrs = ["type_casters.h"],
    copts = ["-fexceptions"],  # Required for pybind11.
    # Required for pybind11.
    features = [
        "-use_header_modules",
        "-parse_headers",
    ],
    deps = [
        "//tensorflow/compiler/mlir/quantization/stablehlo:quantization_config_proto_cc",
        "//tensorflow/compiler/mlir/quantization/tensorflow:exported_model_proto_cc",
        "//tensorflow/compiler/mlir/quantization/tensorflow:quantization_options_proto_cc",
        "//tensorflow/compiler/mlir/quantization/tensorflow/calibrator:calibration_statistics_proto_cc",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/protobuf:for_core_protos_cc",
        "//tensorflow/python/lib/core:pybind11_lib",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
        "@local_tsl//tsl/platform:protobuf",
        "@local_xla//third_party/python_runtime:headers",  # build_cleaner: keep; Required for pybind11.
        "@pybind11",
        "@pybind11_abseil//pybind11_abseil:absl_casters",
    ],
)

cc_library(
    name = "py_function_lib",
    hdrs = ["py_function_lib.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        "//tensorflow/compiler/mlir/quantization/stablehlo:quantization_config_proto_cc",
        "//tensorflow/compiler/mlir/quantization/stablehlo/cc/calibration:min_max_value",
        "//tensorflow/compiler/mlir/quantization/tensorflow:exported_model_proto_cc",
        "//tensorflow/compiler/mlir/quantization/tensorflow:quantization_options_proto_cc",
        "//tensorflow/compiler/mlir/quantization/tensorflow/calibrator:calibration_statistics_proto_cc",
        "//tensorflow/core/protobuf:for_core_protos_cc",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings:string_view",
    ],
)

cc_library(
    name = "unfreeze_constants",
    srcs = ["unfreeze_constants.cc"],
    hdrs = ["unfreeze_constants.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        "//tensorflow/compiler/mlir/quantization/tensorflow:tf_passes",
        "//tensorflow/compiler/mlir/quantization/tensorflow/cc:run_passes",
        "//tensorflow/compiler/mlir/quantization/tensorflow/cc:save_variables",
        "//tensorflow/core:lib",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:string_view",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@local_xla//xla/tsl/platform:errors",
        "@local_xla//xla/tsl/platform:status",
        "@local_xla//xla/tsl/platform:statusor",
    ],
)

tf_python_pybind_extension(
    name = "pywrap_function_lib",
    srcs = ["pywrap_function_lib.cc"],
    pytype_srcs = ["pywrap_function_lib.pyi"],
    visibility = [
        "__subpackages__",
        "//tensorflow/python:__pkg__",
        "//tensorflow/tools/pip_package:__subpackages__",
    ],
    deps = [
        ":py_function_lib",
        ":type_casters",
        "//tensorflow/compiler/mlir/quantization/stablehlo:quantization_config_proto_cc",
        "//tensorflow/compiler/mlir/quantization/stablehlo/cc/calibration:min_max_value",
        "//tensorflow/compiler/mlir/quantization/tensorflow:exported_model_proto_cc",
        "//tensorflow/compiler/mlir/quantization/tensorflow:quantization_options_proto_cc",
        "//tensorflow/compiler/mlir/quantization/tensorflow/calibrator:calibration_statistics_proto_cc",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/protobuf:for_core_protos_cc",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:string_view",
        "@pybind11",
    ],
)

# Exports python symbols via pybind11.
tf_python_pybind_extension(
    name = "pywrap_quantize_model",
    srcs = ["pywrap_quantize_model.cc"],
    pytype_srcs = ["pywrap_quantize_model.pyi"],
    # All deps must be header-only.
    deps = [
        ":py_function_lib",
        ":type_casters",
        "//tensorflow/compiler/mlir/quantization/tensorflow:exported_model_proto_cc",
        "//tensorflow/compiler/mlir/quantization/tensorflow:quantization_options_proto_cc",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/protobuf:for_core_protos_cc",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@pybind11",
        "@pybind11_abseil//pybind11_abseil:absl_casters",
        "@pybind11_abseil//pybind11_abseil:import_status_module",
        "@pybind11_abseil//pybind11_abseil:status_casters",
        "@pybind11_protobuf//pybind11_protobuf:native_proto_caster",
    ] + if_pywrap(
        [":quantize_model_cc_impl"],
        [":quantize_model_cc"],
    ),
)

tf_py_strict_test(
    name = "pywrap_quantize_model_test",
    srcs = [
        "pywrap_quantize_model_test.py",
    ],
    deps = [
        ":py_function_lib_py",
        ":pywrap_quantize_model",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python/platform:client_testlib",
    ],
)

pytype_strict_library(
    name = "save_model",
    srcs = [
        "save_model.py",
    ],
    deps = [
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python/client:session",
        "//tensorflow/python/framework:importer",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/lib/io:file_io",
        "//tensorflow/python/saved_model:builder",
        "//tensorflow/python/saved_model:constants",
        "//tensorflow/python/saved_model:loader",
        "//tensorflow/python/saved_model:tag_constants",
        "//tensorflow/python/training:saver",
        "@absl_py//absl/logging",
    ],
)

pytype_strict_library(
    name = "quantize_model",
    srcs = [
        "quantize_model.py",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":py_function_lib_py",
        ":pywrap_quantize_model",
        ":representative_dataset",
        ":save_model",
        "//tensorflow/compiler/mlir/quantization/stablehlo:quantization_config_proto_py",
        "//tensorflow/compiler/mlir/quantization/tensorflow:quantization_options_proto_py",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python/lib/io:file_io",
        "//tensorflow/python/saved_model:load",
        "//tensorflow/python/saved_model:loader",
        "//tensorflow/python/saved_model:signature_constants",
        "//tensorflow/python/saved_model:tag_constants",
        "//tensorflow/python/trackable:autotrackable",
        "//tensorflow/python/util:tf_export",
        "@absl_py//absl/logging",
    ],
)

tf_py_strict_test(
    name = "quantize_model_test",
    size = "medium",
    srcs = ["integration_test/quantize_model_test.py"],
    shard_count = 50,  # Parallelize the test to avoid timeouts.
    tags = [
        "no_mac",  # TODO(b/292100835): Reenable
    ],
    deps = [
        ":quantize_model",
        ":quantize_model_test_base",
        ":representative_dataset",
        ":save_model",
        "//tensorflow:tensorflow_py",
        "//tensorflow/compiler/mlir/quantization/common/python:testing",
        "//tensorflow/compiler/mlir/quantization/stablehlo:quantization_config_proto_py",
        "//tensorflow/compiler/mlir/quantization/tensorflow:quantization_options_proto_py",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python/client:session",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:tensor_shape",
        "//tensorflow/python/framework:tensor_spec",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/lib/io:file_io",
        "//tensorflow/python/lib/io:tf_record",
        "//tensorflow/python/module",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/ops:nn_ops",
        "//tensorflow/python/ops:random_ops",
        "//tensorflow/python/ops:variables",
        "//tensorflow/python/platform:client_testlib",
        "//tensorflow/python/platform:tf_logging",
        "//tensorflow/python/saved_model:builder",
        "//tensorflow/python/saved_model:loader",
        "//tensorflow/python/saved_model:save",
        "//tensorflow/python/saved_model:save_options",
        "//tensorflow/python/saved_model:signature_constants",
        "//tensorflow/python/saved_model:signature_def_utils",
        "//tensorflow/python/saved_model:tag_constants",
        "//tensorflow/python/training:checkpoint_utils",
        "//tensorflow/python/types:core",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

pytype_strict_library(
    name = "quantize_model_test_base",
    testonly = 1,
    srcs = ["integration_test/quantize_model_test_base.py"],
    tags = ["no_pip"],
    deps = [
        ":representative_dataset",
        "//tensorflow:tensorflow_py",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python/client:session",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:tensor_spec",
        "//tensorflow/python/lib/io:file_io",
        "//tensorflow/python/module",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:array_ops_stack",
        "//tensorflow/python/ops:io_ops",
        "//tensorflow/python/ops:lookup_ops",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/ops:nn_ops",
        "//tensorflow/python/ops:random_ops",
        "//tensorflow/python/ops:string_ops",
        "//tensorflow/python/ops:variables",
        "//tensorflow/python/ops:while_loop",
        "//tensorflow/python/ops/ragged:ragged_string_ops",
        "//tensorflow/python/platform:client_testlib",
        "//tensorflow/python/platform:tf_logging",
        "//tensorflow/python/saved_model:builder",
        "//tensorflow/python/saved_model:save",
        "//tensorflow/python/saved_model:signature_def_utils",
        "//tensorflow/python/trackable:asset",
        "//tensorflow/python/trackable:autotrackable",
        "//tensorflow/python/types:core",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_py_strict_test(
    name = "concurrency_test",
    size = "medium",
    srcs = ["integration_test/concurrency_test.py"],
    deps = [
        ":quantize_model",
        "//tensorflow:tensorflow_py",
        "//tensorflow/compiler/mlir/quantization/tensorflow:quantization_options_proto_py",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:tensor_spec",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/platform:client_testlib",
        "//tensorflow/python/saved_model:save",
        "//tensorflow/python/saved_model:tag_constants",
        "//tensorflow/python/trackable:autotrackable",
    ],
)

pytype_strict_library(
    name = "representative_dataset",
    srcs = [
        "representative_dataset.py",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/compiler/mlir/quantization/tensorflow:quantization_options_proto_py",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python/client:session",
        "//tensorflow/python/data/ops:readers",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/framework:tensor_util",
        "//tensorflow/python/lib/io:python_io",
        "//tensorflow/python/platform:tf_logging",
        "//tensorflow/python/types:core",
        "//tensorflow/python/util:tf_export",
        "//third_party/py/numpy",
    ],
)

tf_py_strict_test(
    name = "representative_dataset_test",
    srcs = ["representative_dataset_test.py"],
    deps = [
        ":representative_dataset",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python/client:session",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/platform:client_testlib",
        "//tensorflow/python/types:core",
        "//third_party/py/numpy",
    ],
)
