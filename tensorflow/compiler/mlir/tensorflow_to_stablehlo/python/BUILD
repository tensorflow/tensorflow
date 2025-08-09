load("@local_xla//xla/tsl/platform:build_config_root.bzl", "if_pywrap")
load("@rules_cc//cc:cc_library.bzl", "cc_library")
load(
    "//tensorflow:tensorflow.default.bzl",
    "get_compatible_with_portable",
    "tf_py_strict_test",
    "tf_python_pybind_extension",
)
load("//tensorflow/compiler/mlir/quantization/stablehlo:internal_visibility_allowlist.bzl", "internal_visibility_allowlist")

package_group(
    name = "internal_visibility_allowlist_package",
    packages = [
        "//tensorflow/compiler/mlir/lite/...",
        "//tensorflow/compiler/mlir/quantization/...",
        "//tensorflow/compiler/mlir/tf2xla/transforms/...",
        "//tensorflow/lite/...",
    ] + internal_visibility_allowlist(),
)

package(
    # copybara:uncomment default_applicable_licenses = ["@stablehlo//:license"],
    default_visibility = [
        ":internal_visibility_allowlist_package",
        "//tensorflow:__pkg__",
        "//tensorflow/python:__pkg__",
    ],
    licenses = ["notice"],
)

# copybara:uncomment_begin(google-only)
# tf_py_strict_test(
#     name = "tensorflow_to_stablehlo_test",
#     testonly = 1,
#     srcs = ["integration_test/tensorflow_to_stablehlo_test.py"],
#     deps = [
#         ":pywrap_tensorflow_to_stablehlo",
#         "//testing/pymocks:matchers",
#         "//third_party/py/mlir",
#         "//third_party/py/mlir:ir",
#         "//third_party/py/mlir:stablehlo_dialect",
#         "//third_party/py/mlir/_mlir_libs:_mlirRegisterEverything",
#         "//tensorflow:tensorflow_py",
#         "//tensorflow/compiler/mlir/stablehlo",
#         "//tensorflow/python/framework:test_lib",
#         "//tensorflow/python/platform:client_testlib",
#         "//tensorflow/python/types:core",
#     ],
# )
# copybara:uncomment_end

# This is a header-only target. The purpose of `pywrap_tensorflow_to_stablehlo_lib_*` targets is to expose only
# the symbols that are required by `pywrap_tensorflow_to_stablehlo` that translates them to python functions.
# The only intended use case of this library is by `pywrap_tensorflow_to_stablehlo`. Not letting
# `pywrap_tensorflow_to_stablehlo` directly depend on sub-libraries like `static_range_srq` and instead haiving
# a consolidated impl library `pywrap_tensorflow_to_stablehlo_lib_impl` allows the maintainers to avoid
# declaring multiple impl libraries to `libtensorflow_cc` and `lib_pywrap_tensorflow_internal`,
# which is required to avoid ODR violations.
cc_library(
    name = "pywrap_tensorflow_to_stablehlo_lib_header_only",
    srcs = [],
    hdrs = ["pywrap_tensorflow_to_stablehlo_lib.h"],
    compatible_with = get_compatible_with_portable(),
    visibility = ["//visibility:private"],  # ONLY for `pywrap_tensorflow_to_stablehlo`.
    deps = [
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
    ],
)

# See the comments for `pywrap_tensorflow_to_stablehlo_lib_header_only`.
cc_library(
    name = "pywrap_tensorflow_to_stablehlo_lib_impl",
    srcs = ["pywrap_tensorflow_to_stablehlo_lib.cc"],
    hdrs = ["pywrap_tensorflow_to_stablehlo_lib.h"],
    compatible_with = get_compatible_with_portable(),
    visibility = [
        "//tensorflow:__pkg__",  # For libtensorflow_cc.so.
        "//tensorflow/python:__pkg__",  # For lib_pywrap_tensorflow_internal.so.
    ],
    deps = [
        "//tensorflow/compiler/mlir/tensorflow",
        "//tensorflow/compiler/mlir/tensorflow_to_stablehlo:tf_to_stablehlo",
        "//tensorflow/core:lib",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:BytecodeWriter",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Parser",
        "@llvm-project//mlir:Support",
        "@local_xla//third_party/python_runtime:headers",
    ],
)

tf_python_pybind_extension(
    name = "pywrap_tensorflow_to_stablehlo",
    srcs = ["pywrap_tensorflow_to_stablehlo.cc"],
    pytype_srcs = ["pywrap_tensorflow_to_stablehlo.pyi"],
    # Each dependency MUST be either header-only or exclusive.
    deps = [
        "@com_google_absl//absl/strings:string_view",
        "@local_xla//third_party/python_runtime:headers",
        "@pybind11",
        "@pybind11_abseil//pybind11_abseil:absl_casters",
        "@pybind11_abseil//pybind11_abseil:status_casters",
    ] + if_pywrap(
        if_false = [":pywrap_tensorflow_to_stablehlo_lib_header_only"],
        if_true = [":pywrap_tensorflow_to_stablehlo_lib_impl"],
    ),
)
