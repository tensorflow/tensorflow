load(
    "//tensorflow:tensorflow.default.bzl",
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

tf_python_pybind_extension(
    name = "pywrap_tensorflow_to_stablehlo",
    srcs = ["pywrap_tensorflow_to_stablehlo.cc"],
    pytype_srcs = ["pywrap_tensorflow_to_stablehlo.pyi"],
    deps = [
        "//tensorflow/compiler/mlir/quantization/tensorflow/python:type_casters",
        "//tensorflow/compiler/mlir/quantization/tensorflow_to_stablehlo:tf_to_stablehlo",
        "//tensorflow/compiler/mlir/tensorflow",
        "@com_google_absl//absl/strings:str_format",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:CAPIIR",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Parser",
        "@llvm-project//mlir:Support",
        "@pybind11",
        "@pybind11_abseil//pybind11_abseil:absl_casters",
        "@pybind11_abseil//pybind11_abseil:status_casters",
    ],
)
