load(
    "@org_tensorflow//:tensorflow.bzl",
    "tf_cc_binary",
    "tf_cc_test",
)

load("@hedron_compile_commands//:refresh_compile_commands.bzl", "refresh_compile_commands")

package(default_visibility = ["//visibility:public"])

tf_cc_binary(
    name = "tf-opt",
    srcs = ["tf-opt.cpp"],
    deps = [
        "@org_tensorflow//tensorflow/compiler/mlir:mlir_cc",
        "@org_tensorflow//tensorflow/compiler/mlir/python:mlir_bindings_cc",
        "@org_tensorflow//tensorflow/compiler/mlir/python:mlir",
        "@org_tensorflow//tensorflow/compiler/mlir/python:mlir_py",
    ],
)

tf_cc_test(
    name = "tf-opt-test",
    srcs = ["tf-opt-test.cpp"],
    deps = [
        "@org_tensorflow//tensorflow/compiler/mlir:mlir_cc",
        "@org_tensorflow//tensorflow/compiler/mlir/python:mlir_bindings_cc",
        "@org_tensorflow//tensorflow/compiler/mlir/python:mlir",
        "@org_tensorflow//tensorflow/compiler/mlir/python:mlir_py",
        "@org_tensorflow//tensorflow/compiler/mlir/python:pywrap_mlir",
    ],
)

refresh_compile_commands(
    name = "tf_opt_compile_commands",
    targets = {"//tensorflow/compiler/mlir:tf-opt": ""}, 
)
