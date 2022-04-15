load("//tensorflow/core/platform:rules_cc.bzl", "cc_library")

package(
    default_visibility = [
        "//visibility:public",
    ],
    licenses = ["notice"],
)

cc_library(
    name = "utils",
    srcs = [
        "utils.cc",
    ],
    hdrs = [
        "utils.h",
    ],
    deps = [
        "//tensorflow/compiler/mlir/lite:flatbuffer_translate_lib",
        "//tensorflow/compiler/mlir/lite:tensorflow_lite",
        "//tensorflow/compiler/mlir/tensorflow:error_util",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:str_format",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ArithmeticDialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Parser",
        "@llvm-project//mlir:Support",
    ],
)
