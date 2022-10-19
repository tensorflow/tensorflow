load("//tensorflow/tsl/platform:rules_cc.bzl", "cc_library")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

cc_library(
    name = "translate_cl_registration",
    testonly = True,
    srcs = ["xla_mlir_translate_registration.cc"],
    deps = [
        "//tensorflow/compiler/jit:xla_cpu_jit",
        "//tensorflow/compiler/jit:xla_gpu_jit",
        "//tensorflow/compiler/mlir/xla:mhlo_to_lhlo_with_xla",
        "@llvm-project//mlir:TranslateLib",
    ],
    alwayslink = 1,
)
