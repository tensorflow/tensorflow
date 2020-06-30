package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],  # Apache 2.0
)

cc_library(
    name = "mlir",
    srcs = ["mlir.cc"],
    hdrs = ["mlir.h"],
    deps = [
        "//tensorflow/c:tf_status",
        "//tensorflow/c:tf_status_helper",
        "//tensorflow/compiler/mlir/tensorflow:convert_graphdef",
        "//tensorflow/compiler/mlir/tensorflow:tensorflow_passes",
        "//tensorflow/compiler/mlir/tensorflow:tf_saved_model_passes",
        "//tensorflow/compiler/mlir/tensorflow:error_util",
        # (yongtang) The graph_optimization_pass_registration needs to be part
        # of a shared object that will be loaded whenever `import tensorflow`
        # is run. The natural place is libtensorflow_framework.so.
        # While adding graph_optimization_pass_registration to
        # libtensorflow_framework.so is possible with some modification in
        # dependency, many tests will fail due to multiple copies of LLVM.
        # See https://github.com/tensorflow/tensorflow/pull/39231 for details.
        # Alternatively, we place graph_optimization_pass_registration here
        # because:
        # - tensorflow/python/_pywrap_mlir.so already depends on LLVM anyway
        # - tensorflow/python/_pywrap_mlir.so always loaded as part of python
        #   binding
        # TODO: It might be still preferrable to place graph_optimization_pass
        # as part of the libtensorflow_framework.so, as it is the central
        # place for core related components.
        "//tensorflow/compiler/mlir/tensorflow:graph_optimization_pass_registration",
        "//tensorflow/compiler/mlir/tensorflow:import_utils",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Parser",
        "@llvm-project//mlir:Pass",
    ],
    alwayslink = 1,
)

filegroup(
    name = "pywrap_mlir_hdrs",
    srcs = [
        "mlir.h",
    ],
    visibility = [
        "//tensorflow/python:__pkg__",
    ],
)
