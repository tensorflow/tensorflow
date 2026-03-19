load("//tensorflow:pytype.default.bzl", "pytype_strict_library")
load("//tensorflow:tensorflow.bzl", "tf_gen_op_wrapper_py")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [
        "//tensorflow:__subpackages__",
    ],
    licenses = ["notice"],
)

pytype_strict_library(
    name = "ops",
    srcs = ["tpu_ops.py"],
    visibility = [
        "//smartass:__subpackages__",
        "//tensorflow:__subpackages__",
    ],
    deps = [
        "//tensorflow/python/framework:for_generated_wrappers",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:tpu_ops_gen",
        "//tensorflow/python/platform:tf_logging",
        "//tensorflow/python/tpu:tpu_function",
        "//tensorflow/python/util:tf_export",
    ],
)

tf_gen_op_wrapper_py(
    name = "gen_tpu_embedding_ops",
    out = "gen_tpu_embedding_ops.py",
    extra_py_deps = [
        "//tensorflow/python:pywrap_tfe",
        "//tensorflow/python/util:dispatch",
        "//tensorflow/python/util:deprecation",
        "//tensorflow/python/util:tf_export",
    ],
    py_lib_rule = pytype_strict_library,
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/core/ops:tpu_configuration_ops_op_lib",
        "//tensorflow/core/ops:tpu_embedding_ops_op_lib",
        "//tensorflow/core/tpu/ops:tpu_embedding_ops",
    ],
)

tf_gen_op_wrapper_py(
    name = "gen_xla_ops",
    out = "gen_xla_ops.py",
    api_def_srcs = [
        "//tensorflow/core/api_def:base_api_def",
        "//tensorflow/core/api_def:python_api_def",
    ],
    op_allowlist = [
        "ConvertToCooTensor",
        "GetMinibatchesInCsrWithPhysicalReplica",
        "GetMinibatchSplitsWithPhysicalReplica",
        "StoreMinibatchStatisticsInFdo",
        "ConvertToListOfSparseCoreCooTensors",
        "SortListOfSparseCoreCooTensors",
        "ConvertToSparseCoreCsrWrappedCooTensor",
        "GetStatsFromListOfSparseCoreCooTensors",
        "GlobalIterId",
        "TPUCopyWithDynamicShape",
        "TPUAnnotateTensorsWithDynamicShape",
        "XlaSparseDenseMatmul",
        "XlaSparseDenseMatmulWithCsrInput",
        "XlaSparseDenseMatmulIntWithCsrInput",
        "XlaSparseDenseMatmulCustomCombinerOnTcWithCsrInput",
        "XlaSparseDenseMatmulGrad",
        "XlaSparseDenseMatmulIntGradWithCsrInput",
        "XlaSparseDenseMatmulGradWithCsrInput",
        "XlaSparseDenseMatmulCustomCombinerOnTcGradWithSgdAndCsrInput",
        "XlaSparseDenseMatmulCustomCombinerOnTcGradWithAdagradAndCsrInput",
        "XlaSparseDenseMatmulCustomCombinerOnTcGradWithAdagradMomentumAndCsrInput",
        "XlaSparseDenseMatmulCustomCombinerOnTcGradWithAdamAndCsrInput",
        "XlaSparseDenseMatmulCustomCombinerOnTcGradWithFtrlAndCsrInput",
        "XlaSparseDenseMatmulCustomCombinerOnTcGradWithCsrInput",
        "XlaSparseDenseMatmulGradWithSgdAndCsrInput",
        "XlaSparseDenseMatmulGradWithAdagradAndCsrInput",
        "XlaSparseDenseMatmulGradWithAdagradMomentumAndCsrInput",
        "XlaSparseDenseMatmulGradWithAdamAndCsrInput",
        "XlaSparseDenseMatmulGradWithFtrlAndCsrInput",
        "XlaSparseDenseMatmulWithStaticBufferSize",
        "XlaSparseDenseMatmulGradWithSgdAndStaticBufferSize",
        "XlaSparseDenseMatmulGradWithAdagradAndStaticBufferSize",
        "XlaSparseDenseMatmulGradWithAdagradMomentumAndStaticBufferSize",
        "XlaSparseDenseMatmulGradWithAdamAndStaticBufferSize",
        "XlaSparseDenseMatmulGradWithFtrlAndStaticBufferSize",
        "XlaSparseCoreSgd",
        "XlaSparseCoreAdagrad",
        "XlaSparseCoreAdagradMomentum",
        "XlaSparseCoreAdam",
        "XlaSparseCoreFtrl",
    ],
    deps = [
        "//tensorflow/core/tpu/ops:sparse_core_ops",
        "//tensorflow/core/tpu/ops:sparse_core_preprocess_ops",
        "//tensorflow/core/tpu/ops:tpu_copy_with_dynamic_shape_op",
    ],
)
