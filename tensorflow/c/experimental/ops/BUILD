load(
    "//tensorflow:tensorflow.bzl",
    "genrule",
    "tf_binary_additional_srcs",
)
load("//tensorflow:tensorflow.bzl", "filegroup")

# buildifier: disable=same-origin-load
load("//tensorflow:tensorflow.bzl", "get_compatible_with_cloud")
load("//tensorflow/core/platform:rules_cc.bzl", "cc_library")

# Experimental ops.
package(
    default_visibility = ["//visibility:private"],
    licenses = ["notice"],
)

OPS = {
    "array": "Identity IdentityN ZerosLike Shape ExpandDims OnesLike",
    "math": "Mul Conj AddV2 MatMul Neg Sum Sub Div DivNoNan Exp Sqrt SqrtGrad Log1p",
    "nn": "SparseSoftmaxCrossEntropyWithLogits ReluGrad Relu BiasAdd BiasAddGrad",
    "resource_variable": "VarHandleOp ReadVariableOp AssignVariableOp DestroyResourceOp",
}

CATEGORIES = OPS.keys()

APIDEF = "//tensorflow/core/api_def/base_api:base_api_def"

APIDIR = "$$(dirname $$(echo $(locations %s) | cut -d\\  -f1)) " % APIDEF

BUILDRULE = "//tensorflow/c/experimental/ops/gen:generate_cpp"

# Rules to generate the source .h/.cc for each category of ops.
[
    genrule(
        name = "gen_%s_ops" % category,
        srcs = [APIDEF],
        outs = [
            "%s_ops.h" % category,
            "%s_ops.cc" % category,
        ],
        cmd = "$(location %s)" % BUILDRULE +
              " --category=" + category +
              " --api_dirs=" + APIDIR +
              " --source_dir=" + "$(rootpath %s)" % BUILDRULE +
              " --output_dir=" + "$(RULEDIR) " +
              OPS[category],
        compatible_with = get_compatible_with_cloud(),
        exec_tools = [BUILDRULE] +
                     # This keeps libtensorflow_framework.so around:
                     tf_binary_additional_srcs(),
    )
    for category in CATEGORIES
]

[
    sh_test(
        name = "gen_%s_ops_test" % category,
        srcs = ["gen_ops_test.sh"],
        args = [category],
        data = [
            "%s_ops.h" % category,
            "%s_ops.cc" % category,
        ],
    )
    for category in CATEGORIES
]

[
    cc_library(
        name = "%s_ops" % category,
        srcs = ["%s_ops.cc" % category],
        hdrs = [
            "%s_ops.h" % category,
        ],
        visibility = [
            "//tensorflow:internal",
        ],
        deps = [
            "//tensorflow/c/eager:abstract_context",
            "//tensorflow/c/eager:abstract_operation",
            "//tensorflow/c/eager:abstract_tensor_handle",
            "//tensorflow/c/eager:c_api_unified_internal",
            "//tensorflow/c/eager:tracing_utils",
            "//tensorflow/core:framework",
            "//tensorflow/core/platform:errors",
            "@com_google_absl//absl/types:span",
        ],
    )
    for category in CATEGORIES
]

cc_library(
    name = "ops",
    hdrs = [
        "array_ops.h",
        "math_ops.h",
        "nn_ops.h",
        "resource_variable_ops.h",
    ],
    visibility = [
        "//tensorflow:internal",
    ],
    deps = [
        ":array_ops",
        ":math_ops",
        ":nn_ops",
        ":resource_variable_ops",
        "//tensorflow/c/eager:abstract_context",
        "//tensorflow/c/eager:abstract_operation",
        "//tensorflow/c/eager:abstract_tensor_handle",
        "//tensorflow/c/eager:c_api_unified_internal",
    ],
)

filegroup(
    name = "pywrap_required_hdrs",
    srcs = [
        "array_ops.h",
        "math_ops.h",
        "nn_ops.h",
        "resource_variable_ops.h",
    ],
    visibility = [
        "//tensorflow/core:__pkg__",
        "//tensorflow/python:__subpackages__",
    ],
)
