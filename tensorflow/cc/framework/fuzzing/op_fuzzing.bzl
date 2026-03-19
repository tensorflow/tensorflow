"""Functions for automatically generating fuzzers."""

load(
    "//tensorflow:tensorflow.bzl",
    "if_not_windows",
    "lrt_if_needed",
    "tf_cc_binary",
    "tf_copts",
)
load(
    "//tensorflow/core/platform:rules_cc.bzl",
    "cc_test",
)

def tf_gen_op_wrappers_fuzz(
        name,
        op_def_src,
        api_def_srcs = [],
        kernel_deps = []):
    """
    Generates fuzzers for several groups of ops.

    For each one we need the corresponding OpDef, ApiDef and KernelDef,
    since they all can contain constraints for the inputs.

    Args:
        name: the name of the fuzz artifact
        op_def_src: op definitions
        api_def_srcs: api definitions
        kernel_deps: op kernel dependencies
    """

    # Create tool to generate .cc fuzzer files.
    tf_cc_binary(
        name = "op_fuzz_gen_tool",
        copts = tf_copts(),
        linkopts = if_not_windows(["-lm", "-Wl,-ldl"]) + lrt_if_needed(),
        linkstatic = 1,  # Faster to link this one-time-use binary dynamically
        deps = [
            "//tensorflow/cc/framework/fuzzing:cc_op_fuzz_gen_main",
            op_def_src,
        ] + kernel_deps,
    )

    # Add relevant locations to look for api_defs.
    api_def_src_locations = ",".join(["$$(dirname $$(echo $(locations " + api_def_src + ") | cut -d\" \" -f1))" for api_def_src in api_def_srcs])

    out_fuzz_files = [op_name + "_fuzz.cc" for op_name in op_names]
    native.genrule(
        name = name + "_genrule",
        outs = out_fuzz_files,
        srcs = api_def_srcs,
        tools = [":op_fuzz_gen_tool"],
        cmd = ("$(location :op_fuzz_gen_tool) " +
               " $$(dirname $(location " + out_fuzz_files[0] + "))" +
               " " + api_def_src_locations + " " + (",".join(op_names))),
    )

    for op_name in op_names:
        cc_test(
            name = op_name.lower() + "_fuzz",
            srcs = [op_name + "_fuzz.cc"],
            deps = kernel_deps +
                   [
                       "//tensorflow/security/fuzzing/cc:fuzz_session",
                       "@com_google_googletest//:gtest_main",
                       "@com_google_fuzztest//fuzztest",
                       "//tensorflow/cc:cc_ops",
                       "//third_party/mediapipe/framework/port:parse_text_proto",
                   ],
        )

op_names = [
    "BatchMatrixBandPart",
    "BatchMatrixDiag",
    "BatchMatrixDiagPart",
    "BatchMatrixSetDiag",
    "BatchToSpace",
    "BatchToSpaceND",
    "Bitcast",
    "BroadcastArgs",
    "BroadcastTo",
    "CheckNumerics",
    "ConcatV2",
    "ConjugateTranspose",
    "DebugGradientIdentity",
    "DeepCopy",
    "DepthToSpace",
    "Dequantize",
    "EditDistance",
    "Empty",
    "EnsureShape",
    "ExpandDims",
    "ExtractImagePatches",
    "ExtractVolumePatches",
    "FakeQuantWithMinMaxArgs",
    "FakeQuantWithMinMaxArgsGradient",
    "FakeQuantWithMinMaxVars",
    "FakeQuantWithMinMaxVarsGradient",
    "FakeQuantWithMinMaxVarsPerChannel",
    "FakeQuantWithMinMaxVarsPerChannelGradient",
    "Fill",
    "Fingerprint",
    "Gather",
    "GuaranteeConst",
    "Identity",
    "IdentityN",
    "InplaceAdd",
    "InplaceSub",
    "InplaceUpdate",
    "InvertPermutation",
    "ListDiff",
    "MatrixBandPart",
    "MatrixDiag",
    "MatrixDiagPart",
    "MatrixDiagPartV2",
    "MatrixDiagPartV3",
    "MatrixDiagV2",
    "MatrixDiagV3",
    "MatrixSetDiag",
    "MatrixSetDiagV2",
    "MatrixSetDiagV3",
    "MirrorPad",
    "OneHot",
    "OnesLike",
    "Pack",
    "Pad",
    "PadV2",
    "ParallelConcat",
    "PlaceholderV2",
    "PlaceholderWithDefault",
    "PreventGradient",
    "QuantizeAndDequantize",
    "QuantizeV2",
    "Rank",
    "Reshape",
    "ResourceStridedSliceAssign",
    "ReverseSequence",
    "ReverseV2",
    "ScatterNdNonAliasingAdd",
    "Shape",
    "ShapeN",
    "Size",
    "Slice",
    "Snapshot",
    "SpaceToBatch",
    "SpaceToBatchND",
    "SpaceToDepth",
    "Split",
    "SplitV",
    "Squeeze",
    "StopGradient",
    "StridedSlice",
    "StridedSliceGrad",
    "TensorScatterAdd",
    "TensorScatterMax",
    "TensorScatterMin",
    "TensorScatterSub",
    "TensorStridedSliceUpdate",
    "Tile",
    "TileGrad",
    "Transpose",
    "Unique",
    "UniqueV2",
    "UniqueWithCounts",
    "UniqueWithCountsV2",
    "Unpack",
    "UnravelIndex",
    "Where",
    "ZerosLike",
]
