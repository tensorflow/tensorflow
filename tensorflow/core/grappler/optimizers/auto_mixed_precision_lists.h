/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef TENSORFLOW_CORE_GRAPPLER_OPTIMIZERS_AUTO_MIXED_PRECISION_LISTS_H_
#define TENSORFLOW_CORE_GRAPPLER_OPTIMIZERS_AUTO_MIXED_PRECISION_LISTS_H_

#include "tensorflow/core/lib/gtl/flatset.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/util/env_var.h"

namespace tensorflow {
namespace grappler {

class AutoMixedPrecisionLists {
 private:
  static void UpdateList(gtl::FlatSet<string>* list, const string& to_add,
                         const string& to_remove) {
    for (auto x : str_util::Split(to_add, ",")) {
      list->insert(x);
    }
    for (auto x : str_util::Split(to_remove, ",")) {
      list->erase(x);
    }
  }

  static bool IsPseudoFastMath() {
    string optimization_level;
    TF_CHECK_OK(
        ReadStringFromEnvVar("TF_AUTO_MIXED_PRECISION_GRAPH_REWRITE_LEVEL", "",
                             &optimization_level));
    optimization_level = str_util::Uppercase(optimization_level);
    return optimization_level == "TENSOR_CORES_ONLY";
  }

 public:
  // Returns the set of ops that are considered numerically-safe (for execution
  // in fp16) and performance-critical. These ops are always converted to fp16.
  static gtl::FlatSet<string> WhiteList(int cuda_version) {
    string to_add, to_remove;
    TF_CHECK_OK(ReadStringFromEnvVar(
        "TF_AUTO_MIXED_PRECISION_GRAPH_REWRITE_WHITELIST_ADD", "", &to_add));
    TF_CHECK_OK(ReadStringFromEnvVar(
        "TF_AUTO_MIXED_PRECISION_GRAPH_REWRITE_WHITELIST_REMOVE", "",
        &to_remove));

    auto list = gtl::FlatSet<string> {
          "BlockLSTM", "BlockLSTMGrad", "Conv2D", "Conv2DBackpropFilter",
          "Conv2DBackpropInput",
          // TODO(benbarsdell): Enable these when Tensor Core kernels are
          // available for 3D convolutions.
          // "Conv3D",
          // "Conv3DBackpropFilter",
          // "Conv3DBackpropFilterV2",
          // "Conv3DBackpropInput",
          // "Conv3DBackpropInputV2",
          "CudnnRNN", "CudnnRNNBackprop", "CudnnRNNBackpropV2",
          "CudnnRNNBackpropV3", "CudnnRNNV2", "CudnnRNNV3", "GRUBlockCell",
          "GRUBlockCellGrad", "LSTMBlockCell", "LSTMBlockCellGrad",
          // TODO(benbarsdell): Enable these when fast and safe fp16 kernels are
          // available for depthwise convolutions.
          // "DepthwiseConv2dNative",
          // "DepthwiseConv2dNativeBackpropFilter",
          // "DepthwiseConv2dNativeBackpropInput",
          "MatMul",
    };
    if (cuda_version >= 9010) {
      // Fp16 BatchMatMul is slow before CUDA 9.1.
      list.insert("BatchMatMul");
      list.insert("BatchMatMulV2");
    }
    UpdateList(&list, to_add, to_remove);
    return list;
  }

  // Returns the set of ops that are considered numerically-safe (for execution
  // in fp16), but which may be made unsafe by an upstream blacklist op.
  static gtl::FlatSet<string> GrayList() {
    if (IsPseudoFastMath()) {
      return gtl::FlatSet<string>{};
    }
    string to_add, to_remove;
    TF_CHECK_OK(ReadStringFromEnvVar(
        "TF_AUTO_MIXED_PRECISION_GRAPH_REWRITE_GRAYLIST_ADD", "", &to_add));
    TF_CHECK_OK(ReadStringFromEnvVar(
        "TF_AUTO_MIXED_PRECISION_GRAPH_REWRITE_GRAYLIST_REMOVE", "",
        &to_remove));

    auto list = gtl::FlatSet<string>{
        "Add",
        "AddN",
        "AddV2",
        "AvgPool",
        "AvgPool3D",
        "AvgPool3DGrad",
        "AvgPoolGrad",
        "BiasAdd",
        "BiasAddGrad",
        "BiasAddV1",
        "Elu",
        "EluGrad",
        "Erf",
        "Erfc",
        "FloorDiv",
        "FusedBatchNormV2",
        "FusedBatchNormGradV2",
        "FusedBatchNormV3",
        "FusedBatchNormGradV3",
        "_FusedBatchNormEx",
        "Inv",
        "LeakyRelu",
        "LeakyReluGrad",
        "Mul",
        "Prod",
        "RealDiv",
        "Reciprocal",
        "Sigmoid",
        "SigmoidGrad",
        "Softplus",
        "SoftplusGrad",
        "Sqrt",
        "Sub",
        "Tanh",
        "TanhGrad",
    };
    UpdateList(&list, to_add, to_remove);
    return list;
  }

  // Returns the set of ops that are considered numerically-dangerous (i.e.,
  // unsafe for execution in fp16) and whose effects may also be observed in
  // downstream nodes (e.g., in Exp -> Add, the Add is unsafe due to the Exp).
  static gtl::FlatSet<string> BlackList() {
    if (IsPseudoFastMath()) {
      return gtl::FlatSet<string>{};
    }
    string to_add, to_remove;
    TF_CHECK_OK(ReadStringFromEnvVar(
        "TF_AUTO_MIXED_PRECISION_GRAPH_REWRITE_BLACKLIST_ADD", "", &to_add));
    TF_CHECK_OK(ReadStringFromEnvVar(
        "TF_AUTO_MIXED_PRECISION_GRAPH_REWRITE_BLACKLIST_REMOVE", "",
        &to_remove));

    auto list = gtl::FlatSet<string>{
        "Exp",
        "Expm1",
        "L2Loss",
        "Log",
        "Log1p",
        "LogSoftmax",
        "Mean",
        "Pow",
        "SaveV2",
        "Softmax",
        "SoftmaxCrossEntropyWithLogits",
        "SparseSoftmaxCrossEntropyWithLogits",
        "Sum",
    };
    UpdateList(&list, to_add, to_remove);
    return list;
  }

  // Returns the set of ops that do not have numerically-significant effects
  // (i.e., they are always considered safe for execution in fp16 precision).
  static gtl::FlatSet<string> ClearList() {
    if (IsPseudoFastMath()) {
      return gtl::FlatSet<string>{};
    }
    // Note: if a stateful op (such as StackPopV2) is added to the clearlist,
    // you must also modify the AutoMixedPrecisionImpl class to call
    // AddDataStructureOpsToMap() with that op
    string to_add, to_remove;
    TF_CHECK_OK(ReadStringFromEnvVar(
        "TF_AUTO_MIXED_PRECISION_GRAPH_REWRITE_CLEARLIST_ADD", "", &to_add));
    TF_CHECK_OK(ReadStringFromEnvVar(
        "TF_AUTO_MIXED_PRECISION_GRAPH_REWRITE_CLEARLIST_REMOVE", "",
        &to_remove));

    auto list = gtl::FlatSet<string>{
        "Abs",
        "ArgMax",
        "ArgMin",
        "BatchToSpace",
        "BatchToSpaceND",
        "BroadcastTo",
        "Ceil",
        "CheckNumerics",
        "ClipByValue",
        "Concat",
        "ConcatV2",
        "DepthToSpace",
        "DynamicPartition",
        "DynamicStitch",
        "Enter",
        "EnsureShape",
        "Equal",
        "Exit",
        "ExpandDims",
        "Fill",
        "Floor",
        "Gather",
        "GatherNd",
        "GatherV2",
        "Greater",
        "GreaterEqual",
        "Identity",
        "IdentityN",
        "IsFinite",
        "IsInf",
        "IsNan",
        "Less",
        "LessEqual",
        "Max",
        "MaxPool",
        "MaxPool3D",
        "MaxPool3DGrad",
        "MaxPool3DGradGrad",
        "MaxPoolGrad",
        "MaxPoolGradGrad",
        "MaxPoolGradGradV2",
        "MaxPoolGradV2",
        "MaxPoolV2",
        "Maximum",
        "Merge",
        "Min",
        "Minimum",
        "MirrorPad",
        "MirrorPadGrad",
        "Neg",
        "NextIteration",
        "NotEqual",
        "OneHot",
        "OnesLike",
        "Pack",
        "Pad",
        "PadV2",
        "PreventGradient",
        "Rank",
        "Relu",
        "Relu6",
        "Relu6Grad",
        "ReluGrad",
        "Reshape",
        "ResizeNearestNeighbor",
        "ResizeNearestNeighborGrad",
        "Reverse",
        "ReverseSequence",
        "ReverseV2",
        "Round",
        "Select",
        "Shape",
        "ShapeN",
        "Sign",
        "Size",
        "Slice",
        "Snapshot",
        "SpaceToBatch",
        "SpaceToBatchND",
        "SpaceToDepth",
        "Split",
        "SplitV",
        "Squeeze",
        "StackPopV2",
        "StackPushV2",
        "StopGradient",
        "StridedSlice",
        "StridedSliceGrad",
        "Switch",
        "TensorArrayConcatV3",
        "TensorArrayGatherV3",
        "TensorArrayReadV3",
        "TensorArrayScatterV3",
        "TensorArraySplitV3",
        "TensorArrayWriteV3",
        "Tile",
        "TopK",
        "TopKV2",
        "Transpose",
        "Where",
        "ZerosLike",
    };
    UpdateList(&list, to_add, to_remove);
    return list;
  }
};

}  // end namespace grappler
}  // end namespace tensorflow

#endif  // TENSORFLOW_CORE_GRAPPLER_OPTIMIZERS_AUTO_MIXED_PRECISION_LISTS_H_
