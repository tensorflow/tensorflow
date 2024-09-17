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

#include <string>

#include "tensorflow/core/grappler/optimizers/auto_mixed_precision.h"
#include "tensorflow/core/lib/gtl/flatset.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/util/env_var.h"

namespace tensorflow {
namespace grappler {

// Represents the four lists of ops: the allow list, infer list, deny list, and
// clear list. These lists determine which ops are converted to fp16/bf16
// (referred to as 'f16' for short) and which ops stay as fp32.
class AutoMixedPrecisionLists {
 public:
  virtual ~AutoMixedPrecisionLists() {}

  // Returns the set of ops that are considered numerically-safe (for execution
  // in f16), performance-critical, and can run in f16. These ops are always
  // converted to f16.
  virtual gtl::FlatSet<string> AllowList() = 0;
  // Returns the set of ops that can run in f16 and are considered numerically-
  // safe (for execution in f16), but which may be made unsafe by an upstream
  // denylist op.
  virtual gtl::FlatSet<string> InferList() = 0;
  // Returns the set of ops that are considered numerically-dangerous (i.e.,
  // unsafe for execution in f16) and whose effects may also be observed in
  // downstream nodes (e.g. for f16, in Exp -> Add, the Add is unsafe due to
  // the Exp).
  virtual gtl::FlatSet<string> DenyList() = 0;
  // Returns the set of ops that do not have numerically-significant effects
  // (i.e., they are always considered safe for execution in f16 precision), and
  // can run in f16.
  virtual gtl::FlatSet<string> ClearList() = 0;

 protected:
  // Adds or removes ops from list if certain environmental variables are set.
  static void UpdateList(const string& list_name, gtl::FlatSet<string>* list) {
    CHECK(list_name == "ALLOWLIST" || list_name == "INFERLIST" ||  // Crash OK.
          list_name == "DENYLIST" || list_name == "CLEARLIST" ||
          // TODO(reedwm): for bkwds compat; remove when no longer necessary:
          list_name == "WHITELIST" || list_name == "GRAYLIST" ||
          list_name == "BLACKLIST");
    string add_env_var =
        "TF_AUTO_MIXED_PRECISION_GRAPH_REWRITE_" + list_name + "_ADD";
    string remove_env_var =
        "TF_AUTO_MIXED_PRECISION_GRAPH_REWRITE_" + list_name + "_REMOVE";
    string to_add, to_remove;
    TF_CHECK_OK(ReadStringFromEnvVar(add_env_var, "", &to_add));
    TF_CHECK_OK(ReadStringFromEnvVar(remove_env_var, "", &to_remove));
    for (const auto& x : str_util::Split(to_add, ",")) {
      list->insert(x);
    }
    for (const auto& x : str_util::Split(to_remove, ",")) {
      list->erase(x);
    }
  }

  // Subclasses should include these on the ClearList.
  static void AddTensorListOps(gtl::FlatSet<string>* list) {
    // Note: if a data structure op (such as TensorListPopBack) is added here,
    // IsTensorListReaderOp or IsTensorListWriterOp may need to be modified
    // LINT.IfChange
    constexpr const char* tensor_list_ops[] = {
        "TensorListConcat",     "TensorListConcatLists",
        "TensorListConcatV2",   "TensorListGather",
        "TensorListGetItem",    "TensorListPopBack",
        "TensorListPushBack",   "TensorListPushBackBatch",
        "TensorListFromTensor", "TensorListScatter",
        "TensorListScatterV2",  "TensorListScatterIntoExistingList",
        "TensorListSetItem",    "TensorListSplit",
        "TensorListStack"};
    // LINT.ThenChange(//tensorflow/core/grappler/optimizers/auto_mixed_precision.cc)
    for (auto op : tensor_list_ops) {
      list->insert(op);
    }
  }
};

class AutoMixedPrecisionListsFp16 : public AutoMixedPrecisionLists {
 private:
  static bool IsPseudoFastMath() {
    string optimization_level;
    TF_CHECK_OK(
        ReadStringFromEnvVar("TF_AUTO_MIXED_PRECISION_GRAPH_REWRITE_LEVEL", "",
                             &optimization_level));
    optimization_level = str_util::Uppercase(optimization_level);
    return optimization_level == "TENSOR_CORES_ONLY";
  }

 public:
  AutoMixedPrecisionListsFp16(
      int cuda_version, int cudnn_version,
      AutoMixedPrecisionMode mode = AutoMixedPrecisionMode::CUDA)
      : cuda_version_(cuda_version), cudnn_version_(cudnn_version) {
    if (mode == AutoMixedPrecisionMode::CUDA ||
        mode == AutoMixedPrecisionMode::CPU) {
      // Note: this is not a typo here. use_cuda_ is set to true for the CPU
      // intentionally to make CPU and GPU have the same fp16 ops.
      use_cuda_ = true;
      use_onednn_ = false;
    } else if (mode == AutoMixedPrecisionMode::FP16_CPU) {
      use_onednn_ = true;
      use_cuda_ = false;
    }
  }

  gtl::FlatSet<string> AllowList() override {
    auto list = gtl::FlatSet<string>{
        "Conv2D", "Conv2DBackpropFilter", "Conv2DBackpropInput", "Einsum",
        "MatMul",
    };
    if (use_cuda_) {
      list.insert("BlockLSTM");
      list.insert("BlockLSTMV2");
      list.insert("BlockLSTMGrad");
      list.insert("BlockLSTMGradV2");
      list.insert("CudnnRNN");
      list.insert("CudnnRNNBackprop");
      list.insert("CudnnRNNBackpropV2");
      list.insert("CudnnRNNBackpropV3");
      list.insert("CudnnRNNV2");
      list.insert("CudnnRNNV3");
      list.insert("FusedConv2DBiasActivation");
      list.insert("FusedSparseConvGpuV2");
      list.insert("GRUBlockCell");
      list.insert("GRUBlockCellGrad");
      list.insert("LSTMBlockCell");
      list.insert("LSTMBlockCellGrad");
      list.insert("Mha");
      list.insert("MhaV2");
      list.insert("Tmlp");
      list.insert("TmlpV2");
      list.insert("TmlpV3");
      list.insert("Pmlp");
      list.insert("FastUnsortedSegmentMax");
    }
#if TENSORFLOW_USE_ROCM
    if (true) {
#else
    if ((use_cuda_ && cuda_version_ >= 9010) || use_onednn_) {
      // Fp16 BatchMatMul is slow before CUDA 9.1.
#endif
      list.insert("BatchMatMul");
      list.insert("BatchMatMulV2");
    }
    if ((use_cuda_ && cudnn_version_ >= 7602) || use_onednn_) {
      // Fp16 3D conv is slow before CUDNN 7.6.2.
      list.insert("Conv3D");
      list.insert("Conv3DBackpropFilter");
      list.insert("Conv3DBackpropFilterV2");
      list.insert("Conv3DBackpropInput");
      list.insert("Conv3DBackpropInputV2");
    }
    if ((use_cuda_ && cudnn_version_ >= 8000) || use_onednn_) {
      list.insert("DepthwiseConv2dNative");
      list.insert("DepthwiseConv2dNativeBackpropFilter");
      list.insert("DepthwiseConv2dNativeBackpropInput");
    }
    UpdateList("ALLOWLIST", &list);
    // For backwards compatibility, keeping the original env variable here.
    // TODO(reedwm): This should be removed if we don't have active users.
    UpdateList("WHITELIST", &list);

    return list;
  }

  gtl::FlatSet<string> InferList() override {
    if (IsPseudoFastMath() && use_cuda_) {
      return gtl::FlatSet<string>{};
    }

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
        "Log",
        "Log1p",
        "LogSoftmax",
        "Mul",
        "Prod",
        "RealDiv",
        "Reciprocal",
        "Selu",
        "SeluGrad",
        "Sigmoid",
        "SigmoidGrad",
        "Softmax",
        "Softplus",
        "SoftplusGrad",
        "Softsign",
        "SoftsignGrad",
        "Sqrt",
        "Sub",
        "Tanh",
        "TanhGrad",
    };
    if (use_onednn_) {
      list.insert("Rsqrt");
      list.insert("Square");
      list.insert("SquaredDifference");
    }
    UpdateList("INFERLIST", &list);
    // For backwards compatibility, keeping the original env variable here.
    // TODO(reedwm): This should be removed if we don't have active users.
    UpdateList("GRAYLIST", &list);
    return list;
  }

  gtl::FlatSet<string> DenyList() override {
    if (IsPseudoFastMath() && use_cuda_) {
      return gtl::FlatSet<string>{};
    }

    auto list = gtl::FlatSet<string>{
        "Exp",
        "Expm1",
        "L2Loss",
        "Mean",
        "Pow",
        "SaveV2",
        "SoftmaxCrossEntropyWithLogits",
        "SparseSoftmaxCrossEntropyWithLogits",
        "Sum",
    };
    UpdateList("DENYLIST", &list);
    // For backwards compatibility, keeping the original env variable here.
    // TODO(reedwm): This should be removed if we don't have active users.
    UpdateList("BLACKLIST", &list);
    return list;
  }

  gtl::FlatSet<string> ClearList() override {
    if (IsPseudoFastMath() && use_cuda_) {
      return gtl::FlatSet<string>{};
    }

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
        "SelectV2",
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
        "StopGradient",
        "StridedSlice",
        "StridedSliceGrad",
        "Switch",
        "Tile",
        "TopK",
        "TopKV2",
        "Transpose",
        "Unpack",
        "Where",
        "ZerosLike",
    };
    AddTensorListOps(&list);
    UpdateList("CLEARLIST", &list);
    return list;
  }

 private:
  int cuda_version_;
  int cudnn_version_;
  bool use_cuda_;
  bool use_onednn_;
};

// TODO(reedwm): Remove this alias. Some Google-internal code still uses the
// AutoMixedPrecisionListsCuda name.
using AutoMixedPrecisionListsCuda = AutoMixedPrecisionListsFp16;

class AutoMixedPrecisionListsMkl : public AutoMixedPrecisionLists {
 public:
  AutoMixedPrecisionListsMkl() {}

  // Only ops which are supported by MKL in bfloat16 should be added to the
  // allow list, infer list, or clear list.
  gtl::FlatSet<string> AllowList() override {
    auto list = gtl::FlatSet<string>{"Conv2D",
                                     "Conv2DBackpropFilter",
                                     "Conv2DBackpropInput",
                                     "Conv3D",
                                     "Conv3DBackpropFilterV2",
                                     "Conv3DBackpropInputV2",
                                     "DepthwiseConv2dNative",
                                     "DepthwiseConv2dNativeBackpropFilter",
                                     "DepthwiseConv2dNativeBackpropInput",
                                     "MatMul",
                                     "FusedPadConv2D",
                                     "BatchMatMul",
                                     "BatchMatMulV2",
                                     "Einsum"};

    UpdateList("ALLOWLIST", &list);
    // For backwards compatibility, keeping the original env variable here.
    // TODO(reedwm): This should be removed if we don't have active users.
    UpdateList("WHITELIST", &list);
    return list;
  }

  gtl::FlatSet<string> InferList() override {
    auto list = gtl::FlatSet<string>{"Add",
                                     "AddN",
                                     "AddV2",
                                     "AvgPool",
                                     "AvgPool3D",
                                     "AvgPool3DGrad",
                                     "AvgPoolGrad",
                                     "BiasAdd",
                                     "BiasAddGrad",
                                     "BiasAddV1",
                                     "Erf",
                                     "Erfc",
                                     "FusedBatchNormV2",
                                     "FusedBatchNormGradV2",
                                     "FusedBatchNormV3",
                                     "FusedBatchNormGradV3",
                                     "LeakyRelu",
                                     "LeakyReluGrad",
                                     "Mul",
                                     "Sub",
                                     "Elu",
                                     "EluGrad",
                                     "FloorDiv",
                                     "_FusedBatchNormEx",
                                     "Inv",
                                     "Log",
                                     "Log1p",
                                     "LogSoftmax",
                                     "Mean",
                                     "Prod",
                                     "RealDiv",
                                     "Reciprocal",
                                     "Rsqrt",
                                     "Selu",
                                     "SeluGrad",
                                     "Sigmoid",
                                     "SigmoidGrad",
                                     "Softmax",
                                     "Softplus",
                                     "SoftplusGrad",
                                     "Softsign",
                                     "SoftsignGrad",
                                     "Sqrt",
                                     "Square",
                                     "SquaredDifference",
                                     "Sum",
                                     "Tanh",
                                     "TanhGrad"};
    UpdateList("INFERLIST", &list);
    // For backwards compatibility, keeping the original env variable here.
    // TODO(reedwm): This should be removed if we don't have active users.
    UpdateList("GRAYLIST", &list);
    return list;
  }

  gtl::FlatSet<string> DenyList() override {
    auto list = gtl::FlatSet<string>{
        "Exp",
        "Expm1",
        "L2Loss",
        "Pow",
        "SaveV2",
        "SoftmaxCrossEntropyWithLogits",
        "SparseSoftmaxCrossEntropyWithLogits",
    };
    UpdateList("DENYLIST", &list);
    // For backwards compatibility, keeping the original env variable here.
    // TODO(reedwm): This should be removed if we don't have active users.
    UpdateList("BLACKLIST", &list);
    return list;
  }

  gtl::FlatSet<string> ClearList() override {
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
        "EnsureShape",
        "Enter",
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
        "Maximum",
        "MaxPool",
        "MaxPool3D",
        "MaxPool3DGrad",
        "MaxPoolGrad",
        "MaxPoolGradGrad",
        "MaxPoolGradGradV2",
        "MaxPoolGradV2",
        "MaxPoolV2",
        "Merge",
        "Min",
        "Minimum",
        "MirrorPad",
        "MirrorPadGrad",
        "Neg",
        "NextIteration",
        "NotEqual",
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
        "ResizeBilinear",
        "Reverse",
        "ReverseSequence",
        "ReverseV2",
        "Round",
        "ScatterNd",
        "Select",
        "SelectV2",
        "Shape",
        "ShapeN",
        "Sign",
        "Slice",
        "Snapshot",
        "SpaceToBatch",
        "SpaceToBatchND",
        "SpaceToDepth",
        "Split",
        "SplitV",
        "Squeeze",
        "StatelessWhile",
        "StopGradient",
        "StridedSlice",
        "StridedSliceGrad",
        "Switch",
        "Tile",
        "TopK",
        "TopKV2",
        "Transpose",
        "Where",
        "While",
        "Unpack",
        "ZerosLike",
    };
    AddTensorListOps(&list);
    UpdateList("CLEARLIST", &list);
    return list;
  }
};

}  // end namespace grappler
}  // end namespace tensorflow

#endif  // TENSORFLOW_CORE_GRAPPLER_OPTIMIZERS_AUTO_MIXED_PRECISION_LISTS_H_
