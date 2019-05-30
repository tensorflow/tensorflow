/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/lite/toco/tflite/op_version.h"

#include <cstring>

#include "tensorflow/lite/toco/model.h"
#include "tensorflow/lite/toco/tflite/operator.h"
#include "tensorflow/lite/toco/tooling_util.h"

namespace toco {
namespace tflite {

string GetMinimumRuntimeVersionForModel(const Model& model) {
  static constexpr char kPendingReleaseOpVersion[] = "UNKNOWN";
  // A map from the version key of an op to its minimum runtime version.
  // For example, {{kAveragePool, 1}, "1.5.0"},  means the 1st version of
  // AveragePool requires a minimum TF Lite runtime version '1.5.0`.
  static const std::map<std::pair<OperatorType, int>, string>* op_version_map =
      new std::map<std::pair<OperatorType, int>, string>(
          {{{OperatorType::kAveragePool, 1}, "1.5.0"},
           {{OperatorType::kAveragePool, 2}, kPendingReleaseOpVersion},
           {{OperatorType::kConv, 1}, "1.5.0"},
           {{OperatorType::kConv, 2}, kPendingReleaseOpVersion},
           {{OperatorType::kConv, 3}, kPendingReleaseOpVersion},
           {{OperatorType::kDepthwiseConv, 1}, "1.5.0"},
           {{OperatorType::kDepthwiseConv, 2}, "1.12.0"},
           {{OperatorType::kDepthwiseConv, 3}, kPendingReleaseOpVersion},
           {{OperatorType::kAdd, 1}, "1.5.0"},
           {{OperatorType::kAdd, 2}, kPendingReleaseOpVersion},
           {{OperatorType::kAddN, 1}, kPendingReleaseOpVersion},
           {{OperatorType::kSpaceToBatchND, 1}, "1.6.0"},
           {{OperatorType::kSpaceToBatchND, 2}, kPendingReleaseOpVersion},
           {{OperatorType::kSub, 1}, "1.6.0"},
           {{OperatorType::kSub, 2}, kPendingReleaseOpVersion},
           {{OperatorType::kDiv, 1}, "1.6.0"},
           {{OperatorType::kBatchToSpaceND, 1}, "1.6.0"},
           {{OperatorType::kBatchToSpaceND, 2}, kPendingReleaseOpVersion},
           {{OperatorType::kCast, 1}, "1.5.0"},
           {{OperatorType::kConcatenation, 1}, "1.5.0"},
           {{OperatorType::kConcatenation, 2}, kPendingReleaseOpVersion},
           {{OperatorType::kDepthToSpace, 1}, "1.5.0"},
           {{OperatorType::kFakeQuant, 1}, "1.5.0"},
           {{OperatorType::kFakeQuant, 2}, "1.10.0"},
           {{OperatorType::kFullyConnected, 1}, "1.5.0"},
           {{OperatorType::kFullyConnected, 2}, "1.10.0"},
           {{OperatorType::kFullyConnected, 3}, kPendingReleaseOpVersion},
           {{OperatorType::kFullyConnected, 4}, kPendingReleaseOpVersion},
           {{OperatorType::kGather, 1}, "1.6.0"},
           {{OperatorType::kGather, 2}, kPendingReleaseOpVersion},
           {{OperatorType::kGatherNd, 1}, kPendingReleaseOpVersion},
           {{OperatorType::kSvdf, 1}, "1.5.0"},
           {{OperatorType::kSvdf, 2}, kPendingReleaseOpVersion},
           {{OperatorType::kL2Normalization, 1}, "1.5.0"},
           {{OperatorType::kL2Normalization, 2}, kPendingReleaseOpVersion},
           {{OperatorType::kL2Pool, 1}, "1.5.0"},
           {{OperatorType::kLocalResponseNormalization, 1}, "1.5.0"},
           {{OperatorType::kMaxPool, 1}, "1.5.0"},
           {{OperatorType::kMaxPool, 2}, kPendingReleaseOpVersion},
           {{OperatorType::kMaximum, 1}, kPendingReleaseOpVersion},
           {{OperatorType::kMaximum, 2}, kPendingReleaseOpVersion},
           {{OperatorType::kMinimum, 1}, kPendingReleaseOpVersion},
           {{OperatorType::kMinimum, 2}, kPendingReleaseOpVersion},
           {{OperatorType::kMul, 1}, "1.5.0"},
           {{OperatorType::kMul, 2}, kPendingReleaseOpVersion},
           {{OperatorType::kPad, 1}, "1.5.0"},
           {{OperatorType::kPad, 2}, kPendingReleaseOpVersion},
           {{OperatorType::kTile, 1}, "1.10.1"},
           {{OperatorType::kPadV2, 1}, "1.9.0"},
           {{OperatorType::kPadV2, 2}, kPendingReleaseOpVersion},
           {{OperatorType::kReshape, 1}, "1.5.0"},
           {{OperatorType::kSoftmax, 1}, "1.5.0"},
           {{OperatorType::kSoftmax, 2}, kPendingReleaseOpVersion},
           {{OperatorType::kSpaceToDepth, 1}, "1.5.0"},
           {{OperatorType::kSpaceToDepth, 2}, kPendingReleaseOpVersion},
           {{OperatorType::kTranspose, 1}, "1.6.0"},
           {{OperatorType::kTranspose, 2}, kPendingReleaseOpVersion},
           {{OperatorType::kLstmCell, 1}, "1.7.0"},
           {{OperatorType::kLstmCell, 2}, "1.10.0"},
           {{OperatorType::kLstmCell, 3}, kPendingReleaseOpVersion},
           {{OperatorType::kUnidirectionalSequenceLstm, 1}, "1.13.1"},
           {{OperatorType::kUnidirectionalSequenceLstm, 1},
            kPendingReleaseOpVersion},
           {{OperatorType::kBidirectionalSequenceLstm, 1},
            kPendingReleaseOpVersion},
           {{OperatorType::kBidirectionalSequenceRnn, 1},
            kPendingReleaseOpVersion},
           {{OperatorType::kMean, 1}, "1.6.0"},
           {{OperatorType::kMean, 2}, kPendingReleaseOpVersion},
           {{OperatorType::kSum, 1}, "1.10.0"},
           {{OperatorType::kReduceMax, 1}, "1.11.0"},
           {{OperatorType::kReduceMax, 2}, kPendingReleaseOpVersion},
           {{OperatorType::kReduceMin, 1}, "1.11.0"},
           {{OperatorType::kReduceMin, 2}, kPendingReleaseOpVersion},
           {{OperatorType::kReduceProd, 1}, "1.11.0"},
           {{OperatorType::kAny, 1}, "1.11.0"},
           {{OperatorType::kRelu6, 1}, kPendingReleaseOpVersion},
           {{OperatorType::kRelu6, 2}, kPendingReleaseOpVersion},
           {{OperatorType::kResizeBilinear, 1}, "1.7.0"},
           {{OperatorType::kResizeBilinear, 2}, kPendingReleaseOpVersion},
           {{OperatorType::kResizeNearestNeighbor, 1}, "1.13.1"},
           {{OperatorType::kResizeNearestNeighbor, 2},
            kPendingReleaseOpVersion},
           {{OperatorType::kSqueeze, 1}, "1.6.0"},
           {{OperatorType::kSplit, 1}, "1.5.0"},
           {{OperatorType::kSplit, 2}, kPendingReleaseOpVersion},
           {{OperatorType::kSplit, 3}, kPendingReleaseOpVersion},
           {{OperatorType::kSplitV, 1}, "1.13.1"},
           {{OperatorType::kStridedSlice, 1}, "1.6.0"},
           {{OperatorType::kStridedSlice, 2}, kPendingReleaseOpVersion},
           {{OperatorType::kTopK_V2, 1}, "1.7.0"},
           {{OperatorType::kTopK_V2, 2}, kPendingReleaseOpVersion},
           {{OperatorType::kArgMax, 1}, "1.9.0"},
           {{OperatorType::kArgMax, 2}, kPendingReleaseOpVersion},
           {{OperatorType::kArgMin, 1}, "1.9.0"},
           {{OperatorType::kArgMin, 2}, kPendingReleaseOpVersion},
           {{OperatorType::kTransposeConv, 1}, "1.9.0"},
           {{OperatorType::kSparseToDense, 1}, "1.9.0"},
           {{OperatorType::kSparseToDense, 2}, kPendingReleaseOpVersion},
           {{OperatorType::kExpandDims, 1}, "1.10.0"},
           {{OperatorType::kPack, 1}, "1.11.0"},
           {{OperatorType::kPack, 2}, kPendingReleaseOpVersion},
           {{OperatorType::kShape, 1}, "1.10.0"},
           {{OperatorType::kSlice, 1}, kPendingReleaseOpVersion},
           {{OperatorType::kSlice, 2}, kPendingReleaseOpVersion},
           {{OperatorType::kSlice, 3}, kPendingReleaseOpVersion},
           {{OperatorType::kTanh, 1}, kPendingReleaseOpVersion},
           {{OperatorType::kTanh, 2}, kPendingReleaseOpVersion},
           {{OperatorType::kOneHot, 1}, "1.11.0"},
           {{OperatorType::kCTCBeamSearchDecoder, 1}, "1.11.0"},
           {{OperatorType::kUnpack, 1}, "1.11.0"},
           {{OperatorType::kLeakyRelu, 1}, "1.13.1"},
           {{OperatorType::kLogistic, 1}, kPendingReleaseOpVersion},
           {{OperatorType::kLogistic, 2}, kPendingReleaseOpVersion},
           {{OperatorType::kLogSoftmax, 1}, kPendingReleaseOpVersion},
           {{OperatorType::kLogSoftmax, 2}, kPendingReleaseOpVersion},
           {{OperatorType::kSquaredDifference, 1}, "1.13.1"},
           {{OperatorType::kMirrorPad, 1}, "1.13.1"},
           {{OperatorType::kUnique, 1}, kPendingReleaseOpVersion},
           {{OperatorType::kUnidirectionalSequenceRnn, 1},
            kPendingReleaseOpVersion},
           {{OperatorType::kWhere, 1}, kPendingReleaseOpVersion},
           {{OperatorType::kDequantize, 1}, "1.13.1"},
           {{OperatorType::kDequantize, 2}, kPendingReleaseOpVersion},
           {{OperatorType::kReverseSequence, 1}, kPendingReleaseOpVersion},
           {{OperatorType::kReverseSequence, 2}, kPendingReleaseOpVersion},
           {{OperatorType::kEqual, 1}, kPendingReleaseOpVersion},
           {{OperatorType::kEqual, 2}, kPendingReleaseOpVersion},
           {{OperatorType::kNotEqual, 1}, kPendingReleaseOpVersion},
           {{OperatorType::kNotEqual, 2}, kPendingReleaseOpVersion},
           {{OperatorType::kGreater, 1}, kPendingReleaseOpVersion},
           {{OperatorType::kGreater, 2}, kPendingReleaseOpVersion},
           {{OperatorType::kGreaterEqual, 1}, kPendingReleaseOpVersion},
           {{OperatorType::kGreaterEqual, 2}, kPendingReleaseOpVersion},
           {{OperatorType::kLess, 1}, kPendingReleaseOpVersion},
           {{OperatorType::kLess, 2}, kPendingReleaseOpVersion},
           {{OperatorType::kLessEqual, 1}, kPendingReleaseOpVersion},
           {{OperatorType::kLessEqual, 2}, kPendingReleaseOpVersion},
           {{OperatorType::kSelect, 1}, kPendingReleaseOpVersion},
           {{OperatorType::kSelect, 2}, kPendingReleaseOpVersion}});

  const auto& op_types_map =
      tflite::BuildOperatorByTypeMap(false /*enable_select_tf_ops=*/);
  OperatorSignature op_signature;
  op_signature.model = &model;
  string model_min_version;
  for (const auto& op : model.operators) {
    op_signature.op = op.get();
    const int version = op_types_map.at(op->type)->GetVersion(op_signature);
    std::pair<OperatorType, int> version_key = {op->type, version};
    auto it = op_version_map->find(version_key);
    if (it == op_version_map->end() || it->second == kPendingReleaseOpVersion) {
      // In case we didn't find the current op in the map, or the operator
      // doesn't have a minimum runtime version associated, continue.
      continue;
    }
    if (strcmp(model_min_version.c_str(), it->second.c_str()) < 0) {
      // Current min model runtime version should be bumped if we see a higher
      // op version.
      model_min_version = it->second;
    }
  }
  return model_min_version;
}

}  // namespace tflite
}  // namespace toco
