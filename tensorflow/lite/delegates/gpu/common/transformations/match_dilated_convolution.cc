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

#include "tensorflow/lite/delegates/gpu/common/transformations/match_dilated_convolution.h"

#include <vector>

#include "absl/memory/memory.h"
#include "absl/types/any.h"
#include "tensorflow/lite/delegates/gpu/common/model.h"
#include "tensorflow/lite/delegates/gpu/common/operations.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"

namespace tflite {
namespace gpu {
namespace {

class MatchDilatedConvolution : public SequenceTransformation {
 public:
  int ExpectedSequenceLength() const final { return 3; }

  // TODO(eignasheva): use span instead of const reference b/131628066.
  TransformResult ApplyToNodesSequence(const std::vector<Node*>& sequence,
                                       GraphFloat32* graph) final {
    auto& sb_node = *sequence[0];
    auto& conv_node = *sequence[1];
    auto& bs_node = *sequence[2];
    if (sb_node.operation.type != ToString(OperationType::SPACE_TO_BATCH) &&
        bs_node.operation.type != ToString(OperationType::BATCH_TO_SPACE)) {
      return {TransformStatus::SKIPPED, ""};
    }
    if (conv_node.operation.type !=
            ToString(OperationType::DEPTHWISE_CONVOLUTION) &&
        conv_node.operation.type != ToString(OperationType::CONVOLUTION_2D)) {
      return {TransformStatus::SKIPPED, ""};
    }

    auto sb_attr =
        absl::any_cast<SpaceToBatchAttributes>(sb_node.operation.attributes);

    auto bs_attr =
        absl::any_cast<BatchToSpaceAttributes>(bs_node.operation.attributes);

    if (sb_attr.block != bs_attr.block) {
      return {TransformStatus::INVALID, "Invalid block size"};
    }

    if (conv_node.operation.type ==
        ToString(OperationType::DEPTHWISE_CONVOLUTION)) {
      auto dw_attr = absl::any_cast<DepthwiseConvolution2DAttributes>(
          conv_node.operation.attributes);
      dw_attr.padding = sb_attr.padding - bs_attr.crop;
      dw_attr.dilations = sb_attr.block;
      conv_node.operation.attributes = std::move(dw_attr);
    } else {
      auto conv2d_attr = absl::any_cast<Convolution2DAttributes>(
          conv_node.operation.attributes);
      conv2d_attr.padding = sb_attr.padding - bs_attr.crop;
      conv2d_attr.dilations = sb_attr.block;
      conv_node.operation.attributes = std::move(conv2d_attr);
    }

    Status status = RemoveFollowingNode(graph, &bs_node, &conv_node);
    if (!status.ok()) {
      return {TransformStatus::INVALID,
              "Unable to remove batch_to_space node after convolution."};
    }
    status = RemovePrecedingNode(graph, &sb_node, &conv_node);
    if (!status.ok()) {
      return {TransformStatus::INVALID,
              "Unable to remove space_to_batch node before convolution."};
    }

    return {TransformStatus::APPLIED, ""};
  }
};

}  // namespace

std::unique_ptr<SequenceTransformation> NewMatchDilatedConvolution() {
  return absl::make_unique<MatchDilatedConvolution>();
}

}  // namespace gpu
}  // namespace tflite
