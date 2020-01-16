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

#include "tensorflow/lite/delegates/gpu/common/transformations/make_padding.h"

#include "absl/memory/memory.h"
#include "absl/types/any.h"
#include "tensorflow/lite/delegates/gpu/common/model.h"
#include "tensorflow/lite/delegates/gpu/common/operations.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"

namespace tflite {
namespace gpu {
namespace {

bool IsConstZeros(const Node& node) {
  if (node.operation.type != ToString(OperationType::CONST)) {
    return false;
  }
  auto& attr =
      absl::any_cast<const ConstTensorAttributes&>(node.operation.attributes);
  for (auto f : attr.tensor.data) {
    if (f != 0) {
      return false;
    }
  }
  return true;
}

class MakePaddingFromZerosConcat : public NodeTransformation {
 public:
  TransformResult ApplyToNode(Node* node, GraphFloat32* graph) final {
    if (node->operation.type != ToString(OperationType::CONCAT)) {
      return {TransformStatus::SKIPPED, ""};
    }
    auto inputs = graph->FindInputs(node->id);
    if (inputs.size() != 2) {
      return {TransformStatus::SKIPPED, ""};
    }

    bool first = true;
    for (auto input : inputs) {
      auto dep = graph->FindProducer(input->id);
      if (dep != nullptr && IsConstZeros(*dep)) {
        auto& concat_attr =
            absl::any_cast<const ConcatAttributes&>(node->operation.attributes);
        PadAttributes pad_attr;
        pad_attr.type = PaddingContentType::ZEROS;
        pad_attr.appended = BHWC(0, 0, 0, 0);
        pad_attr.prepended = BHWC(0, 0, 0, 0);
        BHWC* p = first ? &pad_attr.prepended : &pad_attr.appended;
        switch (concat_attr.axis) {
          case Axis::HEIGHT:
            p->h = input->tensor.shape.h;
            break;
          case Axis::WIDTH:
            p->w = input->tensor.shape.w;
            break;
          case Axis::CHANNELS:
            p->c = input->tensor.shape.c;
            break;
          default:
            return {TransformStatus::DECLINED,
                    "Padding for concat axis is unsupported: " +
                        ToString(concat_attr.axis)};
        }
        Status status = RemovePrecedingNode(graph, dep, node);
        if (!status.ok()) {
          return {TransformStatus::INVALID,
                  "Unable to remove const node: " + status.error_message()};
        }
        node->operation.attributes = pad_attr;
        node->operation.type = ToString(OperationType::PAD);
        return {TransformStatus::APPLIED, "Replaced concat with padding"};
      }
      first = false;
    }
    return {TransformStatus::SKIPPED, ""};
  }
};

}  // namespace

std::unique_ptr<NodeTransformation> NewMakePaddingFromConcat() {
  return absl::make_unique<MakePaddingFromZerosConcat>();
}

}  // namespace gpu
}  // namespace tflite
