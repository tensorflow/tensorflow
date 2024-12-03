/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/delegates/gpu/common/transformations/global_pooling_to_reduce_op.h"

#include <any>
#include <memory>
#include <string>
#include <vector>

#include "absl/types/any.h"
#include "tensorflow/lite/delegates/gpu/common/model.h"
#include "tensorflow/lite/delegates/gpu/common/model_transformer.h"
#include "tensorflow/lite/delegates/gpu/common/operations.h"
#include "tensorflow/lite/delegates/gpu/common/shape.h"
#include "tensorflow/lite/delegates/gpu/common/tensor.h"

namespace tflite {
namespace gpu {
namespace {

bool IsGlobalPooling(const Pooling2DAttributes& attr, const BHWC& src_shape,
                     const BHWC& dst_shape) {
  return dst_shape.w == 1 && dst_shape.h == 1 && attr.kernel.w == src_shape.w &&
         attr.kernel.h == src_shape.h && attr.padding.appended.w == 0 &&
         attr.padding.appended.h == 0 && attr.padding.prepended.w == 0 &&
         attr.padding.prepended.h == 0;
}

bool IsGlobalAveragePooling(const Pooling2DAttributes& attr,
                            const BHWC& src_shape, const BHWC& dst_shape) {
  return attr.type == tflite::gpu::PoolingType::AVERAGE &&
         attr.output_indices == false &&
         IsGlobalPooling(attr, src_shape, dst_shape);
}

class GlobalPoolingToReduceOp : public NodeTransformation {
 public:
  TransformResult ApplyToNode(Node* node, GraphFloat32* graph) final {
    if (node->operation.type != ToString(OperationType::POOLING_2D)) {
      return {TransformStatus::SKIPPED, ""};
    }

    auto inputs = graph->FindInputs(node->id);
    auto outputs = graph->FindOutputs(node->id);
    const auto& pool_attr =
        std::any_cast<const Pooling2DAttributes&>(node->operation.attributes);
    if (!IsGlobalAveragePooling(pool_attr, inputs[0]->tensor.shape,
                                outputs[0]->tensor.shape)) {
      return {TransformStatus::SKIPPED, ""};
    }

    MeanAttributes mean_attr;
    mean_attr.dims = {Axis::WIDTH, Axis::HEIGHT};

    node->operation.attributes = mean_attr;
    node->operation.type = ToString(OperationType::MEAN);
    return {TransformStatus::APPLIED,
            "Replaced global average pooling with mean."};
  }
};

}  // namespace

std::unique_ptr<NodeTransformation> NewGlobalPoolingToReduceOp() {
  return std::make_unique<GlobalPoolingToReduceOp>();
}

}  // namespace gpu
}  // namespace tflite
