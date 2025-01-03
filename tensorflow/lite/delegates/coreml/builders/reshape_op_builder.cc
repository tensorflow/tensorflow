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
#include "tensorflow/lite/delegates/coreml/builders/reshape_op_builder.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <iterator>
#include <memory>
#include <string>

#include "mlmodel/format/NeuralNetwork.pb.h"
#include "tensorflow/lite/core/c/builtin_op_data.h"
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/delegates/coreml/builders/op_builder.h"
#include "tensorflow/lite/delegates/coreml/builders/op_factory.h"
#include "tensorflow/lite/delegates/coreml/builders/op_validator.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"

namespace tflite {
namespace delegates {
namespace coreml {

const std::string& ReshapeOpBuilder::DebugName() {
  if (debug_name_.empty()) {
    SetDebugName("ReshapeOpBuilder", node_id_);
  }
  return debug_name_;
}

CoreML::Specification::NeuralNetworkLayer* ReshapeOpBuilder::Build() {
  if (layer_ == nullptr) {
    layer_ = std::make_unique<CoreML::Specification::NeuralNetworkLayer>();
  }
  layer_->set_name(DebugName());
  for (int dim : shape_) {
    layer_->mutable_reshape()->add_targetshape(dim);
  }
  if (need_transpose_)
    layer_->mutable_reshape()->set_mode(
        CoreML::Specification::ReshapeLayerParams::CHANNEL_LAST);
  return layer_.release();
}

void ReshapeOpBuilder::SetShapeFromTensor(const TfLiteTensor* output_shape,
                                          const TfLiteIntArray* input_shape) {
  TfLiteIntArray* shape = TfLiteIntArrayCreate(output_shape->dims->data[0]);
  std::memcpy(shape->data, GetTensorData<int>(output_shape),
              shape->size * sizeof(int));

  SetShapeFromIntArray(shape, input_shape);
  TfLiteIntArrayFree(shape);
}

void ReshapeOpBuilder::SetShapeFromIntArray(const TfLiteIntArray* output_shape,
                                            const TfLiteIntArray* input_shape) {
  // ignore first dimension (batch)
  std::copy(output_shape->data + 1, output_shape->data + output_shape->size,
            std::back_inserter(shape_));

  int64_t reshape_size = 1;
  int negative_index = -1;
  for (int i = 0; i < shape_.size(); ++i) {
    if (shape_[i] == -1) {
      negative_index = i;
    } else {
      reshape_size *= shape_[i];
    }
  }
  if (negative_index >= 0) {
    int64_t input_size = NumElements(input_shape);
    shape_[negative_index] = input_size / reshape_size;
  }

  if (shape_.size() == 2) {
    shape_ = {shape_[1], 1, shape_[0]};
  } else if (shape_.size() == 3) {
    shape_ = {shape_[2], shape_[0], shape_[1]};
  }
  // When channel dimension is changed, reshape should be done with HWC layout.
  if (shape_[0] != input_shape->data[input_shape->size - 1]) {
    need_transpose_ = true;
  }
}

TfLiteStatus ReshapeOpBuilder::RegisterInputs(const TfLiteIntArray* inputs,
                                              TfLiteContext* context) {
  AddInput(inputs->data[0]);

  if (inputs->size == 2) {
    SetShapeFromTensor(&context->tensors[inputs->data[1]],
                       context->tensors[inputs->data[0]].dims);
  } else {
    const auto* params = reinterpret_cast<TfLiteReshapeParams*>(builtin_data_);
    TfLiteIntArray* output_shape = TfLiteIntArrayCreate(params->num_dimensions);
    std::memcpy(output_shape->data, params->shape,
                params->num_dimensions * sizeof(int));

    SetShapeFromIntArray(output_shape, context->tensors[inputs->data[0]].dims);
    TfLiteIntArrayFree(output_shape);
  }
  return kTfLiteOk;
}

TfLiteStatus ReshapeOpBuilder::RegisterOutputs(const TfLiteIntArray* outputs,
                                               TfLiteContext* context) {
  graph_builder_->AddTensorWithID(outputs->data[0], GetOutput(context));
  return kTfLiteOk;
}

bool IsReshapeOpSupported(const TfLiteRegistration* registration,
                          const TfLiteNode* node, TfLiteContext* context,
                          int coreml_version) {
  if (coreml_version >= 3) {
    return false;
  }
  if (node->inputs->size == 1) {
    const auto* params =
        reinterpret_cast<TfLiteReshapeParams*>(node->builtin_data);
    return params->num_dimensions == 3 || params->num_dimensions == 4;
  }

  const int kShapeTensor = 1;
  const TfLiteTensor* shape;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, kShapeTensor, &shape));
  if (shape->allocation_type != kTfLiteMmapRo) {
    TF_LITE_KERNEL_LOG(context, "Reshape has non-const shape.");
    return false;
  }
  const bool is_shape_tensor =
      shape->dims->size == 1 && shape->type == kTfLiteInt32;
  return is_shape_tensor &&
         (shape->dims->data[0] == 3 || shape->dims->data[0] == 4);
}

OpBuilder* CreateReshapeOpBuilder(GraphBuilder* graph_builder) {
  return new ReshapeOpBuilder(graph_builder);
}

}  // namespace coreml
}  // namespace delegates
}  // namespace tflite
