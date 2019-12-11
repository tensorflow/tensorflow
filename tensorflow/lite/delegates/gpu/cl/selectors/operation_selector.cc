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

#include "tensorflow/lite/delegates/gpu/cl/selectors/operation_selector.h"

#include "absl/strings/str_cat.h"
#include "absl/types/any.h"
#include "tensorflow/lite/delegates/gpu/cl/kernels/elementwise.h"
#include "tensorflow/lite/delegates/gpu/cl/selectors/convolution_selector.h"
#include "tensorflow/lite/delegates/gpu/cl/selectors/convolution_transposed_selector.h"
#include "tensorflow/lite/delegates/gpu/cl/selectors/dw_convolution_selector.h"
#include "tensorflow/lite/delegates/gpu/cl/selectors/fully_connected_selector.h"
#include "tensorflow/lite/delegates/gpu/cl/selectors/simple_selectors.h"
#include "tensorflow/lite/delegates/gpu/common/data_type.h"
#include "tensorflow/lite/delegates/gpu/common/shape.h"
#include "tensorflow/lite/delegates/gpu/common/tensor.h"

namespace tflite {
namespace gpu {
namespace cl {

Status GPUOperationFromNode(const CreationContext& creation_context,
                            const OperationDef& op_def, ModelHints hints,
                            const std::vector<Value<TensorRef<BHWC>>*>& inputs,
                            const std::vector<Value<TensorRef<BHWC>>*>& outputs,
                            const Node& node,
                            GPUOperationsSubgraph* gpu_subgraph) {
  gpu_subgraph->operations.push_back({});
  std::unique_ptr<GPUOperation>* gpu_op =
      &gpu_subgraph->operations[0].operation;
  for (int i = 0; i < inputs.size(); ++i) {
    gpu_subgraph->operations[0].input_ids.push_back(i);
  }
  for (int i = 0; i < outputs.size(); ++i) {
    gpu_subgraph->operations[0].output_ids.push_back(i);
  }
  auto op_type = OperationTypeFromString(node.operation.type);
  switch (op_type) {
    case OperationType::ADD: {
      const auto attr =
          absl::any_cast<AddAttributes>(node.operation.attributes);
      const auto* adds =
          absl::get_if<::tflite::gpu::Tensor<Linear, DataType::FLOAT32>>(
              &attr.param);
      const auto* adds_scalar = absl::get_if<float>(&attr.param);
      if (adds || adds_scalar) {
        return SelectBroadcastAdd(attr, creation_context, op_def, gpu_op);
      } else {
        auto output = outputs[0];
        std::vector<int> channels(inputs.size());
        for (int i = 0; i < inputs.size(); ++i) {
          channels[i] = inputs[i]->tensor.shape.c;
        }
        SelectAdd(op_def, channels, output->tensor.shape.c, gpu_op);
        return OkStatus();
      }
    }
    case OperationType::APPLY_MASK: {
      SelectApplyMask(op_def, inputs[0]->tensor.shape, inputs[1]->tensor.shape,
                      gpu_op);
      return OkStatus();
    }
    case OperationType::CONCAT: {
      auto attr = absl::any_cast<ConcatAttributes>(node.operation.attributes);
      std::vector<int> channels(inputs.size());
      for (int i = 0; i < inputs.size(); ++i) {
        channels[i] = inputs[i]->tensor.shape.c;
      }
      return SelectConcat(attr, channels, op_def, gpu_op);
    }
    case OperationType::CONVOLUTION_2D: {
      auto attr =
          absl::any_cast<Convolution2DAttributes>(node.operation.attributes);
      auto input = inputs[0];
      return SelectConvolution(attr, input->tensor.shape, creation_context,
                               op_def, hints, gpu_op);
    }
    case OperationType::CONVOLUTION_TRANSPOSED: {
      auto attr = absl::any_cast<ConvolutionTransposedAttributes>(
          node.operation.attributes);
      return SelectConvolutionTransposed(attr, creation_context, op_def,
                                         gpu_op);
    }
    case OperationType::DEPTHWISE_CONVOLUTION: {
      auto attr = absl::any_cast<DepthwiseConvolution2DAttributes>(
          node.operation.attributes);
      return SelectDWConvolution(attr, creation_context, op_def, gpu_op);
    }
    case OperationType::FULLY_CONNECTED: {
      auto attr =
          absl::any_cast<FullyConnectedAttributes>(node.operation.attributes);
      return SelectFullyConnected(attr, creation_context, op_def,
                                  inputs[0]->tensor.shape.b, gpu_op);
    }
    case OperationType::LSTM: {
      SelectLSTM(op_def, gpu_op);
      return OkStatus();
    }
    case OperationType::MAX_UNPOOLING_2D: {
      auto attr =
          absl::any_cast<MaxUnpooling2DAttributes>(node.operation.attributes);
      SelectMaxUnpooling(attr, op_def, gpu_op);
      return OkStatus();
    }
    case OperationType::MULTIPLY_SCALAR: {
      auto attr =
          absl::any_cast<MultiplyScalarAttributes>(node.operation.attributes);
      return SelectMultiplyScalar(attr, creation_context, op_def, gpu_op);
    }
    case OperationType::PAD: {
      auto attr = absl::any_cast<PadAttributes>(node.operation.attributes);
      SelectPadding(attr, op_def, gpu_op);
      return OkStatus();
    }
    case OperationType::POOLING_2D: {
      auto attr =
          absl::any_cast<Pooling2DAttributes>(node.operation.attributes);
      SelectPooling(attr, op_def, gpu_op);
      return OkStatus();
    }
    case OperationType::PRELU: {
      auto attr = absl::any_cast<PReLUAttributes>(node.operation.attributes);
      return SelectPReLU(attr, creation_context, op_def, gpu_op);
    }
    case OperationType::RELU: {
      auto attr = absl::any_cast<ReLUAttributes>(node.operation.attributes);
      SelectReLU(creation_context, attr, op_def, gpu_op);
      return OkStatus();
    }
    case OperationType::RESHAPE: {
      const int src_channels = inputs[0]->tensor.shape.c;
      auto attr = absl::any_cast<ReshapeAttributes>(node.operation.attributes);
      SelectReshape(src_channels, attr.new_shape.c, op_def, gpu_op);
      return OkStatus();
    }
    case OperationType::SLICE: {
      auto attr = absl::any_cast<SliceAttributes>(node.operation.attributes);
      SelectStridedSlice(attr, op_def, gpu_op);
      return OkStatus();
    }
    case OperationType::SOFTMAX: {
      SelectSoftmax(inputs[0]->tensor.shape, op_def, gpu_op);
      return OkStatus();
    }
    case OperationType::TRANSPOSE: {
      auto attr =
          absl::any_cast<TransposeAttributes>(node.operation.attributes);
      SelectTranspose(attr, op_def, gpu_op);
      return OkStatus();
    }
    case OperationType::UPSAMPLE_2D: {
      auto attr =
          absl::any_cast<Upsample2DAttributes>(node.operation.attributes);
      return SelectUpsampling(attr, op_def, gpu_op);
    }
    case OperationType::ABS:
    case OperationType::COS:
    case OperationType::HARD_SWISH:
    case OperationType::LOG:
    case OperationType::RSQRT:
    case OperationType::SIGMOID:
    case OperationType::SIN:
    case OperationType::SQRT:
    case OperationType::SQUARE:
    case OperationType::TANH: {
      ElementwiseOneInput operation =
          CreateElementwiseOneInput(op_def, op_type);
      *gpu_op = absl::make_unique<ElementwiseOneInput>(std::move(operation));
      return OkStatus();
    }
    case OperationType::DIV:
    case OperationType::POW:
    case OperationType::SQUARED_DIFF:
    case OperationType::SUB: {
      ElementwiseTwoInput operation =
          CreateElementwiseTwoInput(op_def, op_type);
      *gpu_op = absl::make_unique<ElementwiseTwoInput>(std::move(operation));
      return OkStatus();
    }
    default:
      return UnimplementedError(
          absl::StrCat("No selector for ", node.operation.type));
  }
}

}  // namespace cl
}  // namespace gpu
}  // namespace tflite
