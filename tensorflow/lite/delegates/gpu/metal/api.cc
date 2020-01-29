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

#include "tensorflow/lite/delegates/gpu/metal/api.h"

#include <vector>

#include "absl/strings/substitute.h"
#include "tensorflow/lite/delegates/gpu/common/model.h"
#include "tensorflow/lite/delegates/gpu/common/operations.h"
#include "tensorflow/lite/delegates/gpu/common/shape.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/metal/compiled_model.h"
#include "tensorflow/lite/delegates/gpu/metal/compute_task_descriptor.h"
#include "tensorflow/lite/delegates/gpu/metal/environment.h"
#include "tensorflow/lite/delegates/gpu/metal/kernels/add.h"
#include "tensorflow/lite/delegates/gpu/metal/kernels/concat.h"
#include "tensorflow/lite/delegates/gpu/metal/kernels/conv.h"
#include "tensorflow/lite/delegates/gpu/metal/kernels/custom_registry.h"
#include "tensorflow/lite/delegates/gpu/metal/kernels/depthwise_conv.h"
#include "tensorflow/lite/delegates/gpu/metal/kernels/elementwise.h"
#include "tensorflow/lite/delegates/gpu/metal/kernels/fully_connected.h"
#include "tensorflow/lite/delegates/gpu/metal/kernels/max_unpooling.h"
#include "tensorflow/lite/delegates/gpu/metal/kernels/mean.h"
#include "tensorflow/lite/delegates/gpu/metal/kernels/mul.h"
#include "tensorflow/lite/delegates/gpu/metal/kernels/padding.h"
#include "tensorflow/lite/delegates/gpu/metal/kernels/pooling.h"
#include "tensorflow/lite/delegates/gpu/metal/kernels/prelu.h"
#include "tensorflow/lite/delegates/gpu/metal/kernels/relu.h"
#include "tensorflow/lite/delegates/gpu/metal/kernels/reshape.h"
#include "tensorflow/lite/delegates/gpu/metal/kernels/resize.h"
#include "tensorflow/lite/delegates/gpu/metal/kernels/slice.h"
#include "tensorflow/lite/delegates/gpu/metal/kernels/softmax.h"
#include "tensorflow/lite/delegates/gpu/metal/kernels/transpose_conv.h"
#include "tensorflow/lite/delegates/gpu/metal/runtime_options.h"

namespace tflite {
namespace gpu {
namespace metal {
namespace {

std::vector<ComputeTaskDescriptorPtr> SelectConvolution(
    const GraphFloat32& graph, int id, ValueId input_id, ValueId output_id,
    const Convolution2DAttributes& attr, const metal::RuntimeOptions& options) {
  // Special precise version, in case we cover dst_shape poorly with standard
  // work group size.
  auto gpu_type = GetGpuType();
  bool a11_12 = gpu_type == GpuType::kA11 || gpu_type == GpuType::kA12;
  const auto dst_shape = graph.FindOutputs(id)[0]->tensor.shape;
  if (GetThreadsRatioUsualToPreciseConvolution(dst_shape) >= 1.2f) {
    // Special version for PowerVR >= IPhone6S/SE
    // Metal has bad driver for PowerVR in IPhone6, so for Iphone6 we should use
    // default kernel with shared memory.
    if ((gpu_type == GpuType::kA9 || gpu_type == GpuType::kA10) &&
        CheckConvolutionPrecise1x1Support(attr)) {
      return ConvolutionPrecise1x1PowerVR(id, input_id, output_id, attr,
                                          options);
    }
    if (a11_12 && GetThreadsRatioUsualToPreciseConvolution(dst_shape) >= 1.2f) {
      return ConvolutionPrecise(id, input_id, output_id, attr, options);
    }
  }
  if (a11_12) {
    if (CheckConvolution1x1Support(attr)) {
      return Convolution1x1(id, input_id, output_id, attr, options);
    } else {
      return ConvolutionGeneric(id, input_id, output_id, attr, options);
    }
  } else {
    return Convolution(id, input_id, output_id, attr, options);
  }
}

std::vector<ComputeTaskDescriptorPtr> SelectDepthWiseConv(
    int id, ValueId input_id, ValueId output_id,
    const DepthwiseConvolution2DAttributes& attr,
    const metal::RuntimeOptions& options) {
  if (CheckDepthWiseConv3x3Stride1x1Support(attr)) {
    return DepthWiseConv3x3Stride1x1(id, input_id, output_id, attr, options);
  } else if (CheckDepthWiseConv3x3Stride2Support(attr)) {
    return DepthWiseConv3x3Stride2(id, input_id, output_id, attr, options);
  } else {
    return DepthWiseConvolution(id, input_id, output_id, attr, options);
  }
}

std::vector<ComputeTaskDescriptorPtr> SelectPReLU(
    const GraphFloat32& graph, int id, ValueId input_id, ValueId output_id,
    const PReLUAttributes& attr, const metal::RuntimeOptions& options) {
  auto alpha = absl::get_if<Tensor<Linear, DataType::FLOAT32>>(&attr.alpha);
  if (alpha) {
    return PReLU(id, input_id, output_id, attr, options);
  }
  auto alpha3d = absl::get_if<Tensor<HWC, DataType::FLOAT32>>(&attr.alpha);
  if (!alpha3d) {
    return {};
  }
  const auto shape = graph.FindInputs(id)[0]->tensor.shape;
  if (alpha3d->shape.h != shape.h || alpha3d->shape.w != shape.w ||
      alpha3d->shape.c != shape.c) {
    return {};
  }
  return PReLUFull(id, input_id, output_id, attr, options);
}

std::vector<ComputeTaskDescriptorPtr> SelectReshape(
    const GraphFloat32& graph, int id, ValueId input_id, ValueId output_id,
    const ReshapeAttributes& attr) {
  const auto src_shape = graph.FindInputs(id)[0]->tensor.shape;
  if (src_shape.c % 4 == 0 && attr.new_shape.c % 4 == 0) {
    return Reshapex4(id, input_id, output_id, attr);
  } else {
    return Reshape(id, input_id, output_id, attr);
  }
}

std::vector<ComputeTaskDescriptorPtr> SelectSoftmax(const GraphFloat32& graph,
                                                    int id, ValueId input_id,
                                                    ValueId output_id) {
  const auto src_shape = graph.FindInputs(id)[0]->tensor.shape;
  if (src_shape.w == 1 && src_shape.h == 1) {
    return Softmax1x1(id, input_id, output_id, src_shape.c);
  } else {
    return Softmax(id, input_id, output_id, src_shape.c);
  }
}

Status RegisterPrimaryOps(const GraphFloat32& graph, const Node* node,
                          const std::vector<ValueId>& inputs,
                          const std::vector<ValueId>& outputs,
                          const RuntimeOptions& options,
                          std::vector<ComputeTaskDescriptorPtr>* tasks) {
  if (!IsBatchMatchesForAllValues(graph)) {
    return InvalidArgumentError("Only identical batch dimension is supported");
  }
  int node_id = static_cast<int>(node->id);
  auto op_type = OperationTypeFromString(node->operation.type);
  switch (op_type) {
    case OperationType::ADD:
      *tasks = Add(node_id, inputs, outputs[0],
                   absl::any_cast<AddAttributes>(node->operation.attributes),
                   options);
      break;
    case OperationType::CONCAT: {
      std::vector<BHWC> input_shapes;
      for (auto& input : graph.FindInputs(node->id)) {
        input_shapes.push_back(input->tensor.shape);
      }
      *tasks =
          Concat(node_id, inputs, outputs[0],
                 absl::any_cast<ConcatAttributes>(node->operation.attributes),
                 input_shapes);
      break;
    }
    case OperationType::CONVOLUTION_2D:
      *tasks = SelectConvolution(
          graph, node_id, inputs[0], outputs[0],
          absl::any_cast<Convolution2DAttributes>(node->operation.attributes),
          options);
      break;
    case OperationType::CONVOLUTION_TRANSPOSED:
      *tasks =
          ConvolutionTransposed(node_id, inputs[0], outputs[0],
                                absl::any_cast<ConvolutionTransposedAttributes>(
                                    node->operation.attributes),
                                options);
      break;
    case OperationType::DEPTHWISE_CONVOLUTION:
      *tasks =
          SelectDepthWiseConv(node_id, inputs[0], outputs[0],
                              absl::any_cast<DepthwiseConvolution2DAttributes>(
                                  node->operation.attributes),
                              options);
      break;
    case OperationType::FULLY_CONNECTED:
      *tasks = FullyConnected(
          node_id, inputs[0], outputs[0],
          absl::any_cast<FullyConnectedAttributes>(node->operation.attributes),
          options);
      break;
    case OperationType::MAX_UNPOOLING_2D:
      *tasks = MaxUnpooling(
          node_id, inputs[0], inputs[1], outputs[0],
          absl::any_cast<MaxUnpooling2DAttributes>(node->operation.attributes));
      break;
    case OperationType::MEAN:
      *tasks = Mean(node_id, inputs[0], outputs[0],
                    absl::any_cast<MeanAttributes>(node->operation.attributes));
      break;
    case OperationType::MUL:
      if (node->operation.attributes.has_value()) {
        *tasks = Multiply(
            node_id, inputs[0], outputs[0],
            absl::any_cast<MultiplyAttributes>(node->operation.attributes),
            options);
      } else {
        *tasks = ApplyMask(node_id, inputs[0], inputs[1], outputs[0], options);
      }
      break;
    case OperationType::PAD: {
      auto attr = absl::any_cast<PadAttributes>(node->operation.attributes);
      if (attr.appended.b != 0 || attr.prepended.b != 0) {
        return UnimplementedError("Padding for BATCH is not supported.");
      }
      *tasks = Padding(node_id, inputs[0], outputs[0], attr);
      break;
    }
    case OperationType::POOLING_2D:
      *tasks = Pooling(
          node_id, inputs[0], outputs,
          absl::any_cast<Pooling2DAttributes>(node->operation.attributes));
      break;
    case OperationType::PRELU:
      *tasks = SelectPReLU(
          graph, node_id, inputs[0], outputs[0],
          absl::any_cast<PReLUAttributes>(node->operation.attributes), options);
      break;
    case OperationType::RELU:
      *tasks = ReLU(node_id, inputs[0], outputs[0],
                    absl::any_cast<ReLUAttributes>(node->operation.attributes));
      break;
    case OperationType::RESHAPE:
      *tasks = SelectReshape(
          graph, node_id, inputs[0], outputs[0],
          absl::any_cast<ReshapeAttributes>(node->operation.attributes));
      break;
    case OperationType::RESIZE:
      *tasks = Resize(
          node_id, inputs[0], outputs[0],
          absl::any_cast<Resize2DAttributes>(node->operation.attributes));
      break;
    case OperationType::SLICE:
      *tasks =
          Slice(node_id, inputs[0], outputs[0],
                absl::any_cast<SliceAttributes>(node->operation.attributes));
      break;
    case OperationType::SOFTMAX: {
      auto attr = absl::any_cast<SoftmaxAttributes>(node->operation.attributes);
      if (attr.axis != Axis::CHANNELS) {
        return UnimplementedError("Softmax supports only CHANNELS dimension");
      }
      *tasks = SelectSoftmax(graph, node_id, inputs[0], outputs[0]);
      break;
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
    case OperationType::TANH:
      *tasks = ElementwiseWithOneInput(node_id, inputs[0], outputs[0], op_type);
      break;
    case OperationType::SUB:
    case OperationType::DIV:
    case OperationType::POW:
    case OperationType::SQUARED_DIFF:
      *tasks = ElementwiseWithTwoInputs(node_id, inputs, outputs[0], op_type);
      break;
    case OperationType::BATCH_NORMALIZATION:
    case OperationType::BATCH_TO_SPACE:
    case OperationType::CONST:
    case OperationType::LSTM:
    case OperationType::SPACE_TO_BATCH:
    case OperationType::TRANSPOSE:
    case OperationType::UNKNOWN:
      return UnimplementedError("Unsupported op: " + node->operation.type);
  }
  return OkStatus();
}

}  // namespace

Status Compile(const GraphFloat32& graph, const RuntimeOptions& options,
               CompiledModel* compiled_model) {
  for (const auto& node : graph.nodes()) {
    std::vector<ValueId> inputs;
    for (auto& input : graph.FindInputs(node->id)) {
      inputs.push_back(static_cast<ValueId>(input->id));
    }
    std::vector<ValueId> outputs;
    for (auto& output : graph.FindOutputs(node->id)) {
      outputs.push_back(static_cast<ValueId>(output->id));
    }
    std::vector<ComputeTaskDescriptorPtr> tasks;
    auto custom_status =
        RegisterCustomOps(graph, node, inputs, outputs, options, &tasks);
    if (!custom_status.ok()) {
      auto primary_status =
          RegisterPrimaryOps(graph, node, inputs, outputs, options, &tasks);
      if (!primary_status.ok()) {
        return UnimplementedError(absl::Substitute(
            "Unsupported op type: $0; custom registry error: "
            "$1; primary registry error: $2;",
            node->operation.type, custom_status.error_message(),
            primary_status.error_message()));
      }
    }
    compiled_model->insert(compiled_model->end(), tasks.begin(), tasks.end());
  }
  return OkStatus();
}

}  // namespace metal
}  // namespace gpu
}  // namespace tflite
