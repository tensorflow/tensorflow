/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/delegates/gpu/common/gpu_model_test_util.h"

#include <cmath>

#include "tensorflow/lite/delegates/gpu/common/gpu_model.h"
#include "tensorflow/lite/delegates/gpu/common/operations.h"
#include "tensorflow/lite/delegates/gpu/common/tasks/conv_generic.h"
#include "tensorflow/lite/delegates/gpu/common/tasks/elementwise.h"

namespace tflite {
namespace gpu {

absl::Status TestLinkingConvolutionAndCosOp(TestExecutionEnvironment* env) {
  GraphFloat32 graph;
  auto input = graph.NewValue();
  input->tensor.type = DataType::FLOAT32;
  input->tensor.shape = BHWC(1, 32, 32, 128);

  auto conv_node = graph.NewNode();
  conv_node->operation.type =
      ToString(tflite::gpu::OperationType::CONVOLUTION_2D);

  Convolution2DAttributes conv_attr;
  conv_attr.padding.prepended = HW(0, 0);
  conv_attr.padding.appended = HW(0, 0);
  conv_attr.strides = HW(1, 1);
  conv_attr.dilations = HW(1, 1);
  conv_attr.weights.shape = OHWI(16, 1, 1, 128);
  conv_attr.weights.data.resize(conv_attr.weights.shape.DimensionsProduct());
  for (int i = 0; i < conv_attr.weights.data.size(); ++i) {
    conv_attr.weights.data[i] = std::sin(i * 0.12345f);
  }
  conv_attr.bias.shape = Linear(16);
  conv_attr.bias.data.resize(conv_attr.bias.shape.DimensionsProduct());
  for (int i = 0; i < conv_attr.bias.data.size(); ++i) {
    conv_attr.bias.data[i] = std::sin(i * 0.12345f);
  }
  conv_node->operation.attributes = conv_attr;
  RETURN_IF_ERROR(graph.AddConsumer(conv_node->id, input->id));

  auto cos_node = graph.NewNode();
  cos_node->operation.type = ToString(tflite::gpu::OperationType::COS);
  tflite::gpu::Value* conv_output = nullptr;
  RETURN_IF_ERROR(ConnectTwoNodes(&graph, conv_node, cos_node, &conv_output));
  conv_output->tensor.type = DataType::FLOAT32;
  conv_output->tensor.shape = BHWC(1, 32, 32, 16);

  tflite::gpu::Value* cos_output = nullptr;
  RETURN_IF_ERROR(AddOutput(&graph, cos_node, &cos_output));
  cos_output->tensor.type = DataType::FLOAT32;
  cos_output->tensor.shape = BHWC(1, 32, 32, 16);

  RETURN_IF_ERROR(RunGraphTransformsForGpuModel(&graph));

  for (auto precision : env->GetSupportedPrecisions()) {
    auto data_type = DeduceDataTypeFromPrecision(precision);
    for (auto storage : env->GetSupportedStorages(data_type)) {
      CreateGpuModelInfo create_info;
      create_info.precision = precision;
      create_info.storage_type = storage;
      create_info.hints.Add(ModelHints::kAllowSpecialKernels);

      GpuModel gpu_model;
      RETURN_IF_ERROR(
          GraphToGpuModel(graph, create_info, env->GetGpuInfo(), &gpu_model));

      if (gpu_model.nodes.size() != 1) {
        return absl::InternalError("Expected model with one node.");
      }

      TensorFloat32 src_tensor;
      src_tensor.shape = input->tensor.shape;
      src_tensor.data.resize(src_tensor.shape.DimensionsProduct());
      for (int i = 0; i < src_tensor.data.size(); ++i) {
        src_tensor.data[i] = std::sin(i * 0.12345f);
      }

      TensorFloat32 dst_tensor_v1;
      RETURN_IF_ERROR(
          env->ExecuteGpuModel({src_tensor}, {&dst_tensor_v1}, &gpu_model));

      OperationDef op_def;
      op_def.precision = precision;
      op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
      op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});

      ConvGeneric conv_operation =
          CreateConvGeneric(env->GetGpuInfo(), op_def, conv_attr);
      TensorFloat32 intermediate;
      RETURN_IF_ERROR(env->ExecuteGPUOperation(
          src_tensor, std::make_unique<ConvGeneric>(std::move(conv_operation)),
          conv_output->tensor.shape, &intermediate));

      GPUOperation cos_operation = CreateElementwiseOneInput(
          env->GetGpuInfo(), op_def, OperationType::COS);
      TensorFloat32 dst_tensor_v0;
      RETURN_IF_ERROR(env->ExecuteGPUOperation(
          intermediate,
          std::make_unique<GPUOperation>(std::move(cos_operation)),
          cos_output->tensor.shape, &dst_tensor_v0));

      RETURN_IF_ERROR(
          PointWiseNear(dst_tensor_v0.data, dst_tensor_v1.data, 0.0f));
    }
  }
  return absl::OkStatus();
}

}  // namespace gpu
}  // namespace tflite
