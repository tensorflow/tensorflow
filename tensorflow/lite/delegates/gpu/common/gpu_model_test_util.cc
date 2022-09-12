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
#include "tensorflow/lite/delegates/gpu/common/tasks/prelu.h"

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
  cos_node->operation.type = ToString(OperationType::COS);
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

absl::Status TestLinkingConvolution2InputMul2InputMul(
    TestExecutionEnvironment* env) {
  GraphFloat32 graph;
  auto input0 = graph.NewValue();
  auto input1 = graph.NewValue();
  auto input2 = graph.NewValue();
  input0->tensor.type = DataType::FLOAT32;
  input0->tensor.shape = BHWC(1, 32, 32, 128);
  input1->tensor.type = DataType::FLOAT32;
  input1->tensor.shape = BHWC(1, 32, 32, 16);
  input2->tensor.type = DataType::FLOAT32;
  input2->tensor.shape = BHWC(1, 32, 32, 16);

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
  RETURN_IF_ERROR(graph.AddConsumer(conv_node->id, input0->id));

  auto mul0_node = graph.NewNode();
  mul0_node->operation.type = ToString(OperationType::MUL);
  tflite::gpu::Value* conv_output = nullptr;
  RETURN_IF_ERROR(ConnectTwoNodes(&graph, conv_node, mul0_node, &conv_output));
  RETURN_IF_ERROR(graph.AddConsumer(mul0_node->id, input1->id));
  conv_output->tensor.type = DataType::FLOAT32;
  conv_output->tensor.shape = BHWC(1, 32, 32, 16);

  auto mul1_node = graph.NewNode();
  mul1_node->operation.type = ToString(OperationType::MUL);
  tflite::gpu::Value* mul0_output = nullptr;
  RETURN_IF_ERROR(ConnectTwoNodes(&graph, mul0_node, mul1_node, &mul0_output));
  RETURN_IF_ERROR(graph.AddConsumer(mul1_node->id, input2->id));
  mul0_output->tensor.type = DataType::FLOAT32;
  mul0_output->tensor.shape = BHWC(1, 32, 32, 16);

  tflite::gpu::Value* mul1_output = nullptr;
  RETURN_IF_ERROR(AddOutput(&graph, mul1_node, &mul1_output));
  mul1_output->tensor.type = DataType::FLOAT32;
  mul1_output->tensor.shape = BHWC(1, 32, 32, 16);

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

      TensorFloat32 src0_tensor;
      src0_tensor.shape = input0->tensor.shape;
      src0_tensor.data.resize(src0_tensor.shape.DimensionsProduct());
      for (int i = 0; i < src0_tensor.data.size(); ++i) {
        src0_tensor.data[i] = std::sin(i * 0.12345f);
      }
      TensorFloat32 src1_tensor;
      src1_tensor.shape = input1->tensor.shape;
      src1_tensor.data.resize(src1_tensor.shape.DimensionsProduct());
      for (int i = 0; i < src1_tensor.data.size(); ++i) {
        src1_tensor.data[i] = std::sin(i * 0.12345f);
      }
      TensorFloat32 src2_tensor;
      src2_tensor.shape = input2->tensor.shape;
      src2_tensor.data.resize(src2_tensor.shape.DimensionsProduct());
      for (int i = 0; i < src2_tensor.data.size(); ++i) {
        src2_tensor.data[i] = std::sin(i * 0.12345f);
      }

      TensorFloat32 dst_tensor_v1;
      RETURN_IF_ERROR(
          env->ExecuteGpuModel({src0_tensor, src1_tensor, src2_tensor},
                               {&dst_tensor_v1}, &gpu_model));

      OperationDef op_def;
      op_def.precision = precision;
      op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
      op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});

      ConvGeneric conv_operation =
          CreateConvGeneric(env->GetGpuInfo(), op_def, conv_attr);
      TensorFloat32 intermediate0;
      RETURN_IF_ERROR(env->ExecuteGPUOperation(
          src0_tensor, std::make_unique<ConvGeneric>(std::move(conv_operation)),
          conv_output->tensor.shape, &intermediate0));

      OperationDef op_def_mul;
      op_def_mul.precision = precision;
      op_def_mul.src_tensors.push_back({data_type, storage, Layout::HWC});
      op_def_mul.src_tensors.push_back({data_type, storage, Layout::HWC});
      op_def_mul.dst_tensors.push_back({data_type, storage, Layout::HWC});

      GPUOperation mul0_operation = CreateElementwiseTwoInput(
          op_def_mul, OperationType::MUL, src1_tensor.shape);
      TensorFloat32 intermediate1;
      RETURN_IF_ERROR(env->ExecuteGPUOperation(
          {intermediate0, src1_tensor},
          std::make_unique<GPUOperation>(std::move(mul0_operation)),
          mul0_output->tensor.shape, &intermediate1));

      GPUOperation mul1_operation = CreateElementwiseTwoInput(
          op_def_mul, OperationType::MUL, src2_tensor.shape);
      TensorFloat32 dst_tensor_v0;
      RETURN_IF_ERROR(env->ExecuteGPUOperation(
          {intermediate1, src2_tensor},
          std::make_unique<GPUOperation>(std::move(mul1_operation)),
          mul1_output->tensor.shape, &dst_tensor_v0));

      RETURN_IF_ERROR(
          PointWiseNear(dst_tensor_v0.data, dst_tensor_v1.data, 0.0f));
    }
  }
  return absl::OkStatus();
}

absl::Status TestLinkingConvolution2InputBroadcastMul2InputMul(
    TestExecutionEnvironment* env) {
  GraphFloat32 graph;
  auto input0 = graph.NewValue();
  auto input1 = graph.NewValue();
  auto input2 = graph.NewValue();
  input0->tensor.type = DataType::FLOAT32;
  input0->tensor.shape = BHWC(1, 32, 32, 128);
  input1->tensor.type = DataType::FLOAT32;
  input1->tensor.shape = BHWC(1, 32, 32, 1);
  input2->tensor.type = DataType::FLOAT32;
  input2->tensor.shape = BHWC(1, 32, 32, 16);

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
  RETURN_IF_ERROR(graph.AddConsumer(conv_node->id, input0->id));

  auto mul0_node = graph.NewNode();
  mul0_node->operation.type = ToString(OperationType::MUL);
  tflite::gpu::Value* conv_output = nullptr;
  RETURN_IF_ERROR(ConnectTwoNodes(&graph, conv_node, mul0_node, &conv_output));
  RETURN_IF_ERROR(graph.AddConsumer(mul0_node->id, input1->id));
  conv_output->tensor.type = DataType::FLOAT32;
  conv_output->tensor.shape = BHWC(1, 32, 32, 16);

  auto mul1_node = graph.NewNode();
  mul1_node->operation.type = ToString(OperationType::MUL);
  tflite::gpu::Value* mul0_output = nullptr;
  RETURN_IF_ERROR(ConnectTwoNodes(&graph, mul0_node, mul1_node, &mul0_output));
  RETURN_IF_ERROR(graph.AddConsumer(mul1_node->id, input2->id));
  mul0_output->tensor.type = DataType::FLOAT32;
  mul0_output->tensor.shape = BHWC(1, 32, 32, 16);

  tflite::gpu::Value* mul1_output = nullptr;
  RETURN_IF_ERROR(AddOutput(&graph, mul1_node, &mul1_output));
  mul1_output->tensor.type = DataType::FLOAT32;
  mul1_output->tensor.shape = BHWC(1, 32, 32, 16);

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

      TensorFloat32 src0_tensor;
      src0_tensor.shape = input0->tensor.shape;
      src0_tensor.data.resize(src0_tensor.shape.DimensionsProduct());
      for (int i = 0; i < src0_tensor.data.size(); ++i) {
        src0_tensor.data[i] = std::sin(i * 0.12345f);
      }
      TensorFloat32 src1_tensor;
      src1_tensor.shape = input1->tensor.shape;
      src1_tensor.data.resize(src1_tensor.shape.DimensionsProduct());
      for (int i = 0; i < src1_tensor.data.size(); ++i) {
        src1_tensor.data[i] = std::sin(i * 0.12345f);
      }
      TensorFloat32 src2_tensor;
      src2_tensor.shape = input2->tensor.shape;
      src2_tensor.data.resize(src2_tensor.shape.DimensionsProduct());
      for (int i = 0; i < src2_tensor.data.size(); ++i) {
        src2_tensor.data[i] = std::sin(i * 0.12345f);
      }

      TensorFloat32 dst_tensor_v1;
      RETURN_IF_ERROR(
          env->ExecuteGpuModel({src0_tensor, src1_tensor, src2_tensor},
                               {&dst_tensor_v1}, &gpu_model));

      OperationDef op_def;
      op_def.precision = precision;
      op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
      op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});

      ConvGeneric conv_operation =
          CreateConvGeneric(env->GetGpuInfo(), op_def, conv_attr);
      TensorFloat32 intermediate0;
      RETURN_IF_ERROR(env->ExecuteGPUOperation(
          src0_tensor, std::make_unique<ConvGeneric>(std::move(conv_operation)),
          conv_output->tensor.shape, &intermediate0));

      OperationDef op_def_mul;
      op_def_mul.precision = precision;
      op_def_mul.src_tensors.push_back({data_type, storage, Layout::HWC});
      op_def_mul.src_tensors.push_back({data_type, storage, Layout::HWC});
      op_def_mul.dst_tensors.push_back({data_type, storage, Layout::HWC});

      GPUOperation mul0_operation = CreateElementwiseTwoInput(
          op_def_mul, OperationType::MUL, src1_tensor.shape);
      TensorFloat32 intermediate1;
      RETURN_IF_ERROR(env->ExecuteGPUOperation(
          {intermediate0, src1_tensor},
          std::make_unique<GPUOperation>(std::move(mul0_operation)),
          mul0_output->tensor.shape, &intermediate1));

      GPUOperation mul1_operation = CreateElementwiseTwoInput(
          op_def_mul, OperationType::MUL, src2_tensor.shape);
      TensorFloat32 dst_tensor_v0;
      RETURN_IF_ERROR(env->ExecuteGPUOperation(
          {intermediate1, src2_tensor},
          std::make_unique<GPUOperation>(std::move(mul1_operation)),
          mul1_output->tensor.shape, &dst_tensor_v0));

      RETURN_IF_ERROR(
          PointWiseNear(dst_tensor_v0.data, dst_tensor_v1.data, 0.0f));
    }
  }
  return absl::OkStatus();
}

absl::Status TestLinkingConvolution2InputMul2InputBroadcastMul(
    TestExecutionEnvironment* env) {
  GraphFloat32 graph;
  auto input0 = graph.NewValue();
  auto input1 = graph.NewValue();
  auto input2 = graph.NewValue();
  input0->tensor.type = DataType::FLOAT32;
  input0->tensor.shape = BHWC(1, 32, 32, 128);
  input1->tensor.type = DataType::FLOAT32;
  input1->tensor.shape = BHWC(1, 32, 32, 16);
  input2->tensor.type = DataType::FLOAT32;
  input2->tensor.shape = BHWC(1, 1, 1, 16);

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
  RETURN_IF_ERROR(graph.AddConsumer(conv_node->id, input0->id));

  auto mul0_node = graph.NewNode();
  mul0_node->operation.type = ToString(OperationType::MUL);
  tflite::gpu::Value* conv_output = nullptr;
  RETURN_IF_ERROR(ConnectTwoNodes(&graph, conv_node, mul0_node, &conv_output));
  RETURN_IF_ERROR(graph.AddConsumer(mul0_node->id, input1->id));
  conv_output->tensor.type = DataType::FLOAT32;
  conv_output->tensor.shape = BHWC(1, 32, 32, 16);

  auto mul1_node = graph.NewNode();
  mul1_node->operation.type = ToString(OperationType::MUL);
  tflite::gpu::Value* mul0_output = nullptr;
  RETURN_IF_ERROR(ConnectTwoNodes(&graph, mul0_node, mul1_node, &mul0_output));
  RETURN_IF_ERROR(graph.AddConsumer(mul1_node->id, input2->id));
  mul0_output->tensor.type = DataType::FLOAT32;
  mul0_output->tensor.shape = BHWC(1, 32, 32, 16);

  tflite::gpu::Value* mul1_output = nullptr;
  RETURN_IF_ERROR(AddOutput(&graph, mul1_node, &mul1_output));
  mul1_output->tensor.type = DataType::FLOAT32;
  mul1_output->tensor.shape = BHWC(1, 32, 32, 16);

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

      TensorFloat32 src0_tensor;
      src0_tensor.shape = input0->tensor.shape;
      src0_tensor.data.resize(src0_tensor.shape.DimensionsProduct());
      for (int i = 0; i < src0_tensor.data.size(); ++i) {
        src0_tensor.data[i] = std::sin(i * 0.12345f);
      }
      TensorFloat32 src1_tensor;
      src1_tensor.shape = input1->tensor.shape;
      src1_tensor.data.resize(src1_tensor.shape.DimensionsProduct());
      for (int i = 0; i < src1_tensor.data.size(); ++i) {
        src1_tensor.data[i] = std::sin(i * 0.12345f);
      }
      TensorFloat32 src2_tensor;
      src2_tensor.shape = input2->tensor.shape;
      src2_tensor.data.resize(src2_tensor.shape.DimensionsProduct());
      for (int i = 0; i < src2_tensor.data.size(); ++i) {
        src2_tensor.data[i] = std::sin(i * 0.12345f);
      }

      TensorFloat32 dst_tensor_v1;
      RETURN_IF_ERROR(
          env->ExecuteGpuModel({src0_tensor, src1_tensor, src2_tensor},
                               {&dst_tensor_v1}, &gpu_model));

      OperationDef op_def;
      op_def.precision = precision;
      op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
      op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});

      ConvGeneric conv_operation =
          CreateConvGeneric(env->GetGpuInfo(), op_def, conv_attr);
      TensorFloat32 intermediate0;
      RETURN_IF_ERROR(env->ExecuteGPUOperation(
          src0_tensor, std::make_unique<ConvGeneric>(std::move(conv_operation)),
          conv_output->tensor.shape, &intermediate0));

      OperationDef op_def_mul;
      op_def_mul.precision = precision;
      op_def_mul.src_tensors.push_back({data_type, storage, Layout::HWC});
      op_def_mul.src_tensors.push_back({data_type, storage, Layout::HWC});
      op_def_mul.dst_tensors.push_back({data_type, storage, Layout::HWC});

      GPUOperation mul0_operation = CreateElementwiseTwoInput(
          op_def_mul, OperationType::MUL, src1_tensor.shape);
      TensorFloat32 intermediate1;
      RETURN_IF_ERROR(env->ExecuteGPUOperation(
          {intermediate0, src1_tensor},
          std::make_unique<GPUOperation>(std::move(mul0_operation)),
          mul0_output->tensor.shape, &intermediate1));

      GPUOperation mul1_operation = CreateElementwiseTwoInput(
          op_def_mul, OperationType::MUL, src2_tensor.shape);
      TensorFloat32 dst_tensor_v0;
      RETURN_IF_ERROR(env->ExecuteGPUOperation(
          {intermediate1, src2_tensor},
          std::make_unique<GPUOperation>(std::move(mul1_operation)),
          mul1_output->tensor.shape, &dst_tensor_v0));

      RETURN_IF_ERROR(
          PointWiseNear(dst_tensor_v0.data, dst_tensor_v1.data, 0.0f));
    }
  }
  return absl::OkStatus();
}

absl::Status TestLinkingConvolution2InputMul2InputMulCos(
    TestExecutionEnvironment* env) {
  GraphFloat32 graph;
  auto input0 = graph.NewValue();
  auto input1 = graph.NewValue();
  auto input2 = graph.NewValue();
  input0->tensor.type = DataType::FLOAT32;
  input0->tensor.shape = BHWC(1, 32, 32, 128);
  input1->tensor.type = DataType::FLOAT32;
  input1->tensor.shape = BHWC(1, 32, 32, 16);
  input2->tensor.type = DataType::FLOAT32;
  input2->tensor.shape = BHWC(1, 32, 32, 16);

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
  RETURN_IF_ERROR(graph.AddConsumer(conv_node->id, input0->id));

  auto mul0_node = graph.NewNode();
  mul0_node->operation.type = ToString(OperationType::MUL);
  tflite::gpu::Value* conv_output = nullptr;
  RETURN_IF_ERROR(ConnectTwoNodes(&graph, conv_node, mul0_node, &conv_output));
  RETURN_IF_ERROR(graph.AddConsumer(mul0_node->id, input1->id));
  conv_output->tensor.type = DataType::FLOAT32;
  conv_output->tensor.shape = BHWC(1, 32, 32, 16);

  auto mul1_node = graph.NewNode();
  mul1_node->operation.type = ToString(OperationType::MUL);
  tflite::gpu::Value* mul0_output = nullptr;
  RETURN_IF_ERROR(ConnectTwoNodes(&graph, mul0_node, mul1_node, &mul0_output));
  RETURN_IF_ERROR(graph.AddConsumer(mul1_node->id, input2->id));
  mul0_output->tensor.type = DataType::FLOAT32;
  mul0_output->tensor.shape = BHWC(1, 32, 32, 16);

  auto cos_node = graph.NewNode();
  cos_node->operation.type = ToString(OperationType::COS);
  tflite::gpu::Value* mul1_output = nullptr;
  RETURN_IF_ERROR(ConnectTwoNodes(&graph, mul1_node, cos_node, &mul1_output));
  mul1_output->tensor.type = DataType::FLOAT32;
  mul1_output->tensor.shape = BHWC(1, 32, 32, 16);

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

      TensorFloat32 src0_tensor;
      src0_tensor.shape = input0->tensor.shape;
      src0_tensor.data.resize(src0_tensor.shape.DimensionsProduct());
      for (int i = 0; i < src0_tensor.data.size(); ++i) {
        src0_tensor.data[i] = std::sin(i * 0.12345f);
      }
      TensorFloat32 src1_tensor;
      src1_tensor.shape = input1->tensor.shape;
      src1_tensor.data.resize(src1_tensor.shape.DimensionsProduct());
      for (int i = 0; i < src1_tensor.data.size(); ++i) {
        src1_tensor.data[i] = std::sin(i * 0.12345f);
      }
      TensorFloat32 src2_tensor;
      src2_tensor.shape = input2->tensor.shape;
      src2_tensor.data.resize(src2_tensor.shape.DimensionsProduct());
      for (int i = 0; i < src2_tensor.data.size(); ++i) {
        src2_tensor.data[i] = std::sin(i * 0.12345f);
      }

      TensorFloat32 dst_tensor_v1;
      RETURN_IF_ERROR(
          env->ExecuteGpuModel({src0_tensor, src1_tensor, src2_tensor},
                               {&dst_tensor_v1}, &gpu_model));

      OperationDef op_def;
      op_def.precision = precision;
      op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
      op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});

      ConvGeneric conv_operation =
          CreateConvGeneric(env->GetGpuInfo(), op_def, conv_attr);
      TensorFloat32 intermediate0;
      RETURN_IF_ERROR(env->ExecuteGPUOperation(
          src0_tensor, std::make_unique<ConvGeneric>(std::move(conv_operation)),
          conv_output->tensor.shape, &intermediate0));

      OperationDef op_def_mul;
      op_def_mul.precision = precision;
      op_def_mul.src_tensors.push_back({data_type, storage, Layout::HWC});
      op_def_mul.src_tensors.push_back({data_type, storage, Layout::HWC});
      op_def_mul.dst_tensors.push_back({data_type, storage, Layout::HWC});

      GPUOperation mul0_operation = CreateElementwiseTwoInput(
          op_def_mul, OperationType::MUL, src1_tensor.shape);
      TensorFloat32 intermediate1;
      RETURN_IF_ERROR(env->ExecuteGPUOperation(
          {intermediate0, src1_tensor},
          std::make_unique<GPUOperation>(std::move(mul0_operation)),
          mul0_output->tensor.shape, &intermediate1));

      GPUOperation mul1_operation = CreateElementwiseTwoInput(
          op_def_mul, OperationType::MUL, src2_tensor.shape);
      TensorFloat32 intermediate2;
      RETURN_IF_ERROR(env->ExecuteGPUOperation(
          {intermediate1, src2_tensor},
          std::make_unique<GPUOperation>(std::move(mul1_operation)),
          mul1_output->tensor.shape, &intermediate2));

      GPUOperation cos_operation = CreateElementwiseOneInput(
          env->GetGpuInfo(), op_def, OperationType::COS);
      TensorFloat32 dst_tensor_v0;
      RETURN_IF_ERROR(env->ExecuteGPUOperation(
          intermediate2,
          std::make_unique<GPUOperation>(std::move(cos_operation)),
          cos_output->tensor.shape, &dst_tensor_v0));

      RETURN_IF_ERROR(
          PointWiseNear(dst_tensor_v0.data, dst_tensor_v1.data, 0.0f));
    }
  }
  return absl::OkStatus();
}

absl::Status TestLinkingConvolutionFirstTanh2InputDiff(
    TestExecutionEnvironment* env) {
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

  auto tanh_node = graph.NewNode();
  tanh_node->operation.type = ToString(OperationType::TANH);
  tflite::gpu::Value* conv_output = nullptr;
  RETURN_IF_ERROR(ConnectTwoNodes(&graph, conv_node, tanh_node, &conv_output));
  conv_output->tensor.type = DataType::FLOAT32;
  conv_output->tensor.shape = BHWC(1, 32, 32, 16);

  auto sub_node = graph.NewNode();
  sub_node->operation.type = ToString(OperationType::SUB);
  auto tanh_output = graph.NewValue();
  tanh_output->tensor.type = DataType::FLOAT32;
  tanh_output->tensor.shape = BHWC(1, 32, 32, 16);
  auto sub_output = graph.NewValue();
  sub_output->tensor.type = DataType::FLOAT32;
  sub_output->tensor.shape = BHWC(1, 32, 32, 16);
  RETURN_IF_ERROR(graph.SetProducer(tanh_node->id, tanh_output->id));
  RETURN_IF_ERROR(graph.AddConsumer(sub_node->id, tanh_output->id));
  RETURN_IF_ERROR(graph.AddConsumer(sub_node->id, conv_output->id));
  RETURN_IF_ERROR(graph.SetProducer(sub_node->id, sub_output->id));

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
      TensorFloat32 intermediate0;
      RETURN_IF_ERROR(env->ExecuteGPUOperation(
          src_tensor, std::make_unique<ConvGeneric>(std::move(conv_operation)),
          conv_output->tensor.shape, &intermediate0));

      GPUOperation tanh_operation = CreateElementwiseOneInput(
          env->GetGpuInfo(), op_def, OperationType::TANH);
      TensorFloat32 intermediate1;
      RETURN_IF_ERROR(env->ExecuteGPUOperation(
          intermediate0,
          std::make_unique<GPUOperation>(std::move(tanh_operation)),
          tanh_output->tensor.shape, &intermediate1));

      OperationDef op_def_sub;
      op_def_sub.precision = precision;
      op_def_sub.src_tensors.push_back({data_type, storage, Layout::HWC});
      op_def_sub.src_tensors.push_back({data_type, storage, Layout::HWC});
      op_def_sub.dst_tensors.push_back({data_type, storage, Layout::HWC});
      GPUOperation sub_operation = CreateElementwiseTwoInput(
          op_def_sub, OperationType::SUB, conv_output->tensor.shape);
      TensorFloat32 dst_tensor_v0;
      RETURN_IF_ERROR(env->ExecuteGPUOperation(
          {intermediate1, intermediate0},
          std::make_unique<GPUOperation>(std::move(sub_operation)),
          sub_output->tensor.shape, &dst_tensor_v0));

      RETURN_IF_ERROR(
          PointWiseNear(dst_tensor_v0.data, dst_tensor_v1.data, 0.0f));
    }
  }
  return absl::OkStatus();
}

absl::Status TestLinkingConvolutionSecondTanh2InputDiff(
    TestExecutionEnvironment* env) {
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

  auto tanh_node = graph.NewNode();
  tanh_node->operation.type = ToString(OperationType::TANH);
  tflite::gpu::Value* conv_output = nullptr;
  RETURN_IF_ERROR(ConnectTwoNodes(&graph, conv_node, tanh_node, &conv_output));
  conv_output->tensor.type = DataType::FLOAT32;
  conv_output->tensor.shape = BHWC(1, 32, 32, 16);

  auto sub_node = graph.NewNode();
  sub_node->operation.type = ToString(OperationType::SUB);
  auto tanh_output = graph.NewValue();
  tanh_output->tensor.type = DataType::FLOAT32;
  tanh_output->tensor.shape = BHWC(1, 32, 32, 16);
  auto sub_output = graph.NewValue();
  sub_output->tensor.type = DataType::FLOAT32;
  sub_output->tensor.shape = BHWC(1, 32, 32, 16);
  RETURN_IF_ERROR(graph.SetProducer(tanh_node->id, tanh_output->id));
  RETURN_IF_ERROR(graph.AddConsumer(sub_node->id, conv_output->id));
  RETURN_IF_ERROR(graph.AddConsumer(sub_node->id, tanh_output->id));
  RETURN_IF_ERROR(graph.SetProducer(sub_node->id, sub_output->id));

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
      TensorFloat32 intermediate0;
      RETURN_IF_ERROR(env->ExecuteGPUOperation(
          src_tensor, std::make_unique<ConvGeneric>(std::move(conv_operation)),
          conv_output->tensor.shape, &intermediate0));

      GPUOperation tanh_operation = CreateElementwiseOneInput(
          env->GetGpuInfo(), op_def, OperationType::TANH);
      TensorFloat32 intermediate1;
      RETURN_IF_ERROR(env->ExecuteGPUOperation(
          intermediate0,
          std::make_unique<GPUOperation>(std::move(tanh_operation)),
          tanh_output->tensor.shape, &intermediate1));

      OperationDef op_def_sub;
      op_def_sub.precision = precision;
      op_def_sub.src_tensors.push_back({data_type, storage, Layout::HWC});
      op_def_sub.src_tensors.push_back({data_type, storage, Layout::HWC});
      op_def_sub.dst_tensors.push_back({data_type, storage, Layout::HWC});
      GPUOperation sub_operation = CreateElementwiseTwoInput(
          op_def_sub, OperationType::SUB, conv_output->tensor.shape);
      TensorFloat32 dst_tensor_v0;
      RETURN_IF_ERROR(env->ExecuteGPUOperation(
          {intermediate0, intermediate1},
          std::make_unique<GPUOperation>(std::move(sub_operation)),
          sub_output->tensor.shape, &dst_tensor_v0));

      RETURN_IF_ERROR(
          PointWiseNear(dst_tensor_v0.data, dst_tensor_v1.data, 0.0f));
    }
  }
  return absl::OkStatus();
}

//      input
//        |
//   convolution
//     /     \
//   tanh    cos
//     \     /
//  substraction
//        |
//     output
absl::Status TestLinkingConvolutionFirstTanhSecondCos2InputDiff(
    TestExecutionEnvironment* env) {
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

  auto tanh_node = graph.NewNode();
  tanh_node->operation.type = ToString(OperationType::TANH);
  tflite::gpu::Value* conv_output = nullptr;
  RETURN_IF_ERROR(ConnectTwoNodes(&graph, conv_node, tanh_node, &conv_output));
  conv_output->tensor.type = DataType::FLOAT32;
  conv_output->tensor.shape = BHWC(1, 32, 32, 16);

  auto cos_node = graph.NewNode();
  cos_node->operation.type = ToString(OperationType::COS);
  auto cos_output = graph.NewValue();
  cos_output->tensor.type = DataType::FLOAT32;
  cos_output->tensor.shape = BHWC(1, 32, 32, 16);
  RETURN_IF_ERROR(graph.AddConsumer(cos_node->id, conv_output->id));
  RETURN_IF_ERROR(graph.SetProducer(cos_node->id, cos_output->id));

  auto sub_node = graph.NewNode();
  sub_node->operation.type = ToString(OperationType::SUB);
  auto tanh_output = graph.NewValue();
  tanh_output->tensor.type = DataType::FLOAT32;
  tanh_output->tensor.shape = BHWC(1, 32, 32, 16);
  auto sub_output = graph.NewValue();
  sub_output->tensor.type = DataType::FLOAT32;
  sub_output->tensor.shape = BHWC(1, 32, 32, 16);
  RETURN_IF_ERROR(graph.SetProducer(tanh_node->id, tanh_output->id));
  RETURN_IF_ERROR(graph.AddConsumer(sub_node->id, tanh_output->id));
  RETURN_IF_ERROR(graph.AddConsumer(sub_node->id, cos_output->id));
  RETURN_IF_ERROR(graph.SetProducer(sub_node->id, sub_output->id));

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
      TensorFloat32 intermediate0;
      RETURN_IF_ERROR(env->ExecuteGPUOperation(
          src_tensor, std::make_unique<ConvGeneric>(std::move(conv_operation)),
          conv_output->tensor.shape, &intermediate0));

      GPUOperation tanh_operation = CreateElementwiseOneInput(
          env->GetGpuInfo(), op_def, OperationType::TANH);
      TensorFloat32 intermediate1;
      RETURN_IF_ERROR(env->ExecuteGPUOperation(
          intermediate0,
          std::make_unique<GPUOperation>(std::move(tanh_operation)),
          tanh_output->tensor.shape, &intermediate1));

      GPUOperation cos_operation = CreateElementwiseOneInput(
          env->GetGpuInfo(), op_def, OperationType::COS);
      TensorFloat32 intermediate2;
      RETURN_IF_ERROR(env->ExecuteGPUOperation(
          intermediate0,
          std::make_unique<GPUOperation>(std::move(cos_operation)),
          cos_output->tensor.shape, &intermediate2));

      OperationDef op_def_sub;
      op_def_sub.precision = precision;
      op_def_sub.src_tensors.push_back({data_type, storage, Layout::HWC});
      op_def_sub.src_tensors.push_back({data_type, storage, Layout::HWC});
      op_def_sub.dst_tensors.push_back({data_type, storage, Layout::HWC});
      GPUOperation sub_operation = CreateElementwiseTwoInput(
          op_def_sub, OperationType::SUB, conv_output->tensor.shape);
      TensorFloat32 dst_tensor_v0;
      RETURN_IF_ERROR(env->ExecuteGPUOperation(
          {intermediate1, intermediate2},
          std::make_unique<GPUOperation>(std::move(sub_operation)),
          sub_output->tensor.shape, &dst_tensor_v0));

      RETURN_IF_ERROR(
          PointWiseNear(dst_tensor_v0.data, dst_tensor_v1.data, 0.0f));
    }
  }
  return absl::OkStatus();
}

//      input
//        |
//   convolution
//      /    \
//   tanh    cos
//    /     /   \
//   |    prelu  sin
//   |      \   /
//   |       pow
//   |        |
//   |       exp
//    \       |
//  substraction
//        |
//     output
absl::Status TestLinkingComplex0(TestExecutionEnvironment* env) {
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
  auto conv_output = graph.NewValue();
  conv_output->tensor.type = DataType::FLOAT32;
  conv_output->tensor.shape = BHWC(1, 32, 32, 16);
  RETURN_IF_ERROR(graph.AddConsumer(conv_node->id, input->id));
  RETURN_IF_ERROR(graph.SetProducer(conv_node->id, conv_output->id));

  auto tanh_node = graph.NewNode();
  tanh_node->operation.type = ToString(OperationType::TANH);
  auto tanh_output = graph.NewValue();
  tanh_output->tensor.type = DataType::FLOAT32;
  tanh_output->tensor.shape = BHWC(1, 32, 32, 16);
  RETURN_IF_ERROR(graph.AddConsumer(tanh_node->id, conv_output->id));
  RETURN_IF_ERROR(graph.SetProducer(tanh_node->id, tanh_output->id));

  auto cos_node = graph.NewNode();
  cos_node->operation.type = ToString(OperationType::COS);
  auto cos_output = graph.NewValue();
  cos_output->tensor.type = DataType::FLOAT32;
  cos_output->tensor.shape = BHWC(1, 32, 32, 16);
  RETURN_IF_ERROR(graph.AddConsumer(cos_node->id, conv_output->id));
  RETURN_IF_ERROR(graph.SetProducer(cos_node->id, cos_output->id));

  auto prelu_node = graph.NewNode();
  prelu_node->operation.type = ToString(OperationType::PRELU);
  PReLUAttributes prelu_attr;
  tflite::gpu::Tensor<Linear, DataType::FLOAT32> parameters;
  parameters.shape = Linear(16);
  parameters.data.resize(parameters.shape.DimensionsProduct());
  for (int i = 0; i < parameters.data.size(); ++i) {
    parameters.data[i] = std::sin(i * 0.5f);
  }
  prelu_attr.alpha = parameters;
  prelu_node->operation.attributes = prelu_attr;
  auto prelu_output = graph.NewValue();
  prelu_output->tensor.type = DataType::FLOAT32;
  prelu_output->tensor.shape = BHWC(1, 32, 32, 16);
  RETURN_IF_ERROR(graph.AddConsumer(prelu_node->id, cos_output->id));
  RETURN_IF_ERROR(graph.SetProducer(prelu_node->id, prelu_output->id));

  auto sin_node = graph.NewNode();
  sin_node->operation.type = ToString(OperationType::SIN);
  auto sin_output = graph.NewValue();
  sin_output->tensor.type = DataType::FLOAT32;
  sin_output->tensor.shape = BHWC(1, 32, 32, 16);
  RETURN_IF_ERROR(graph.AddConsumer(sin_node->id, cos_output->id));
  RETURN_IF_ERROR(graph.SetProducer(sin_node->id, sin_output->id));

  auto pow_node = graph.NewNode();
  pow_node->operation.type = ToString(OperationType::POW);
  auto pow_output = graph.NewValue();
  pow_output->tensor.type = DataType::FLOAT32;
  pow_output->tensor.shape = BHWC(1, 32, 32, 16);
  RETURN_IF_ERROR(graph.AddConsumer(pow_node->id, prelu_output->id));
  RETURN_IF_ERROR(graph.AddConsumer(pow_node->id, sin_output->id));
  RETURN_IF_ERROR(graph.SetProducer(pow_node->id, pow_output->id));

  auto exp_node = graph.NewNode();
  exp_node->operation.type = ToString(OperationType::EXP);
  auto exp_output = graph.NewValue();
  exp_output->tensor.type = DataType::FLOAT32;
  exp_output->tensor.shape = BHWC(1, 32, 32, 16);
  RETURN_IF_ERROR(graph.AddConsumer(exp_node->id, pow_output->id));
  RETURN_IF_ERROR(graph.SetProducer(exp_node->id, exp_output->id));

  auto sub_node = graph.NewNode();
  sub_node->operation.type = ToString(OperationType::SUB);
  auto sub_output = graph.NewValue();
  sub_output->tensor.type = DataType::FLOAT32;
  sub_output->tensor.shape = BHWC(1, 32, 32, 16);
  RETURN_IF_ERROR(graph.AddConsumer(sub_node->id, tanh_output->id));
  RETURN_IF_ERROR(graph.AddConsumer(sub_node->id, exp_output->id));
  RETURN_IF_ERROR(graph.SetProducer(sub_node->id, sub_output->id));

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

      OperationDef op_def_two_input;
      op_def_two_input.precision = precision;
      op_def_two_input.src_tensors.push_back({data_type, storage, Layout::HWC});
      op_def_two_input.src_tensors.push_back({data_type, storage, Layout::HWC});
      op_def_two_input.dst_tensors.push_back({data_type, storage, Layout::HWC});

      ConvGeneric conv_operation =
          CreateConvGeneric(env->GetGpuInfo(), op_def, conv_attr);
      TensorFloat32 intermediate0;
      RETURN_IF_ERROR(env->ExecuteGPUOperation(
          src_tensor, std::make_unique<ConvGeneric>(std::move(conv_operation)),
          conv_output->tensor.shape, &intermediate0));

      GPUOperation tanh_operation = CreateElementwiseOneInput(
          env->GetGpuInfo(), op_def, OperationType::TANH);
      TensorFloat32 intermediate1;
      RETURN_IF_ERROR(env->ExecuteGPUOperation(
          intermediate0,
          std::make_unique<GPUOperation>(std::move(tanh_operation)),
          tanh_output->tensor.shape, &intermediate1));

      GPUOperation cos_operation = CreateElementwiseOneInput(
          env->GetGpuInfo(), op_def, OperationType::COS);
      TensorFloat32 intermediate2;
      RETURN_IF_ERROR(env->ExecuteGPUOperation(
          intermediate0,
          std::make_unique<GPUOperation>(std::move(cos_operation)),
          cos_output->tensor.shape, &intermediate2));

      GPUOperation prelu_operation =
          CreatePReLU(env->GetGpuInfo(), op_def, prelu_attr);
      TensorFloat32 intermediate3;
      RETURN_IF_ERROR(env->ExecuteGPUOperation(
          intermediate2,
          std::make_unique<GPUOperation>(std::move(prelu_operation)),
          prelu_output->tensor.shape, &intermediate3));

      GPUOperation sin_operation = CreateElementwiseOneInput(
          env->GetGpuInfo(), op_def, OperationType::SIN);
      TensorFloat32 intermediate4;
      RETURN_IF_ERROR(env->ExecuteGPUOperation(
          intermediate2,
          std::make_unique<GPUOperation>(std::move(sin_operation)),
          sin_output->tensor.shape, &intermediate4));

      GPUOperation pow_operation = CreateElementwiseTwoInput(
          op_def_two_input, OperationType::POW, sin_output->tensor.shape);
      TensorFloat32 intermediate5;
      RETURN_IF_ERROR(env->ExecuteGPUOperation(
          {intermediate3, intermediate4},
          std::make_unique<GPUOperation>(std::move(pow_operation)),
          pow_output->tensor.shape, &intermediate5));

      GPUOperation exp_operation = CreateElementwiseOneInput(
          env->GetGpuInfo(), op_def, OperationType::EXP);
      TensorFloat32 intermediate6;
      RETURN_IF_ERROR(env->ExecuteGPUOperation(
          intermediate5,
          std::make_unique<GPUOperation>(std::move(exp_operation)),
          exp_output->tensor.shape, &intermediate6));

      GPUOperation sub_operation = CreateElementwiseTwoInput(
          op_def_two_input, OperationType::SUB, conv_output->tensor.shape);
      TensorFloat32 dst_tensor_v0;
      RETURN_IF_ERROR(env->ExecuteGPUOperation(
          {intermediate1, intermediate6},
          std::make_unique<GPUOperation>(std::move(sub_operation)),
          sub_output->tensor.shape, &dst_tensor_v0));

      RETURN_IF_ERROR(
          PointWiseNear(dst_tensor_v0.data, dst_tensor_v1.data, 0.0f));
    }
  }
  return absl::OkStatus();
}

//                input1
//                  |
//              convolution
//                  |
//         input0  cos0
//             \   /
//              add
//               |
//              cos1
//               |
//              sin
//               |
//              abs
//               |
//             output
absl::Status TestLinkingConvElem2InputAddElemsOp(
    TestExecutionEnvironment* env) {
  GraphFloat32 graph;
  auto input0 = graph.NewValue();
  auto input1 = graph.NewValue();
  input0->tensor.type = DataType::FLOAT32;
  input0->tensor.shape = BHWC(1, 32, 32, 16);
  input1->tensor.type = DataType::FLOAT32;
  input1->tensor.shape = BHWC(1, 32, 32, 8);

  auto conv_node = graph.NewNode();
  conv_node->operation.type =
      ToString(tflite::gpu::OperationType::CONVOLUTION_2D);

  Convolution2DAttributes conv_attr;
  conv_attr.padding.prepended = HW(0, 0);
  conv_attr.padding.appended = HW(0, 0);
  conv_attr.strides = HW(1, 1);
  conv_attr.dilations = HW(1, 1);
  conv_attr.weights.shape = OHWI(16, 1, 1, 8);
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
  auto conv_output = graph.NewValue();
  conv_output->tensor.type = DataType::FLOAT32;
  conv_output->tensor.shape = BHWC(1, 32, 32, 16);
  RETURN_IF_ERROR(graph.AddConsumer(conv_node->id, input1->id));
  RETURN_IF_ERROR(graph.SetProducer(conv_node->id, conv_output->id));

  auto cos0_node = graph.NewNode();
  cos0_node->operation.type = ToString(OperationType::COS);
  auto cos0_output = graph.NewValue();
  cos0_output->tensor.type = DataType::FLOAT32;
  cos0_output->tensor.shape = BHWC(1, 32, 32, 16);
  RETURN_IF_ERROR(graph.AddConsumer(cos0_node->id, conv_output->id));
  RETURN_IF_ERROR(graph.SetProducer(cos0_node->id, cos0_output->id));

  auto add_node = graph.NewNode();
  add_node->operation.type = ToString(OperationType::ADD);
  auto add_output = graph.NewValue();
  add_output->tensor.type = DataType::FLOAT32;
  add_output->tensor.shape = BHWC(1, 32, 32, 16);
  RETURN_IF_ERROR(graph.AddConsumer(add_node->id, input0->id));
  RETURN_IF_ERROR(graph.AddConsumer(add_node->id, cos0_output->id));
  RETURN_IF_ERROR(graph.SetProducer(add_node->id, add_output->id));

  auto cos1_node = graph.NewNode();
  cos1_node->operation.type = ToString(OperationType::COS);
  auto cos1_output = graph.NewValue();
  cos1_output->tensor.type = DataType::FLOAT32;
  cos1_output->tensor.shape = BHWC(1, 32, 32, 16);
  RETURN_IF_ERROR(graph.AddConsumer(cos1_node->id, add_output->id));
  RETURN_IF_ERROR(graph.SetProducer(cos1_node->id, cos1_output->id));

  auto sin_node = graph.NewNode();
  sin_node->operation.type = ToString(OperationType::SIN);
  auto sin_output = graph.NewValue();
  sin_output->tensor.type = DataType::FLOAT32;
  sin_output->tensor.shape = BHWC(1, 32, 32, 16);
  RETURN_IF_ERROR(graph.AddConsumer(sin_node->id, cos1_output->id));
  RETURN_IF_ERROR(graph.SetProducer(sin_node->id, sin_output->id));

  auto abs_node = graph.NewNode();
  abs_node->operation.type = ToString(OperationType::ABS);
  auto abs_output = graph.NewValue();
  abs_output->tensor.type = DataType::FLOAT32;
  abs_output->tensor.shape = BHWC(1, 32, 32, 16);
  RETURN_IF_ERROR(graph.AddConsumer(abs_node->id, sin_output->id));
  RETURN_IF_ERROR(graph.SetProducer(abs_node->id, abs_output->id));

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

      TensorFloat32 src0_tensor, src1_tensor;
      src0_tensor.shape = input0->tensor.shape;
      src0_tensor.data.resize(src0_tensor.shape.DimensionsProduct());
      for (int i = 0; i < src0_tensor.data.size(); ++i) {
        src0_tensor.data[i] = std::sin(i * 0.12345f);
      }
      src1_tensor.shape = input1->tensor.shape;
      src1_tensor.data.resize(src1_tensor.shape.DimensionsProduct());
      for (int i = 0; i < src1_tensor.data.size(); ++i) {
        src1_tensor.data[i] = std::sin(i * 0.12345f);
      }

      TensorFloat32 dst_tensor_v1;
      RETURN_IF_ERROR(env->ExecuteGpuModel({src1_tensor, src0_tensor},
                                           {&dst_tensor_v1}, &gpu_model));

      OperationDef op_def;
      op_def.precision = precision;
      op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
      op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});

      OperationDef op_def_two_input;
      op_def_two_input.precision = precision;
      op_def_two_input.src_tensors.push_back({data_type, storage, Layout::HWC});
      op_def_two_input.src_tensors.push_back({data_type, storage, Layout::HWC});
      op_def_two_input.dst_tensors.push_back({data_type, storage, Layout::HWC});

      ConvGeneric conv_operation =
          CreateConvGeneric(env->GetGpuInfo(), op_def, conv_attr);
      TensorFloat32 intermediate1;
      RETURN_IF_ERROR(env->ExecuteGPUOperation(
          src1_tensor, std::make_unique<ConvGeneric>(std::move(conv_operation)),
          conv_output->tensor.shape, &intermediate1));

      GPUOperation cos0_operation = CreateElementwiseOneInput(
          env->GetGpuInfo(), op_def, OperationType::COS);
      TensorFloat32 intermediate2;
      RETURN_IF_ERROR(env->ExecuteGPUOperation(
          intermediate1,
          std::make_unique<GPUOperation>(std::move(cos0_operation)),
          cos0_output->tensor.shape, &intermediate2));

      GPUOperation add_operation = CreateElementwiseTwoInput(
          op_def_two_input, OperationType::ADD, add_output->tensor.shape);
      TensorFloat32 intermediate3;
      RETURN_IF_ERROR(env->ExecuteGPUOperation(
          {src0_tensor, intermediate2},
          std::make_unique<GPUOperation>(std::move(add_operation)),
          add_output->tensor.shape, &intermediate3));

      GPUOperation cos1_operation = CreateElementwiseOneInput(
          env->GetGpuInfo(), op_def, OperationType::COS);
      TensorFloat32 intermediate4;
      RETURN_IF_ERROR(env->ExecuteGPUOperation(
          intermediate3,
          std::make_unique<GPUOperation>(std::move(cos1_operation)),
          cos1_output->tensor.shape, &intermediate4));

      GPUOperation sin_operation = CreateElementwiseOneInput(
          env->GetGpuInfo(), op_def, OperationType::SIN);
      TensorFloat32 intermediate5;
      RETURN_IF_ERROR(env->ExecuteGPUOperation(
          intermediate4,
          std::make_unique<GPUOperation>(std::move(sin_operation)),
          sin_output->tensor.shape, &intermediate5));

      GPUOperation abs_operation = CreateElementwiseOneInput(
          env->GetGpuInfo(), op_def, OperationType::ABS);
      TensorFloat32 dst_tensor_v0;
      RETURN_IF_ERROR(env->ExecuteGPUOperation(
          intermediate5,
          std::make_unique<GPUOperation>(std::move(abs_operation)),
          abs_output->tensor.shape, &dst_tensor_v0));

      RETURN_IF_ERROR(
          PointWiseNear(dst_tensor_v0.data, dst_tensor_v1.data, 0.0f));
    }
  }
  return absl::OkStatus();
}

}  // namespace gpu
}  // namespace tflite
