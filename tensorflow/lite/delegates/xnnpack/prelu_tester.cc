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

#include "tensorflow/lite/delegates/xnnpack/prelu_tester.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <numeric>
#include <random>
#include <vector>

#include <gtest/gtest.h>
#include "fp16.h"  // from @FP16
#include "flatbuffers/buffer.h"  // from @flatbuffers
#include "flatbuffers/flatbuffer_builder.h"  // from @flatbuffers
#include "flatbuffers/string.h"  // from @flatbuffers
#include "tensorflow/compiler/mlir/lite/schema/schema_conversion_utils.h"
#include "tensorflow/lite/core/interpreter_builder.h"
#include "tensorflow/lite/core/kernels/register.h"
#include "tensorflow/lite/delegates/xnnpack/test_util.h"
#include "tensorflow/lite/delegates/xnnpack/xnnpack_delegate.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

namespace tflite {
namespace xnnpack {

void PreluTester::Test(TfLiteDelegate* delegate) const {
  if (INT8ChannelWiseWeights()) {
    ASSERT_FALSE(SlopeShape().empty());
  }

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto input_rng = std::bind(std::uniform_real_distribution<float>(-1.0f, 1.0f),
                             std::ref(rng));

  std::vector<char> buffer = CreateTfLiteModel();
  const Model* model = GetModel(buffer.data());

  std::unique_ptr<Interpreter> delegate_interpreter;
  ASSERT_EQ(
      InterpreterBuilder(
          model,
          ::tflite::ops::builtin::BuiltinOpResolverWithoutDefaultDelegates())(
          &delegate_interpreter),
      kTfLiteOk);
  std::unique_ptr<Interpreter> default_interpreter;
  ASSERT_EQ(
      InterpreterBuilder(
          model,
          ::tflite::ops::builtin::BuiltinOpResolverWithoutDefaultDelegates())(
          &default_interpreter),
      kTfLiteOk);

  ASSERT_TRUE(delegate_interpreter);
  ASSERT_TRUE(default_interpreter);

  ASSERT_EQ(delegate_interpreter->inputs().size(), 1);
  ASSERT_EQ(default_interpreter->inputs().size(), 1);

  ASSERT_EQ(delegate_interpreter->outputs().size(), 1);
  ASSERT_EQ(default_interpreter->outputs().size(), 1);

  ASSERT_EQ(delegate_interpreter->AllocateTensors(), kTfLiteOk);
  ASSERT_EQ(default_interpreter->AllocateTensors(), kTfLiteOk);

  ASSERT_EQ(delegate_interpreter->ModifyGraphWithDelegate(delegate), kTfLiteOk);

  if (weights_cache_ != nullptr) {
    TfLiteXNNPackDelegateWeightsCacheFinalizeHard(weights_cache_);
  }

  float* default_input_data = default_interpreter->typed_input_tensor<float>(0);
  std::generate_n(default_input_data, ComputeSize(InputShape()),
                  std::ref(input_rng));

  float* xnnpack_input_data =
      delegate_interpreter->typed_input_tensor<float>(0);
  std::copy_n(default_input_data, ComputeSize(InputShape()),
              xnnpack_input_data);

  ASSERT_EQ(default_interpreter->Invoke(), kTfLiteOk);
  ASSERT_EQ(delegate_interpreter->Invoke(), kTfLiteOk);

  float* default_output_data =
      default_interpreter->typed_output_tensor<float>(0);
  float* xnnpack_output_data =
      delegate_interpreter->typed_output_tensor<float>(0);

  for (size_t i = 0; i < ComputeSize(OutputShape()); i++) {
    ASSERT_EQ(default_output_data[i], xnnpack_output_data[i]);
  }
}

std::vector<char> PreluTester::CreateTfLiteModel() const {
  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto slope_rng = std::bind(std::uniform_real_distribution<float>(0.25f, 0.5f),
                             std::ref(rng));

  flatbuffers::FlatBufferBuilder builder;
  std::vector<flatbuffers::Offset<OperatorCode>> operator_codes{
      {CreateOperatorCode(builder, BuiltinOperator_PRELU)}};
  if (FP16Weights() || INT8Weights() || INT8ChannelWiseWeights()) {
    operator_codes.emplace_back(
        CreateOperatorCode(builder, BuiltinOperator_DEQUANTIZE));
  } else if (SparseWeights()) {
    operator_codes.emplace_back(
        CreateOperatorCode(builder, BuiltinOperator_DENSIFY));
  }

  std::vector<flatbuffers::Offset<Buffer>> buffers{{
      CreateBuffer(builder, builder.CreateVector({})),
  }};

  std::vector<float> slope_scales;
  std::vector<int64_t> slope_zero_points;
  int32_t slope_quantized_dimension = 0;
  if (FP16Weights()) {
    std::vector<uint16_t> slope_data(ComputeSize(SlopeShape()));
    std::generate(slope_data.begin(), slope_data.end(),
                  std::bind(fp16_ieee_from_fp32_value, slope_rng));

    buffers.push_back(CreateBuffer(
        builder, builder.CreateVector(
                     reinterpret_cast<const uint8_t*>(slope_data.data()),
                     sizeof(uint16_t) * slope_data.size())));
  } else {
    std::vector<float> slope_data(ComputeSize(SlopeShape()));
    std::generate(slope_data.begin(), slope_data.end(), slope_rng);

    if (INT8Weights()) {
      std::vector<int8_t> quantized_slope_data(slope_data.size());
      slope_scales.resize(1, GetInt8QuantizationScale(slope_data));
      slope_zero_points.resize(1, 0);
      std::transform(
          slope_data.begin(), slope_data.end(), quantized_slope_data.begin(),
          std::bind(QuantizeInt8, std::placeholders::_1, 0, slope_scales[0]));
      buffers.push_back(CreateBuffer(
          builder,
          builder.CreateVector(
              reinterpret_cast<const uint8_t*>(quantized_slope_data.data()),
              sizeof(int8_t) * quantized_slope_data.size())));
    } else if (INT8ChannelWiseWeights()) {
      std::vector<int8_t> quantized_slope_data(slope_data.size());
      slope_quantized_dimension = static_cast<int32_t>(SlopeShape().size()) - 1;
      const int32_t num_scales = SlopeShape()[slope_quantized_dimension];
      slope_scales = GetInt8QuantizationScalePerChannel(
          slope_data.data(), slope_quantized_dimension, SlopeShape());
      slope_zero_points.resize(num_scales, 0);
      QuantizeInt8PerChannel(slope_scales.data(), slope_zero_points.data(),
                             slope_quantized_dimension, slope_data.data(),
                             quantized_slope_data.data(), SlopeShape());
      buffers.push_back(CreateBuffer(
          builder,
          builder.CreateVector(
              reinterpret_cast<const uint8_t*>(quantized_slope_data.data()),
              sizeof(int8_t) * quantized_slope_data.size())));
    } else {
      buffers.push_back(CreateBuffer(
          builder, builder.CreateVector(
                       reinterpret_cast<const uint8_t*>(slope_data.data()),
                       sizeof(float) * slope_data.size())));
    }
  }

  std::vector<flatbuffers::Offset<Tensor>> tensors;
  std::vector<flatbuffers::Offset<Operator>> operators;
  if (FP16Weights()) {
    tensors.emplace_back(CreateTensor(
        builder,
        builder.CreateVector<int32_t>(SlopeShape().data(), SlopeShape().size()),
        TensorType_FLOAT16, /*buffer=*/1));
  } else if (INT8Weights() || INT8ChannelWiseWeights()) {
    tensors.emplace_back(CreateTensor(
        builder,
        builder.CreateVector<int32_t>(SlopeShape().data(), SlopeShape().size()),
        TensorType_INT8, /*buffer=*/1, /*name=*/0,
        CreateQuantizationParameters(
            builder, /*min=*/0, /*max=*/0,
            builder.CreateVector<float>(slope_scales),
            builder.CreateVector<int64_t>(slope_zero_points),
            /*details_type=*/QuantizationDetails_NONE,
            /*details=*/0, slope_quantized_dimension)));
  } else if (SparseWeights()) {
    const int dims_count = SlopeShape().size();
    std::vector<flatbuffers::Offset<DimensionMetadata>> dim_metadata(
        dims_count);
    std::vector<int> traversal_order(dims_count);
    for (int i = 0; i < dims_count; i++) {
      traversal_order[i] = i;
      dim_metadata[i] = CreateDimensionMetadata(builder, DimensionType_DENSE,
                                                SlopeShape()[i]);
    }
    const flatbuffers::Offset<SparsityParameters> sparsity_param =
        CreateSparsityParameters(builder, builder.CreateVector(traversal_order),
                                 0, builder.CreateVector(dim_metadata));
    tensors.emplace_back(CreateTensor(
        builder,
        builder.CreateVector<int32_t>(SlopeShape().data(), SlopeShape().size()),
        TensorType_FLOAT32, /*buffer=*/1, /*name=*/0, /*quantization=*/0,
        /*is_variable=*/false, /*sparsity=*/sparsity_param));
  }
  if (FP16Weights() || INT8Weights() || INT8ChannelWiseWeights()) {
    const std::array<int32_t, 1> dequantize_inputs{{0}};
    const std::array<int32_t, 1> dequantize_outputs{{2}};
    operators.emplace_back(CreateOperator(
        builder, /*opcode_index=*/1,
        builder.CreateVector<int32_t>(dequantize_inputs.data(),
                                      dequantize_inputs.size()),
        builder.CreateVector<int32_t>(dequantize_outputs.data(),
                                      dequantize_outputs.size())));
  } else if (SparseWeights()) {
    const std::array<int32_t, 1> densify_inputs{{0}};
    const std::array<int32_t, 1> densify_outputs{{2}};
    operators.emplace_back(
        CreateOperator(builder, /*opcode_index=*/1,
                       builder.CreateVector<int32_t>(densify_inputs.data(),
                                                     densify_inputs.size()),
                       builder.CreateVector<int32_t>(densify_outputs.data(),
                                                     densify_outputs.size())));
  }
  tensors.emplace_back(CreateTensor(
      builder,
      builder.CreateVector<int32_t>(InputShape().data(), InputShape().size()),
      TensorType_FLOAT32));
  tensors.emplace_back(CreateTensor(
      builder,
      builder.CreateVector<int32_t>(SlopeShape().data(), SlopeShape().size()),
      TensorType_FLOAT32,
      /*buffer=*/
      (FP16Weights() || INT8Weights() || INT8ChannelWiseWeights() ||
       SparseWeights())
          ? 0
          : 1));
  tensors.emplace_back(CreateTensor(
      builder,
      builder.CreateVector<int32_t>(OutputShape().data(), OutputShape().size()),
      TensorType_FLOAT32));

  const std::array<int32_t, 2> op_inputs{
      {static_cast<int>(tensors.size()) - 3,
       static_cast<int>(tensors.size()) - 2}};
  const std::array<int32_t, 1> op_outputs{
      {static_cast<int>(tensors.size()) - 1}};
  operators.emplace_back(CreateOperator(
      builder, /*opcode_index=*/0,
      builder.CreateVector<int32_t>(op_inputs.data(), op_inputs.size()),
      builder.CreateVector<int32_t>(op_outputs.data(), op_outputs.size())));

  const std::array<int32_t, 1> subgraph_inputs{
      {static_cast<int32_t>(tensors.size() - 3)}};
  const std::array<int32_t, 1> subgraph_outputs{
      {static_cast<int32_t>(tensors.size()) - 1}};
  flatbuffers::Offset<SubGraph> subgraph = CreateSubGraph(
      builder, builder.CreateVector(tensors.data(), tensors.size()),
      builder.CreateVector<int32_t>(subgraph_inputs.data(),
                                    subgraph_inputs.size()),
      builder.CreateVector<int32_t>(subgraph_outputs.data(),
                                    subgraph_outputs.size()),
      builder.CreateVector(operators.data(), operators.size()));

  flatbuffers::Offset<flatbuffers::String> description =
      builder.CreateString("PReLU model");

  flatbuffers::Offset<Model> model_buffer = CreateModel(
      builder, TFLITE_SCHEMA_VERSION,
      builder.CreateVector(operator_codes.data(), operator_codes.size()),
      builder.CreateVector(&subgraph, 1), description,
      builder.CreateVector(buffers.data(), buffers.size()));

  builder.Finish(model_buffer);

  return std::vector<char>(builder.GetBufferPointer(),
                           builder.GetBufferPointer() + builder.GetSize());
}

int32_t PreluTester::ComputeSize(const std::vector<int32_t>& shape) {
  return std::accumulate(shape.cbegin(), shape.cend(), 1,
                         std::multiplies<int32_t>());
}

}  // namespace xnnpack
}  // namespace tflite
