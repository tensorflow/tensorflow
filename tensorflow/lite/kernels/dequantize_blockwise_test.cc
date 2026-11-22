/* Copyright 2026 The TensorFlow Authors. All Rights Reserved.

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

// Tests DEQUANTIZE on constant tensors quantized with N-D blockwise
// quantization (`BlockwiseQuantization.block_shape`). The models are built as
// flatbuffers so that the whole path, from parsing the quantization parameters
// to the kernel, is exercised.

#include <cstdint>
#include <cstring>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "Eigen/Core"  // from @eigen_archive
#include "flatbuffers/buffer.h"  // from @flatbuffers
#include "flatbuffers/flatbuffer_builder.h"  // from @flatbuffers
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/core/interpreter.h"
#include "tensorflow/lite/core/interpreter_builder.h"
#include "tensorflow/lite/core/kernels/register.h"
#include "tensorflow/lite/core/model_builder.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/test_util.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {
namespace {

using ::testing::ElementsAreArray;

// A tensor holding blockwise quantization parameters.
struct ParamsTensor {
  TensorType type;
  std::vector<int32_t> shape;
  std::vector<uint8_t> data;
};

template <typename T>
ParamsTensor MakeParams(TensorType type, std::vector<int32_t> shape,
                        const std::vector<T>& values) {
  ParamsTensor params{type, std::move(shape), {}};
  params.data.resize(values.size() * sizeof(T));
  std::memcpy(params.data.data(), values.data(), params.data.size());
  return params;
}

ParamsTensor MakeF16Params(std::vector<int32_t> shape,
                           const std::vector<float>& values) {
  std::vector<uint16_t> halves;
  for (float value : values) {
    halves.push_back(Eigen::numext::bit_cast<uint16_t>(Eigen::half(value)));
  }
  return MakeParams(TensorType_FLOAT16, std::move(shape), halves);
}

// Packs signed 4-bit values two per byte, low nibble first.
std::vector<uint8_t> PackInt4(const std::vector<int8_t>& values) {
  std::vector<uint8_t> packed((values.size() + 1) / 2, 0);
  for (size_t i = 0; i < values.size(); ++i) {
    const uint8_t nibble = static_cast<uint8_t>(values[i]) & 0x0F;
    packed[i / 2] |= (i % 2 == 0) ? nibble : static_cast<uint8_t>(nibble << 4);
  }
  return packed;
}

// Builds a model computing DEQUANTIZE on a constant blockwise quantized input
// of type `input_type` and shape `shape`, holding `input_data`.
std::vector<uint8_t> BuildBlockwiseDequantizeModel(
    TensorType input_type, const std::vector<int32_t>& shape,
    const std::vector<uint8_t>& input_data,
    const std::vector<int32_t>& block_shape, const ParamsTensor& scales,
    const std::optional<ParamsTensor>& zero_points) {
  flatbuffers::FlatBufferBuilder builder;
  std::vector<flatbuffers::Offset<Buffer>> buffers = {
      CreateBuffer(builder, builder.CreateVector(std::vector<uint8_t>{}))};
  auto add_buffer = [&](const std::vector<uint8_t>& data) {
    buffers.push_back(CreateBuffer(builder, builder.CreateVector(data)));
    return static_cast<uint32_t>(buffers.size() - 1);
  };

  // Tensors: 0 = quantized input, 1 = scales, 2 = zero points (optional),
  // last = output.
  const int32_t scales_index = 1;
  const int32_t zero_points_index = zero_points.has_value() ? 2 : -1;
  const int32_t output_index = zero_points.has_value() ? 3 : 2;

  auto details = CreateBlockwiseQuantization(
      builder, scales_index, zero_points_index, /*block_size=*/0,
      builder.CreateVector(block_shape));
  auto quantization = CreateQuantizationParameters(
      builder, /*min=*/0, /*max=*/0, /*scale=*/0, /*zero_point=*/0,
      QuantizationDetails_BlockwiseQuantization, details.Union());

  std::vector<flatbuffers::Offset<Tensor>> tensors;
  tensors.push_back(CreateTensor(builder, builder.CreateVector(shape),
                                 input_type, add_buffer(input_data),
                                 builder.CreateString("input"), quantization));
  tensors.push_back(CreateTensor(builder, builder.CreateVector(scales.shape),
                                 scales.type, add_buffer(scales.data),
                                 builder.CreateString("scales")));
  if (zero_points.has_value()) {
    tensors.push_back(CreateTensor(
        builder, builder.CreateVector(zero_points->shape), zero_points->type,
        add_buffer(zero_points->data), builder.CreateString("zero_points")));
  }
  tensors.push_back(CreateTensor(builder, builder.CreateVector(shape),
                                 TensorType_FLOAT32, /*buffer=*/0,
                                 builder.CreateString("output")));

  auto op =
      CreateOperator(builder, /*opcode_index=*/0,
                     builder.CreateVector(std::vector<int32_t>{0}),
                     builder.CreateVector(std::vector<int32_t>{output_index}));
  auto subgraph = CreateSubGraph(
      builder, builder.CreateVector(tensors),
      builder.CreateVector(std::vector<int32_t>{}),
      builder.CreateVector(std::vector<int32_t>{output_index}),
      builder.CreateVector(std::vector<flatbuffers::Offset<Operator>>{op}));
  auto op_code = CreateOperatorCode(
      builder, /*deprecated_builtin_code=*/BuiltinOperator_DEQUANTIZE,
      /*custom_code=*/0, /*version=*/10, BuiltinOperator_DEQUANTIZE);
  auto model =
      CreateModel(builder, /*version=*/3, builder.CreateVector({op_code}),
                  builder.CreateVector({subgraph}),
                  builder.CreateString("test"), builder.CreateVector(buffers));
  FinishModelBuffer(builder, model);
  return std::vector<uint8_t>(builder.GetBufferPointer(),
                              builder.GetBufferPointer() + builder.GetSize());
}

class BlockwiseDequantizeModel {
 public:
  explicit BlockwiseDequantizeModel(std::vector<uint8_t> buffer)
      : buffer_(std::move(buffer)) {
    model_ = FlatBufferModel::BuildFromBuffer(
        reinterpret_cast<const char*>(buffer_.data()), buffer_.size());
    ops::builtin::BuiltinOpResolverWithoutDefaultDelegates resolver;
    InterpreterBuilder(*model_, resolver)(&interpreter_);
  }

  bool Built() const { return interpreter_ != nullptr; }

  TfLiteStatus AllocateAndInvoke() {
    if (TfLiteStatus status = interpreter_->AllocateTensors();
        status != kTfLiteOk) {
      return status;
    }
    return interpreter_->Invoke();
  }

  std::vector<float> GetOutput() const {
    const TfLiteTensor* output =
        interpreter_->tensor(interpreter_->outputs()[0]);
    const float* data = output->data.f;
    return std::vector<float>(data, data + NumElements(output));
  }

 private:
  std::vector<uint8_t> buffer_;
  std::unique_ptr<FlatBufferModel> model_;
  std::unique_ptr<Interpreter> interpreter_;
};

std::vector<uint8_t> AsBytes(const std::vector<int8_t>& values) {
  return std::vector<uint8_t>(values.begin(), values.end());
}

TEST(BlockwiseDequantizeTest, GroupedChannels) {
  // Two adjacent rows share a scale: block_shape = [2, 4] on a [4, 4] tensor.
  BlockwiseDequantizeModel m(BuildBlockwiseDequantizeModel(
      TensorType_INT8, {4, 4},
      AsBytes({1, 2, 3, 4, 5, 6, 7, 8, -1, -2, -3, -4, -5, -6, -7, -8}),
      /*block_shape=*/{2, 4},
      MakeParams<float>(TensorType_FLOAT32, {2, 1}, {0.5f, 0.25f}),
      std::nullopt));
  ASSERT_TRUE(m.Built());
  ASSERT_EQ(m.AllocateAndInvoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput(), ElementsAreArray(ArrayFloatNear(
                                 {0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, -0.25, -0.5,
                                  -0.75, -1, -1.25, -1.5, -1.75, -2})));
}

TEST(BlockwiseDequantizeTest, PerChannelAndBlockedWithBroadcastZeroPoint) {
  // One scale per row and per block of 2 columns, and a zero point that is
  // shared by all rows (extent 1 on axis 0).
  BlockwiseDequantizeModel m(BuildBlockwiseDequantizeModel(
      TensorType_INT8, {2, 4}, AsBytes({1, 2, 3, 4, -1, -2, -3, -4}),
      /*block_shape=*/{1, 2},
      MakeParams<float>(TensorType_FLOAT32, {2, 2}, {1, 2, 3, 4}),
      MakeParams<int32_t>(TensorType_INT32, {1, 2}, {1, -1})));
  ASSERT_TRUE(m.Built());
  ASSERT_EQ(m.AllocateAndInvoke(), kTfLiteOk);
  // Row 0: (1-1)*1, (2-1)*1, (3+1)*2, (4+1)*2.
  // Row 1: (-1-1)*3, (-2-1)*3, (-3+1)*4, (-4+1)*4.
  EXPECT_THAT(m.GetOutput(),
              ElementsAreArray(ArrayFloatNear({0, 1, 8, 10, -6, -9, -8, -12})));
}

TEST(BlockwiseDequantizeTest, Int4) {
  BlockwiseDequantizeModel m(BuildBlockwiseDequantizeModel(
      TensorType_INT4, {2, 4}, PackInt4({1, 2, 3, 4, -1, -2, -3, -4}),
      /*block_shape=*/{1, 2},
      MakeParams<float>(TensorType_FLOAT32, {2, 2}, {1, 2, 3, 4}),
      MakeParams<int32_t>(TensorType_INT32, {1, 2}, {1, -1})));
  ASSERT_TRUE(m.Built());
  ASSERT_EQ(m.AllocateAndInvoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput(),
              ElementsAreArray(ArrayFloatNear({0, 1, 8, 10, -6, -9, -8, -12})));
}

TEST(BlockwiseDequantizeTest, PartialBlocks) {
  // 5 columns in blocks of 2: the last block only has one column.
  BlockwiseDequantizeModel m(BuildBlockwiseDequantizeModel(
      TensorType_INT8, {1, 5}, AsBytes({1, 1, 1, 1, 1}),
      /*block_shape=*/{1, 2},
      MakeParams<float>(TensorType_FLOAT32, {1, 3}, {1, 2, 3}), std::nullopt));
  ASSERT_TRUE(m.Built());
  ASSERT_EQ(m.AllocateAndInvoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput(), ElementsAreArray(ArrayFloatNear({1, 1, 2, 2, 3})));
}

TEST(BlockwiseDequantizeTest, Rank3) {
  // [2, 2, 4] with one scale per (expert, block of 2 columns), shared across
  // the middle axis.
  BlockwiseDequantizeModel m(BuildBlockwiseDequantizeModel(
      TensorType_INT8, {2, 2, 4},
      AsBytes({1, 1, 1, 1, 2, 2, 2, 2, 1, 1, 1, 1, 2, 2, 2, 2}),
      /*block_shape=*/{1, 2, 2},
      MakeParams<float>(TensorType_FLOAT32, {2, 1, 2}, {1, 10, 100, 1000}),
      std::nullopt));
  ASSERT_TRUE(m.Built());
  ASSERT_EQ(m.AllocateAndInvoke(), kTfLiteOk);
  EXPECT_THAT(
      m.GetOutput(),
      ElementsAreArray(ArrayFloatNear({1, 1, 10, 10, 2, 2, 20, 20, 100, 100,
                                       1000, 1000, 200, 200, 2000, 2000})));
}

TEST(BlockwiseDequantizeTest, F16ScalesAndFloatZeroPoint) {
  // A grid centered between the integers: w = (q + 0.5) * scale.
  BlockwiseDequantizeModel m(BuildBlockwiseDequantizeModel(
      TensorType_INT8, {2, 2}, AsBytes({-2, 1, -1, 0}),
      /*block_shape=*/{1, 2}, MakeF16Params({2, 1}, {1.0f, 2.0f}),
      MakeParams<float>(TensorType_FLOAT32, {1, 1}, {-0.5f})));
  ASSERT_TRUE(m.Built());
  ASSERT_EQ(m.AllocateAndInvoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput(),
              ElementsAreArray(ArrayFloatNear({-1.5, 1.5, -1, 1})));
}

TEST(BlockwiseDequantizeTest, RejectsScalesNotShapedLikeTheBlockGrid) {
  // 2 blocks per row, but 3 scales per row.
  BlockwiseDequantizeModel m(BuildBlockwiseDequantizeModel(
      TensorType_INT8, {2, 4}, AsBytes({1, 2, 3, 4, 5, 6, 7, 8}),
      /*block_shape=*/{1, 2},
      MakeParams<float>(TensorType_FLOAT32, {2, 3}, {1, 2, 3, 4, 5, 6}),
      std::nullopt));
  ASSERT_TRUE(m.Built());
  EXPECT_NE(m.AllocateAndInvoke(), kTfLiteOk);
}

}  // namespace
}  // namespace tflite
