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

// Tests EMBEDDING_LOOKUP on tables quantized with N-D blockwise quantization
// (`BlockwiseQuantization.block_shape`), including the centered int2 grid
// (fractional zero point -0.5). The models are built as flatbuffers so that
// the whole path, from parsing the quantization parameters to the kernel, is
// exercised.

#include <cstdint>
#include <cstring>
#include <memory>
#include <optional>
#include <tuple>
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
#include "tensorflow/lite/kernels/register_ref.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {
namespace {

template <typename T>
std::vector<uint8_t> ToBytes(const std::vector<T>& values) {
  std::vector<uint8_t> bytes(values.size() * sizeof(T));
  if (!bytes.empty()) std::memcpy(bytes.data(), values.data(), bytes.size());
  return bytes;
}

// Packs signed 2-bit values four per byte, least-significant bits first.
std::vector<uint8_t> PackInt2(const std::vector<int32_t>& values) {
  std::vector<uint8_t> packed((values.size() + 3) / 4, 0);
  for (size_t i = 0; i < values.size(); ++i) {
    packed[i / 4] |= static_cast<uint8_t>(
        (static_cast<uint8_t>(values[i]) & 0x03) << ((i % 4) * 2));
  }
  return packed;
}

// Packs signed 4-bit values two per byte, low nibble first.
std::vector<uint8_t> PackInt4(const std::vector<int32_t>& values) {
  std::vector<uint8_t> packed((values.size() + 1) / 2, 0);
  for (size_t i = 0; i < values.size(); ++i) {
    const uint8_t nibble = static_cast<uint8_t>(values[i]) & 0x0F;
    packed[i / 2] |= (i % 2 == 0) ? nibble : static_cast<uint8_t>(nibble << 4);
  }
  return packed;
}

struct TableSpec {
  TensorType type;
  int rows;
  int cols;
  std::vector<int32_t> values;
  std::vector<int32_t> block_shape;
  std::vector<int32_t> scale_shape;
  std::vector<float> scales;
  std::optional<float> zero_point;  // One zero point, broadcast.
};

std::vector<uint8_t> BuildModel(const TableSpec& table, int num_lookups,
                                TensorType output_type) {
  flatbuffers::FlatBufferBuilder builder;
  std::vector<flatbuffers::Offset<Buffer>> buffers = {
      CreateBuffer(builder, builder.CreateVector(std::vector<uint8_t>{}))};
  auto add_buffer = [&](const std::vector<uint8_t>& data) {
    buffers.push_back(CreateBuffer(builder, builder.CreateVector(data)));
    return static_cast<uint32_t>(buffers.size() - 1);
  };
  std::vector<uint8_t> table_bytes;
  if (table.type == TensorType_INT2) {
    table_bytes = PackInt2(table.values);
  } else if (table.type == TensorType_INT4) {
    table_bytes = PackInt4(table.values);
  } else {
    table_bytes =
        ToBytes(std::vector<int8_t>(table.values.begin(), table.values.end()));
  }
  // Tensors: 0 = lookup, 1 = table, 2 = scales, [3 = zero point], output.
  const int32_t zero_point_index = table.zero_point.has_value() ? 3 : -1;
  const int32_t output_index = table.zero_point.has_value() ? 4 : 3;
  auto details = CreateBlockwiseQuantization(
      builder, /*scales=*/2, zero_point_index, /*block_size=*/0,
      builder.CreateVector(table.block_shape));
  auto quantization = CreateQuantizationParameters(
      builder, /*min=*/0, /*max=*/0, /*scale=*/0, /*zero_point=*/0,
      QuantizationDetails_BlockwiseQuantization, details.Union());
  std::vector<flatbuffers::Offset<Tensor>> tensors;
  tensors.push_back(CreateTensor(
      builder, builder.CreateVector(std::vector<int32_t>{num_lookups}),
      TensorType_INT32, /*buffer=*/0, builder.CreateString("lookup")));
  tensors.push_back(CreateTensor(
      builder,
      builder.CreateVector(std::vector<int32_t>{table.rows, table.cols}),
      table.type, add_buffer(table_bytes), builder.CreateString("table"),
      quantization));
  tensors.push_back(CreateTensor(
      builder, builder.CreateVector(table.scale_shape), TensorType_FLOAT32,
      add_buffer(ToBytes(table.scales)), builder.CreateString("scales")));
  if (table.zero_point.has_value()) {
    tensors.push_back(
        CreateTensor(builder, builder.CreateVector(std::vector<int32_t>{1, 1}),
                     TensorType_FLOAT32,
                     add_buffer(ToBytes(std::vector<float>{*table.zero_point})),
                     builder.CreateString("zero_points")));
  }
  tensors.push_back(CreateTensor(
      builder,
      builder.CreateVector(std::vector<int32_t>{num_lookups, table.cols}),
      output_type, /*buffer=*/0, builder.CreateString("output")));
  auto op =
      CreateOperator(builder, /*opcode_index=*/0,
                     builder.CreateVector(std::vector<int32_t>{0, 1}),
                     builder.CreateVector(std::vector<int32_t>{output_index}));
  auto subgraph = CreateSubGraph(
      builder, builder.CreateVector(tensors),
      builder.CreateVector(std::vector<int32_t>{0}),
      builder.CreateVector(std::vector<int32_t>{output_index}),
      builder.CreateVector(std::vector<flatbuffers::Offset<Operator>>{op}));
  auto op_code = CreateOperatorCode(
      builder, /*deprecated_builtin_code=*/BuiltinOperator_EMBEDDING_LOOKUP,
      /*custom_code=*/0, /*version=*/6, BuiltinOperator_EMBEDDING_LOOKUP);
  auto model =
      CreateModel(builder, /*version=*/3, builder.CreateVector({op_code}),
                  builder.CreateVector({subgraph}),
                  builder.CreateString("test"), builder.CreateVector(buffers));
  FinishModelBuffer(builder, model);
  return std::vector<uint8_t>(builder.GetBufferPointer(),
                              builder.GetBufferPointer() + builder.GetSize());
}

// Runs the lookup; returns the outputs as floats, or nullopt if the model
// fails to allocate.
std::optional<std::vector<float>> RunLookup(const TableSpec& table,
                                            const std::vector<int32_t>& lookup,
                                            TensorType output_type,
                                            bool reference) {
  const std::vector<uint8_t> buffer =
      BuildModel(table, lookup.size(), output_type);
  std::unique_ptr<FlatBufferModel> model = FlatBufferModel::BuildFromBuffer(
      reinterpret_cast<const char*>(buffer.data()), buffer.size());
  std::unique_ptr<Interpreter> interpreter;
  if (reference) {
    ops::builtin::BuiltinRefOpResolver resolver;
    InterpreterBuilder(*model, resolver)(&interpreter);
  } else {
    ops::builtin::BuiltinOpResolver resolver;
    InterpreterBuilder(*model, resolver)(&interpreter);
  }
  if (interpreter == nullptr || interpreter->AllocateTensors() != kTfLiteOk) {
    return std::nullopt;
  }
  std::copy(lookup.begin(), lookup.end(),
            interpreter->typed_input_tensor<int32_t>(0));
  if (interpreter->Invoke() != kTfLiteOk) return std::nullopt;
  const TfLiteTensor* out = interpreter->tensor(interpreter->outputs()[0]);
  std::vector<float> values(lookup.size() * table.cols);
  for (size_t i = 0; i < values.size(); ++i) {
    values[i] = output_type == TensorType_FLOAT16
                    ? static_cast<float>(Eigen::numext::bit_cast<Eigen::half>(
                          out->data.f16[i].data))
                    : out->data.f[i];
  }
  return values;
}

// What the table dequantizes to at `(row, col)`.
float Expected(const TableSpec& table, int row, int col) {
  const int rb = row / table.block_shape[0];
  const int cb = col / table.block_shape[1];
  const int scale_rows = table.scale_shape[0];
  const int scale_cols = table.scale_shape[1];
  const float scale = table.scales[(scale_rows == 1 ? 0 : rb) * scale_cols +
                                   (scale_cols == 1 ? 0 : cb)];
  return scale * (static_cast<float>(table.values[row * table.cols + col]) -
                  table.zero_point.value_or(0.0f));
}

class EmbeddingLookupNdBlockwiseTest
    : public ::testing::TestWithParam<std::tuple<bool, TensorType>> {};

TEST_P(EmbeddingLookupNdBlockwiseTest, CenteredInt2OneBlockPerRow) {
  const auto [reference, output_type] = GetParam();
  TableSpec table{TensorType_INT2, /*rows=*/5, /*cols=*/12};
  for (int i = 0; i < table.rows * table.cols; ++i) {
    table.values.push_back((i * 7 + i / 5) % 4 - 2);  // [-2, 1]
  }
  table.block_shape = {1, table.cols};
  table.scale_shape = {table.rows, 1};
  table.scales = {0.5f, 0.25f, 0.125f, 1.0f, 2.0f};
  table.zero_point = -0.5f;
  const std::vector<int32_t> lookup = {3, 0, 4, 3};
  std::optional<std::vector<float>> got =
      RunLookup(table, lookup, output_type, reference);
  ASSERT_TRUE(got.has_value());
  for (size_t i = 0; i < lookup.size(); ++i) {
    for (int c = 0; c < table.cols; ++c) {
      // The centered grid's values are exact in float16 too.
      EXPECT_EQ((*got)[i * table.cols + c], Expected(table, lookup[i], c))
          << "lookup " << i << " col " << c;
    }
  }
}

TEST_P(EmbeddingLookupNdBlockwiseTest, Int4BlocksAlongColumns) {
  const auto [reference, output_type] = GetParam();
  TableSpec table{TensorType_INT4, /*rows=*/4, /*cols=*/8};
  for (int i = 0; i < table.rows * table.cols; ++i) {
    table.values.push_back((i * 5 + 3) % 16 - 8);  // [-8, 7]
  }
  // Two rows and four columns per block; no zero point.
  table.block_shape = {2, 4};
  table.scale_shape = {2, 2};
  table.scales = {0.5f, 0.25f, 1.0f, 0.125f};
  const std::vector<int32_t> lookup = {1, 2};
  std::optional<std::vector<float>> got =
      RunLookup(table, lookup, output_type, reference);
  ASSERT_TRUE(got.has_value());
  for (size_t i = 0; i < lookup.size(); ++i) {
    for (int c = 0; c < table.cols; ++c) {
      EXPECT_EQ((*got)[i * table.cols + c], Expected(table, lookup[i], c));
    }
  }
}

TEST_P(EmbeddingLookupNdBlockwiseTest, OutOfBoundsLookupFails) {
  const auto [reference, output_type] = GetParam();
  TableSpec table{TensorType_INT2,
                  /*rows=*/2,
                  /*cols=*/4,
                  /*values=*/{0, 1, -1, -2, 1, 0, -2, -1},
                  /*block_shape=*/{1, 4},
                  /*scale_shape=*/{2, 1},
                  /*scales=*/{1.0f, 2.0f},
                  /*zero_point=*/-0.5f};
  EXPECT_FALSE(RunLookup(table, {2}, output_type, reference).has_value());
}

INSTANTIATE_TEST_SUITE_P(
    EmbeddingLookupNdBlockwiseTest, EmbeddingLookupNdBlockwiseTest,
    ::testing::Combine(::testing::Bool(),
                       ::testing::Values(TensorType_FLOAT32,
                                         TensorType_FLOAT16)));

}  // namespace
}  // namespace tflite
