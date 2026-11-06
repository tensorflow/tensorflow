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

// Tests EMBEDDING_LOOKUP on constant tables quantized with N-D blockwise
// quantization (`BlockwiseQuantization.block_shape`). The models are built as
// flatbuffers so that the whole path, from parsing the quantization parameters
// to the kernel, is exercised.

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <memory>
#include <optional>
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
#include "tensorflow/lite/kernels/register_ref.h"
#include "tensorflow/lite/kernels/test_util.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {
namespace {

using ::testing::ElementsAreArray;

// A tensor holding blockwise quantization parameters. `values` are the
// parameters as floats, used to compute the expected outputs.
struct ParamsTensor {
  TensorType type;
  std::vector<int32_t> shape;
  std::vector<uint8_t> data;
  std::vector<float> values;
};

template <typename T>
ParamsTensor MakeParams(TensorType type, std::vector<int32_t> shape,
                        const std::vector<T>& values) {
  ParamsTensor params{type, std::move(shape), {}, {}};
  params.data.resize(values.size() * sizeof(T));
  std::memcpy(params.data.data(), values.data(), params.data.size());
  params.values.assign(values.begin(), values.end());
  return params;
}

ParamsTensor MakeF16Params(std::vector<int32_t> shape,
                           const std::vector<float>& values) {
  std::vector<uint16_t> halves;
  std::vector<float> rounded;
  for (float value : values) {
    halves.push_back(Eigen::numext::bit_cast<uint16_t>(Eigen::half(value)));
    rounded.push_back(static_cast<float>(Eigen::half(value)));
  }
  ParamsTensor params =
      MakeParams(TensorType_FLOAT16, std::move(shape), halves);
  params.values = std::move(rounded);
  return params;
}

ParamsTensor MakeBf16Params(std::vector<int32_t> shape,
                            const std::vector<float>& values) {
  std::vector<uint16_t> bf16;
  std::vector<float> rounded;
  for (float value : values) {
    bf16.push_back(Eigen::numext::bit_cast<uint16_t>(Eigen::bfloat16(value)));
    rounded.push_back(static_cast<float>(Eigen::bfloat16(value)));
  }
  ParamsTensor params = MakeParams(TensorType_BFLOAT16, std::move(shape), bf16);
  params.values = std::move(rounded);
  return params;
}

// Packs signed 2-bit values four per byte, least-significant bits first.
std::vector<uint8_t> PackInt2(const std::vector<int32_t>& values) {
  std::vector<uint8_t> packed((values.size() + 3) / 4, 0);
  for (size_t i = 0; i < values.size(); ++i) {
    const uint8_t two_bits = static_cast<uint8_t>(values[i]) & 0x03;
    packed[i / 4] |= static_cast<uint8_t>(two_bits << ((i % 4) * 2));
  }
  return packed;
}

// Packs 4-bit values (signed or unsigned) two per byte, low nibble first.
std::vector<uint8_t> PackNibbles(const std::vector<int32_t>& values) {
  std::vector<uint8_t> packed((values.size() + 1) / 2, 0);
  for (size_t i = 0; i < values.size(); ++i) {
    const uint8_t nibble = static_cast<uint8_t>(values[i]) & 0x0F;
    packed[i / 2] |= (i % 2 == 0) ? nibble : static_cast<uint8_t>(nibble << 4);
  }
  return packed;
}

template <typename T>
std::vector<uint8_t> PackAs(const std::vector<int32_t>& values) {
  std::vector<T> typed(values.begin(), values.end());
  std::vector<uint8_t> bytes(typed.size() * sizeof(T));
  std::memcpy(bytes.data(), typed.data(), bytes.size());
  return bytes;
}

std::vector<uint8_t> Pack(TensorType type, const std::vector<int32_t>& values) {
  switch (type) {
    case TensorType_INT2:
      return PackInt2(values);
    case TensorType_INT4:
    case TensorType_UINT4:
      return PackNibbles(values);
    case TensorType_INT8:
      return PackAs<int8_t>(values);
    case TensorType_UINT8:
      return PackAs<uint8_t>(values);
    case TensorType_INT16:
      return PackAs<int16_t>(values);
    default:
      return {};
  }
}

struct BlockwiseTable {
  TensorType type;
  std::vector<int32_t> shape;
  // Unpacked integer values, in row-major order.
  std::vector<int32_t> values;
  std::vector<int32_t> block_shape;
  ParamsTensor scales;
  std::optional<ParamsTensor> zero_points;
};

// Dequantizes `table` element by element, following the definition of N-D
// blockwise quantization.
std::vector<float> Dequantize(const BlockwiseTable& table) {
  const int rank = table.shape.size();
  auto param_index = [&](const ParamsTensor& params,
                         const std::vector<int>& coords) {
    int64_t index = 0;
    for (int d = 0; d < rank; ++d) {
      const int extent = params.shape[d];
      const int block_coord =
          extent == 1 ? 0 : coords[d] / table.block_shape[d];
      index = index * extent + block_coord;
    }
    return index;
  };
  std::vector<float> result(table.values.size());
  std::vector<int> coords(rank, 0);
  for (size_t i = 0; i < table.values.size(); ++i) {
    int64_t rest = i;
    for (int d = rank - 1; d >= 0; --d) {
      coords[d] = rest % table.shape[d];
      rest /= table.shape[d];
    }
    const float scale = table.scales.values[param_index(table.scales, coords)];
    const float zero_point =
        table.zero_points.has_value()
            ? table.zero_points->values[param_index(*table.zero_points, coords)]
            : 0.0f;
    result[i] = scale * (table.values[i] - zero_point);
  }
  return result;
}

// Returns the rows `lookup` of the dequantized `table`.
std::vector<float> ExpectedLookup(const BlockwiseTable& table,
                                  const std::vector<int32_t>& lookup) {
  const std::vector<float> dequantized = Dequantize(table);
  const size_t row_size = dequantized.size() / table.shape[0];
  std::vector<float> result;
  for (int32_t idx : lookup) {
    result.insert(result.end(), dequantized.begin() + idx * row_size,
                  dequantized.begin() + (idx + 1) * row_size);
  }
  return result;
}

// Builds a model computing EMBEDDING_LOOKUP of `num_lookups` ids (the model
// input) in the constant blockwise quantized `table`, with an `output_type`
// output.
std::vector<uint8_t> BuildBlockwiseEmbeddingLookupModel(
    const BlockwiseTable& table, int num_lookups, TensorType output_type) {
  flatbuffers::FlatBufferBuilder builder;
  std::vector<flatbuffers::Offset<Buffer>> buffers = {
      CreateBuffer(builder, builder.CreateVector(std::vector<uint8_t>{}))};
  auto add_buffer = [&](const std::vector<uint8_t>& data) {
    buffers.push_back(CreateBuffer(builder, builder.CreateVector(data)));
    return static_cast<uint32_t>(buffers.size() - 1);
  };

  // Tensors: 0 = lookup, 1 = table, 2 = scales, 3 = zero points (optional),
  // last = output.
  const int32_t scales_index = 2;
  const int32_t zero_points_index = table.zero_points.has_value() ? 3 : -1;
  const int32_t output_index = table.zero_points.has_value() ? 4 : 3;

  auto details = CreateBlockwiseQuantization(
      builder, scales_index, zero_points_index, /*block_size=*/0,
      builder.CreateVector(table.block_shape));
  auto quantization = CreateQuantizationParameters(
      builder, /*min=*/0, /*max=*/0, /*scale=*/0, /*zero_point=*/0,
      QuantizationDetails_BlockwiseQuantization, details.Union());

  std::vector<flatbuffers::Offset<Tensor>> tensors;
  tensors.push_back(CreateTensor(
      builder, builder.CreateVector(std::vector<int32_t>{num_lookups}),
      TensorType_INT32, /*buffer=*/0, builder.CreateString("lookup")));
  tensors.push_back(CreateTensor(builder, builder.CreateVector(table.shape),
                                 table.type,
                                 add_buffer(Pack(table.type, table.values)),
                                 builder.CreateString("table"), quantization));
  tensors.push_back(CreateTensor(
      builder, builder.CreateVector(table.scales.shape), table.scales.type,
      add_buffer(table.scales.data), builder.CreateString("scales")));
  if (table.zero_points.has_value()) {
    tensors.push_back(CreateTensor(
        builder, builder.CreateVector(table.zero_points->shape),
        table.zero_points->type, add_buffer(table.zero_points->data),
        builder.CreateString("zero_points")));
  }
  std::vector<int32_t> output_shape = table.shape;
  output_shape[0] = num_lookups;
  tensors.push_back(CreateTensor(builder, builder.CreateVector(output_shape),
                                 output_type, /*buffer=*/0,
                                 builder.CreateString("output")));

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

enum class Resolver { kBuiltin, kReference };

class BlockwiseEmbeddingLookupModel {
 public:
  BlockwiseEmbeddingLookupModel(std::vector<uint8_t> buffer, Resolver resolver)
      : buffer_(std::move(buffer)) {
    model_ = FlatBufferModel::BuildFromBuffer(
        reinterpret_cast<const char*>(buffer_.data()), buffer_.size());
    if (resolver == Resolver::kBuiltin) {
      ops::builtin::BuiltinOpResolverWithoutDefaultDelegates op_resolver;
      InterpreterBuilder(*model_, op_resolver)(&interpreter_);
    } else {
      ops::builtin::BuiltinRefOpResolver op_resolver;
      InterpreterBuilder(*model_, op_resolver)(&interpreter_);
    }
  }

  bool Built() const { return interpreter_ != nullptr; }

  TfLiteStatus Run(const std::vector<int32_t>& lookup) {
    if (TfLiteStatus status = interpreter_->AllocateTensors();
        status != kTfLiteOk) {
      return status;
    }
    TfLiteTensor* input = interpreter_->tensor(interpreter_->inputs()[0]);
    std::copy(lookup.begin(), lookup.end(), input->data.i32);
    return interpreter_->Invoke();
  }

  std::vector<float> GetOutput() const {
    const TfLiteTensor* output =
        interpreter_->tensor(interpreter_->outputs()[0]);
    std::vector<float> values(NumElements(output));
    for (size_t i = 0; i < values.size(); ++i) {
      values[i] = output->type == kTfLiteFloat16
                      ? static_cast<float>(Eigen::numext::bit_cast<Eigen::half>(
                            output->data.f16[i].data))
                      : output->data.f[i];
    }
    return values;
  }

 private:
  std::vector<uint8_t> buffer_;
  std::unique_ptr<FlatBufferModel> model_;
  std::unique_ptr<Interpreter> interpreter_;
};

// Deterministic integer values in [lo, hi].
std::vector<int32_t> Pattern(int size, int lo, int hi) {
  std::vector<int32_t> values(size);
  const int range = hi - lo + 1;
  for (int i = 0; i < size; ++i) {
    values[i] = lo + (i * 7 + (i / 3) * 5) % range;
  }
  return values;
}

// Runs the lookup with both resolvers and both output types and compares it
// against the dequantized rows of `table`.
void ExpectLookupMatchesDequantizedRows(const BlockwiseTable& table,
                                        const std::vector<int32_t>& lookup) {
  const std::vector<float> expected = ExpectedLookup(table, lookup);
  for (Resolver resolver : {Resolver::kBuiltin, Resolver::kReference}) {
    for (TensorType output_type : {TensorType_FLOAT32, TensorType_FLOAT16}) {
      SCOPED_TRACE(testing::Message()
                   << "resolver=" << static_cast<int>(resolver)
                   << " output_type=" << EnumNameTensorType(output_type));
      BlockwiseEmbeddingLookupModel m(
          BuildBlockwiseEmbeddingLookupModel(
              table, static_cast<int>(lookup.size()), output_type),
          resolver);
      ASSERT_TRUE(m.Built());
      ASSERT_EQ(m.Run(lookup), kTfLiteOk);
      // FLOAT16 outputs keep 11 significant bits.
      const bool is_f16 = output_type == TensorType_FLOAT16;
      EXPECT_THAT(m.GetOutput(),
                  ElementsAreArray(ArrayFloatNear(
                      expected, /*max_abs_err=*/is_f16 ? 1e-3f : 1e-5f,
                      kFpErrorAuto, /*max_rel_err=*/is_f16 ? 1e-3f : 0.0f)));
    }
  }
}

TEST(BlockwiseEmbeddingLookupTest, CenteredInt2PerRow) {
  // The centered int2 grid w = scale * (q + 0.5), q in [-2, 1], with one scale
  // per row: values in scale * {-1.5, -0.5, 0.5, 1.5}.
  const BlockwiseTable table = {
      .type = TensorType_INT2,
      .shape = {4, 8},
      .values = Pattern(32, -2, 1),
      .block_shape = {1, 8},
      .scales = MakeF16Params({4, 1}, {0.5f, 1.0f, 1.5f, 0.25f}),
      .zero_points = MakeParams<float>(TensorType_FLOAT32, {1, 1}, {-0.5f}),
  };
  ExpectLookupMatchesDequantizedRows(table, {2, 0, 3, 2});

  // Spot check the first looked up row (row 2, scale 1.5) by hand.
  BlockwiseEmbeddingLookupModel m(
      BuildBlockwiseEmbeddingLookupModel(table, 1, TensorType_FLOAT32),
      Resolver::kBuiltin);
  ASSERT_TRUE(m.Built());
  ASSERT_EQ(m.Run({2}), kTfLiteOk);
  std::vector<float> row_2;
  for (int j = 0; j < 8; ++j) {
    row_2.push_back(1.5f * (table.values[16 + j] + 0.5f));
  }
  EXPECT_THAT(m.GetOutput(), ElementsAreArray(ArrayFloatNear(row_2)));
}

TEST(BlockwiseEmbeddingLookupTest, Int2RowsStraddlingBytes) {
  // Rows of 5 int2 values do not start on byte boundaries.
  ExpectLookupMatchesDequantizedRows(
      {
          .type = TensorType_INT2,
          .shape = {5, 5},
          .values = Pattern(25, -2, 1),
          .block_shape = {1, 5},
          .scales = MakeParams<float>(TensorType_FLOAT32, {5, 1},
                                      {0.5f, 1.0f, 2.0f, 4.0f, 8.0f}),
          .zero_points = MakeParams<float>(TensorType_FLOAT32, {1, 1}, {-0.5f}),
      },
      {4, 1, 3, 0, 2});
}

TEST(BlockwiseEmbeddingLookupTest, Int4SubChannelPartialBlocks) {
  // 5 columns in blocks of 2 (the last block has one column), with one zero
  // point per block.
  ExpectLookupMatchesDequantizedRows(
      {
          .type = TensorType_INT4,
          .shape = {3, 5},
          .values = Pattern(15, -8, 7),
          .block_shape = {1, 2},
          .scales = MakeParams<float>(
              TensorType_FLOAT32, {3, 3},
              {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.9f}),
          .zero_points = MakeParams<int32_t>(TensorType_INT32, {3, 3},
                                             {0, 1, -1, 2, -2, 3, -3, 4, -4}),
      },
      {1, 2, 0});
}

TEST(BlockwiseEmbeddingLookupTest, GroupedRowsWithoutZeroPoint) {
  // Two adjacent rows share a scale.
  ExpectLookupMatchesDequantizedRows(
      {
          .type = TensorType_INT8,
          .shape = {4, 4},
          .values = Pattern(16, -128, 127),
          .block_shape = {2, 4},
          .scales =
              MakeParams<float>(TensorType_FLOAT32, {2, 1}, {0.5f, 0.25f}),
          .zero_points = std::nullopt,
      },
      {3, 0, 1, 2});
}

TEST(BlockwiseEmbeddingLookupTest, Rank3TableWithBroadcastParams) {
  // [3, 2, 4] with one scale per (row, block of 2 columns), shared across the
  // middle axis, and a zero point shared by all rows.
  ExpectLookupMatchesDequantizedRows(
      {
          .type = TensorType_INT8,
          .shape = {3, 2, 4},
          .values = Pattern(24, -20, 20),
          .block_shape = {1, 1, 2},
          .scales = MakeParams<float>(TensorType_FLOAT32, {3, 1, 2},
                                      {1.0f, 10.0f, 0.5f, 5.0f, 0.25f, 2.5f}),
          .zero_points =
              MakeParams<int8_t>(TensorType_INT8, {1, 1, 2}, {1, -1}),
      },
      {2, 0, 1});
}

TEST(BlockwiseEmbeddingLookupTest, Rank3GroupedRowsAndPartialColumnBlocks) {
  // [4, 2, 6] with pairs of rows sharing parameters, one parameter per middle
  // index, and blocks of 4 columns (the last block is partial), with BF16
  // scales and INT16 zero points that broadcast along the middle axis.
  ExpectLookupMatchesDequantizedRows(
      {
          .type = TensorType_INT4,
          .shape = {4, 2, 6},
          .values = Pattern(48, -8, 7),
          .block_shape = {2, 1, 4},
          .scales = MakeBf16Params(
              {2, 2, 2}, {0.5f, 1.0f, 0.25f, 2.0f, 4.0f, 0.125f, 1.5f, 0.75f}),
          .zero_points =
              MakeParams<int16_t>(TensorType_INT16, {2, 1, 2}, {1, -2, 0, 3}),
      },
      {3, 0, 2, 1});
}

TEST(BlockwiseEmbeddingLookupTest, OtherStorageTypes) {
  struct Case {
    TensorType type;
    int lo;
    int hi;
  };
  for (const Case& c :
       {Case{TensorType_UINT4, 0, 15}, Case{TensorType_UINT8, 0, 255},
        Case{TensorType_INT16, -1000, 1000}}) {
    SCOPED_TRACE(EnumNameTensorType(c.type));
    ExpectLookupMatchesDequantizedRows(
        {
            .type = c.type,
            .shape = {3, 6},
            .values = Pattern(18, c.lo, c.hi),
            .block_shape = {1, 3},
            .scales = MakeF16Params({3, 2}, {0.5f, 1, 2, 0.25f, 4, 0.125f}),
            .zero_points = MakeParams<float>(TensorType_FLOAT32, {1, 1},
                                             {(c.lo + c.hi) / 2.0f}),
        },
        {0, 2, 1});
  }
}

TEST(BlockwiseEmbeddingLookupTest, EmptyLookup) {
  BlockwiseEmbeddingLookupModel m(
      BuildBlockwiseEmbeddingLookupModel(
          {
              .type = TensorType_INT2,
              .shape = {2, 4},
              .values = Pattern(8, -2, 1),
              .block_shape = {1, 4},
              .scales = MakeParams<float>(TensorType_FLOAT32, {2, 1}, {1, 2}),
              .zero_points = std::nullopt,
          },
          /*num_lookups=*/0, TensorType_FLOAT32),
      Resolver::kBuiltin);
  ASSERT_TRUE(m.Built());
  ASSERT_EQ(m.Run({}), kTfLiteOk);
  EXPECT_TRUE(m.GetOutput().empty());
}

TEST(BlockwiseEmbeddingLookupTest, RejectsOutOfBoundsIndex) {
  BlockwiseEmbeddingLookupModel m(
      BuildBlockwiseEmbeddingLookupModel(
          {
              .type = TensorType_INT2,
              .shape = {2, 4},
              .values = Pattern(8, -2, 1),
              .block_shape = {1, 4},
              .scales = MakeParams<float>(TensorType_FLOAT32, {2, 1}, {1, 2}),
              .zero_points = std::nullopt,
          },
          /*num_lookups=*/1, TensorType_FLOAT32),
      Resolver::kBuiltin);
  ASSERT_TRUE(m.Built());
  EXPECT_NE(m.Run({2}), kTfLiteOk);
}

TEST(BlockwiseEmbeddingLookupTest, RejectsScalesNotShapedLikeTheBlockGrid) {
  // 2 blocks per row, but 3 scales per row.
  BlockwiseEmbeddingLookupModel m(
      BuildBlockwiseEmbeddingLookupModel(
          {
              .type = TensorType_INT8,
              .shape = {2, 4},
              .values = Pattern(8, -8, 7),
              .block_shape = {1, 2},
              .scales = MakeParams<float>(TensorType_FLOAT32, {2, 3},
                                          {1, 2, 3, 4, 5, 6}),
              .zero_points = std::nullopt,
          },
          /*num_lookups=*/1, TensorType_FLOAT32),
      Resolver::kBuiltin);
  ASSERT_TRUE(m.Built());
  EXPECT_NE(m.Run({0}), kTfLiteOk);
}

TEST(BlockwiseEmbeddingLookupTest, RejectsIntegerOutput) {
  BlockwiseEmbeddingLookupModel m(
      BuildBlockwiseEmbeddingLookupModel(
          {
              .type = TensorType_INT8,
              .shape = {2, 4},
              .values = Pattern(8, -8, 7),
              .block_shape = {1, 4},
              .scales = MakeParams<float>(TensorType_FLOAT32, {2, 1}, {1, 2}),
              .zero_points = std::nullopt,
          },
          /*num_lookups=*/1, TensorType_INT8),
      Resolver::kBuiltin);
  ASSERT_TRUE(m.Built());
  EXPECT_NE(m.Run({0}), kTfLiteOk);
}

}  // namespace
}  // namespace tflite
