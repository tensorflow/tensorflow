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

// Tests FULLY_CONNECTED with an N-D blockwise quantized filter
// (`BlockwiseQuantization` with `block_shape`), both with an int8 per-tensor
// quantized input and output (static range) and with a float32 or float16 input
// and output (dynamic range). Models are built as flatbuffers to exercise
// flatbuffer parsing, delegate fallback, and both kernel variants end to end.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
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
#include "flatbuffers/flexbuffers.h"  // from @flatbuffers
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/core/interpreter.h"
#include "tensorflow/lite/core/interpreter_builder.h"
#include "tensorflow/lite/core/kernels/register.h"
#include "tensorflow/lite/core/model_builder.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/register_ref.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {
namespace {

using ::testing::ElementsAreArray;
using ::testing::FloatEq;
using ::testing::FloatNear;
using ::testing::Pointwise;

struct ParamsTensor {
  TensorType type;
  std::vector<int32_t> shape;
  std::vector<uint8_t> data;
};

template <typename T>
std::vector<uint8_t> ToBytes(const std::vector<T>& values) {
  // memcpy with null pointers is undefined even for a size of 0.
  if (values.empty()) return {};
  std::vector<uint8_t> bytes(values.size() * sizeof(T));
  std::memcpy(bytes.data(), values.data(), bytes.size());
  return bytes;
}

template <typename T>
ParamsTensor MakeParams(TensorType type, std::vector<int32_t> shape,
                        const std::vector<T>& values) {
  return ParamsTensor{type, std::move(shape), ToBytes(values)};
}

ParamsTensor MakeF16Params(std::vector<int32_t> shape,
                           const std::vector<float>& values) {
  std::vector<uint16_t> halves;
  halves.reserve(values.size());
  for (float value : values) {
    halves.push_back(Eigen::numext::bit_cast<uint16_t>(Eigen::half(value)));
  }
  return MakeParams(TensorType_FLOAT16, std::move(shape), halves);
}

ParamsTensor MakeBF16Params(std::vector<int32_t> shape,
                            const std::vector<float>& values) {
  std::vector<uint16_t> bf16;
  bf16.reserve(values.size());
  for (float value : values) {
    bf16.push_back(Eigen::numext::bit_cast<uint16_t>(Eigen::bfloat16(value)));
  }
  return MakeParams(TensorType_BFLOAT16, std::move(shape), bf16);
}

// Packs signed 2-bit values [-2, 1] four per byte, least-significant bits
// first.
std::vector<uint8_t> PackInt2(const std::vector<int32_t>& values) {
  std::vector<uint8_t> packed((values.size() + 3) / 4, 0);
  for (size_t i = 0; i < values.size(); ++i) {
    const uint8_t two_bits = static_cast<uint8_t>(values[i]) & 0x03;
    packed[i / 4] |= static_cast<uint8_t>(two_bits << ((i % 4) * 2));
  }
  return packed;
}

// Packs 4-bit values (signed [-8, 7] or unsigned [0, 15]) two per byte, low
// nibble first.
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
  return ToBytes(std::vector<T>(values.begin(), values.end()));
}

// Deterministic integer values in [lo, hi].
std::vector<int32_t> Pattern(int size, int lo, int hi, int seed) {
  std::vector<int32_t> values(size);
  const int range = hi - lo + 1;
  for (int i = 0; i < size; ++i) {
    values[i] = lo + (i * 7 + seed * 13 + (i / 3) * 5) % range;
  }
  return values;
}

struct AffineParams {
  float scale = 1.0f;
  int64_t zero_point = 0;
};

struct BiasSpec {
  TensorType type = TensorType_INT32;
  std::vector<uint8_t> data;
  std::vector<float> scales;
  std::vector<int64_t> zero_points;
};

struct BlockwiseFcSpec {
  TensorType input_type = TensorType_INT8;
  std::vector<int32_t> input_shape;
  AffineParams input_quant;
  TensorType filter_type = TensorType_INT2;
  std::vector<int32_t> filter_shape;
  std::vector<uint8_t> filter_data;
  std::vector<int32_t> block_shape;
  ParamsTensor scales;
  bool scales_as_rw_tensor = false;
  std::optional<ParamsTensor> zero_points;
  std::optional<BiasSpec> bias;
  // Passes a third, omitted (-1) bias input when there is no bias.
  bool omitted_bias_input = false;
  std::vector<int32_t> output_shape;
  TensorType output_type = TensorType_INT8;
  AffineParams output_quant;
  ActivationFunctionType activation = ActivationFunctionType_NONE;
  bool keep_num_dims = false;
  bool asymmetric_quantize_inputs = false;
  std::vector<uint8_t> quant_spec;
  int op_version = 16;
};

std::vector<uint8_t> BuildBlockwiseFullyConnectedModel(
    const BlockwiseFcSpec& spec) {
  flatbuffers::FlatBufferBuilder builder;
  std::vector<flatbuffers::Offset<Buffer>> buffers = {
      CreateBuffer(builder, builder.CreateVector(std::vector<uint8_t>{}))};
  auto add_buffer = [&](const std::vector<uint8_t>& data) {
    buffers.push_back(CreateBuffer(builder, builder.CreateVector(data)));
    return static_cast<uint32_t>(buffers.size() - 1);
  };
  // Float tensors (and a bias without scales) carry no quantization.
  auto make_affine_quant = [&](const std::vector<float>& scales,
                               const std::vector<int64_t>& zero_points)
      -> flatbuffers::Offset<QuantizationParameters> {
    if (scales.empty()) return 0;
    return CreateQuantizationParameters(builder, /*min=*/0, /*max=*/0,
                                        builder.CreateVector(scales),
                                        builder.CreateVector(zero_points));
  };
  auto make_activation_quant = [&](TensorType type, AffineParams quant) {
    return (type == TensorType_FLOAT32 || type == TensorType_FLOAT16)
               ? make_affine_quant({}, {})
               : make_affine_quant({quant.scale}, {quant.zero_point});
  };

  // Tensors: input, filter, scales, [zero points], [bias], output.
  std::vector<flatbuffers::Offset<Tensor>> tensors;
  tensors.push_back(CreateTensor(
      builder, builder.CreateVector(spec.input_shape), spec.input_type,
      /*buffer=*/0, builder.CreateString("input"),
      make_activation_quant(spec.input_type, spec.input_quant)));
  const int32_t scales_index = 2;
  const int32_t zero_points_index = spec.zero_points.has_value() ? 3 : -1;
  auto filter_details = CreateBlockwiseQuantization(
      builder, scales_index, zero_points_index, /*block_size=*/0,
      builder.CreateVector(spec.block_shape));
  auto filter_quant = CreateQuantizationParameters(
      builder, /*min=*/0, /*max=*/0, /*scale=*/0, /*zero_point=*/0,
      QuantizationDetails_BlockwiseQuantization, filter_details.Union());
  tensors.push_back(CreateTensor(builder,
                                 builder.CreateVector(spec.filter_shape),
                                 spec.filter_type, add_buffer(spec.filter_data),
                                 builder.CreateString("filter"), filter_quant));
  tensors.push_back(CreateTensor(
      builder, builder.CreateVector(spec.scales.shape), spec.scales.type,
      spec.scales_as_rw_tensor ? 0u : add_buffer(spec.scales.data),
      builder.CreateString("filter_scales")));
  if (spec.zero_points.has_value()) {
    tensors.push_back(
        CreateTensor(builder, builder.CreateVector(spec.zero_points->shape),
                     spec.zero_points->type, add_buffer(spec.zero_points->data),
                     builder.CreateString("filter_zero_points")));
  }
  std::vector<int32_t> op_inputs = {0, 1};
  if (spec.bias.has_value()) {
    op_inputs.push_back(static_cast<int32_t>(tensors.size()));
    tensors.push_back(CreateTensor(
        builder,
        builder.CreateVector(std::vector<int32_t>{
            spec.filter_shape.empty() ? 0 : spec.filter_shape[0]}),
        spec.bias->type, add_buffer(spec.bias->data),
        builder.CreateString("bias"),
        make_affine_quant(spec.bias->scales, spec.bias->zero_points)));
  } else if (spec.omitted_bias_input) {
    op_inputs.push_back(-1);
  }
  const int32_t output_index = static_cast<int32_t>(tensors.size());
  tensors.push_back(CreateTensor(
      builder, builder.CreateVector(spec.output_shape), spec.output_type,
      /*buffer=*/0, builder.CreateString("output"),
      make_activation_quant(spec.output_type, spec.output_quant)));

  auto fc_options = CreateFullyConnectedOptions(
      builder, spec.activation, FullyConnectedOptionsWeightsFormat_DEFAULT,
      spec.keep_num_dims, spec.asymmetric_quantize_inputs, TensorType_FLOAT32,
      spec.quant_spec.empty() ? 0 : builder.CreateVector(spec.quant_spec));
  auto op = CreateOperator(
      builder, /*opcode_index=*/0, builder.CreateVector(op_inputs),
      builder.CreateVector(std::vector<int32_t>{output_index}),
      BuiltinOptions_FullyConnectedOptions, fc_options.Union());
  auto subgraph = CreateSubGraph(
      builder, builder.CreateVector(tensors),
      builder.CreateVector(std::vector<int32_t>{0}),
      builder.CreateVector(std::vector<int32_t>{output_index}),
      builder.CreateVector(std::vector<flatbuffers::Offset<Operator>>{op}));
  auto op_code = CreateOperatorCode(
      builder, /*deprecated_builtin_code=*/BuiltinOperator_FULLY_CONNECTED,
      /*custom_code=*/0, spec.op_version, BuiltinOperator_FULLY_CONNECTED);
  auto model =
      CreateModel(builder, /*version=*/3, builder.CreateVector({op_code}),
                  builder.CreateVector({subgraph}),
                  builder.CreateString("test"), builder.CreateVector(buffers));
  FinishModelBuffer(builder, model);
  return std::vector<uint8_t>(builder.GetBufferPointer(),
                              builder.GetBufferPointer() + builder.GetSize());
}

enum class Resolver {
  kBuiltinWithoutDefaultDelegates,
  kBuiltinWithDefaultDelegates,
  kReference,
};

constexpr Resolver kAllResolvers[] = {
    Resolver::kBuiltinWithoutDefaultDelegates,
    Resolver::kBuiltinWithDefaultDelegates,
    Resolver::kReference,
};

class BlockwiseFullyConnectedModel {
 public:
  explicit BlockwiseFullyConnectedModel(
      std::vector<uint8_t> buffer,
      Resolver resolver = Resolver::kBuiltinWithoutDefaultDelegates)
      : buffer_(std::move(buffer)) {
    model_ = FlatBufferModel::BuildFromBuffer(
        reinterpret_cast<const char*>(buffer_.data()), buffer_.size());
    switch (resolver) {
      case Resolver::kBuiltinWithoutDefaultDelegates: {
        ops::builtin::BuiltinOpResolverWithoutDefaultDelegates op_resolver;
        InterpreterBuilder(*model_, op_resolver)(&interpreter_);
        break;
      }
      case Resolver::kBuiltinWithDefaultDelegates: {
        ops::builtin::BuiltinOpResolver op_resolver;
        InterpreterBuilder(*model_, op_resolver)(&interpreter_);
        break;
      }
      case Resolver::kReference: {
        ops::builtin::BuiltinRefOpResolver op_resolver;
        InterpreterBuilder(*model_, op_resolver)(&interpreter_);
        break;
      }
    }
  }

  bool Built() const { return interpreter_ != nullptr; }
  TfLiteStatus Allocate() { return interpreter_->AllocateTensors(); }
  TfLiteStatus Invoke() { return interpreter_->Invoke(); }

  void SetInput(const std::vector<int32_t>& values) {
    TfLiteTensor* input = interpreter_->tensor(interpreter_->inputs()[0]);
    ASSERT_EQ(NumElements(input), static_cast<int64_t>(values.size()));
    if (values.empty()) return;
    std::copy(values.begin(), values.end(), input->data.int8);
  }

  void SetFloatInput(const std::vector<float>& values) {
    TfLiteTensor* input = interpreter_->tensor(interpreter_->inputs()[0]);
    ASSERT_EQ(NumElements(input), static_cast<int64_t>(values.size()));
    if (values.empty()) return;
    std::copy(values.begin(), values.end(), input->data.f);
  }

  void SetHalfInput(const std::vector<float>& values) {
    TfLiteTensor* input = interpreter_->tensor(interpreter_->inputs()[0]);
    ASSERT_EQ(input->type, kTfLiteFloat16);
    ASSERT_EQ(NumElements(input), static_cast<int64_t>(values.size()));
    TfLiteFloat16* data = input->data.f16;
    for (size_t i = 0; i < values.size(); ++i) {
      data[i].data = Eigen::numext::bit_cast<uint16_t>(Eigen::half(values[i]));
    }
  }

  // Returns the float16 output widened to float.
  std::vector<float> GetHalfOutput() const {
    const TfLiteTensor* output =
        interpreter_->tensor(interpreter_->outputs()[0]);
    const int64_t num_elements = NumElements(output);
    std::vector<float> values(num_elements);
    const TfLiteFloat16* data = output->data.f16;
    for (int64_t i = 0; i < num_elements; ++i) {
      values[i] = static_cast<float>(
          Eigen::numext::bit_cast<Eigen::half>(data[i].data));
    }
    return values;
  }

  std::vector<float> GetFloatOutput() const {
    const TfLiteTensor* output =
        interpreter_->tensor(interpreter_->outputs()[0]);
    const int64_t num_elements = NumElements(output);
    if (num_elements == 0) return {};
    return std::vector<float>(output->data.f, output->data.f + num_elements);
  }

  std::vector<int8_t> GetOutput() const {
    const TfLiteTensor* output =
        interpreter_->tensor(interpreter_->outputs()[0]);
    const int64_t num_elements = NumElements(output);
    if (num_elements == 0) return {};
    return std::vector<int8_t>(output->data.int8,
                               output->data.int8 + num_elements);
  }

  std::vector<int> GetOutputShape() const {
    const TfLiteTensor* output =
        interpreter_->tensor(interpreter_->outputs()[0]);
    return std::vector<int>(output->dims->data,
                            output->dims->data + output->dims->size);
  }

 private:
  std::vector<uint8_t> buffer_;
  std::unique_ptr<FlatBufferModel> model_;
  std::unique_ptr<Interpreter> interpreter_;
};

// A row-major [extent0, extent1] block parameter table, broadcasting along
// extents of 1.
struct BlockTable {
  std::vector<float> values;
  std::array<int, 2> extents;

  double At(int n_block, int k_block) const {
    const int i = extents[0] == 1 ? 0 : n_block;
    const int j = extents[1] == 1 ? 0 : k_block;
    return values[i * extents[1] + j];
  }
};

// Dequantizes a [num_units, depth] filter with block shape
// [unit_block, depth_block].
std::vector<double> DequantizeFilter(const std::vector<int32_t>& q,
                                     int num_units, int depth, int unit_block,
                                     int depth_block, const BlockTable& scales,
                                     const BlockTable& zero_points) {
  std::vector<double> filter(q.size());
  for (int n = 0; n < num_units; ++n) {
    for (int k = 0; k < depth; ++k) {
      const int nb = n / unit_block;
      const int kb = k / depth_block;
      filter[n * depth + k] =
          scales.At(nb, kb) * (q[n * depth + k] - zero_points.At(nb, kb));
    }
  }
  return filter;
}

// Float oracle: dequantize, accumulate, add the bias, requantize and clamp.
// The tests use power-of-two scales so that this double computation is exact
// and matches the kernel bit for bit, including on rounding ties.
std::vector<int8_t> ReferenceFullyConnected(
    const std::vector<int32_t>& input, AffineParams input_quant,
    const std::vector<double>& filter, const std::vector<double>& bias,
    AffineParams output_quant, int batches, int num_units, int depth,
    int32_t act_min = -128, int32_t act_max = 127) {
  std::vector<int8_t> output;
  output.reserve(batches * num_units);
  for (int b = 0; b < batches; ++b) {
    for (int n = 0; n < num_units; ++n) {
      double acc = bias.empty() ? 0.0 : bias[n];
      for (int k = 0; k < depth; ++k) {
        const double x =
            input_quant.scale * (input[b * depth + k] - input_quant.zero_point);
        acc += x * filter[n * depth + k];
      }
      const double rounded =
          std::round(acc / output_quant.scale) + output_quant.zero_point;
      output.push_back(static_cast<int8_t>(
          std::clamp<double>(rounded, static_cast<double>(act_min),
                             static_cast<double>(act_max))));
    }
  }
  return output;
}

void ExpectOutput(const BlockwiseFcSpec& spec,
                  const std::vector<int32_t>& input,
                  const std::vector<int8_t>& expected,
                  const std::vector<int>& expected_shape) {
  for (Resolver resolver : kAllResolvers) {
    SCOPED_TRACE(static_cast<int>(resolver));
    BlockwiseFullyConnectedModel m(BuildBlockwiseFullyConnectedModel(spec),
                                   resolver);
    ASSERT_TRUE(m.Built());
    ASSERT_EQ(m.Allocate(), kTfLiteOk);
    m.SetInput(input);
    ASSERT_EQ(m.Invoke(), kTfLiteOk);
    EXPECT_THAT(m.GetOutputShape(), ElementsAreArray(expected_shape));
    EXPECT_THAT(m.GetOutput(), ElementsAreArray(expected));
  }
}

// A8W2: centered int2 filter (zero point -0.5), one scale per output channel
// (block shape [1, depth]), multiple input rows, with and without a bias.
class CenteredInt2PerChannelTest : public ::testing::TestWithParam<int> {};

TEST_P(CenteredInt2PerChannelTest, MatchesFloatReference) {
  // 0: no bias input, 1: omitted bias input, 2: per-channel bias,
  // 3: per-tensor bias.
  const int bias_mode = GetParam();
  constexpr int kBatches = 3;
  constexpr int kUnits = 4;
  constexpr int kDepth = 8;
  const AffineParams input_quant = {/*scale=*/0.5f, /*zero_point=*/3};
  const AffineParams output_quant = {/*scale=*/1.0f, /*zero_point=*/-2};
  const std::vector<int32_t> input = Pattern(kBatches * kDepth, -40, 40, 1);
  const std::vector<int32_t> filter_q = Pattern(kUnits * kDepth, -2, 1, 2);
  const BlockTable scales = {{0.25f, 0.5f, 1.0f, 0.125f}, {kUnits, 1}};
  const BlockTable zero_points = {{-0.5f}, {1, 1}};

  BlockwiseFcSpec spec;
  spec.input_shape = {kBatches, kDepth};
  spec.input_quant = input_quant;
  spec.filter_type = TensorType_INT2;
  spec.filter_shape = {kUnits, kDepth};
  spec.filter_data = PackInt2(filter_q);
  spec.block_shape = {1, kDepth};
  spec.scales = MakeParams(TensorType_FLOAT32, {kUnits, 1}, scales.values);
  spec.zero_points = MakeParams(TensorType_FLOAT32, {1, 1}, zero_points.values);
  spec.output_shape = {kBatches, kUnits};
  spec.output_quant = output_quant;

  std::vector<double> bias_real;
  if (bias_mode == 1) {
    spec.omitted_bias_input = true;
  } else if (bias_mode >= 2) {
    const std::vector<int32_t> bias_q = {37, -120, 5, 999};
    BiasSpec bias;
    bias.data = ToBytes(bias_q);
    if (bias_mode == 2) {
      // The conventional input_scale * filter_scale per channel.
      for (int n = 0; n < kUnits; ++n) {
        bias.scales.push_back(input_quant.scale * scales.values[n]);
        bias.zero_points.push_back(0);
      }
    } else {
      bias.scales = {0.0625f};
      bias.zero_points = {0};
    }
    for (int n = 0; n < kUnits; ++n) {
      bias_real.push_back(bias_q[n] *
                          bias.scales[bias.scales.size() == 1 ? 0 : n]);
    }
    spec.bias = bias;
  }

  const std::vector<double> filter = DequantizeFilter(
      filter_q, kUnits, kDepth, 1, kDepth, scales, zero_points);
  ExpectOutput(spec, input,
               ReferenceFullyConnected(input, input_quant, filter, bias_real,
                                       output_quant, kBatches, kUnits, kDepth),
               {kBatches, kUnits});
}

INSTANTIATE_TEST_SUITE_P(BiasModes, CenteredInt2PerChannelTest,
                         ::testing::Values(0, 1, 2, 3));

TEST(BlockwiseFullyConnectedTest, Int4PartialDepthBlocksWithKeepNumDims) {
  // Depth 10 with depth blocks of 4 leaves a partial last block of 2. The
  // zero points vary per block and the scales are float16.
  constexpr int kUnits = 3;
  constexpr int kDepth = 10;
  const AffineParams input_quant = {/*scale=*/0.25f, /*zero_point=*/-5};
  const AffineParams output_quant = {/*scale=*/0.5f, /*zero_point=*/4};
  const std::vector<int32_t> input = Pattern(2 * kDepth, -60, 60, 3);
  const std::vector<int32_t> filter_q = Pattern(kUnits * kDepth, -8, 7, 4);
  const BlockTable scales = {
      {0.5f, 0.25f, 1.0f, 0.125f, 0.5f, 0.25f, 1.0f, 2.0f, 0.5f}, {kUnits, 3}};
  const BlockTable zero_points = {{1, -2, 0, 3, -1, 2, 0, 0, -3}, {kUnits, 3}};

  BlockwiseFcSpec spec;
  spec.input_shape = {2, 1, kDepth};
  spec.input_quant = input_quant;
  spec.filter_type = TensorType_INT4;
  spec.filter_shape = {kUnits, kDepth};
  spec.filter_data = PackNibbles(filter_q);
  spec.block_shape = {1, 4};
  spec.scales = MakeF16Params({kUnits, 3}, scales.values);
  spec.zero_points = MakeParams(TensorType_INT8, {kUnits, 3},
                                std::vector<int8_t>(zero_points.values.begin(),
                                                    zero_points.values.end()));
  spec.output_shape = {2, 1, kUnits};
  spec.output_quant = output_quant;
  spec.keep_num_dims = true;

  const std::vector<double> filter =
      DequantizeFilter(filter_q, kUnits, kDepth, 1, 4, scales, zero_points);
  ExpectOutput(spec, input,
               ReferenceFullyConnected(input, input_quant, filter, {},
                                       output_quant, 2, kUnits, kDepth),
               {2, 1, kUnits});
}

TEST(BlockwiseFullyConnectedTest, Int8UnitBlocksWithReluAndFlattenedInput) {
  // Output channel blocks of 2 (with a partial last block of 1) and depth
  // blocks of 5; the [2, 2, 10] input flattens to 4 rows.
  constexpr int kUnits = 5;
  constexpr int kDepth = 10;
  const AffineParams input_quant = {/*scale=*/0.125f, /*zero_point=*/0};
  const AffineParams output_quant = {/*scale=*/0.25f, /*zero_point=*/-10};
  const std::vector<int32_t> input = Pattern(4 * kDepth, -128, 127, 5);
  const std::vector<int32_t> filter_q = Pattern(kUnits * kDepth, -127, 127, 6);
  const BlockTable scales = {
      {0.0078125f, 0.015625f, 0.03125f, 0.0078125f, 0.015625f, 0.0625f},
      {3, 2}};
  const BlockTable zero_points = {{0.0f}, {1, 1}};
  const std::vector<int32_t> bias_q = {100, -200, 300, -400, 500};

  BlockwiseFcSpec spec;
  spec.input_shape = {2, 2, kDepth};
  spec.input_quant = input_quant;
  spec.filter_type = TensorType_INT8;
  spec.filter_shape = {kUnits, kDepth};
  spec.filter_data = PackAs<int8_t>(filter_q);
  spec.block_shape = {2, 5};
  spec.scales = MakeParams(TensorType_FLOAT32, {3, 2}, scales.values);
  spec.bias = BiasSpec{TensorType_INT32, ToBytes(bias_q), {0.03125f}, {0}};
  spec.output_shape = {4, kUnits};
  spec.output_quant = output_quant;
  spec.activation = ActivationFunctionType_RELU;

  std::vector<double> bias_real;
  for (int32_t b : bias_q) bias_real.push_back(b * 0.03125);
  const std::vector<double> filter =
      DequantizeFilter(filter_q, kUnits, kDepth, 2, 5, scales, zero_points);
  // RELU clamps at the output zero point.
  ExpectOutput(spec, input,
               ReferenceFullyConnected(input, input_quant, filter, bias_real,
                                       output_quant, 4, kUnits, kDepth,
                                       /*act_min=*/-10, /*act_max=*/127),
               {4, kUnits});
}

TEST(BlockwiseFullyConnectedTest, UInt4UInt8AndInt16Filters) {
  constexpr int kUnits = 3;
  constexpr int kDepth = 6;
  const AffineParams input_quant = {/*scale=*/0.5f, /*zero_point=*/1};
  const AffineParams output_quant = {/*scale=*/2.0f, /*zero_point=*/0};
  const std::vector<int32_t> input = Pattern(2 * kDepth, -20, 20, 7);
  const BlockTable scales = {{0.25f, 0.5f, 0.125f, 1.0f, 0.5f, 0.25f},
                             {kUnits, 2}};

  struct Case {
    TensorType type;
    std::vector<int32_t> q;
    float zero_point;
  };
  const std::vector<int32_t> u4 = Pattern(kUnits * kDepth, 0, 15, 8);
  const std::vector<int32_t> u8 = Pattern(kUnits * kDepth, 0, 255, 9);
  const std::vector<int32_t> i16 = Pattern(kUnits * kDepth, -1000, 1000, 10);
  for (const Case& c :
       {Case{TensorType_UINT4, u4, 8.0f}, Case{TensorType_UINT8, u8, 128.0f},
        Case{TensorType_INT16, i16, 0.0f}}) {
    SCOPED_TRACE(EnumNameTensorType(c.type));
    BlockwiseFcSpec spec;
    spec.input_shape = {2, kDepth};
    spec.input_quant = input_quant;
    spec.filter_type = c.type;
    spec.filter_shape = {kUnits, kDepth};
    if (c.type == TensorType_UINT4) {
      spec.filter_data = PackNibbles(c.q);
    } else if (c.type == TensorType_UINT8) {
      spec.filter_data = PackAs<uint8_t>(c.q);
    } else {
      spec.filter_data = PackAs<int16_t>(c.q);
    }
    spec.block_shape = {1, 3};
    spec.scales = MakeParams(TensorType_FLOAT32, {kUnits, 2}, scales.values);
    spec.zero_points =
        MakeParams(TensorType_INT32, {1, 1},
                   std::vector<int32_t>{static_cast<int32_t>(c.zero_point)});
    spec.output_shape = {2, kUnits};
    spec.output_quant = output_quant;

    const std::vector<double> filter = DequantizeFilter(
        c.q, kUnits, kDepth, 1, 3, scales, {{c.zero_point}, {1, 1}});
    ExpectOutput(spec, input,
                 ReferenceFullyConnected(input, input_quant, filter, {},
                                         output_quant, 2, kUnits, kDepth),
                 {2, kUnits});
  }
}

TEST(BlockwiseFullyConnectedTest, OddPackedSizesWithDepthOnlyScalesAndRelu6) {
  // [3, 5] filters do not fill their last packed byte. The scales vary only
  // along depth (extent [1, 3]: a unit block of 3 covers all units, depth
  // blocks of 2 leave a partial last block), so they broadcast along units.
  constexpr int kUnits = 3;
  constexpr int kDepth = 5;
  const AffineParams input_quant = {/*scale=*/0.5f, /*zero_point=*/-3};
  const AffineParams output_quant = {/*scale=*/0.0625f, /*zero_point=*/-100};
  const std::vector<int32_t> input = Pattern(2 * kDepth, -30, 30, 12);
  const BlockTable scales = {{0.25f, 0.125f, 0.5f}, {1, 3}};
  // RELU6 clamps to [zero_point, zero_point + 6 / scale] = [-100, -4].
  constexpr int32_t kActMin = -100;
  constexpr int32_t kActMax = -4;

  for (TensorType type : {TensorType_INT2, TensorType_INT4}) {
    SCOPED_TRACE(EnumNameTensorType(type));
    const bool is_int2 = type == TensorType_INT2;
    const std::vector<int32_t> filter_q = Pattern(
        kUnits * kDepth, is_int2 ? -2 : -8, is_int2 ? 1 : 7, is_int2 ? 13 : 14);
    BlockwiseFcSpec spec;
    spec.input_shape = {2, kDepth};
    spec.input_quant = input_quant;
    spec.filter_type = type;
    spec.filter_shape = {kUnits, kDepth};
    spec.filter_data = is_int2 ? PackInt2(filter_q) : PackNibbles(filter_q);
    spec.block_shape = {kUnits, 2};
    spec.scales = MakeParams(TensorType_FLOAT32, {1, 3}, scales.values);
    spec.zero_points =
        MakeParams(TensorType_FLOAT32, {1, 1}, std::vector<float>{-0.5f});
    spec.output_shape = {2, kUnits};
    spec.output_quant = output_quant;
    spec.activation = ActivationFunctionType_RELU6;

    const std::vector<double> filter = DequantizeFilter(
        filter_q, kUnits, kDepth, kUnits, 2, scales, {{-0.5f}, {1, 1}});
    ExpectOutput(
        spec, input,
        ReferenceFullyConnected(input, input_quant, filter, {}, output_quant, 2,
                                kUnits, kDepth, kActMin, kActMax),
        {2, kUnits});
  }
}

TEST(BlockwiseFullyConnectedTest, ZeroDepthOutputsQuantizedBias) {
  BlockwiseFcSpec spec;
  spec.input_shape = {3, 0};
  spec.input_quant = {/*scale=*/0.5f, /*zero_point=*/0};
  spec.filter_type = TensorType_INT8;
  spec.filter_shape = {2, 0};
  spec.block_shape = {1, 4};
  spec.scales = MakeParams(TensorType_FLOAT32, {2, 0}, std::vector<float>{});
  spec.bias = BiasSpec{
      TensorType_INT32, ToBytes(std::vector<int32_t>{8, -12}), {0.25f}, {0}};
  spec.output_shape = {3, 2};
  spec.output_quant = {/*scale=*/0.5f, /*zero_point=*/-7};

  for (Resolver resolver : kAllResolvers) {
    SCOPED_TRACE(static_cast<int>(resolver));
    BlockwiseFullyConnectedModel m(BuildBlockwiseFullyConnectedModel(spec),
                                   resolver);
    ASSERT_TRUE(m.Built());
    ASSERT_EQ(m.Allocate(), kTfLiteOk);
    ASSERT_EQ(m.Invoke(), kTfLiteOk);
    // 8 * 0.25 / 0.5 - 7 = -3 and -12 * 0.25 / 0.5 - 7 = -13.
    EXPECT_THAT(m.GetOutput(), ElementsAreArray({-3, -13, -3, -13, -3, -13}));
  }
}

TEST(BlockwiseFullyConnectedTest, ZeroBatchesProducesEmptyOutput) {
  BlockwiseFcSpec spec;
  spec.input_shape = {0, 4};
  spec.input_quant = {/*scale=*/0.5f, /*zero_point=*/0};
  spec.filter_type = TensorType_INT8;
  spec.filter_shape = {2, 4};
  spec.filter_data = {1, 2, 3, 4, 5, 6, 7, 8};
  spec.block_shape = {1, 4};
  spec.scales =
      MakeParams(TensorType_FLOAT32, {2, 1}, std::vector<float>{0.5f, 0.25f});
  spec.output_shape = {0, 2};
  spec.output_quant = {/*scale=*/0.5f, /*zero_point=*/0};

  for (Resolver resolver : kAllResolvers) {
    SCOPED_TRACE(static_cast<int>(resolver));
    BlockwiseFullyConnectedModel m(BuildBlockwiseFullyConnectedModel(spec),
                                   resolver);
    ASSERT_TRUE(m.Built());
    ASSERT_EQ(m.Allocate(), kTfLiteOk);
    ASSERT_EQ(m.Invoke(), kTfLiteOk);
    EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({0, 2}));
    EXPECT_TRUE(m.GetOutput().empty());
  }
}

// A valid A8W2 spec that the rejection tests below break one way each.
BlockwiseFcSpec ValidInt2Spec() {
  BlockwiseFcSpec spec;
  spec.input_shape = {1, 8};
  spec.input_quant = {/*scale=*/0.5f, /*zero_point=*/0};
  spec.filter_type = TensorType_INT2;
  spec.filter_shape = {2, 8};
  spec.filter_data = PackInt2(Pattern(16, -2, 1, 0));
  spec.block_shape = {1, 8};
  spec.scales =
      MakeParams(TensorType_FLOAT32, {2, 1}, std::vector<float>{0.5f, 0.25f});
  spec.zero_points =
      MakeParams(TensorType_FLOAT32, {1, 1}, std::vector<float>{-0.5f});
  spec.output_shape = {1, 2};
  spec.output_quant = {/*scale=*/0.5f, /*zero_point=*/0};
  return spec;
}

TfLiteStatus AllocateSpec(const BlockwiseFcSpec& spec) {
  BlockwiseFullyConnectedModel m(BuildBlockwiseFullyConnectedModel(spec));
  if (!m.Built()) return kTfLiteError;
  return m.Allocate();
}

TEST(BlockwiseFullyConnectedTest, ValidSpecAllocates) {
  EXPECT_EQ(AllocateSpec(ValidInt2Spec()), kTfLiteOk);
}

TEST(BlockwiseFullyConnectedTest, RejectsNonInt8Input) {
  BlockwiseFcSpec spec = ValidInt2Spec();
  spec.input_type = TensorType_INT16;
  EXPECT_EQ(AllocateSpec(spec), kTfLiteError);
}

TEST(BlockwiseFullyConnectedTest, RejectsFloatInput) {
  BlockwiseFcSpec spec = ValidInt2Spec();
  spec.input_type = TensorType_FLOAT32;
  EXPECT_EQ(AllocateSpec(spec), kTfLiteError);
}

TEST(BlockwiseFullyConnectedTest, RejectsMismatchedBlockShapeRank) {
  BlockwiseFcSpec spec = ValidInt2Spec();
  spec.block_shape = {8};
  BlockwiseFullyConnectedModel m(BuildBlockwiseFullyConnectedModel(spec));
  EXPECT_FALSE(m.Built());
}

TEST(BlockwiseFullyConnectedTest, RejectsMismatchedScalesExtent) {
  BlockwiseFcSpec spec = ValidInt2Spec();
  spec.scales = MakeParams(TensorType_FLOAT32, {3, 1},
                           std::vector<float>{0.5f, 0.25f, 1.0f});
  EXPECT_EQ(AllocateSpec(spec), kTfLiteError);
}

TEST(BlockwiseFullyConnectedTest, RejectsNonConstantScales) {
  BlockwiseFcSpec spec = ValidInt2Spec();
  spec.scales = ParamsTensor{TensorType_FLOAT32, {2, 1}, /*data=*/{}};
  spec.scales_as_rw_tensor = true;
  EXPECT_EQ(AllocateSpec(spec), kTfLiteError);
}

TEST(BlockwiseFullyConnectedTest, RejectsFloatBias) {
  BlockwiseFcSpec spec = ValidInt2Spec();
  spec.bias = BiasSpec{
      TensorType_FLOAT32, ToBytes(std::vector<float>{1.0f, 2.0f}), {1.0f}, {0}};
  EXPECT_EQ(AllocateSpec(spec), kTfLiteError);
}

TEST(BlockwiseFullyConnectedTest, RejectsBiasWithNonZeroZeroPoint) {
  BlockwiseFcSpec spec = ValidInt2Spec();
  spec.bias = BiasSpec{
      TensorType_INT32, ToBytes(std::vector<int32_t>{1, 2}), {0.25f}, {3}};
  EXPECT_EQ(AllocateSpec(spec), kTfLiteError);
}

TEST(BlockwiseFullyConnectedTest, RejectsNonPositiveOutputScale) {
  BlockwiseFcSpec spec = ValidInt2Spec();
  spec.output_quant.scale = 0.0f;
  EXPECT_EQ(AllocateSpec(spec), kTfLiteError);
}

TEST(BlockwiseFullyConnectedTest, RejectsInputSizeNotMultipleOfDepth) {
  BlockwiseFcSpec spec = ValidInt2Spec();
  spec.input_shape = {1, 9};
  EXPECT_EQ(AllocateSpec(spec), kTfLiteError);
}

TEST(BlockwiseFullyConnectedTest, RejectsTanhActivation) {
  BlockwiseFcSpec spec = ValidInt2Spec();
  spec.activation = ActivationFunctionType_TANH;
  EXPECT_EQ(AllocateSpec(spec), kTfLiteError);
}

TEST(BlockwiseFullyConnectedTest, RejectsOutOfRangeZeroPoints) {
  BlockwiseFcSpec spec = ValidInt2Spec();
  spec.output_quant.zero_point = 200;
  spec.activation = ActivationFunctionType_RELU;
  EXPECT_EQ(AllocateSpec(spec), kTfLiteError);

  spec = ValidInt2Spec();
  spec.input_quant.zero_point = -129;
  EXPECT_EQ(AllocateSpec(spec), kTfLiteError);
}

TEST(BlockwiseFullyConnectedTest,
     BF16ScalesInt16ZeroPointsAndOmittedZeroPointsWithInputOffset) {
  constexpr int kUnits = 2;
  constexpr int kDepth = 4;
  const AffineParams input_quant = {/*scale=*/0.5f, /*zero_point=*/-7};
  const AffineParams output_quant = {/*scale=*/0.5f, /*zero_point=*/3};
  const std::vector<int32_t> input = Pattern(2 * kDepth, -30, 30, 21);
  const std::vector<int32_t> filter_q = Pattern(kUnits * kDepth, -8, 7, 22);
  const BlockTable scales = {{0.25f, 0.5f, 1.0f, 0.125f}, {kUnits, 2}};

  // Case 1: BFLOAT16 scales + INT16 zero points.
  {
    const BlockTable zero_points = {{-2.0f, 3.0f}, {1, 2}};
    BlockwiseFcSpec spec;
    spec.input_shape = {2, kDepth};
    spec.input_quant = input_quant;
    spec.filter_type = TensorType_INT4;
    spec.filter_shape = {kUnits, kDepth};
    spec.filter_data = PackNibbles(filter_q);
    spec.block_shape = {1, 2};
    spec.scales = MakeBF16Params({kUnits, 2}, scales.values);
    spec.zero_points =
        MakeParams(TensorType_INT16, {1, 2}, std::vector<int16_t>{-2, 3});
    spec.output_shape = {2, kUnits};
    spec.output_quant = output_quant;

    const std::vector<double> filter =
        DequantizeFilter(filter_q, kUnits, kDepth, 1, 2, scales, zero_points);
    ExpectOutput(spec, input,
                 ReferenceFullyConnected(input, input_quant, filter, {},
                                         output_quant, 2, kUnits, kDepth),
                 {2, kUnits});
  }

  // Case 2: Omitted zero_points tensor with non-zero input zero point, plus
  // INT32_MAX depth block size.
  {
    const BlockTable single_scales = {{0.25f, 0.5f}, {kUnits, 1}};
    BlockwiseFcSpec spec;
    spec.input_shape = {2, kDepth};
    spec.input_quant = input_quant;
    spec.filter_type = TensorType_INT4;
    spec.filter_shape = {kUnits, kDepth};
    spec.filter_data = PackNibbles(filter_q);
    spec.block_shape = {1, std::numeric_limits<int32_t>::max()};
    spec.scales = MakeBF16Params({kUnits, 1}, single_scales.values);
    spec.output_shape = {2, kUnits};
    spec.output_quant = output_quant;

    const std::vector<double> filter = DequantizeFilter(
        filter_q, kUnits, kDepth, 1, kDepth, single_scales, {{0.0f}, {1, 1}});
    ExpectOutput(spec, input,
                 ReferenceFullyConnected(input, input_quant, filter, {},
                                         output_quant, 2, kUnits, kDepth),
                 {2, kUnits});
  }
}

TEST(BlockwiseFullyConnectedTest, RejectsA4W2QuantSpec) {
  flexbuffers::Builder fbb;
  const size_t map_start = fbb.StartMap();
  fbb.String("spec", "cint2_fp32_int4_e8m0_drq");
  fbb.Double("act_dilation", 1.5);
  fbb.EndMap(map_start);
  fbb.Finish();

  BlockwiseFcSpec spec = ValidInt2Spec();
  spec.quant_spec = fbb.GetBuffer();
  EXPECT_EQ(AllocateSpec(spec), kTfLiteError);
}

// Dynamic range (hybrid) FULLY_CONNECTED: float32 or float16 input and output,
// each input row quantized to int8 at runtime (symmetrically by default,
// asymmetrically with `asymmetric_quantize_inputs`), blockwise quantized
// filter.

// Quantizes one row like the kernel: scale = max(|x|) / 127 and
// q = clamp(round(x * (127 / max(|x|))), -127, 127); an all-zero row gets
// scale 1.
void QuantizeRowOracle(const float* row, int depth, std::vector<int32_t>* q,
                       float* scale) {
  float max_abs = 0.0f;
  for (int k = 0; k < depth; ++k) max_abs = std::max(max_abs, std::abs(row[k]));
  q->assign(depth, 0);
  if (max_abs == 0.0f) {
    *scale = 1.0f;
    return;
  }
  *scale = max_abs / 127.0f;
  const float inverse_scale = 127.0f / max_abs;
  for (int k = 0; k < depth; ++k) {
    (*q)[k] = static_cast<int32_t>(
        std::clamp(std::round(row[k] * inverse_scale), -127.0f, 127.0f));
  }
}

// Quantizes one row asymmetrically to [-128, 127] like
// `tensor_utils::PortableAsymmetricQuantizeFloats`, which the kernel uses:
// `x ~ scale * (q - zero_point)`.
void AsymmetricQuantizeRowOracle(const float* row, int depth,
                                 std::vector<int32_t>* q, float* scale,
                                 int32_t* zero_point) {
  q->assign(depth, 0);
  *scale = 1.0f;
  *zero_point = 0;
  if (depth == 0) return;
  const auto [min_it, max_it] = std::minmax_element(row, row + depth);
  const double rmin = std::min(0.0f, *min_it);
  const double rmax = std::max(0.0f, *max_it);
  if (rmin == rmax) return;
  const double qmin = -128.0;
  const double qmax = 127.0;
  const double scale_double = (rmax - rmin) / (qmax - qmin);
  const double zero_point_from_min = qmin - rmin / scale_double;
  const double zero_point_from_max = qmax - rmax / scale_double;
  const double zero_point_double =
      std::abs(qmin) + std::abs(rmin / scale_double) <
              std::abs(qmax) + std::abs(rmax / scale_double)
          ? zero_point_from_min
          : zero_point_from_max;
  *zero_point = zero_point_double <= qmin ? -128
                : zero_point_double >= qmax
                    ? 127
                    : static_cast<int32_t>(std::round(zero_point_double));
  *scale = static_cast<float>(scale_double);
  const float inverse_scale = 1.0f / *scale;
  for (int k = 0; k < depth; ++k) {
    (*q)[k] = std::clamp(
        static_cast<int32_t>(std::round(*zero_point + row[k] * inverse_scale)),
        -128, 127);
  }
}

// Float oracle of the dynamic range kernel: quantize each input row (per-row
// asymmetric with `asymmetric`), multiply with the dequantized filter, scale by
// the row scale, add the bias and clamp. With power-of-two filter scales the
// accumulation is exact in double, so this matches the kernel bit for bit.
std::vector<float> ReferenceHybridFullyConnected(
    const std::vector<float>& input, const std::vector<double>& filter,
    const std::vector<float>& bias, int batches, int num_units, int depth,
    float act_min = std::numeric_limits<float>::lowest(),
    float act_max = std::numeric_limits<float>::max(),
    bool asymmetric = false) {
  std::vector<float> output;
  output.reserve(batches * num_units);
  std::vector<int32_t> q;
  for (int b = 0; b < batches; ++b) {
    float row_scale = 0.0f;
    int32_t row_zero_point = 0;
    if (asymmetric) {
      AsymmetricQuantizeRowOracle(input.data() + b * depth, depth, &q,
                                  &row_scale, &row_zero_point);
    } else {
      QuantizeRowOracle(input.data() + b * depth, depth, &q, &row_scale);
    }
    for (int n = 0; n < num_units; ++n) {
      double acc = 0.0;
      for (int k = 0; k < depth; ++k) {
        acc += (q[k] - row_zero_point) * filter[n * depth + k];
      }
      double value = acc * static_cast<double>(row_scale);
      if (!bias.empty()) value += static_cast<double>(bias[n]);
      output.push_back(
          std::min(std::max(static_cast<float>(value), act_min), act_max));
    }
  }
  return output;
}

// Unquantized float FULLY_CONNECTED with the dequantized filter.
std::vector<float> FloatFullyConnected(const std::vector<float>& input,
                                       const std::vector<double>& filter,
                                       const std::vector<float>& bias,
                                       int batches, int num_units, int depth) {
  std::vector<float> output;
  for (int b = 0; b < batches; ++b) {
    for (int n = 0; n < num_units; ++n) {
      double acc = bias.empty() ? 0.0 : bias[n];
      for (int k = 0; k < depth; ++k) {
        acc += input[b * depth + k] * filter[n * depth + k];
      }
      output.push_back(static_cast<float>(acc));
    }
  }
  return output;
}

// Deterministic float values in [-amplitude, amplitude] that are not on a
// coarse grid.
std::vector<float> FloatPattern(int size, float amplitude, int seed) {
  std::vector<float> values(size);
  for (int i = 0; i < size; ++i) {
    values[i] = amplitude * std::sin(0.37f * i + 1.7f * seed);
  }
  return values;
}

BlockwiseFcSpec MakeHybridSpec() {
  BlockwiseFcSpec spec;
  spec.input_type = TensorType_FLOAT32;
  spec.output_type = TensorType_FLOAT32;
  spec.op_version = 17;
  return spec;
}

void ExpectHybridOutput(const BlockwiseFcSpec& spec,
                        const std::vector<float>& input,
                        const std::vector<float>& expected,
                        const std::vector<int>& expected_shape) {
  for (Resolver resolver : kAllResolvers) {
    SCOPED_TRACE(static_cast<int>(resolver));
    BlockwiseFullyConnectedModel m(BuildBlockwiseFullyConnectedModel(spec),
                                   resolver);
    ASSERT_TRUE(m.Built());
    ASSERT_EQ(m.Allocate(), kTfLiteOk);
    m.SetFloatInput(input);
    ASSERT_EQ(m.Invoke(), kTfLiteOk);
    EXPECT_THAT(m.GetOutputShape(), ElementsAreArray(expected_shape));
    EXPECT_THAT(m.GetFloatOutput(), Pointwise(FloatEq(), expected));
  }
}

// DRQ A8W2: float activations against a centered int2 filter (zero point -0.5)
// with one scale per output channel. One input row is all zeros.
class HybridCenteredInt2PerChannelTest : public ::testing::TestWithParam<bool> {
};

TEST_P(HybridCenteredInt2PerChannelTest, MatchesReference) {
  const bool with_bias = GetParam();
  constexpr int kBatches = 3;
  constexpr int kUnits = 4;
  constexpr int kDepth = 16;
  std::vector<float> input = FloatPattern(kBatches * kDepth, 3.0f, 1);
  std::fill_n(input.begin() + kDepth, kDepth, 0.0f);
  const std::vector<int32_t> filter_q = Pattern(kUnits * kDepth, -2, 1, 2);
  const BlockTable scales = {{0.25f, 0.5f, 1.0f, 0.125f}, {kUnits, 1}};
  const BlockTable zero_points = {{-0.5f}, {1, 1}};

  BlockwiseFcSpec spec = MakeHybridSpec();
  spec.input_shape = {kBatches, kDepth};
  spec.filter_type = TensorType_INT2;
  spec.filter_shape = {kUnits, kDepth};
  spec.filter_data = PackInt2(filter_q);
  spec.block_shape = {1, kDepth};
  spec.scales = MakeParams(TensorType_FLOAT32, {kUnits, 1}, scales.values);
  spec.zero_points = MakeParams(TensorType_FLOAT32, {1, 1}, zero_points.values);
  spec.output_shape = {kBatches, kUnits};
  std::vector<float> bias;
  if (with_bias) {
    bias = {0.75f, -1.25f, 0.1f, 3.0f};
    spec.bias = BiasSpec{TensorType_FLOAT32, ToBytes(bias), {}, {}};
  }

  const std::vector<double> filter = DequantizeFilter(
      filter_q, kUnits, kDepth, 1, kDepth, scales, zero_points);
  const std::vector<float> expected = ReferenceHybridFullyConnected(
      input, filter, bias, kBatches, kUnits, kDepth);
  ExpectHybridOutput(spec, input, expected, {kBatches, kUnits});

  // The all-zero row outputs exactly the bias.
  for (int n = 0; n < kUnits; ++n) {
    EXPECT_EQ(expected[kUnits + n], with_bias ? bias[n] : 0.0f);
  }
  // Int8 activation quantization stays close to the float computation.
  EXPECT_THAT(expected,
              Pointwise(FloatNear(0.3f),
                        FloatFullyConnected(input, filter, bias, kBatches,
                                            kUnits, kDepth)));
}

INSTANTIATE_TEST_SUITE_P(WithAndWithoutBias, HybridCenteredInt2PerChannelTest,
                         ::testing::Bool());

TEST(HybridBlockwiseFullyConnectedTest,
     CenteredInt2SubChannelBlocksWithKeepNumDimsAndRelu6) {
  // Depth blocks of 4 over a depth of 10 leave a partial last block of 2, and
  // output channel blocks of 2 leave a partial last block of 1. The scales are
  // float16 and RELU6 clamps the float output.
  constexpr int kUnits = 3;
  constexpr int kDepth = 10;
  const std::vector<float> input = FloatPattern(2 * kDepth, 5.0f, 3);
  const std::vector<int32_t> filter_q = Pattern(kUnits * kDepth, -2, 1, 4);
  const BlockTable scales = {{0.5f, 0.25f, 1.0f, 2.0f, 0.125f, 0.5f}, {2, 3}};
  const BlockTable zero_points = {{-0.5f}, {1, 1}};

  BlockwiseFcSpec spec = MakeHybridSpec();
  spec.input_shape = {2, 1, kDepth};
  spec.filter_type = TensorType_INT2;
  spec.filter_shape = {kUnits, kDepth};
  spec.filter_data = PackInt2(filter_q);
  spec.block_shape = {2, 4};
  spec.scales = MakeF16Params({2, 3}, scales.values);
  spec.zero_points = MakeParams(TensorType_FLOAT32, {1, 1}, zero_points.values);
  spec.output_shape = {2, 1, kUnits};
  spec.keep_num_dims = true;
  spec.activation = ActivationFunctionType_RELU6;

  const std::vector<double> filter =
      DequantizeFilter(filter_q, kUnits, kDepth, 2, 4, scales, zero_points);
  ExpectHybridOutput(spec, input,
                     ReferenceHybridFullyConnected(input, filter, {}, 2, kUnits,
                                                   kDepth, 0.0f, 6.0f),
                     {2, 1, kUnits});
}

TEST(HybridBlockwiseFullyConnectedTest, Int4And8FiltersWithZeroPoints) {
  // The dynamic range path is not specific to int2: other storage types with
  // per-block integer zero points and an omitted zero point tensor work too.
  constexpr int kUnits = 2;
  constexpr int kDepth = 8;
  const std::vector<float> input = FloatPattern(2 * kDepth, 1.5f, 5);
  const BlockTable scales = {{0.25f, 0.5f, 0.125f, 1.0f}, {kUnits, 2}};

  {
    SCOPED_TRACE("INT4");
    const std::vector<int32_t> filter_q = Pattern(kUnits * kDepth, -8, 7, 6);
    const BlockTable zero_points = {{1, -2, 0, 3}, {kUnits, 2}};
    BlockwiseFcSpec spec = MakeHybridSpec();
    spec.input_shape = {2, kDepth};
    spec.filter_type = TensorType_INT4;
    spec.filter_shape = {kUnits, kDepth};
    spec.filter_data = PackNibbles(filter_q);
    spec.block_shape = {1, 4};
    spec.scales = MakeParams(TensorType_FLOAT32, {kUnits, 2}, scales.values);
    spec.zero_points = MakeParams(TensorType_INT8, {kUnits, 2},
                                  std::vector<int8_t>{1, -2, 0, 3});
    spec.output_shape = {2, kUnits};
    const std::vector<double> filter =
        DequantizeFilter(filter_q, kUnits, kDepth, 1, 4, scales, zero_points);
    ExpectHybridOutput(
        spec, input,
        ReferenceHybridFullyConnected(input, filter, {}, 2, kUnits, kDepth),
        {2, kUnits});
  }
  {
    SCOPED_TRACE("INT8");
    const std::vector<int32_t> filter_q =
        Pattern(kUnits * kDepth, -127, 127, 7);
    BlockwiseFcSpec spec = MakeHybridSpec();
    spec.input_shape = {2, kDepth};
    spec.filter_type = TensorType_INT8;
    spec.filter_shape = {kUnits, kDepth};
    spec.filter_data = PackAs<int8_t>(filter_q);
    spec.block_shape = {1, 4};
    spec.scales = MakeParams(TensorType_FLOAT32, {kUnits, 2}, scales.values);
    spec.output_shape = {2, kUnits};
    spec.activation = ActivationFunctionType_RELU;
    const std::vector<double> filter = DequantizeFilter(
        filter_q, kUnits, kDepth, 1, 4, scales, {{0.0f}, {1, 1}});
    ExpectHybridOutput(spec, input,
                       ReferenceHybridFullyConnected(input, filter, {}, 2,
                                                     kUnits, kDepth, 0.0f),
                       {2, kUnits});
  }
}

TEST(HybridBlockwiseFullyConnectedTest, ZeroDepthOutputsBias) {
  BlockwiseFcSpec spec = MakeHybridSpec();
  spec.input_shape = {2, 0};
  spec.filter_type = TensorType_INT2;
  spec.filter_shape = {2, 0};
  spec.block_shape = {1, 4};
  spec.scales = MakeParams(TensorType_FLOAT32, {2, 0}, std::vector<float>{});
  spec.bias = BiasSpec{
      TensorType_FLOAT32, ToBytes(std::vector<float>{1.5f, -2.25f}), {}, {}};
  spec.output_shape = {2, 2};

  for (Resolver resolver : kAllResolvers) {
    SCOPED_TRACE(static_cast<int>(resolver));
    BlockwiseFullyConnectedModel m(BuildBlockwiseFullyConnectedModel(spec),
                                   resolver);
    ASSERT_TRUE(m.Built());
    ASSERT_EQ(m.Allocate(), kTfLiteOk);
    ASSERT_EQ(m.Invoke(), kTfLiteOk);
    EXPECT_THAT(m.GetFloatOutput(),
                ElementsAreArray({1.5f, -2.25f, 1.5f, -2.25f}));
  }
}

// Rounds `value` to the nearest float16.
float RoundToHalf(float value) {
  return static_cast<float>(Eigen::half(value));
}

std::vector<float> RoundToHalf(const std::vector<float>& values) {
  std::vector<float> rounded;
  rounded.reserve(values.size());
  for (float value : values) rounded.push_back(RoundToHalf(value));
  return rounded;
}

std::vector<uint8_t> HalfBytes(const std::vector<float>& values) {
  std::vector<uint16_t> halves;
  halves.reserve(values.size());
  for (float value : values) {
    halves.push_back(Eigen::numext::bit_cast<uint16_t>(Eigen::half(value)));
  }
  return ToBytes(halves);
}

// DRQ A8W2 with float16 activations: the input, output and bias are float16.
// The kernel widens them to float32, so the result matches the float32
// reference on the float16-rounded input and bias, rounded to float16 once.
class HybridHalfCenteredInt2Test
    : public ::testing::TestWithParam<
          std::tuple<bool, ActivationFunctionType>> {};

TEST_P(HybridHalfCenteredInt2Test, MatchesReference) {
  const auto [with_bias, activation] = GetParam();
  const bool relu6 = activation == ActivationFunctionType_RELU6;
  const float act_min = relu6 ? 0.0f : std::numeric_limits<float>::lowest();
  const float act_max = relu6 ? 6.0f : std::numeric_limits<float>::max();
  constexpr int kBatches = 3;
  constexpr int kUnits = 4;
  constexpr int kDepth = 16;
  std::vector<float> input =
      RoundToHalf(FloatPattern(kBatches * kDepth, 3.0f, 9));
  std::fill_n(input.begin() + kDepth, kDepth, 0.0f);
  const std::vector<int32_t> filter_q = Pattern(kUnits * kDepth, -2, 1, 10);
  const BlockTable scales = {{0.25f, 0.5f, 1.0f, 0.125f}, {kUnits, 1}};
  const BlockTable zero_points = {{-0.5f}, {1, 1}};

  BlockwiseFcSpec spec = MakeHybridSpec();
  spec.input_type = TensorType_FLOAT16;
  spec.output_type = TensorType_FLOAT16;
  spec.input_shape = {kBatches, kDepth};
  spec.filter_type = TensorType_INT2;
  spec.filter_shape = {kUnits, kDepth};
  spec.filter_data = PackInt2(filter_q);
  spec.block_shape = {1, 8};
  spec.scales = MakeF16Params({kUnits, 1}, scales.values);
  spec.zero_points = MakeParams(TensorType_FLOAT32, {1, 1}, zero_points.values);
  spec.output_shape = {kBatches, kUnits};
  spec.activation = activation;
  std::vector<float> bias;
  if (with_bias) {
    bias = RoundToHalf({0.75f, -1.25f, 0.1f, 3.0f});
    spec.bias = BiasSpec{TensorType_FLOAT16, HalfBytes(bias), {}, {}};
  }

  const std::vector<double> filter =
      DequantizeFilter(filter_q, kUnits, kDepth, 1, 8, scales, zero_points);
  const std::vector<float> expected = RoundToHalf(ReferenceHybridFullyConnected(
      input, filter, bias, kBatches, kUnits, kDepth, act_min, act_max));
  for (Resolver resolver : kAllResolvers) {
    SCOPED_TRACE(static_cast<int>(resolver));
    BlockwiseFullyConnectedModel m(BuildBlockwiseFullyConnectedModel(spec),
                                   resolver);
    ASSERT_TRUE(m.Built());
    ASSERT_EQ(m.Allocate(), kTfLiteOk);
    m.SetHalfInput(input);
    ASSERT_EQ(m.Invoke(), kTfLiteOk);
    EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({kBatches, kUnits}));
    EXPECT_THAT(m.GetHalfOutput(), Pointwise(FloatEq(), expected));
  }
}

INSTANTIATE_TEST_SUITE_P(
    BiasAndActivation, HybridHalfCenteredInt2Test,
    ::testing::Combine(::testing::Bool(),
                       ::testing::Values(ActivationFunctionType_NONE,
                                         ActivationFunctionType_RELU6)));

TEST(HybridBlockwiseFullyConnectedTest, HalfZeroDepthOutputsBias) {
  BlockwiseFcSpec spec = MakeHybridSpec();
  spec.input_type = TensorType_FLOAT16;
  spec.output_type = TensorType_FLOAT16;
  spec.input_shape = {2, 0};
  spec.filter_type = TensorType_INT2;
  spec.filter_shape = {2, 0};
  spec.block_shape = {1, 4};
  spec.scales = MakeParams(TensorType_FLOAT32, {2, 0}, std::vector<float>{});
  spec.bias = BiasSpec{TensorType_FLOAT16, HalfBytes({1.5f, -2.25f}), {}, {}};
  spec.output_shape = {2, 2};
  for (Resolver resolver : kAllResolvers) {
    SCOPED_TRACE(static_cast<int>(resolver));
    BlockwiseFullyConnectedModel m(BuildBlockwiseFullyConnectedModel(spec),
                                   resolver);
    ASSERT_TRUE(m.Built());
    ASSERT_EQ(m.Allocate(), kTfLiteOk);
    ASSERT_EQ(m.Invoke(), kTfLiteOk);
    EXPECT_THAT(m.GetHalfOutput(),
                ElementsAreArray({1.5f, -2.25f, 1.5f, -2.25f}));
  }
}

// A valid DRQ A8W2 spec that the rejection tests below break one way each.
BlockwiseFcSpec ValidHybridInt2Spec() {
  BlockwiseFcSpec spec = MakeHybridSpec();
  spec.input_shape = {1, 8};
  spec.filter_type = TensorType_INT2;
  spec.filter_shape = {2, 8};
  spec.filter_data = PackInt2(Pattern(16, -2, 1, 0));
  spec.block_shape = {1, 8};
  spec.scales =
      MakeParams(TensorType_FLOAT32, {2, 1}, std::vector<float>{0.5f, 0.25f});
  spec.zero_points =
      MakeParams(TensorType_FLOAT32, {1, 1}, std::vector<float>{-0.5f});
  spec.output_shape = {1, 2};
  return spec;
}

TEST(HybridBlockwiseFullyConnectedTest, ValidSpecAllocates) {
  EXPECT_EQ(AllocateSpec(ValidHybridInt2Spec()), kTfLiteOk);
}

TEST(HybridBlockwiseFullyConnectedTest, RejectsMixedInputAndOutputTypes) {
  BlockwiseFcSpec spec = ValidHybridInt2Spec();
  spec.output_type = TensorType_INT8;
  EXPECT_EQ(AllocateSpec(spec), kTfLiteError);

  spec = ValidInt2Spec();
  spec.output_type = TensorType_FLOAT32;
  EXPECT_EQ(AllocateSpec(spec), kTfLiteError);
}

// A float16 op may keep a float32 bias, e.g. when float16 casts around a
// float32 op are folded into it.
TEST(HybridBlockwiseFullyConnectedTest, HalfWithFloatBias) {
  constexpr int kUnits = 2;
  constexpr int kDepth = 8;
  const std::vector<float> input = RoundToHalf(FloatPattern(kDepth, 2.0f, 13));
  const std::vector<int32_t> filter_q = Pattern(kUnits * kDepth, -2, 1, 14);
  const BlockTable scales = {{0.5f, 0.25f}, {kUnits, 1}};
  const BlockTable zero_points = {{-0.5f}, {1, 1}};
  const std::vector<float> bias = {0.1f, -0.3f};

  BlockwiseFcSpec spec = MakeHybridSpec();
  spec.input_type = TensorType_FLOAT16;
  spec.output_type = TensorType_FLOAT16;
  spec.input_shape = {1, kDepth};
  spec.filter_type = TensorType_INT2;
  spec.filter_shape = {kUnits, kDepth};
  spec.filter_data = PackInt2(filter_q);
  spec.block_shape = {1, kDepth};
  spec.scales = MakeParams(TensorType_FLOAT32, {kUnits, 1}, scales.values);
  spec.zero_points = MakeParams(TensorType_FLOAT32, {1, 1}, zero_points.values);
  spec.bias = BiasSpec{TensorType_FLOAT32, ToBytes(bias), {}, {}};
  spec.output_shape = {1, kUnits};

  const std::vector<double> filter = DequantizeFilter(
      filter_q, kUnits, kDepth, 1, kDepth, scales, zero_points);
  const std::vector<float> expected = RoundToHalf(
      ReferenceHybridFullyConnected(input, filter, bias, 1, kUnits, kDepth));
  BlockwiseFullyConnectedModel m(BuildBlockwiseFullyConnectedModel(spec));
  ASSERT_TRUE(m.Built());
  ASSERT_EQ(m.Allocate(), kTfLiteOk);
  m.SetHalfInput(input);
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetHalfOutput(), Pointwise(FloatEq(), expected));
}

TEST(HybridBlockwiseFullyConnectedTest, HalfSpecAllocates) {
  BlockwiseFcSpec spec = ValidHybridInt2Spec();
  spec.input_type = TensorType_FLOAT16;
  spec.output_type = TensorType_FLOAT16;
  spec.bias = BiasSpec{TensorType_FLOAT16, HalfBytes({1.0f, 2.0f}), {}, {}};
  EXPECT_EQ(AllocateSpec(spec), kTfLiteOk);
}

TEST(HybridBlockwiseFullyConnectedTest, RejectsMixedFloatTypes) {
  // Float16 input with a float32 output.
  BlockwiseFcSpec spec = ValidHybridInt2Spec();
  spec.input_type = TensorType_FLOAT16;
  EXPECT_EQ(AllocateSpec(spec), kTfLiteError);

  // Float32 input and output with a float16 bias.
  spec = ValidHybridInt2Spec();
  spec.bias = BiasSpec{TensorType_FLOAT16, HalfBytes({1.0f, 2.0f}), {}, {}};
  EXPECT_EQ(AllocateSpec(spec), kTfLiteError);
}

// `asymmetric_quantize_inputs` quantizes each input row asymmetrically with its
// own zero point, which the blockwise accumulation subtracts. Covers float32
// and float16, a non-negative row (zero point at the bottom of the range), a
// mixed-sign row and an all-zero row.
class HybridAsymmetricInputsTest : public ::testing::TestWithParam<bool> {};

TEST_P(HybridAsymmetricInputsTest, MatchesReference) {
  const bool half = GetParam();
  constexpr int kBatches = 3;
  constexpr int kUnits = 4;
  constexpr int kDepth = 16;
  std::vector<float> input = FloatPattern(kBatches * kDepth, 2.0f, 11);
  // Row 0 is non-negative and shifted, row 1 is all zeros.
  for (int k = 0; k < kDepth; ++k) input[k] = std::abs(input[k]) + 0.5f;
  std::fill_n(input.begin() + kDepth, kDepth, 0.0f);
  if (half) input = RoundToHalf(input);
  const std::vector<int32_t> filter_q = Pattern(kUnits * kDepth, -2, 1, 12);
  const BlockTable scales = {
      {0.25f, 0.5f, 1.0f, 0.125f, 0.5f, 0.25f, 0.125f, 1.0f}, {kUnits, 2}};
  const BlockTable zero_points = {{-0.5f}, {1, 1}};
  std::vector<float> bias = {0.75f, -1.25f, 0.5f, 3.0f};

  BlockwiseFcSpec spec = MakeHybridSpec();
  if (half) {
    spec.input_type = TensorType_FLOAT16;
    spec.output_type = TensorType_FLOAT16;
    spec.bias = BiasSpec{TensorType_FLOAT16, HalfBytes(bias), {}, {}};
  } else {
    spec.bias = BiasSpec{TensorType_FLOAT32, ToBytes(bias), {}, {}};
  }
  spec.asymmetric_quantize_inputs = true;
  spec.input_shape = {kBatches, kDepth};
  spec.filter_type = TensorType_INT2;
  spec.filter_shape = {kUnits, kDepth};
  spec.filter_data = PackInt2(filter_q);
  spec.block_shape = {1, 8};
  spec.scales = MakeParams(TensorType_FLOAT32, {kUnits, 2}, scales.values);
  spec.zero_points = MakeParams(TensorType_FLOAT32, {1, 1}, zero_points.values);
  spec.output_shape = {kBatches, kUnits};

  const std::vector<double> filter =
      DequantizeFilter(filter_q, kUnits, kDepth, 1, 8, scales, zero_points);
  std::vector<float> expected = ReferenceHybridFullyConnected(
      input, filter, bias, kBatches, kUnits, kDepth,
      std::numeric_limits<float>::lowest(), std::numeric_limits<float>::max(),
      /*asymmetric=*/true);
  if (half) expected = RoundToHalf(expected);
  // The all-zero row outputs exactly the bias.
  for (int n = 0; n < kUnits; ++n) EXPECT_EQ(expected[kUnits + n], bias[n]);
  // The asymmetric scheme differs from the symmetric one on these inputs.
  EXPECT_NE(expected, half
                          ? RoundToHalf(ReferenceHybridFullyConnected(
                                input, filter, bias, kBatches, kUnits, kDepth))
                          : ReferenceHybridFullyConnected(
                                input, filter, bias, kBatches, kUnits, kDepth));

  for (Resolver resolver : kAllResolvers) {
    SCOPED_TRACE(static_cast<int>(resolver));
    BlockwiseFullyConnectedModel m(BuildBlockwiseFullyConnectedModel(spec),
                                   resolver);
    ASSERT_TRUE(m.Built());
    ASSERT_EQ(m.Allocate(), kTfLiteOk);
    if (half) {
      m.SetHalfInput(input);
    } else {
      m.SetFloatInput(input);
    }
    ASSERT_EQ(m.Invoke(), kTfLiteOk);
    EXPECT_THAT(half ? m.GetHalfOutput() : m.GetFloatOutput(),
                Pointwise(FloatEq(), expected));
  }
}

INSTANTIATE_TEST_SUITE_P(FloatAndHalf, HybridAsymmetricInputsTest,
                         ::testing::Bool());

TEST(HybridBlockwiseFullyConnectedTest, AsymmetricZeroDepthOutputsBias) {
  BlockwiseFcSpec spec = MakeHybridSpec();
  spec.asymmetric_quantize_inputs = true;
  spec.input_shape = {2, 0};
  spec.filter_type = TensorType_INT2;
  spec.filter_shape = {2, 0};
  spec.block_shape = {1, 4};
  spec.scales = MakeParams(TensorType_FLOAT32, {2, 0}, std::vector<float>{});
  spec.bias = BiasSpec{
      TensorType_FLOAT32, ToBytes(std::vector<float>{1.5f, -2.25f}), {}, {}};
  spec.output_shape = {2, 2};
  BlockwiseFullyConnectedModel m(BuildBlockwiseFullyConnectedModel(spec));
  ASSERT_TRUE(m.Built());
  ASSERT_EQ(m.Allocate(), kTfLiteOk);
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetFloatOutput(),
              ElementsAreArray({1.5f, -2.25f, 1.5f, -2.25f}));
}

TEST(HybridBlockwiseFullyConnectedTest, RejectsInt32Bias) {
  BlockwiseFcSpec spec = ValidHybridInt2Spec();
  spec.bias = BiasSpec{
      TensorType_INT32, ToBytes(std::vector<int32_t>{1, 2}), {0.25f}, {0}};
  EXPECT_EQ(AllocateSpec(spec), kTfLiteError);
}

TEST(HybridBlockwiseFullyConnectedTest, RejectsTanhActivation) {
  BlockwiseFcSpec spec = ValidHybridInt2Spec();
  spec.activation = ActivationFunctionType_TANH;
  EXPECT_EQ(AllocateSpec(spec), kTfLiteError);
}

TEST(HybridBlockwiseFullyConnectedTest, RejectsA4W2QuantSpec) {
  flexbuffers::Builder fbb;
  const size_t map_start = fbb.StartMap();
  fbb.String("spec", "cint2_fp32_int4_e8m0_drq");
  fbb.Double("act_dilation", 1.5);
  fbb.EndMap(map_start);
  fbb.Finish();

  BlockwiseFcSpec spec = ValidHybridInt2Spec();
  spec.quant_spec = fbb.GetBuffer();
  EXPECT_EQ(AllocateSpec(spec), kTfLiteError);
}

}  // namespace
}  // namespace tflite
