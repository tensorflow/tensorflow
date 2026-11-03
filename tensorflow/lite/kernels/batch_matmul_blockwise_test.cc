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

// Tests BATCH_MATMUL with an int8 per-tensor quantized LHS, int8 per-tensor
// quantized output, and N-D blockwise-quantized RHS (`BlockwiseQuantization`
// with `block_shape`). Models are built as flatbuffers to exercise flatbuffer
// parsing, delegate fallback, and kernel execution end to end.

#include <algorithm>
#include <cmath>
#include <cstddef>
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
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {
namespace {

using ::testing::ElementsAreArray;

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
  halves.reserve(values.size());
  for (float value : values) {
    halves.push_back(Eigen::numext::bit_cast<uint16_t>(Eigen::half(value)));
  }
  return MakeParams(TensorType_FLOAT16, std::move(shape), halves);
}

// Packs signed 2-bit values [-2, 1] four per byte, least-significant bits
// first.
std::vector<uint8_t> PackInt2(const std::vector<int8_t>& values) {
  std::vector<uint8_t> packed((values.size() + 3) / 4, 0);
  for (size_t i = 0; i < values.size(); ++i) {
    const uint8_t two_bits = static_cast<uint8_t>(values[i]) & 0x03;
    packed[i / 4] |= static_cast<uint8_t>(two_bits << ((i % 4) * 2));
  }
  return packed;
}

// Packs signed 4-bit values [-8, 7] two per byte, low nibble first.
std::vector<uint8_t> PackInt4(const std::vector<int8_t>& values) {
  std::vector<uint8_t> packed((values.size() + 1) / 2, 0);
  for (size_t i = 0; i < values.size(); ++i) {
    const uint8_t nibble = static_cast<uint8_t>(values[i]) & 0x0F;
    packed[i / 2] |= (i % 2 == 0) ? nibble : static_cast<uint8_t>(nibble << 4);
  }
  return packed;
}

// Packs unsigned 4-bit values [0, 15] two per byte, low nibble first.
std::vector<uint8_t> PackUInt4(const std::vector<uint8_t>& values) {
  std::vector<uint8_t> packed((values.size() + 1) / 2, 0);
  for (size_t i = 0; i < values.size(); ++i) {
    const uint8_t nibble = values[i] & 0x0F;
    packed[i / 2] |= (i % 2 == 0) ? nibble : static_cast<uint8_t>(nibble << 4);
  }
  return packed;
}

std::vector<uint8_t> AsBytes(const std::vector<int8_t>& values) {
  return std::vector<uint8_t>(values.begin(), values.end());
}

struct AffineParams {
  float scale = 1.0f;
  int64_t zero_point = 0;
};

struct BlockwiseBmmSpec {
  std::vector<int32_t> lhs_shape;
  AffineParams lhs_quant;
  TensorType rhs_type = TensorType_INT2;
  std::vector<int32_t> rhs_shape;
  std::vector<uint8_t> rhs_data;
  std::vector<int32_t> block_shape;
  ParamsTensor scales;
  bool scales_as_rw_tensor = false;
  std::optional<ParamsTensor> zero_points;
  std::vector<int32_t> output_shape;
  AffineParams output_quant;
  bool adj_x = false;
  bool adj_y = false;
};

std::vector<uint8_t> BuildBlockwiseBatchMatMulModel(
    const BlockwiseBmmSpec& spec) {
  flatbuffers::FlatBufferBuilder builder;
  std::vector<flatbuffers::Offset<Buffer>> buffers = {
      CreateBuffer(builder, builder.CreateVector(std::vector<uint8_t>{}))};
  auto add_buffer = [&](const std::vector<uint8_t>& data) {
    buffers.push_back(CreateBuffer(builder, builder.CreateVector(data)));
    return static_cast<uint32_t>(buffers.size() - 1);
  };
  auto make_affine_quant = [&](const AffineParams& q) {
    return CreateQuantizationParameters(
        builder, /*min=*/0, /*max=*/0,
        builder.CreateVector(std::vector<float>{q.scale}),
        builder.CreateVector(std::vector<int64_t>{q.zero_point}));
  };

  // Tensors:
  // 0: lhs (int8)
  // 1: rhs (blockwise quantized constant)
  // 2: rhs scales
  // 3: rhs zero_points (if present)
  // last: output (int8)
  const int32_t scales_index = 2;
  const int32_t zp_index = spec.zero_points.has_value() ? 3 : -1;
  const int32_t output_index = spec.zero_points.has_value() ? 4 : 3;

  auto rhs_details = CreateBlockwiseQuantization(
      builder, scales_index, zp_index, /*block_size=*/0,
      builder.CreateVector(spec.block_shape));
  auto rhs_quant = CreateQuantizationParameters(
      builder, /*min=*/0, /*max=*/0, /*scale=*/0, /*zero_point=*/0,
      QuantizationDetails_BlockwiseQuantization, rhs_details.Union());

  std::vector<flatbuffers::Offset<Tensor>> tensors;
  tensors.push_back(CreateTensor(builder, builder.CreateVector(spec.lhs_shape),
                                 TensorType_INT8,
                                 /*buffer=*/0, builder.CreateString("lhs"),
                                 make_affine_quant(spec.lhs_quant)));
  tensors.push_back(CreateTensor(builder, builder.CreateVector(spec.rhs_shape),
                                 spec.rhs_type, add_buffer(spec.rhs_data),
                                 builder.CreateString("rhs"), rhs_quant));
  tensors.push_back(CreateTensor(
      builder, builder.CreateVector(spec.scales.shape), spec.scales.type,
      spec.scales_as_rw_tensor ? 0u : add_buffer(spec.scales.data),
      builder.CreateString("rhs_scales")));
  if (spec.zero_points.has_value()) {
    tensors.push_back(
        CreateTensor(builder, builder.CreateVector(spec.zero_points->shape),
                     spec.zero_points->type, add_buffer(spec.zero_points->data),
                     builder.CreateString("rhs_zero_points")));
  }
  tensors.push_back(CreateTensor(
      builder, builder.CreateVector(spec.output_shape), TensorType_INT8,
      /*buffer=*/0, builder.CreateString("output"),
      make_affine_quant(spec.output_quant)));

  auto bmm_options =
      CreateBatchMatMulOptions(builder, spec.adj_x, spec.adj_y,
                               /*asymmetric_quantize_inputs=*/false);
  auto op =
      CreateOperator(builder, /*opcode_index=*/0,
                     builder.CreateVector(std::vector<int32_t>{0, 1}),
                     builder.CreateVector(std::vector<int32_t>{output_index}),
                     BuiltinOptions_BatchMatMulOptions, bmm_options.Union());
  auto subgraph = CreateSubGraph(
      builder, builder.CreateVector(tensors),
      builder.CreateVector(std::vector<int32_t>{0}),
      builder.CreateVector(std::vector<int32_t>{output_index}),
      builder.CreateVector(std::vector<flatbuffers::Offset<Operator>>{op}));
  auto op_code = CreateOperatorCode(
      builder, /*deprecated_builtin_code=*/BuiltinOperator_BATCH_MATMUL,
      /*custom_code=*/0, /*version=*/5, BuiltinOperator_BATCH_MATMUL);
  auto model =
      CreateModel(builder, /*version=*/3, builder.CreateVector({op_code}),
                  builder.CreateVector({subgraph}),
                  builder.CreateString("test"), builder.CreateVector(buffers));
  FinishModelBuffer(builder, model);
  return std::vector<uint8_t>(builder.GetBufferPointer(),
                              builder.GetBufferPointer() + builder.GetSize());
}

class BlockwiseBatchMatMulModel {
 public:
  explicit BlockwiseBatchMatMulModel(std::vector<uint8_t> buffer,
                                     bool use_default_delegates = false)
      : buffer_(std::move(buffer)) {
    model_ = FlatBufferModel::BuildFromBuffer(
        reinterpret_cast<const char*>(buffer_.data()), buffer_.size());
    if (use_default_delegates) {
      ops::builtin::BuiltinOpResolver resolver;
      InterpreterBuilder(*model_, resolver)(&interpreter_);
    } else {
      ops::builtin::BuiltinOpResolverWithoutDefaultDelegates resolver;
      InterpreterBuilder(*model_, resolver)(&interpreter_);
    }
  }

  bool Built() const { return interpreter_ != nullptr; }

  TfLiteStatus Allocate() { return interpreter_->AllocateTensors(); }
  TfLiteStatus Invoke() { return interpreter_->Invoke(); }

  void SetInput(const std::vector<int8_t>& values) {
    TfLiteTensor* input = interpreter_->tensor(interpreter_->inputs()[0]);
    std::memcpy(input->data.int8, values.data(),
                values.size() * sizeof(int8_t));
  }

  std::vector<int8_t> GetOutput() const {
    const TfLiteTensor* output =
        interpreter_->tensor(interpreter_->outputs()[0]);
    const int8_t* data = output->data.int8;
    return std::vector<int8_t>(data, data + NumElements(output));
  }

 private:
  std::vector<uint8_t> buffer_;
  std::unique_ptr<FlatBufferModel> model_;
  std::unique_ptr<Interpreter> interpreter_;
};

// Reference calculation on unpacked RHS for test oracle verification.
int8_t QuantizeRef(double acc_float, float out_scale, int32_t out_zp) {
  const double rounded =
      std::round(acc_float / static_cast<double>(out_scale)) + out_zp;
  return static_cast<int8_t>(
      static_cast<int32_t>(std::clamp(rounded, -128.0, 127.0)));
}

TEST(BlockwiseBatchMatMulTest, CenteredInt2BroadcastBatchAndTwoQuantDims) {
  // Rank-4 batched matmul with 3-D blockwise RHS:
  //   LHS: [N=2, K2=2, M=1, K=4] (int8, scale=0.5, zp=-2)
  //   RHS: [K2=2, K=4, Out=3] (int2, block_shape=[1, 4, 1], scales=[2, 1, 3],
  //        centered zp=-0.5 broadcast from [1, 1, 1]).
  //   Output: [2, 2, 1, 3] (int8, scale=0.25, zp=1).
  const std::vector<int8_t> lhs = {
      // n=0, k2=0: (x - zp) = [4, 0, -2, 6]
      2,
      -2,
      -4,
      4,
      // n=0, k2=1: (x - zp) = [2, 4, 2, 0]
      0,
      2,
      0,
      -2,
      // n=1, k2=0: (x - zp) = [-4, 2, 2, 0]
      -6,
      0,
      0,
      -2,
      // n=1, k2=1: (x - zp) = [6, -2, 4, 2]
      4,
      -4,
      2,
      0,
  };
  // RHS [K2=2, K=4, Out=3] in [-2, 1]; dequantized value is (q + 0.5) * scale.
  const std::vector<int8_t> rhs_q = {
      // k2=0 (K=4 x Out=3):
      -2,
      -1,
      0,
      1,
      -2,
      -1,
      0,
      1,
      -2,
      1,
      0,
      1,
      // k2=1 (K=4 x Out=3):
      1,
      0,
      -1,
      0,
      1,
      -2,
      -1,
      -2,
      1,
      -2,
      1,
      0,
  };
  // Scales [2, 1, 3] in float16 to cover f16 scale tensors + f32 zero-point.
  const std::vector<float> scales = {
      // k2=0:
      0.5f,
      1.0f,
      1.5f,
      // k2=1:
      2.0f,
      0.25f,
      0.5f,
  };

  BlockwiseBmmSpec spec;
  spec.lhs_shape = {2, 2, 1, 4};
  spec.lhs_quant = {/*scale=*/0.5f, /*zero_point=*/-2};
  spec.rhs_type = TensorType_INT2;
  spec.rhs_shape = {2, 4, 3};
  spec.rhs_data = PackInt2(rhs_q);
  spec.block_shape = {1, 4, 1};
  spec.scales = MakeF16Params({2, 1, 3}, scales);
  spec.zero_points = MakeParams<float>(TensorType_FLOAT32, {1, 1, 1}, {-0.5f});
  spec.output_shape = {2, 2, 1, 3};
  spec.output_quant = {/*scale=*/0.25f, /*zero_point=*/1};

  std::vector<int8_t> expected;
  expected.reserve(2 * 2 * 3);
  for (int n = 0; n < 2; ++n) {
    for (int k2 = 0; k2 < 2; ++k2) {
      for (int out = 0; out < 3; ++out) {
        double acc = 0.0;
        const double s_w = scales[k2 * 3 + out];
        for (int k = 0; k < 4; ++k) {
          const double x_f =
              0.5 * (static_cast<int32_t>(lhs[(n * 2 + k2) * 4 + k]) - (-2));
          const double w_f =
              s_w * (static_cast<double>(rhs_q[(k2 * 4 + k) * 3 + out]) + 0.5);
          acc += x_f * w_f;
        }
        expected.push_back(QuantizeRef(acc, /*out_scale=*/0.25f, /*out_zp=*/1));
      }
    }
  }

  for (bool use_default_delegates : {false, true}) {
    BlockwiseBatchMatMulModel m(BuildBlockwiseBatchMatMulModel(spec),
                                use_default_delegates);
    ASSERT_TRUE(m.Built());
    ASSERT_EQ(m.Allocate(), kTfLiteOk);
    m.SetInput(lhs);
    ASSERT_EQ(m.Invoke(), kTfLiteOk);
    EXPECT_THAT(m.GetOutput(), ElementsAreArray(expected));
  }
}

TEST(BlockwiseBatchMatMulTest, Int4SubChannelBlocksAndSymmetricNoZeroPoints) {
  // Rank-4 A8W4 batched matmul with sub-channel blocks along K:
  //   LHS: [1, 2, 2, 4] (int8, scale=0.25, zp=0)
  //   RHS: [2, 4, 2] (int4, block_shape=[1, 2, 1], scales=[2, 2, 2], no zp)
  //   Output: [1, 2, 2, 2] (int8, scale=0.5, zp=0).
  const std::vector<int8_t> lhs = {
      // k2=0, m=0..1:
      2,
      4,
      -2,
      -4,
      1,
      -1,
      3,
      -3,
      // k2=1, m=0..1:
      4,
      0,
      2,
      2,
      -2,
      2,
      -4,
      4,
  };
  const std::vector<int8_t> rhs_q = {
      // k2=0 (K=4 x Out=2):
      -3,
      2,
      1,
      -4,
      5,
      -1,
      -2,
      3,
      // k2=1 (K=4 x Out=2):
      2,
      -5,
      4,
      1,
      -6,
      2,
      3,
      -2,
  };
  // Scales [2, 2, 2]: (k2, kb, out).
  const std::vector<float> scales = {
      // k2=0, kb=0:
      0.5f,
      1.0f,
      // k2=0, kb=1:
      1.5f,
      0.25f,
      // k2=1, kb=0:
      0.25f,
      0.5f,
      // k2=1, kb=1:
      1.0f,
      2.0f,
  };

  BlockwiseBmmSpec spec;
  spec.lhs_shape = {1, 2, 2, 4};
  spec.lhs_quant = {/*scale=*/0.25f, /*zero_point=*/0};
  spec.rhs_type = TensorType_INT4;
  spec.rhs_shape = {2, 4, 2};
  spec.rhs_data = PackInt4(rhs_q);
  spec.block_shape = {1, 2, 1};
  spec.scales = MakeParams<float>(TensorType_FLOAT32, {2, 2, 2}, scales);
  spec.zero_points = std::nullopt;
  spec.output_shape = {1, 2, 2, 2};
  spec.output_quant = {/*scale=*/0.5f, /*zero_point=*/0};

  std::vector<int8_t> expected;
  for (int k2 = 0; k2 < 2; ++k2) {
    for (int m = 0; m < 2; ++m) {
      for (int out = 0; out < 2; ++out) {
        double acc = 0.0;
        for (int k = 0; k < 4; ++k) {
          const int kb = k / 2;
          const double s_w = scales[(k2 * 2 + kb) * 2 + out];
          const double x_f = 0.25 * lhs[(k2 * 2 + m) * 4 + k];
          const double w_f = s_w * rhs_q[(k2 * 4 + k) * 2 + out];
          acc += x_f * w_f;
        }
        expected.push_back(QuantizeRef(acc, /*out_scale=*/0.5f, /*out_zp=*/0));
      }
    }
  }

  for (bool use_default_delegates : {false, true}) {
    BlockwiseBatchMatMulModel m(BuildBlockwiseBatchMatMulModel(spec),
                                use_default_delegates);
    ASSERT_TRUE(m.Built());
    ASSERT_EQ(m.Allocate(), kTfLiteOk);
    m.SetInput(lhs);
    ASSERT_EQ(m.Invoke(), kTfLiteOk);
    EXPECT_THAT(m.GetOutput(), ElementsAreArray(expected));
  }
}

TEST(BlockwiseBatchMatMulTest, AdjointLhsAndAdjointRhsWithInt8Fallback) {
  // Exercises adj_x = true, adj_y = true, and int8 blockwise RHS (verifying
  // XNNPack delegate cleanly falls back to CPU when RHS has blockwise quant).
  //   LHS: [2, K=3, M=2] with adj_x=true -> effective [2, M=2, K=3]
  //   RHS: [2, N=2, K=3] with adj_y=true -> effective [2, K=3, N=2]
  //   block_shape on RHS [2, 2, 3] = [1, 1, 2] (partial block on K=3!).
  const std::vector<int8_t> lhs = {
      // b=0 (K=3 x M=2): m=0 is [1, 2, 3], m=1 is [-1, 0, 2]
      1,
      -1,
      2,
      0,
      3,
      2,
      // b=1 (K=3 x M=2): m=0 is [2, -2, 1], m=1 is [0, 3, -1]
      2,
      0,
      -2,
      3,
      1,
      -1,
  };
  const std::vector<int8_t> rhs_q = {
      // b=0 (N=2 x K=3):
      2,
      -1,
      3,
      -2,
      4,
      1,
      // b=1 (N=2 x K=3):
      1,
      2,
      -3,
      3,
      -1,
      2,
  };
  // Scales shape: [2, 2, 2] (b, n, kb where K=3 with block 2 -> 2 blocks).
  const std::vector<float> scales = {
      // b=0, n=0: kb=0,1
      0.5f,
      1.0f,
      // b=0, n=1: kb=0,1
      1.5f,
      0.5f,
      // b=1, n=0: kb=0,1
      1.0f,
      2.0f,
      // b=1, n=1: kb=0,1
      0.25f,
      1.0f,
  };

  BlockwiseBmmSpec spec;
  spec.lhs_shape = {2, 3, 2};
  spec.lhs_quant = {/*scale=*/1.0f, /*zero_point=*/0};
  spec.rhs_type = TensorType_INT8;
  spec.rhs_shape = {2, 2, 3};
  spec.rhs_data = AsBytes(rhs_q);
  spec.block_shape = {1, 1, 2};
  spec.scales = MakeParams<float>(TensorType_FLOAT32, {2, 2, 2}, scales);
  spec.zero_points = MakeParams<int32_t>(TensorType_INT32, {1, 1, 2}, {1, -1});
  spec.output_shape = {2, 2, 2};
  spec.output_quant = {/*scale=*/0.5f, /*zero_point=*/0};
  spec.adj_x = true;
  spec.adj_y = true;

  const int32_t zps[2] = {1, -1};
  std::vector<int8_t> expected;
  for (int b = 0; b < 2; ++b) {
    for (int m = 0; m < 2; ++m) {
      for (int n = 0; n < 2; ++n) {
        double acc = 0.0;
        for (int k = 0; k < 3; ++k) {
          const int kb = k / 2;
          const double x_f = lhs[(b * 3 + k) * 2 + m];
          const double s_w = scales[(b * 2 + n) * 2 + kb];
          const double w_f = s_w * (rhs_q[(b * 2 + n) * 3 + k] - zps[kb]);
          acc += x_f * w_f;
        }
        expected.push_back(QuantizeRef(acc, /*out_scale=*/0.5f, /*out_zp=*/0));
      }
    }
  }

  for (bool use_default_delegates : {false, true}) {
    BlockwiseBatchMatMulModel m(BuildBlockwiseBatchMatMulModel(spec),
                                use_default_delegates);
    ASSERT_TRUE(m.Built());
    ASSERT_EQ(m.Allocate(), kTfLiteOk);
    m.SetInput(lhs);
    ASSERT_EQ(m.Invoke(), kTfLiteOk);
    EXPECT_THAT(m.GetOutput(), ElementsAreArray(expected));
  }
}

TEST(BlockwiseBatchMatMulTest, RejectsMismatchedBlockShapeRank) {
  BlockwiseBmmSpec spec;
  spec.lhs_shape = {1, 2, 4};
  spec.lhs_quant = {/*scale=*/0.5f, /*zero_point=*/0};
  spec.rhs_type = TensorType_INT2;
  spec.rhs_shape = {1, 4, 2};
  spec.rhs_data = PackInt2({0, 1, -1, -2, 0, 1, -1, -2});
  // Rank-2 block_shape for a rank-3 RHS is rejected when building the subgraph.
  spec.block_shape = {4, 1};
  spec.scales = MakeParams<float>(TensorType_FLOAT32, {1, 1, 2}, {0.5f, 1.0f});
  spec.output_shape = {1, 2, 2};
  spec.output_quant = {/*scale=*/0.5f, /*zero_point=*/0};

  BlockwiseBatchMatMulModel m(BuildBlockwiseBatchMatMulModel(spec));
  EXPECT_FALSE(m.Built());
}

TEST(BlockwiseBatchMatMulTest, RejectsOutOfBoundsScalesExtentInPrepare) {
  BlockwiseBmmSpec spec;
  spec.lhs_shape = {1, 2, 4};
  spec.lhs_quant = {/*scale=*/0.5f, /*zero_point=*/0};
  spec.rhs_type = TensorType_INT2;
  spec.rhs_shape = {1, 4, 2};
  spec.rhs_data = PackInt2({0, 1, -1, -2, 0, 1, -1, -2});
  spec.block_shape = {1, 4, 1};
  // Extent 3 along output axis does not match num_blocks = 2.
  spec.scales =
      MakeParams<float>(TensorType_FLOAT32, {1, 1, 3}, {0.5f, 1.0f, 1.5f});
  spec.output_shape = {1, 2, 2};
  spec.output_quant = {/*scale=*/0.5f, /*zero_point=*/0};

  BlockwiseBatchMatMulModel m(BuildBlockwiseBatchMatMulModel(spec));
  ASSERT_TRUE(m.Built());
  EXPECT_EQ(m.Allocate(), kTfLiteError);
}

TEST(BlockwiseBatchMatMulTest, BatchAndOutputChannelBlocksWithUInt4) {
  // Exercises non-unit block sizes along batch and N axes (including partial
  // blocks: batch 3 with block 2 -> 2 blocks; K=4 with block 2 -> 2 blocks;
  // N=3 with block 2 -> 2 blocks) and unsigned 4-bit RHS unpacking [0, 15].
  const std::vector<int8_t> lhs = {
      // b=0 (1x4):
      1,
      -2,
      3,
      -1,
      // b=1 (1x4):
      2,
      1,
      -1,
      2,
      // b=2 (1x4):
      -1,
      3,
      2,
      -2,
  };
  const std::vector<uint8_t> rhs_u4 = {
      // b=0 (4x3):
      0,
      4,
      8,  //
      12,
      15,
      3,  //
      7,
      9,
      2,  //
      1,
      11,
      14,  //
      // b=1 (4x3):
      5,
      10,
      6,  //
      13,
      1,
      8,  //
      4,
      15,
      7,  //
      2,
      3,
      12,  //
      // b=2 (4x3):
      14,
      0,
      9,  //
      6,
      8,
      11,  //
      3,
      13,
      5,  //
      10,
      7,
      1,  //
  };
  // Scales shape: [2, 2, 2] for rhs_shape [3, 4, 3] with block_shape [2, 2, 2].
  const std::vector<float> scales = {
      0.25f, 0.5f,    // bb=0, kb=0
      0.5f,  0.25f,   // bb=0, kb=1
      0.5f,  0.125f,  // bb=1, kb=0
      0.25f, 0.5f,    // bb=1, kb=1
  };
  const std::vector<float> zps = {8.0f};

  BlockwiseBmmSpec spec;
  spec.lhs_shape = {3, 1, 4};
  spec.lhs_quant = {/*scale=*/0.5f, /*zero_point=*/0};
  spec.rhs_type = TensorType_UINT4;
  spec.rhs_shape = {3, 4, 3};
  spec.rhs_data = PackUInt4(rhs_u4);
  spec.block_shape = {2, 2, 2};
  spec.scales = MakeParams<float>(TensorType_FLOAT32, {2, 2, 2}, scales);
  spec.zero_points = MakeParams<float>(TensorType_FLOAT32, {1, 1, 1}, zps);
  spec.output_shape = {3, 1, 3};
  spec.output_quant = {/*scale=*/0.25f, /*zero_point=*/0};

  std::vector<int8_t> expected;
  for (int b = 0; b < 3; ++b) {
    const int bb = b / 2;
    for (int n = 0; n < 3; ++n) {
      const int nb = n / 2;
      double acc = 0.0;
      for (int k = 0; k < 4; ++k) {
        const int kb = k / 2;
        const double x_f = 0.5 * lhs[b * 4 + k];
        const double s_w = scales[(bb * 2 + kb) * 2 + nb];
        const double w_f =
            s_w * (static_cast<double>(rhs_u4[(b * 4 + k) * 3 + n]) - 8.0);
        acc += x_f * w_f;
      }
      expected.push_back(QuantizeRef(acc, /*out_scale=*/0.25f, /*out_zp=*/0));
    }
  }

  for (bool use_default_delegates : {false, true}) {
    BlockwiseBatchMatMulModel m(BuildBlockwiseBatchMatMulModel(spec),
                                use_default_delegates);
    ASSERT_TRUE(m.Built());
    ASSERT_EQ(m.Allocate(), kTfLiteOk);
    m.SetInput(lhs);
    ASSERT_EQ(m.Invoke(), kTfLiteOk);
    EXPECT_THAT(m.GetOutput(), ElementsAreArray(expected));
  }
}

TEST(BlockwiseBatchMatMulTest, UInt8AndInt16BlockwiseRhs) {
  const std::vector<int8_t> lhs = {2, -1, 3, -2};
  const std::vector<uint8_t> rhs_u8 = {
      130, 125,  //
      128, 132,  //
      126, 131,  //
      129, 124,  //
  };
  const std::vector<int16_t> rhs_i16 = {
      2,  -3,  //
      0,  4,   //
      -2, 3,   //
      1,  -4,  //
  };
  const std::vector<float> scales = {0.5f, 0.25f, 1.0f, 0.5f};

  std::vector<int8_t> expected;
  for (int n = 0; n < 2; ++n) {
    double acc = 0.0;
    for (int k = 0; k < 4; ++k) {
      const int kb = k / 2;
      const double x_f = 0.5 * lhs[k];
      const double s_w = scales[kb * 2 + n];
      acc += x_f * (s_w * rhs_i16[k * 2 + n]);
    }
    expected.push_back(QuantizeRef(acc, /*out_scale=*/0.25f, /*out_zp=*/0));
  }

  // Test UINT8 with zero_point = 128 (equivalent to rhs_i16).
  {
    BlockwiseBmmSpec spec;
    spec.lhs_shape = {1, 1, 4};
    spec.lhs_quant = {/*scale=*/0.5f, /*zero_point=*/0};
    spec.rhs_type = TensorType_UINT8;
    spec.rhs_shape = {1, 4, 2};
    spec.rhs_data = rhs_u8;
    spec.block_shape = {1, 2, 1};
    spec.scales = MakeParams<float>(TensorType_FLOAT32, {1, 2, 2}, scales);
    spec.zero_points = MakeParams<int32_t>(TensorType_INT32, {1, 1, 1}, {128});
    spec.output_shape = {1, 1, 2};
    spec.output_quant = {/*scale=*/0.25f, /*zero_point=*/0};

    BlockwiseBatchMatMulModel m(BuildBlockwiseBatchMatMulModel(spec));
    ASSERT_TRUE(m.Built());
    ASSERT_EQ(m.Allocate(), kTfLiteOk);
    m.SetInput(lhs);
    ASSERT_EQ(m.Invoke(), kTfLiteOk);
    EXPECT_THAT(m.GetOutput(), ElementsAreArray(expected));
  }

  // Test INT16 with implicit zero_point = 0.
  {
    BlockwiseBmmSpec spec;
    spec.lhs_shape = {1, 1, 4};
    spec.lhs_quant = {/*scale=*/0.5f, /*zero_point=*/0};
    spec.rhs_type = TensorType_INT16;
    spec.rhs_shape = {1, 4, 2};
    spec.rhs_data.resize(rhs_i16.size() * sizeof(int16_t));
    std::memcpy(spec.rhs_data.data(), rhs_i16.data(), spec.rhs_data.size());
    spec.block_shape = {1, 2, 1};
    spec.scales = MakeParams<float>(TensorType_FLOAT32, {1, 2, 2}, scales);
    spec.output_shape = {1, 1, 2};
    spec.output_quant = {/*scale=*/0.25f, /*zero_point=*/0};

    BlockwiseBatchMatMulModel m(BuildBlockwiseBatchMatMulModel(spec));
    ASSERT_TRUE(m.Built());
    ASSERT_EQ(m.Allocate(), kTfLiteOk);
    m.SetInput(lhs);
    ASSERT_EQ(m.Invoke(), kTfLiteOk);
    EXPECT_THAT(m.GetOutput(), ElementsAreArray(expected));
  }
}

TEST(BlockwiseBatchMatMulTest, ZeroContractingDimensionFillsOutputZeroPoint) {
  BlockwiseBmmSpec spec;
  spec.lhs_shape = {1, 2, 0};
  spec.lhs_quant = {/*scale=*/0.5f, /*zero_point=*/0};
  spec.rhs_type = TensorType_INT8;
  spec.rhs_shape = {1, 0, 3};
  spec.rhs_data = {};
  spec.block_shape = {1, 2, 1};
  spec.scales = MakeParams<float>(TensorType_FLOAT32, {1, 0, 3}, {});
  spec.output_shape = {1, 2, 3};
  spec.output_quant = {/*scale=*/0.5f, /*zero_point=*/-7};

  BlockwiseBatchMatMulModel m(BuildBlockwiseBatchMatMulModel(spec));
  ASSERT_TRUE(m.Built());
  ASSERT_EQ(m.Allocate(), kTfLiteOk);
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput(), ElementsAreArray(std::vector<int8_t>(6, -7)));
}

TEST(BlockwiseBatchMatMulTest, RejectsNonConstantScalesBuffer) {
  BlockwiseBmmSpec spec;
  spec.lhs_shape = {1, 1, 4};
  spec.lhs_quant = {/*scale=*/0.5f, /*zero_point=*/0};
  spec.rhs_type = TensorType_INT8;
  spec.rhs_shape = {1, 4, 2};
  spec.rhs_data = AsBytes({1, -1, 2, -2, 1, -1, 2, -2});
  spec.block_shape = {1, 4, 1};
  // Buffer 0 causes InterpreterBuilder to register rhs_scales as a read-write
  // arena tensor (not constant/persistent), which Invoke() must reject cleanly
  // rather than dereferencing a null pointer.
  spec.scales = ParamsTensor{TensorType_FLOAT32, {1, 1, 2}, /*data=*/{}};
  spec.scales_as_rw_tensor = true;
  spec.output_shape = {1, 1, 2};
  spec.output_quant = {/*scale=*/0.5f, /*zero_point=*/0};

  BlockwiseBatchMatMulModel m(BuildBlockwiseBatchMatMulModel(spec));
  ASSERT_TRUE(m.Built());
  ASSERT_EQ(m.Allocate(), kTfLiteOk);
  m.SetInput({1, 2, 3, 4});
  EXPECT_EQ(m.Invoke(), kTfLiteError);
}

}  // namespace
}  // namespace tflite
