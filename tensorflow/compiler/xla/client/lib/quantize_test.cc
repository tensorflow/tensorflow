/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/client/lib/quantize.h"

#include <limits>

#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/tests/client_library_test_base.h"
#include "tensorflow/compiler/xla/tests/test_macros.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/util.h"

namespace xla {
namespace {

using bfloat16 = tsl::bfloat16;

template <typename NativeT>
std::vector<NativeT> GenerateInput() {
  std::vector<NativeT> input;
  const auto n = std::numeric_limits<NativeT>::max();
  input.reserve(n);

  for (int64_t i = std::numeric_limits<NativeT>::min(); i < n; ++i) {
    input.push_back(static_cast<NativeT>(i));
  }

  return input;
}

template <typename NativeT>
Array2D<NativeT> GenerateLargeSizeInput(int num_columns, int num_rows) {
  Array2D<NativeT> input(num_columns, num_rows);

  input.FillRandom(6, 128);

  return input;
}

template <typename NativeT>
Array2D<uint32_t> PackLargeInput(Array2D<NativeT> &input) {
  const int64_t size_per_pack = sizeof(uint32_t) / sizeof(NativeT);
  int64_t width = input.width();

  int64_t padded_output_width = CeilOfRatio(width, size_per_pack);

  Array2D<uint32_t> pack_input(input.height(), padded_output_width);

  for (int h = 0; h < input.height(); h++) {
    std::vector<NativeT> input_row;
    input_row.reserve(width);
    for (int w = 0; w < width; w++) {
      input_row.push_back(input({h, w}));
    }

    auto pack_input_vec = PackToUint32<uint8_t>(input_row);

    for (int w = 0; w < padded_output_width; w++) {
      pack_input(h, w) = pack_input_vec[w];
    }
  }

  return pack_input;
}

template <typename NativeT>
Array2D<bfloat16> GenerateLargeSizeMinCombinedOutput(
    Array2D<NativeT> &input, const QuantizedRange &range,
    bool transpose_output = false) {
  const int64_t size_per_pack = sizeof(uint32_t) / sizeof(NativeT);
  int64_t width = input.width();

  int64_t padded_output_width =
      CeilOfRatio(width, size_per_pack) * size_per_pack;

  int64_t output_height;
  int64_t output_width;

  if (transpose_output) {
    output_height = padded_output_width;
    output_width = input.height();
  } else {
    output_height = input.height();
    output_width = padded_output_width;
  }

  Array2D<bfloat16> output(output_height, output_width, bfloat16(0.0));

  float half_range =
      !std::is_signed<NativeT>::value
          ? 0.0f
          : (static_cast<float>(std::numeric_limits<NativeT>::max() -
                                std::numeric_limits<NativeT>::min() + 1)) /
                2.0f;
  const bfloat16 scale_factor =
      (range.max - range.min) /
      (static_cast<bfloat16>(std::numeric_limits<NativeT>::max() -
                             std::numeric_limits<NativeT>::min()));

  for (int h = 0; h < input.height(); h++) {
    std::vector<NativeT> input_row;
    input_row.reserve(width);
    for (int w = 0; w < width; w++) {
      bfloat16 result =
          static_cast<bfloat16>(input(h, w) + half_range) * scale_factor +
          range.min;
      if (transpose_output) {
        output(w, h) = result;
      } else {
        output(h, w) = result;
      }
    }
  }

  return output;
}

template <typename NativeT>
std::vector<bfloat16> GenerateMinCombinedOutput(const QuantizedRange &range) {
  float half_range =
      !std::is_signed<NativeT>::value
          ? 0.0f
          : (static_cast<float>(std::numeric_limits<NativeT>::max() -
                                std::numeric_limits<NativeT>::min() + 1)) /
                2.0f;
  const bfloat16 scale_factor =
      (range.max - range.min) /
      (static_cast<bfloat16>(std::numeric_limits<NativeT>::max() -
                             std::numeric_limits<NativeT>::min()));
  std::vector<bfloat16> output;
  const auto n = std::numeric_limits<NativeT>::max();
  output.reserve(n);
  for (int64_t i = std::numeric_limits<NativeT>::min(); i < n; ++i) {
    bfloat16 result =
        static_cast<bfloat16>(i + half_range) * scale_factor + range.min;
    output.push_back(result);
  }

  const int64_t pack_size = sizeof(uint32_t) / sizeof(NativeT);
  const int64_t output_size = output.size();

  int64_t num_tailing_zeros =
      CeilOfRatio(output_size, pack_size) * pack_size - output_size;

  output.insert(output.end(), num_tailing_zeros, bfloat16(0.0));
  return output;
}

// TODO(wangtao): add a test to make sure this op is the inverse of the existing
// TF quantize op defined in: third_party/tensorflow/core/kernels/quantize_op.cc

using DequantizeTest = ClientLibraryTestBase;

TEST(PackTest, PackUint8ToUint32) {
  std::vector<uint8_t> input = {0xAB, 0x0B, 0x00, 0xF0, 0x01};
  auto output = PackToUint32<uint8_t>(input);
  EXPECT_THAT(output, ::testing::ElementsAre(0xAB0B00F0, 0x01000000));
}

TEST(PackTest, PackInt8ToUint32) {
  std::vector<int8_t> input = {static_cast<signed char>(0x81), 0x0B, 0x00, 0x20,
                               0x01};
  auto output = PackToUint32<int8_t>(input);
  EXPECT_THAT(output, ::testing::ElementsAre(0x810B0020, 0x01000000));
}

TEST(PackTest, PackUint8ToUint32PerfectSize) {
  std::vector<uint8_t> input = {3, 2, 1, 0};
  auto output = PackToUint32<uint8_t>(input);
  EXPECT_THAT(output, ::testing::ElementsAre(0x03020100));
}

XLA_TEST_F(DequantizeTest, MinCombinedUint16R1) {
  XlaBuilder builder(TestName());
  auto input = GenerateInput<uint16_t>();
  auto x = ConstantR1<uint32_t>(&builder, PackToUint32<uint16_t>(input));
  QuantizedRange range(0, 255.0f);
  xla::Dequantize<uint16_t>(x, range, "MIN_COMBINED");
  auto expected = GenerateMinCombinedOutput<uint16_t>(range);
  ComputeAndCompareR1<bfloat16>(&builder, expected, {});
}

XLA_TEST_F(DequantizeTest, MinCombinedUint8R1) {
  XlaBuilder builder(TestName());
  auto input = GenerateInput<uint8_t>();
  auto x = ConstantR1<uint32_t>(&builder, PackToUint32<uint8_t>(input));
  QuantizedRange range(0, 127.0f);
  xla::Dequantize<uint8_t>(x, range, "MIN_COMBINED");
  auto expected = GenerateMinCombinedOutput<uint8_t>(range);
  ComputeAndCompareR1<bfloat16>(&builder, expected, {});
}

XLA_TEST_F(DequantizeTest, MinCombinedUint8R2) {
  XlaBuilder builder(TestName());
  std::vector<std::vector<uint8_t>> input = {
      {0, 1, 2, 3},
      {4, 5, 6, 7},
      {8, 9, 10, 11},
      {12, 13, 16, 15},
  };
  auto x =
      ConstantR2<uint32_t>(&builder, {{PackToUint32<uint8_t>(input[0])[0]},
                                      {PackToUint32<uint8_t>(input[1])[0]},
                                      {PackToUint32<uint8_t>(input[2])[0]},
                                      {PackToUint32<uint8_t>(input[3])[0]}});
  QuantizedRange range(0, 255.0f);
  xla::Dequantize<uint8_t>(x, range, "MIN_COMBINED");
  const Array2D<bfloat16> expected = {
      {bfloat16(0.0), bfloat16(1.0), bfloat16(2.0), bfloat16(3.0)},
      {bfloat16(4.0), bfloat16(5.0), bfloat16(6.0), bfloat16(7.0)},
      {bfloat16(8.0), bfloat16(9.0), bfloat16(10.0), bfloat16(11.0)},
      {bfloat16(12.0), bfloat16(13.0), bfloat16(16.0), bfloat16(15.0)},
  };
  ComputeAndCompareR2<bfloat16>(&builder, expected, {});
}

XLA_TEST_F(DequantizeTest, MinCombinedUint8R2TransposeOutput) {
  XlaBuilder builder(TestName());
  std::vector<std::vector<uint8_t>> input = {
      {0, 1, 2, 3},
      {4, 5, 6, 7},
      {8, 9, 10, 11},
      {12, 13, 16, 15},
  };
  auto x =
      ConstantR2<uint32_t>(&builder, {{PackToUint32<uint8_t>(input[0])[0]},
                                      {PackToUint32<uint8_t>(input[1])[0]},
                                      {PackToUint32<uint8_t>(input[2])[0]},
                                      {PackToUint32<uint8_t>(input[3])[0]}});
  QuantizedRange range(0, 255.0f);
  xla::Dequantize<uint8_t>(x, range, "MIN_COMBINED", /*transpose_output=*/true);
  const Array2D<bfloat16> expected = {
      {bfloat16(0.0), bfloat16(4.0), bfloat16(8.0), bfloat16(12.0)},
      {bfloat16(1.0), bfloat16(5.0), bfloat16(9.0), bfloat16(13.0)},
      {bfloat16(2.0), bfloat16(6.0), bfloat16(10.0), bfloat16(16.0)},
      {bfloat16(3.0), bfloat16(7.0), bfloat16(11.0), bfloat16(15.0)},
  };
  ComputeAndCompareR2<bfloat16>(&builder, expected, {});
}

XLA_TEST_F(DequantizeTest, MinCombinedUint8R2TailingZero) {
  XlaBuilder builder(TestName());
  std::vector<std::vector<uint8_t>> input = {
      {0, 1, 2, 3, 16},
      {4, 5, 6, 7, 17},
      {8, 9, 10, 11, 18},
      {12, 13, 16, 15, 19},
  };
  auto x = ConstantR2<uint32_t>(
      &builder,
      {{PackToUint32<uint8_t>(input[0])[0], PackToUint32<uint8_t>(input[0])[1]},
       {PackToUint32<uint8_t>(input[1])[0], PackToUint32<uint8_t>(input[1])[1]},
       {PackToUint32<uint8_t>(input[2])[0], PackToUint32<uint8_t>(input[2])[1]},
       {PackToUint32<uint8_t>(input[3])[0],
        PackToUint32<uint8_t>(input[3])[1]}});
  QuantizedRange range(0, 255.0f);
  xla::Dequantize<uint8_t>(x, range, "MIN_COMBINED");

  const Array2D<bfloat16> expected = {
      {bfloat16(0.0), bfloat16(1.0), bfloat16(2.0), bfloat16(3.0),
       bfloat16(16.0), bfloat16(0.0), bfloat16(0.0), bfloat16(0.0)},
      {bfloat16(4.0), bfloat16(5.0), bfloat16(6.0), bfloat16(7.0),
       bfloat16(17.0), bfloat16(0.0), bfloat16(0.0), bfloat16(0.0)},
      {bfloat16(8.0), bfloat16(9.0), bfloat16(10.0), bfloat16(11.0),
       bfloat16(18.0), bfloat16(0.0), bfloat16(0.0), bfloat16(0.0)},
      {bfloat16(12.0), bfloat16(13.0), bfloat16(16.0), bfloat16(15.0),
       bfloat16(19.0), bfloat16(0.0), bfloat16(0.0), bfloat16(0.0)},
  };
  ComputeAndCompareR2<bfloat16>(&builder, expected, {});
}

XLA_TEST_F(DequantizeTest, MinCombinedUint8R2TailingZeroTransposeOutput) {
  XlaBuilder builder(TestName());
  std::vector<std::vector<uint8_t>> input = {
      {0, 1, 2, 3, 16},
      {4, 5, 6, 7, 17},
      {8, 9, 10, 11, 18},
      {12, 13, 16, 15, 19},
  };
  auto x = ConstantR2<uint32_t>(
      &builder,
      {{PackToUint32<uint8_t>(input[0])[0], PackToUint32<uint8_t>(input[0])[1]},
       {PackToUint32<uint8_t>(input[1])[0], PackToUint32<uint8_t>(input[1])[1]},
       {PackToUint32<uint8_t>(input[2])[0], PackToUint32<uint8_t>(input[2])[1]},
       {PackToUint32<uint8_t>(input[3])[0],
        PackToUint32<uint8_t>(input[3])[1]}});
  QuantizedRange range(0, 255.0f);
  xla::Dequantize<uint8_t>(x, range, "MIN_COMBINED", /*transpose_output=*/true);

  const Array2D<bfloat16> expected = {
      {bfloat16(0.0), bfloat16(4.0), bfloat16(8.0), bfloat16(12.0)},
      {bfloat16(1.0), bfloat16(5.0), bfloat16(9.0), bfloat16(13.0)},
      {bfloat16(2.0), bfloat16(6.0), bfloat16(10.0), bfloat16(16.0)},
      {bfloat16(3.0), bfloat16(7.0), bfloat16(11.0), bfloat16(15.0)},
      {bfloat16(16.0), bfloat16(17.0), bfloat16(18.0), bfloat16(19.0)},
      {bfloat16(0.0), bfloat16(0.0), bfloat16(0.0), bfloat16(0.0)},
      {bfloat16(0.0), bfloat16(0.0), bfloat16(0.0), bfloat16(0.0)},
      {bfloat16(0.0), bfloat16(0.0), bfloat16(0.0), bfloat16(0.0)},
  };
  ComputeAndCompareR2<bfloat16>(&builder, expected, {});
}

XLA_TEST_F(DequantizeTest, MinCombinedUint8LargeSizeTest) {
  XlaBuilder builder(TestName());
  Array2D<uint8_t> input = GenerateLargeSizeInput<uint8_t>(500, 3547);
  Array2D<uint32_t> input_packed = PackLargeInput<uint8_t>(input);

  auto x = ConstantR2FromArray2D<uint32_t>(&builder, input_packed);
  QuantizedRange range(0, 255.0f);
  xla::Dequantize<uint8_t>(x, range, "MIN_COMBINED");

  const Array2D<bfloat16> expected =
      GenerateLargeSizeMinCombinedOutput<uint8_t>(input, range);
  ComputeAndCompareR2<bfloat16>(&builder, expected, {});
}

XLA_TEST_F(DequantizeTest, MinCombinedUint8LargeSizeTestTransposeOutput) {
  XlaBuilder builder(TestName());
  Array2D<uint8_t> input = GenerateLargeSizeInput<uint8_t>(500, 3547);
  Array2D<uint32_t> input_packed = PackLargeInput<uint8_t>(input);

  auto x = ConstantR2FromArray2D<uint32_t>(&builder, input_packed);
  QuantizedRange range(0, 255.0f);
  xla::Dequantize<uint8_t>(x, range, "MIN_COMBINED", /*transpose_output=*/true);

  const Array2D<bfloat16> expected =
      GenerateLargeSizeMinCombinedOutput<uint8_t>(input, range,
                                                  /*transpose_output=*/true);
  ComputeAndCompareR2<bfloat16>(&builder, expected, {});
}

}  // namespace
}  // namespace xla
