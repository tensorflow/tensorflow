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

#ifndef TENSORFLOW_COMPILER_XLA_CLIENT_LIB_QUANTIZE_H_
#define TENSORFLOW_COMPILER_XLA_CLIENT_LIB_QUANTIZE_H_

#include <limits>
#include <numeric>
#include <vector>

#include "tensorflow/compiler/xla/client/lib/constants.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/lib/bfloat16/bfloat16.h"

namespace xla {

constexpr int64 kBitsOfByte = 8;

// Represents the range used for quantization
struct QuantizedRange {
  QuantizedRange() = default;
  QuantizedRange(float min_in, float max_in) : min(min_in), max(max_in) {}

  bool operator==(const QuantizedRange& rhs) const {
    return this->min == rhs.min && this->max == rhs.max;
  }

  bool operator!=(const QuantizedRange& rhs) const { return !(*this == rhs); }

  tensorflow::bfloat16 min = tensorflow::bfloat16(0.0f);
  tensorflow::bfloat16 max = tensorflow::bfloat16(0.0f);
};

template <typename T>
inline std::vector<uint32> PackToUint32(absl::Span<const T> input) {
  const int64 kElementsPerPack = sizeof(uint32) / sizeof(T);
  const int64 input_size = input.size();
  const int64 output_size = CeilOfRatio(input_size, kElementsPerPack);

  std::vector<uint32> output_vec;
  constexpr int64 kShiftBits = sizeof(T) / sizeof(uint8) * kBitsOfByte;

  for (int64 i = 0; i < output_size; i++) {
    uint32 result = 0;
    for (int64 p = 0; p < kElementsPerPack; p++) {
      int64 index = i * kElementsPerPack + p;
      if (index < input_size) {
        int64 total_shift_bits = kShiftBits * (kElementsPerPack - p - 1);
        result |= (input[index] << total_shift_bits);
      }
    }
    output_vec.push_back(result);
  }

  return output_vec;
}

// Dequantize the quantized input of packed uint32 to bfloat16.
// Only uint8 or uint16 is supported for the original unpacked input.
// Returns a tensor of shape [d0,..., dn * unpack_size] if
// input shape is [d0, ..., dn], where unpack_size = sizeof(unit32) / sizeof(T).
template <typename T>
inline XlaOp Dequantize(XlaOp input, const QuantizedRange& range,
                        absl::string_view mode_string = "MIN_COMBINED") {
  XlaBuilder* const builder = input.builder();
  return builder->ReportErrorOrReturn([&]() -> StatusOr<XlaOp> {
    float half_range =
        !std::is_signed<T>::value
            ? 0.0f
            : (static_cast<float>(std::numeric_limits<T>::max()) -
               std::numeric_limits<T>::min() + 1) /
                  2.0f;
    const int64 unpack_size = sizeof(uint32) / sizeof(T);
    TF_ASSIGN_OR_RETURN(Shape shape, builder->GetShape(input));

    auto element_type = shape.element_type();
    if (element_type != U32) {
      return InvalidArgument(
          "Only U32 is supported for input type of xla::Dequantize Op.");
    }

    auto broadcast_size = shape.dimensions();
    broadcast_size.push_back(unpack_size);
    std::vector<int64> broadcast_dimensions(shape.dimensions_size());
    std::iota(broadcast_dimensions.begin(), broadcast_dimensions.end(), 0);
    // Broadcast the input to [d0, ..., dn, unpack_size] if input size is
    // [d0, ..., dn].
    auto broadcast_input =
        BroadcastInDim(input, broadcast_size, broadcast_dimensions);

    XlaOp iota_r1 = Iota(builder, U32, unpack_size);
    // Highest significant bytes needs to shift more bytes than lower
    // significant bytes.
    XlaOp shift_bytes =
        xla::ConstantR0<uint32>(builder, unpack_size - 1) - iota_r1;

    const int bytes_of_type = sizeof(T) / sizeof(uint8);
    XlaOp shift_bits = shift_bytes * xla::ConstantR0<uint32>(
                                         builder, kBitsOfByte * bytes_of_type);

    // Make bit_mask for different data type T.
    uint32 bit_mask = 0x00000000;
    for (int i = 0; i < bytes_of_type; i++) {
      bit_mask <<= kBitsOfByte;
      bit_mask |= 0x000000ff;
    }

    // Shift the input by sizeof(T) bytes and apply bit_mask to unpack.
    XlaOp shifted_input = ShiftRightLogical(
        broadcast_input, Broadcast(shift_bits, shape.dimensions()));
    XlaOp unpack_input =
        And(shifted_input, xla::ConstantR0<uint32>(builder, bit_mask));

    XlaOp result;

    if (mode_string == "MIN_COMBINED") {
      const tensorflow::bfloat16 scale_factor =
          (range.max - range.min) /
          (static_cast<tensorflow::bfloat16>(std::numeric_limits<T>::max() -
                                             std::numeric_limits<T>::min()));
      // result = bfloat16(input + half_range) * scale_factor + range.min
      XlaOp unpack_input_bf16 = ConvertElementType(unpack_input, BF16);
      XlaOp half_range_bf16 = xla::ConstantR0<tensorflow::bfloat16>(
          builder, static_cast<bfloat16>(half_range));
      XlaOp sum = unpack_input_bf16 + half_range_bf16;

      result =
          sum * xla::ConstantR0<tensorflow::bfloat16>(builder, scale_factor) +
          xla::ConstantR0<tensorflow::bfloat16>(builder, range.min);
    } else {
      // TODO(wangtao): support other modes.
      return InvalidArgument(
          "Only MIN_COMBINED mode is supported in xla::Dequantize Op.");
    }

    // Reshape the result to [d0,..., dn * unpack_size] if
    // input shape is [d0, ..., dn].
    std::vector<int64> result_shape(shape.dimensions());
    result_shape[shape.dimensions_size() - 1] =
        shape.dimensions(shape.dimensions_size() - 1) * unpack_size;
    return Reshape(result, result_shape);
  });
}

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_CLIENT_LIB_QUANTIZE_H_
