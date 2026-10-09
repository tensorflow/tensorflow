/* Copyright 2026 The OpenXLA Authors.

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

#include "xla/packing.h"

#include <cstddef>
#include <cstdint>

#include "absl/types/span.h"
#include "xla/tsl/lib/math/math_util.h"
#include "xla/tsl/platform/logging.h"

// First undef to prevent error when re-included.
#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "xla/packing.cc"
#include "hwy//foreach_target.h"  // IWYU pragma: keep
#include "hwy//highway.h"

HWY_BEFORE_NAMESPACE();
namespace xla {
namespace HWY_NAMESPACE {

namespace hn = hwy::HWY_NAMESPACE;

void Pack4Impl(const uint8_t* HWY_RESTRICT in, size_t count,
               uint8_t* HWY_RESTRICT out) {
  const hn::ScalableTag<uint8_t> d8;
  const size_t lanes = hn::Lanes(d8);
  const auto mask_4 = hn::Set(d8, 0x0F);
  const size_t vector_stride = 2 * lanes;
  const size_t vector_inputs = (count / vector_stride) * vector_stride;

  for (size_t i = 0; i < vector_inputs; i += vector_stride) {
    hn::Vec<decltype(d8)> v0, v1;
    hn::LoadInterleaved2(d8, in + i, v0, v1);
    auto packed =
        hn::Or(hn::And(v0, mask_4), hn::ShiftLeft<4>(hn::And(v1, mask_4)));
    hn::StoreU(packed, d8, out + i / 2);
  }

  constexpr uint8_t mask = 0x0F;
  const size_t aligned_inputs = count / 2;
  for (size_t i = vector_inputs / 2; i < aligned_inputs; ++i) {
    out[i] = (in[2 * i] & mask) | ((in[2 * i + 1] & mask) << 4);
  }
  if (count % 2 != 0) {
    out[aligned_inputs] = in[2 * aligned_inputs] & mask;
  }
}

void Pack2Impl(const uint8_t* HWY_RESTRICT in, size_t count,
               uint8_t* HWY_RESTRICT out) {
  const hn::ScalableTag<uint8_t> d8;
  const size_t lanes = hn::Lanes(d8);
  const auto mask_2 = hn::Set(d8, 0x03);
  const size_t vector_stride = 4 * lanes;
  const size_t vector_inputs = (count / vector_stride) * vector_stride;

  for (size_t i = 0; i < vector_inputs; i += vector_stride) {
    hn::Vec<decltype(d8)> v0, v1, v2, v3;
    hn::LoadInterleaved4(d8, in + i, v0, v1, v2, v3);
    auto p01 =
        hn::Or(hn::And(v0, mask_2), hn::ShiftLeft<2>(hn::And(v1, mask_2)));
    auto p23 = hn::Or(hn::ShiftLeft<4>(hn::And(v2, mask_2)),
                      hn::ShiftLeft<6>(hn::And(v3, mask_2)));
    auto packed = hn::Or(p01, p23);
    hn::StoreU(packed, d8, out + i / 4);
  }

  constexpr uint8_t mask = 0x03;
  const size_t aligned_inputs = count / 4;
  for (size_t i = vector_inputs / 4; i < aligned_inputs; ++i) {
    out[i] = (in[4 * i] & mask) | ((in[4 * i + 1] & mask) << 2) |
             ((in[4 * i + 2] & mask) << 4) | ((in[4 * i + 3] & mask) << 6);
  }
  if (const size_t remainder = count % 4; remainder != 0) {
    uint8_t byte = 0;
    for (size_t j = 0; j < remainder; ++j) {
      byte |= (in[4 * aligned_inputs + j] & mask) << (2 * j);
    }
    out[aligned_inputs] = byte;
  }
}

void Pack1Impl(const uint8_t* HWY_RESTRICT in, size_t count,
               uint8_t* HWY_RESTRICT out) {
  const hn::ScalableTag<uint8_t> d8;
  const size_t lanes = hn::Lanes(d8);
  const auto mask_1 = hn::Set(d8, 0x01);
  const size_t vector_inputs = lanes >= 8 ? (count / lanes) * lanes : 0;

  for (size_t i = 0; i < vector_inputs; i += lanes) {
    auto v = hn::LoadU(d8, in + i);
    auto m = hn::TestBit(v, mask_1);
    hn::StoreMaskBits(d8, m, out + i / 8);
  }

  const size_t aligned_inputs = count / 8;
  for (size_t i = vector_inputs / 8; i < aligned_inputs; ++i) {
    uint8_t byte = 0;
    for (size_t j = 0; j < 8; ++j) {
      byte |= (in[8 * i + j] & 1) << j;
    }
    out[i] = byte;
  }
  if (const size_t remainder = count % 8; remainder != 0) {
    uint8_t byte = 0;
    for (size_t j = 0; j < remainder; ++j) {
      byte |= (in[8 * aligned_inputs + j] & 1) << j;
    }
    out[aligned_inputs] = byte;
  }
}

void Unpack4UnsignedImpl(const uint8_t* HWY_RESTRICT in, size_t count,
                         uint8_t* HWY_RESTRICT out) {
  const hn::ScalableTag<uint8_t> d8;
  const size_t lanes = hn::Lanes(d8);
  const auto mask_4 = hn::Set(d8, 0x0F);
  const size_t vector_stride = 2 * lanes;
  const size_t vector_outputs = (count / vector_stride) * vector_stride;

  for (size_t i = 0; i < vector_outputs; i += vector_stride) {
    auto packed = hn::LoadU(d8, in + i / 2);
    auto low = hn::And(packed, mask_4);
    auto high = hn::And(hn::ShiftRight<4>(packed), mask_4);
    hn::StoreInterleaved2(low, high, d8, out + i);
  }

  constexpr uint8_t mask = 0x0F;
  const size_t aligned_outputs = count / 2;
  for (size_t i = vector_outputs / 2; i < aligned_outputs; ++i) {
    const uint8_t byte = in[i];
    out[2 * i] = byte & mask;
    out[2 * i + 1] = (byte >> 4) & mask;
  }
  if (count % 2 != 0) {
    out[2 * aligned_outputs] = in[aligned_outputs] & mask;
  }
}

void Unpack4SignedImpl(const uint8_t* HWY_RESTRICT in, size_t count,
                       uint8_t* HWY_RESTRICT out) {
  const hn::ScalableTag<uint8_t> d8;
  const hn::ScalableTag<int8_t> d_i8;
  const size_t lanes = hn::Lanes(d8);
  const size_t vector_stride = 2 * lanes;
  const size_t vector_outputs = (count / vector_stride) * vector_stride;

  for (size_t i = 0; i < vector_outputs; i += vector_stride) {
    auto packed = hn::LoadU(d8, in + i / 2);
    auto packed_i8 = hn::BitCast(d_i8, packed);
    auto low = hn::BitCast(d8, hn::ShiftRight<4>(hn::ShiftLeft<4>(packed_i8)));
    auto high = hn::BitCast(d8, hn::ShiftRight<4>(packed_i8));
    hn::StoreInterleaved2(low, high, d8, out + i);
  }

  const size_t aligned_outputs = count / 2;
  for (size_t i = vector_outputs / 2; i < aligned_outputs; ++i) {
    const uint8_t byte = in[i];
    out[2 * i] = static_cast<uint8_t>(static_cast<int8_t>(byte << 4) >> 4);
    out[2 * i + 1] = static_cast<uint8_t>(static_cast<int8_t>(byte) >> 4);
  }
  if (count % 2 != 0) {
    out[2 * aligned_outputs] = static_cast<uint8_t>(
        static_cast<int8_t>(in[aligned_outputs] << 4) >> 4);
  }
}

void Unpack2UnsignedImpl(const uint8_t* HWY_RESTRICT in, size_t count,
                         uint8_t* HWY_RESTRICT out) {
  const hn::ScalableTag<uint8_t> d8;
  const size_t lanes = hn::Lanes(d8);
  const auto mask_2 = hn::Set(d8, 0x03);
  const size_t vector_stride = 4 * lanes;
  const size_t vector_outputs = (count / vector_stride) * vector_stride;

  for (size_t i = 0; i < vector_outputs; i += vector_stride) {
    auto packed = hn::LoadU(d8, in + i / 4);
    auto b0 = hn::And(packed, mask_2);
    auto b1 = hn::And(hn::ShiftRight<2>(packed), mask_2);
    auto b2 = hn::And(hn::ShiftRight<4>(packed), mask_2);
    auto b3 = hn::And(hn::ShiftRight<6>(packed), mask_2);
    hn::StoreInterleaved4(b0, b1, b2, b3, d8, out + i);
  }

  constexpr uint8_t mask = 0x03;
  const size_t aligned_outputs = count / 4;
  for (size_t i = vector_outputs / 4; i < aligned_outputs; ++i) {
    const uint8_t byte = in[i];
    out[4 * i] = byte & mask;
    out[4 * i + 1] = (byte >> 2) & mask;
    out[4 * i + 2] = (byte >> 4) & mask;
    out[4 * i + 3] = (byte >> 6) & mask;
  }
  if (const size_t remainder = count % 4; remainder != 0) {
    const uint8_t byte = in[aligned_outputs];
    for (size_t j = 0; j < remainder; ++j) {
      out[4 * aligned_outputs + j] = (byte >> (2 * j)) & mask;
    }
  }
}

void Unpack2SignedImpl(const uint8_t* HWY_RESTRICT in, size_t count,
                       uint8_t* HWY_RESTRICT out) {
  const hn::ScalableTag<uint8_t> d8;
  const hn::ScalableTag<int8_t> d_i8;
  const size_t lanes = hn::Lanes(d8);
  const size_t vector_stride = 4 * lanes;
  const size_t vector_outputs = (count / vector_stride) * vector_stride;

  for (size_t i = 0; i < vector_outputs; i += vector_stride) {
    auto packed = hn::LoadU(d8, in + i / 4);
    auto packed_i8 = hn::BitCast(d_i8, packed);
    auto b0 = hn::BitCast(d8, hn::ShiftRight<6>(hn::ShiftLeft<6>(packed_i8)));
    auto b1 = hn::BitCast(d8, hn::ShiftRight<6>(hn::ShiftLeft<4>(packed_i8)));
    auto b2 = hn::BitCast(d8, hn::ShiftRight<6>(hn::ShiftLeft<2>(packed_i8)));
    auto b3 = hn::BitCast(d8, hn::ShiftRight<6>(packed_i8));
    hn::StoreInterleaved4(b0, b1, b2, b3, d8, out + i);
  }

  const size_t aligned_outputs = count / 4;
  for (size_t i = vector_outputs / 4; i < aligned_outputs; ++i) {
    const uint8_t byte = in[i];
    out[4 * i] = static_cast<uint8_t>(static_cast<int8_t>(byte << 6) >> 6);
    out[4 * i + 1] = static_cast<uint8_t>(static_cast<int8_t>(byte << 4) >> 6);
    out[4 * i + 2] = static_cast<uint8_t>(static_cast<int8_t>(byte << 2) >> 6);
    out[4 * i + 3] = static_cast<uint8_t>(static_cast<int8_t>(byte) >> 6);
  }
  if (const size_t remainder = count % 4; remainder != 0) {
    const uint8_t byte = in[aligned_outputs];
    for (size_t j = 0; j < remainder; ++j) {
      out[4 * aligned_outputs + j] =
          static_cast<uint8_t>(static_cast<int8_t>(byte << (6 - 2 * j)) >> 6);
    }
  }
}

void Unpack1UnsignedImpl(const uint8_t* HWY_RESTRICT in, size_t count,
                         uint8_t* HWY_RESTRICT out) {
  const hn::ScalableTag<uint8_t> d8;
  const size_t lanes = hn::Lanes(d8);
  const auto val = hn::Set(d8, 0x01);
  const size_t vector_outputs = lanes >= 8 ? (count / lanes) * lanes : 0;

  for (size_t i = 0; i < vector_outputs; i += lanes) {
    auto m = hn::LoadMaskBits(d8, in + i / 8);
    auto v = hn::IfThenElseZero(m, val);
    hn::StoreU(v, d8, out + i);
  }

  constexpr uint8_t fill = 0x01;
  const size_t aligned_outputs = count / 8;
  for (size_t i = vector_outputs / 8; i < aligned_outputs; ++i) {
    const uint8_t byte = in[i];
    for (int j = 0; j < 8; ++j) {
      out[8 * i + j] = ((byte >> j) & 1) ? fill : 0;
    }
  }
  if (const size_t remainder = count % 8; remainder != 0) {
    const uint8_t byte = in[aligned_outputs];
    for (size_t j = 0; j < remainder; ++j) {
      out[8 * aligned_outputs + j] = ((byte >> j) & 1) ? fill : 0;
    }
  }
}

void Unpack1SignedImpl(const uint8_t* HWY_RESTRICT in, size_t count,
                       uint8_t* HWY_RESTRICT out) {
  const hn::ScalableTag<uint8_t> d8;
  const size_t lanes = hn::Lanes(d8);
  const auto val = hn::Set(d8, 0xFF);
  const size_t vector_outputs = lanes >= 8 ? (count / lanes) * lanes : 0;

  for (size_t i = 0; i < vector_outputs; i += lanes) {
    auto m = hn::LoadMaskBits(d8, in + i / 8);
    auto v = hn::IfThenElseZero(m, val);
    hn::StoreU(v, d8, out + i);
  }

  constexpr uint8_t fill = 0xFF;
  const size_t aligned_outputs = count / 8;
  for (size_t i = vector_outputs / 8; i < aligned_outputs; ++i) {
    const uint8_t byte = in[i];
    for (int j = 0; j < 8; ++j) {
      out[8 * i + j] = ((byte >> j) & 1) ? fill : 0;
    }
  }
  if (const size_t remainder = count % 8; remainder != 0) {
    const uint8_t byte = in[aligned_outputs];
    for (size_t j = 0; j < remainder; ++j) {
      out[8 * aligned_outputs + j] = ((byte >> j) & 1) ? fill : 0;
    }
  }
}

}  // namespace HWY_NAMESPACE
}  // namespace xla
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace xla {

HWY_EXPORT(Pack4Impl);
HWY_EXPORT(Pack2Impl);
HWY_EXPORT(Pack1Impl);
HWY_EXPORT(Unpack4UnsignedImpl);
HWY_EXPORT(Unpack4SignedImpl);
HWY_EXPORT(Unpack2UnsignedImpl);
HWY_EXPORT(Unpack2SignedImpl);
HWY_EXPORT(Unpack1UnsignedImpl);
HWY_EXPORT(Unpack1SignedImpl);

template <>
void PackIntNHwy<4>(absl::Span<const char> input, absl::Span<char> output) {
  constexpr size_t kElementsPerByte = 2;
  const size_t required_output_size =
      tsl::MathUtil::CeilOfRatio(input.size(), kElementsPerByte);
  CHECK_GE(output.size(), required_output_size)
      << "Output span too small for packed elements: " << output.size() << " < "
      << required_output_size;
  HWY_DYNAMIC_DISPATCH(Pack4Impl)(
      reinterpret_cast<const uint8_t*>(input.data()), input.size(),
      reinterpret_cast<uint8_t*>(output.data()));
}

template <>
void PackIntNHwy<2>(absl::Span<const char> input, absl::Span<char> output) {
  constexpr size_t kElementsPerByte = 4;
  const size_t required_output_size =
      tsl::MathUtil::CeilOfRatio(input.size(), kElementsPerByte);
  CHECK_GE(output.size(), required_output_size)
      << "Output span too small for packed elements: " << output.size() << " < "
      << required_output_size;
  HWY_DYNAMIC_DISPATCH(Pack2Impl)(
      reinterpret_cast<const uint8_t*>(input.data()), input.size(),
      reinterpret_cast<uint8_t*>(output.data()));
}

template <>
void PackIntNHwy<1>(absl::Span<const char> input, absl::Span<char> output) {
  constexpr size_t kElementsPerByte = 8;
  const size_t required_output_size =
      tsl::MathUtil::CeilOfRatio(input.size(), kElementsPerByte);
  CHECK_GE(output.size(), required_output_size)
      << "Output span too small for packed elements: " << output.size() << " < "
      << required_output_size;
  HWY_DYNAMIC_DISPATCH(Pack1Impl)(
      reinterpret_cast<const uint8_t*>(input.data()), input.size(),
      reinterpret_cast<uint8_t*>(output.data()));
}

void PackIntNHwy(int bits_per_element, absl::Span<const char> input,
                 absl::Span<char> output) {
  if (bits_per_element == 1) {
    PackIntNHwy<1>(input, output);
  } else if (bits_per_element == 2) {
    PackIntNHwy<2>(input, output);
  } else if (bits_per_element == 4) {
    PackIntNHwy<4>(input, output);
  } else {
    LOG(FATAL) << "Invalid bits_per_element: " << bits_per_element;
  }
}

template <>
void UnpackIntNHwy<4, false>(absl::Span<const char> input,
                             absl::Span<char> output) {
  constexpr size_t kElementsPerByte = 2;
  const size_t required_input_size =
      tsl::MathUtil::CeilOfRatio(output.size(), kElementsPerByte);
  CHECK_GE(input.size(), required_input_size)
      << "Input span too small for unpacked elements: " << input.size() << " < "
      << required_input_size;
  HWY_DYNAMIC_DISPATCH(Unpack4UnsignedImpl)(
      reinterpret_cast<const uint8_t*>(input.data()), output.size(),
      reinterpret_cast<uint8_t*>(output.data()));
}

template <>
void UnpackIntNHwy<4, true>(absl::Span<const char> input,
                            absl::Span<char> output) {
  constexpr size_t kElementsPerByte = 2;
  const size_t required_input_size =
      tsl::MathUtil::CeilOfRatio(output.size(), kElementsPerByte);
  CHECK_GE(input.size(), required_input_size)
      << "Input span too small for unpacked elements: " << input.size() << " < "
      << required_input_size;
  HWY_DYNAMIC_DISPATCH(Unpack4SignedImpl)(
      reinterpret_cast<const uint8_t*>(input.data()), output.size(),
      reinterpret_cast<uint8_t*>(output.data()));
}

template <>
void UnpackIntNHwy<2, false>(absl::Span<const char> input,
                             absl::Span<char> output) {
  constexpr size_t kElementsPerByte = 4;
  const size_t required_input_size =
      tsl::MathUtil::CeilOfRatio(output.size(), kElementsPerByte);
  CHECK_GE(input.size(), required_input_size)
      << "Input span too small for unpacked elements: " << input.size() << " < "
      << required_input_size;
  HWY_DYNAMIC_DISPATCH(Unpack2UnsignedImpl)(
      reinterpret_cast<const uint8_t*>(input.data()), output.size(),
      reinterpret_cast<uint8_t*>(output.data()));
}

template <>
void UnpackIntNHwy<2, true>(absl::Span<const char> input,
                            absl::Span<char> output) {
  constexpr size_t kElementsPerByte = 4;
  const size_t required_input_size =
      tsl::MathUtil::CeilOfRatio(output.size(), kElementsPerByte);
  CHECK_GE(input.size(), required_input_size)
      << "Input span too small for unpacked elements: " << input.size() << " < "
      << required_input_size;
  HWY_DYNAMIC_DISPATCH(Unpack2SignedImpl)(
      reinterpret_cast<const uint8_t*>(input.data()), output.size(),
      reinterpret_cast<uint8_t*>(output.data()));
}

template <>
void UnpackIntNHwy<1, false>(absl::Span<const char> input,
                             absl::Span<char> output) {
  constexpr size_t kElementsPerByte = 8;
  const size_t required_input_size =
      tsl::MathUtil::CeilOfRatio(output.size(), kElementsPerByte);
  CHECK_GE(input.size(), required_input_size)
      << "Input span too small for unpacked elements: " << input.size() << " < "
      << required_input_size;
  HWY_DYNAMIC_DISPATCH(Unpack1UnsignedImpl)(
      reinterpret_cast<const uint8_t*>(input.data()), output.size(),
      reinterpret_cast<uint8_t*>(output.data()));
}

template <>
void UnpackIntNHwy<1, true>(absl::Span<const char> input,
                            absl::Span<char> output) {
  constexpr size_t kElementsPerByte = 8;
  const size_t required_input_size =
      tsl::MathUtil::CeilOfRatio(output.size(), kElementsPerByte);
  CHECK_GE(input.size(), required_input_size)
      << "Input span too small for unpacked elements: " << input.size() << " < "
      << required_input_size;
  HWY_DYNAMIC_DISPATCH(Unpack1SignedImpl)(
      reinterpret_cast<const uint8_t*>(input.data()), output.size(),
      reinterpret_cast<uint8_t*>(output.data()));
}

void UnpackIntNHwy(int bits_per_element, absl::Span<const char> input,
                   absl::Span<char> output, bool is_signed) {
  if (bits_per_element == 1) {
    if (is_signed) {
      UnpackIntNHwy<1, true>(input, output);
    } else {
      UnpackIntNHwy<1, false>(input, output);
    }
  } else if (bits_per_element == 2) {
    if (is_signed) {
      UnpackIntNHwy<2, true>(input, output);
    } else {
      UnpackIntNHwy<2, false>(input, output);
    }
  } else if (bits_per_element == 4) {
    if (is_signed) {
      UnpackIntNHwy<4, true>(input, output);
    } else {
      UnpackIntNHwy<4, false>(input, output);
    }
  } else {
    LOG(FATAL) << "Invalid bits_per_element: " << bits_per_element;
  }
}

}  // namespace xla
#endif  // HWY_ONCE
