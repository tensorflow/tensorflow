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
#include "hwy/foreach_target.h"  // IWYU pragma: keep
#include "hwy/highway.h"

HWY_BEFORE_NAMESPACE();
namespace xla {
namespace HWY_NAMESPACE {

namespace hn = hwy::HWY_NAMESPACE;

// Packs pairs of 4-bit values one byte at a time. Used for HWY_SCALAR and for
// tails on targets without native masking.
HWY_INLINE void Pack4Scalar(const uint8_t* HWY_RESTRICT in, size_t count,
                            uint8_t* HWY_RESTRICT out) {
  for (size_t i = 0; i < count / 2; ++i) {
    out[i] = (in[2 * i] & 0x0F) | ((in[2 * i + 1] & 0x0F) << 4);
  }
  if (count % 2 != 0) {
    out[count / 2] = in[count - 1] & 0x0F;
  }
}

// Masks each input byte to 4 bits, combines each adjacent byte pair [0x0L,
// 0x0H] into (H << 4) | L within a uint16_t lane without LoadInterleaved2, and
// compacts the packed bytes via ConcatEven/ConcatOdd.
template <class D8>
HWY_INLINE void Pack4Vector(D8 d8, const uint8_t* HWY_RESTRICT in, size_t count,
                            uint8_t* HWY_RESTRICT out) {
  if constexpr (hn::MaxLanes(d8) == 1) {
    Pack4Scalar(in, count, out);
  } else {
    const size_t lanes = hn::Lanes(d8);
    const size_t vector_stride = 2 * lanes;
    const hn::Half<D8> d8_half;
    const hn::Repartition<uint16_t, D8> du16;
    const auto mask_4 = hn::Set(d8, uint8_t{0x0F});

    const auto merge_pairs = [&](const auto v) HWY_ATTR {
      const auto masked = hn::And(v, mask_4);
      if constexpr (HWY_TARGET <= HWY_SSSE3) {
        // On x86, a single pmaddubsw computes L + 16 * H and zeroes the upper
        // byte, so the compiler also drops the masking in ConcatEven.
        const hn::Repartition<int16_t, D8> di16;
        const auto mul_1_16 =
            hn::Dup128VecFromValues(hn::RebindToSigned<D8>(), 1, 16, 1, 16, 1,
                                    16, 1, 16, 1, 16, 1, 16, 1, 16, 1, 16);
        return hn::BitCast(du16,
                           hn::SatWidenMulPairwiseAdd(di16, masked, mul_1_16));
      } else if constexpr (HWY_IS_BIG_ENDIAN) {
        const auto w = hn::BitCast(du16, masked);
        return hn::Or(hn::ShiftRight<8>(w), hn::ShiftLeft<4>(w));
      } else {
        const auto w = hn::BitCast(du16, masked);
        return hn::Or(w, hn::ShiftRight<4>(w));
      }
    };
    const auto pack_pair = [&](const auto v0, const auto v1) HWY_ATTR {
      const auto b0 = hn::BitCast(d8, merge_pairs(v0));
      const auto b1 = hn::BitCast(d8, merge_pairs(v1));
      // Unqualified for ADL so HWY_SCALAR (where ConcatEven/ConcatOdd are not
      // declared) skips lookup in this discarded if-constexpr branch.
      if constexpr (HWY_IS_BIG_ENDIAN) {
        return ConcatOdd(d8, b1, b0);
      } else {
        return ConcatEven(d8, b1, b0);
      }
    };

    size_t remaining = count;
    for (; remaining >= vector_stride;
         in += vector_stride, out += lanes, remaining -= vector_stride) {
      hn::StoreU(pack_pair(hn::LoadU(d8, in), hn::LoadU(d8, in + lanes)), d8,
                 out);
    }
    if constexpr (HWY_NATIVE_MASK) {
      if (remaining > lanes) {
        const auto packed = pack_pair(
            hn::LoadU(d8, in), hn::LoadN(d8, in + lanes, remaining - lanes));
        hn::StoreN(packed, d8, out, (remaining + 1) / 2);
      } else if (remaining > 0) {
        const auto v = hn::LoadN(d8, in, remaining);
        hn::StoreN(hn::TruncateTo(d8_half, merge_pairs(v)), d8_half, out,
                   (remaining + 1) / 2);
      }
    } else {
      // Without native masking, LoadN and StoreN typically branch on the
      // count, which is slower than finishing with scalar code.
      Pack4Scalar(in, remaining, out);
    }
  }
}

void Pack4Impl(const uint8_t* HWY_RESTRICT in, size_t count,
               uint8_t* HWY_RESTRICT out) {
  Pack4Vector(hn::ScalableTag<uint8_t>(), in, count, out);
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

// Unpacks bytes into pairs of 4-bit values one byte at a time. Used for
// HWY_SCALAR and for tails on targets without native masking.
HWY_INLINE void Unpack4Scalar(const uint8_t* HWY_RESTRICT in, size_t count,
                              uint8_t* HWY_RESTRICT out) {
  for (size_t i = 0; i < count / 2; ++i) {
    const uint8_t byte = in[i];
    out[2 * i] = byte & 0x0F;
    out[2 * i + 1] = (byte >> 4) & 0x0F;
  }
  if (count % 2 != 0) {
    out[count - 1] = in[count / 2] & 0x0F;
  }
}

// Zero-extends each packed byte (0xHL) to a uint16_t lane and expands the low
// and high nibbles into adjacent bytes [0x0L, 0x0H] without StoreInterleaved2.
template <class D8>
HWY_INLINE void Unpack4Vector(D8 d8, const uint8_t* HWY_RESTRICT in,
                              size_t count, uint8_t* HWY_RESTRICT out) {
  if constexpr (hn::MaxLanes(d8) == 1) {
    Unpack4Scalar(in, count, out);
  } else {
    const size_t lanes = hn::Lanes(d8);
    const hn::Half<D8> d8_half;
    const hn::Repartition<uint16_t, D8> du16;
    const auto mask_0f0f = hn::Set(du16, uint16_t{0x0F0F});
    const auto unpack_half = [&](const auto v_packed) HWY_ATTR {
      const auto wide = hn::PromoteTo(du16, v_packed);
      if constexpr (HWY_IS_BIG_ENDIAN) {
        return hn::BitCast(
            d8, hn::And(hn::Or(hn::ShiftLeft<8>(wide), hn::ShiftRight<4>(wide)),
                        mask_0f0f));
      } else {
        return hn::BitCast(
            d8, hn::And(hn::Or(wide, hn::ShiftLeft<4>(wide)), mask_0f0f));
      }
    };

    size_t remaining = count;
    for (; remaining >= lanes;
         in += lanes / 2, out += lanes, remaining -= lanes) {
      hn::StoreU(unpack_half(hn::LoadU(d8_half, in)), d8, out);
    }
    if constexpr (HWY_NATIVE_MASK) {
      if (remaining > 0) {
        hn::StoreN(unpack_half(hn::LoadN(d8_half, in, (remaining + 1) / 2)), d8,
                   out, remaining);
      }
    } else {
      // As in Pack4Vector.
      Unpack4Scalar(in, remaining, out);
    }
  }
}

void Unpack4Impl(const uint8_t* HWY_RESTRICT in, size_t count,
                 uint8_t* HWY_RESTRICT out) {
  Unpack4Vector(hn::ScalableTag<uint8_t>(), in, count, out);
}

void Unpack2Impl(const uint8_t* HWY_RESTRICT in, size_t count,
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

void Unpack1Impl(const uint8_t* HWY_RESTRICT in, size_t count,
                 uint8_t* HWY_RESTRICT out) {
  const hn::ScalableTag<uint8_t> d8;
  const size_t lanes = hn::Lanes(d8);
  const auto one = hn::Set(d8, 0x01);
  const size_t vector_outputs = lanes >= 8 ? (count / lanes) * lanes : 0;

  for (size_t i = 0; i < vector_outputs; i += lanes) {
    auto m = hn::LoadMaskBits(d8, in + i / 8);
    auto v = hn::IfThenElseZero(m, one);
    hn::StoreU(v, d8, out + i);
  }

  const size_t aligned_outputs = count / 8;
  for (size_t i = vector_outputs / 8; i < aligned_outputs; ++i) {
    const uint8_t byte = in[i];
    for (int j = 0; j < 8; ++j) {
      out[8 * i + j] = (byte >> j) & 1;
    }
  }
  if (const size_t remainder = count % 8; remainder != 0) {
    const uint8_t byte = in[aligned_outputs];
    for (size_t j = 0; j < remainder; ++j) {
      out[8 * aligned_outputs + j] = (byte >> j) & 1;
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
HWY_EXPORT(Unpack4Impl);
HWY_EXPORT(Unpack2Impl);
HWY_EXPORT(Unpack1Impl);

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
void UnpackIntNHwy<4>(absl::Span<const char> input, absl::Span<char> output) {
  constexpr size_t kElementsPerByte = 2;
  const size_t required_input_size =
      tsl::MathUtil::CeilOfRatio(output.size(), kElementsPerByte);
  CHECK_GE(input.size(), required_input_size)
      << "Input span too small for unpacked elements: " << input.size() << " < "
      << required_input_size;
  HWY_DYNAMIC_DISPATCH(Unpack4Impl)(
      reinterpret_cast<const uint8_t*>(input.data()), output.size(),
      reinterpret_cast<uint8_t*>(output.data()));
}

template <>
void UnpackIntNHwy<2>(absl::Span<const char> input, absl::Span<char> output) {
  constexpr size_t kElementsPerByte = 4;
  const size_t required_input_size =
      tsl::MathUtil::CeilOfRatio(output.size(), kElementsPerByte);
  CHECK_GE(input.size(), required_input_size)
      << "Input span too small for unpacked elements: " << input.size() << " < "
      << required_input_size;
  HWY_DYNAMIC_DISPATCH(Unpack2Impl)(
      reinterpret_cast<const uint8_t*>(input.data()), output.size(),
      reinterpret_cast<uint8_t*>(output.data()));
}

template <>
void UnpackIntNHwy<1>(absl::Span<const char> input, absl::Span<char> output) {
  constexpr size_t kElementsPerByte = 8;
  const size_t required_input_size =
      tsl::MathUtil::CeilOfRatio(output.size(), kElementsPerByte);
  CHECK_GE(input.size(), required_input_size)
      << "Input span too small for unpacked elements: " << input.size() << " < "
      << required_input_size;
  HWY_DYNAMIC_DISPATCH(Unpack1Impl)(
      reinterpret_cast<const uint8_t*>(input.data()), output.size(),
      reinterpret_cast<uint8_t*>(output.data()));
}

void UnpackIntNHwy(int bits_per_element, absl::Span<const char> input,
                   absl::Span<char> output) {
  if (bits_per_element == 1) {
    UnpackIntNHwy<1>(input, output);
  } else if (bits_per_element == 2) {
    UnpackIntNHwy<2>(input, output);
  } else if (bits_per_element == 4) {
    UnpackIntNHwy<4>(input, output);
  } else {
    LOG(FATAL) << "Invalid bits_per_element: " << bits_per_element;
  }
}

}  // namespace xla
#endif  // HWY_ONCE
