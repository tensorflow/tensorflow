/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/framework/float8.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <type_traits>

namespace tensorflow {
namespace float8_internal {

namespace {

// Helper struct for getting a bit representation provided a byte size.
template <int kNumBytes>
struct GetUnsignedInteger;

template <>
struct GetUnsignedInteger<1> {
  using type = uint8_t;
};

template <>
struct GetUnsignedInteger<2> {
  using type = uint16_t;
};

template <>
struct GetUnsignedInteger<4> {
  using type = uint32_t;
};

template <>
struct GetUnsignedInteger<8> {
  using type = uint64_t;
};

// Converts between two floating-point types.
template <typename From, typename To, bool kSaturate, bool kTruncate,
          typename EnableIf = void>
struct ConvertImpl;

// Convert to same type.  We need explicit specializations for all combinations
// of template parameters to avoid ambiguities.
template <typename Scalar>
struct IdentityConversion {
  static EIGEN_DEVICE_FUNC inline Scalar run(const Scalar& from) {
    return from;
  }
};

template <typename Scalar>
struct ConvertImpl<Scalar, Scalar, /*kSaturate=*/false, /*kTruncate=*/false,
                   /*EnableIf=*/void> : public IdentityConversion<Scalar> {};
template <typename Scalar>
struct ConvertImpl<Scalar, Scalar, /*kSaturate=*/false, /*kTruncate=*/true,
                   /*EnableIf=*/void> : public IdentityConversion<Scalar> {};
template <typename Scalar>
struct ConvertImpl<Scalar, Scalar, /*kSaturate=*/true, /*kTruncate=*/false,
                   /*EnableIf=*/void> : public IdentityConversion<Scalar> {};
template <typename Scalar>
struct ConvertImpl<Scalar, Scalar, /*kSaturate=*/true, /*kTruncate=*/true,
                   /*EnableIf=*/void> : public IdentityConversion<Scalar> {};

// Convert float8 to larger types.
template <typename From, typename To, bool kSaturate, bool kTruncate>
struct ConvertImpl<
    From, To, kSaturate, kTruncate,
    std::enable_if_t<std::is_base_of_v<float8_base<From>, From> &&
                     (sizeof(From) < sizeof(To))>> {
  using FromBits = typename GetUnsignedInteger<sizeof(From)>::type;
  static constexpr int kFromBits = sizeof(From) * CHAR_BIT;
  static constexpr int kFromMantissaBits = Eigen::NumTraits<From>::digits() - 1;
  static constexpr int kFromExponentBits = kFromBits - kFromMantissaBits - 1;
  static constexpr int kFromExponentBias = (1 << (kFromExponentBits - 1)) - 1;
  static constexpr FromBits kFromExponentMask =
      ((static_cast<FromBits>(1) << kFromExponentBits) - 1)
      << kFromMantissaBits;

  using ToBits = typename GetUnsignedInteger<sizeof(To)>::type;
  static constexpr int kToBits = sizeof(To) * CHAR_BIT;
  static constexpr int kToMantissaBits = Eigen::NumTraits<To>::digits() - 1;
  static constexpr int kToExponentBits = kToBits - kToMantissaBits - 1;
  static constexpr int kToExponentBias = (1 << (kToExponentBits - 1)) - 1;

  static constexpr int kExponentOffset = kToExponentBias - kFromExponentBias;
  static constexpr int kDigitShift = kToMantissaBits - kFromMantissaBits;

  static EIGEN_DEVICE_FUNC inline To run(const From& from) {
    // Shift bits to destination type, without sign bit.
    const FromBits from_bits = from.rep() & 0x7F;
    ToBits bits = ToBits{from_bits} << kDigitShift;

    // Adjust the exponent.
    // Special cases.
    if (Eigen::numext::isinf(from) || Eigen::numext::isnan(from)) {
      // Inf or NaN, fill exponent bits with all ones and preserve digits.
      bits |= ((ToBits{1} << kToExponentBits) - 1) << kToMantissaBits;
    } else if ((from.rep() & kFromExponentMask) == 0) {
      // Subnormals.

      // All float8 subnormals become normalized when casting to a type
      // with a larger number of exponent bits.  To do the conversion, we
      // construct an explicit map of all subnormal values to the
      // corresponding normalized values in the destination type.  We do this
      // by setting the normalized mantissa bits in the source type, shifting
      // it up to the destination type, then inserting the exponent bits.
      if constexpr (kFromMantissaBits == 2) {
        // e5m2, only 4 options:
        constexpr ToBits kNormalized[4] = {
            // Mantissa | Exponent
            ToBits{0x00},
            ToBits{0x00} | ToBits{kExponentOffset - 1} << kToMantissaBits,
            ToBits{0x00} | ToBits{kExponentOffset} << kToMantissaBits,
            (ToBits{0x02} << kDigitShift) |
                (ToBits{kExponentOffset} << kToMantissaBits),
        };
        bits = kNormalized[from_bits];
      } else if constexpr (kFromMantissaBits == 3) {
        // e4m3, only 8 options
        constexpr ToBits kNormalized[8] = {
            // Mantissa | Exponent
            ToBits{0x00},
            ToBits{0x00} | (ToBits{kExponentOffset - 2} << kToMantissaBits),
            ToBits{0x00} | (ToBits{kExponentOffset - 1} << kToMantissaBits),
            (ToBits{0x04} << kDigitShift) |
                (ToBits{kExponentOffset - 1} << kToMantissaBits),
            ToBits{0x00} | (ToBits{kExponentOffset} << kToMantissaBits),
            (ToBits{0x02} << kDigitShift) |
                (ToBits{kExponentOffset} << kToMantissaBits),
            (ToBits{0x04} << kDigitShift) |
                (ToBits{kExponentOffset} << kToMantissaBits),
            (ToBits{0x06} << kDigitShift) |
                (ToBits{kExponentOffset} << kToMantissaBits),
        };
        bits = kNormalized[from_bits];
      }
    } else {
      // Increase exponent by offset difference.
      bits += ToBits{kExponentOffset} << kToMantissaBits;
    }

    // Insert sign bit.
    bits |= static_cast<ToBits>(from.rep() & 0x80) << (kToBits - kFromBits);
    return Eigen::numext::bit_cast<To>(bits);
  }
};

template <typename Bits>
constexpr inline Bits RoundBitsToNearestEven(Bits bits, int roundoff) {
  // Round to nearest even by adding a bias term.
  // Consider a bit pattern
  //   FFF...FLRTT...T,
  // where bits RTT...T need to be rounded-off.  We add a bias term to the
  // bit pattern s.t. a carry is introduced to round up only if
  // - L is 1, R is 1, OR
  // - L is 0, R is 1, any T is one.
  // We do this by adding L to a bit pattern consisting of all T = 1.
  Bits bias = roundoff == 0 ? 0
                            : ((bits >> roundoff) & 1) +
                                  (static_cast<Bits>(1) << (roundoff - 1)) - 1;
  return bits + bias;
}

// Convert larger types to float8.
template <typename From, typename To, bool kSaturate, bool kTruncate>
struct ConvertImpl<From, To, kSaturate, kTruncate,
                   std::enable_if_t<std::is_base_of_v<float8_base<To>, To> &&
                                    (sizeof(To) < sizeof(From))>> {
  using FromBits = typename GetUnsignedInteger<sizeof(From)>::type;
  static constexpr int kFromBits = sizeof(From) * CHAR_BIT;
  static constexpr int kFromMantissaBits = Eigen::NumTraits<From>::digits() - 1;
  static constexpr int kFromExponentBits = kFromBits - kFromMantissaBits - 1;
  static constexpr int kFromExponentBias = (1 << (kFromExponentBits - 1)) - 1;
  static constexpr FromBits kFromExponentMask =
      ((static_cast<FromBits>(1) << kFromExponentBits) - 1)
      << kFromMantissaBits;

  using ToBits = typename GetUnsignedInteger<sizeof(To)>::type;
  static constexpr int kToBits = sizeof(To) * CHAR_BIT;
  static constexpr int kToMantissaBits = Eigen::NumTraits<To>::digits() - 1;
  static constexpr int kToExponentBits = kToBits - kToMantissaBits - 1;
  static constexpr int kToExponentBias = (1 << (kToExponentBits - 1)) - 1;

  static constexpr int kExponentOffset = kFromExponentBias - kToExponentBias;
  static constexpr int kDigitShift = kFromMantissaBits - kToMantissaBits;

  static_assert(kFromExponentBits > kToExponentBits,
                "This implementation assumes down-casting to types with fewer "
                "exponent bits.");
  static_assert(kDigitShift > 0,
                "This implementations assumes down-casting to types with fewer "
                "mantissa bits.");

  // Shift bits in the appropriate directions and add the exponent offset
  // to convert between bit representations.  The input `in` must be a
  // positive normalized value.
  static constexpr inline FromBits ToFromBits(ToBits in) {
    FromBits out = static_cast<FromBits>(in) << kDigitShift;
    out += static_cast<FromBits>(kExponentOffset) << kFromMantissaBits;
    return out;
  }

  static constexpr inline FromBits SetFromBit(int idx) {
    return static_cast<FromBits>(1) << idx;
  }

  static EIGEN_DEVICE_FUNC inline To run(const From& from) {
    FromBits from_bits = Eigen::numext::bit_cast<FromBits>(from);
    FromBits from_sign = from_bits & SetFromBit(kFromBits - 1);
    from_bits ^= from_sign;  // Zeros sign bit to obtain absolute value.
    ToBits sign = from_sign >> (kFromBits - kToBits);

    // Special values, preserving sign.
    if (Eigen::numext::isinf(from)) {
      return sign != 0 ? -Eigen::NumTraits<To>::infinity()
                       : Eigen::NumTraits<To>::infinity();
    } else if (Eigen::numext::isnan(from)) {
      return Eigen::numext::bit_cast<To>(
          static_cast<uint8_t>(Eigen::NumTraits<To>::quiet_NaN().rep() | sign));
    }

    // Adjust mantissa.
    if constexpr (!kTruncate) {
      from_bits = RoundBitsToNearestEven(from_bits, kDigitShift);
    }
    // Zero-out tail bits.
    from_bits &= ~(SetFromBit(kDigitShift) - 1);

    // Check for overflows.
    if constexpr (kExponentOffset > 0) {
      // Shift up exponent and mantissa, add offset to adjust exponent to
      // source type.
      constexpr ToBits kToHighest = Eigen::NumTraits<To>::highest().rep();
      constexpr FromBits kHighest = ToFromBits(kToHighest);

      if (from_bits > kHighest) {
        ToBits bits =
            kSaturate ? kToHighest : Eigen::NumTraits<To>::infinity().rep();
        return Eigen::numext::bit_cast<To>(static_cast<ToBits>(bits | sign));
      }
    }

    // Subnormals and zero.
    constexpr FromBits kLowestNormal =
        ToFromBits(std::numeric_limits<To>::min().rep());
    if (from_bits < kLowestNormal) {
      // Round and shift mantissa down.
      constexpr FromBits kMantissaMask = SetFromBit(kFromMantissaBits) - 1;
      int exponent = ((from_bits >> kFromMantissaBits) - kFromExponentBias);
      int exponent_shift = kDigitShift - exponent - kToExponentBias + 1;

      // Insert the implicit leading 1 bit on the mantissa.  This assumes
      // the input is normalized.  If it is not, then the mantissa bits -
      // including the implicit one - will be shifted to zero.
      from_bits = (SetFromBit(kFromMantissaBits) | (from_bits & kMantissaMask));
      ToBits bits = 0;
      // To avoid UB, limit rounding and shifting to the full mantissa plus
      // leading 1.
      if (exponent_shift <= kFromMantissaBits + 1) {
        if constexpr (!kTruncate) {
          from_bits = RoundBitsToNearestEven(from_bits, exponent_shift);
        }
        bits = (from_bits >> exponent_shift);
      }
      // Insert sign and return.
      return Eigen::numext::bit_cast<To>(static_cast<ToBits>(bits | sign));
    }

    // Adjust exponent.
    from_bits += static_cast<FromBits>(-kExponentOffset) << kFromMantissaBits;

    // Shift bits and insert sign.
    ToBits bits = static_cast<ToBits>((from_bits >> kDigitShift) | sign);
    return Eigen::numext::bit_cast<To>(bits);
  }
};

template <bool kSaturate, bool kTruncate>
struct ConvertImpl<float8_e5m2, float8_e4m3, kSaturate, kTruncate> {
  static EIGEN_DEVICE_FUNC inline float8_e4m3 run(const float8_e5m2& from) {
    uint8_t from_bits = from.rep();
    uint8_t sign = from_bits & 0x80;
    from_bits ^= sign;

    // Special values (NaN/Inf).
    if (from_bits > 0x7C) {
      return float8_e4m3::FromRep(sign | 0x7F);
    }

    // Subnormals or overflow.
    if (from_bits < 0x24) {
      // Subnormal output.
      int negative_exponent = 15 - (from_bits >> 2);
      int exponent_shift = negative_exponent - 7;
      uint8_t bits = ((from_bits & 0x03) | 0x04);
      if constexpr (!kTruncate) {
        bits = RoundBitsToNearestEven(bits, exponent_shift);
      }
      bits >>= exponent_shift;
      return float8_e4m3::FromRep(sign | bits);
    } else if (from_bits > 0x5F) {
      uint8_t bits = kSaturate ? 0x7E : 0x7F;
      return float8_e4m3::FromRep(sign | bits);
    }

    // Subtract exponent offset and shift.
    uint8_t bits = (from_bits - 0x20) << 1;
    return float8_e4m3::FromRep(sign | bits);
  }
};

template <bool kTruncate>
struct ConvertImpl<float8_e4m3, float8_e5m2, kTruncate, false> {
  static EIGEN_DEVICE_FUNC inline float8_e5m2 run(const float8_e4m3& from) {
    uint8_t from_bits = from.rep();
    uint8_t sign = from_bits & 0x80;
    from_bits ^= sign;

    // Special values (NaN).
    if (from_bits == 0x7F) {
      return float8_e5m2::FromRep(sign | from_bits);
    }

    // Subnormals.
    if (from_bits < 0x08) {
      // Complete map between types, all are normal in e5m3.
      static constexpr uint8_t kNormalized[8] = {0x00, 0x18, 0x1C, 0x1E,
                                                 0x20, 0x21, 0x22, 0x23};
      uint8_t bits = kNormalized[from_bits];
      return float8_e5m2::FromRep(sign | bits);
    }

    // Round, truncate to destination type, and add exponent offset.
    if (!kTruncate) {
      from_bits = RoundBitsToNearestEven(from_bits, 1);
    }
    from_bits = (from_bits >> 1) + 0x20;
    return float8_e5m2::FromRep(sign | from_bits);
  }
};

// Saturation has no impact when casting e4m3 to e5m2.
template <bool kSaturate, bool kTruncate>
struct ConvertImpl<float8_e4m3, float8_e5m2, kSaturate, kTruncate> {
  static EIGEN_DEVICE_FUNC inline float8_e5m2 run(const float8_e4m3& from) {
    return ConvertImpl<float8_e4m3, float8_e5m2, kTruncate, false>::run(from);
  }
};

template <bool kTruncate>
struct ConvertImpl<Eigen::half, float8_e5m2, kTruncate, false> {
  static EIGEN_DEVICE_FUNC inline float8_e5m2 run(const Eigen::half& from) {
    uint16_t from_bits = Eigen::numext::bit_cast<uint16_t>(from);

    // Special values (Inf or NaN).
    uint16_t abs_bits = from_bits & 0x7FFF;
    if (abs_bits == 0x7C00) {
      return float8_e5m2::FromRep(from_bits >> 8);
    } else if (abs_bits > 0x7C00) {
      return float8_e5m2::FromRep((from_bits >> 8) | 0x01);
    }

    if constexpr (!kTruncate) {
      from_bits = RoundBitsToNearestEven(from_bits, 8);
    }
    return float8_e5m2::FromRep(from_bits >> 8);
  }
};

// Saturation has no impact when casting Eigen::half to e5m2.
template <bool kSaturate, bool kTruncate>
struct ConvertImpl<Eigen::half, float8_e5m2, kSaturate, kTruncate> {
  static EIGEN_DEVICE_FUNC inline float8_e5m2 run(const Eigen::half& from) {
    return ConvertImpl<Eigen::half, float8_e5m2, kTruncate, false>::run(from);
  }
};

template <>
struct ConvertImpl<float8_e5m2, Eigen::half, false, false> {
  static EIGEN_DEVICE_FUNC inline Eigen::half run(const float8_e5m2& from) {
    return Eigen::numext::bit_cast<Eigen::half>(
        static_cast<uint16_t>(static_cast<uint16_t>(from.rep()) << 8));
  }
};

// Saturation and truncation have no impact when casting e5m2 to Eigen::half.
template <bool kSaturate, bool kTruncate>
struct ConvertImpl<float8_e5m2, Eigen::half, kSaturate, kTruncate> {
  static EIGEN_DEVICE_FUNC inline Eigen::half run(const float8_e5m2& from) {
    return ConvertImpl<float8_e5m2, Eigen::half, false, false>::run(from);
  }
};

}  // namespace

template <typename Derived>
template <bool kSaturate, bool kTruncate, typename From>
EIGEN_DEVICE_FUNC Derived float8_base<Derived>::ConvertFrom(const From& from) {
  return ConvertImpl<From, Derived, kSaturate, kTruncate>::run(from);
}

template <typename Derived>
template <typename To, bool kSaturate, bool kTruncate>
EIGEN_DEVICE_FUNC To float8_base<Derived>::ConvertTo(const Derived& from) {
  return ConvertImpl<Derived, To, kSaturate, kTruncate>::run(from);
}

#define DECLARE_CONVERT(Derived, Other)                                 \
  template EIGEN_DEVICE_FUNC Derived                                    \
  float8_base<Derived>::ConvertFrom<false, false, Other>(const Other&); \
  template EIGEN_DEVICE_FUNC Derived                                    \
  float8_base<Derived>::ConvertFrom<false, true, Other>(const Other&);  \
  template EIGEN_DEVICE_FUNC Derived                                    \
  float8_base<Derived>::ConvertFrom<true, false, Other>(const Other&);  \
  template EIGEN_DEVICE_FUNC Derived                                    \
  float8_base<Derived>::ConvertFrom<true, true, Other>(const Other&);   \
  template EIGEN_DEVICE_FUNC Other                                      \
  float8_base<Derived>::ConvertTo<Other, false, false>(const Derived&); \
  template EIGEN_DEVICE_FUNC Other                                      \
  float8_base<Derived>::ConvertTo<Other, false, true>(const Derived&);  \
  template EIGEN_DEVICE_FUNC Other                                      \
  float8_base<Derived>::ConvertTo<Other, true, false>(const Derived&);  \
  template EIGEN_DEVICE_FUNC Other                                      \
  float8_base<Derived>::ConvertTo<Other, true, true>(const Derived&)

DECLARE_CONVERT(float8_e4m3, double);
DECLARE_CONVERT(float8_e4m3, float);
DECLARE_CONVERT(float8_e4m3, Eigen::bfloat16);
DECLARE_CONVERT(float8_e4m3, Eigen::half);
DECLARE_CONVERT(float8_e4m3, float8_e5m2);
DECLARE_CONVERT(float8_e4m3, float8_e4m3);

DECLARE_CONVERT(float8_e5m2, double);
DECLARE_CONVERT(float8_e5m2, float);
DECLARE_CONVERT(float8_e5m2, Eigen::bfloat16);
DECLARE_CONVERT(float8_e5m2, Eigen::half);
DECLARE_CONVERT(float8_e5m2, float8_e5m2);
DECLARE_CONVERT(float8_e5m2, float8_e4m3);

#undef DECLARE_CONVERT

}  // namespace float8_internal
}  // namespace tensorflow
