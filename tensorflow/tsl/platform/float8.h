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

#ifndef TENSORFLOW_TSL_PLATFORM_FLOAT8_H_
#define TENSORFLOW_TSL_PLATFORM_FLOAT8_H_

// 8-bit Floating Point Interchange Format, as described by
//   https://arxiv.org/abs/2209.05433

#include <cstdint>
#include <ostream>

#include "absl/numeric/bits.h"
#include "third_party/eigen3/Eigen/Core"

namespace tsl {

namespace float8_internal {

// Forward-declarations of classes.
class float8_e4m3fn;
class float8_e5m2;

template <typename Derived>
class float8_base {
 protected:
  // Constructor tag to allow constexpr construction from bit representation.
  struct ConstructFromRepTag {};

  constexpr float8_base() : rep_(0) {}
  constexpr float8_base(uint8_t rep, ConstructFromRepTag) : rep_{rep} {}

 public:
  constexpr uint8_t rep() const { return rep_; }

  constexpr Derived operator-() const {
    return Derived(static_cast<uint8_t>(rep() ^ 0x80), ConstructFromRepTag{});
  }

  constexpr bool operator==(const Derived& other) const {
    if (Eigen::numext::isnan(derived()) || Eigen::numext::isnan(other)) {
      return false;
    } else if ((rep() & 0x7F) == 0) {
      return (other.rep() & 0x7F) == 0;
    }
    return rep() == other.rep();
  }

  constexpr bool operator!=(const Derived& other) const {
    return !(derived() == other);
  }

  constexpr const Derived& derived() const {
    return *static_cast<const Derived*>(this);
  }

  constexpr Derived& derived() { return *static_cast<Derived*>(this); }

  static constexpr Derived FromRep(uint8_t rep) {
    return Derived(rep, ConstructFromRepTag{});
  }

  // Conversions allowing saturation and truncation.
  template <bool kSaturate = false, bool kTruncate = false, typename From>
  static inline EIGEN_DEVICE_FUNC Derived ConvertFrom(const From& from);

  template <typename To, bool kSaturate = false, bool kTruncate = false>
  static inline EIGEN_DEVICE_FUNC To ConvertTo(const Derived& from);

  // Operators via float32.
  EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC Derived
  operator+(const Derived& other) const {
    return Derived{float{derived()} + float{other}};
  }

  EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC Derived
  operator-(const Derived& other) const {
    return Derived{float{derived()} - float{other}};
  }

  EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC Derived
  operator*(const Derived& other) const {
    return Derived{float{derived()} * float{other}};
  }

  EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC Derived
  operator/(const Derived& other) const {
    return Derived{float{derived()} / float{other}};
  }

  EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC bool operator<(
      const Derived& other) const {
    return float{derived()} < float{other};
  }

  EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC bool operator<=(
      const Derived& other) const {
    return float{derived()} <= float{other};
  }

  EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC bool operator>(
      const Derived& other) const {
    return float{derived()} > float{other};
  }

  EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC bool operator>=(
      const Derived& other) const {
    return float{derived()} >= float{other};
  }

  // Compound assignment.
  EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC Derived& operator+=(
      const Derived& other) {
    derived() = derived() + other;
    return derived();
  }

  EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC Derived& operator-=(
      const Derived& other) {
    derived() = derived() - other;
    return derived();
  }

  EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC Derived& operator*=(
      const Derived& other) {
    derived() = derived() * other;
    return derived();
  }

  EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC Derived& operator/=(
      const Derived& other) {
    derived() = derived() / other;
    return derived();
  }

 private:
  uint8_t rep_;
};

class float8_e4m3fn : public float8_base<float8_e4m3fn> {
  // Exponent: 4, Mantissa: 3, bias: 7.
  // Extended range: no inf, NaN represented by 0bS111'1111.
  // The "fn" suffix is for consistency with the corresponding LLVM/MLIR type,
  // signaling this type is not consistent with IEEE-754.  The "f" indicates
  // it is finite values only. The "n" indicates it includes NaNs, but only
  // at the outer range.
 private:
  using Base = float8_base<float8_e4m3fn>;
  friend class float8_base<float8_e4m3fn>;

  constexpr float8_e4m3fn(uint8_t rep, ConstructFromRepTag)
      : Base(rep, ConstructFromRepTag{}) {}

 public:
  constexpr float8_e4m3fn() = default;

  template <typename T,
            typename EnableIf = std::enable_if<std::is_arithmetic_v<T>>>
  explicit EIGEN_DEVICE_FUNC float8_e4m3fn(T f)
      : float8_e4m3fn(ConvertFrom(static_cast<float>(f))) {}
  explicit EIGEN_DEVICE_FUNC float8_e4m3fn(double f64)
      : float8_e4m3fn(ConvertFrom(f64)) {}
  explicit EIGEN_DEVICE_FUNC float8_e4m3fn(float f32)
      : float8_e4m3fn(ConvertFrom(f32)) {}
  explicit EIGEN_DEVICE_FUNC float8_e4m3fn(Eigen::bfloat16 bf16)
      : float8_e4m3fn(ConvertFrom(bf16)) {}
  explicit EIGEN_DEVICE_FUNC float8_e4m3fn(Eigen::half f16)
      : float8_e4m3fn(ConvertFrom(f16)) {}
  explicit EIGEN_DEVICE_FUNC float8_e4m3fn(const float8_e5m2& f8)
      : float8_e4m3fn(ConvertFrom(f8)) {}

  template <typename T,
            typename EnableIf = std::enable_if<std::is_arithmetic_v<T>>>
  explicit EIGEN_DEVICE_FUNC operator T() const {
    return static_cast<T>(static_cast<float>(*this));
  }
  explicit EIGEN_DEVICE_FUNC operator double() const {
    return ConvertTo<double>(*this);
  }
  explicit EIGEN_DEVICE_FUNC operator float() const {
    return ConvertTo<float>(*this);
  }
  explicit EIGEN_DEVICE_FUNC operator Eigen::bfloat16() const {
    return ConvertTo<Eigen::bfloat16>(*this);
  }
  explicit EIGEN_DEVICE_FUNC operator Eigen::half() const {
    return ConvertTo<Eigen::half>(*this);
  }
  explicit EIGEN_DEVICE_FUNC operator bool() const {
    return (rep() & 0x7F) != 0;
  }
};

class float8_e5m2 : public float8_base<float8_e5m2> {
  // Exponent: 5, Mantissa: 2, bias: 15.
  // IEEE 754.
 private:
  using Base = float8_base<float8_e5m2>;
  friend class float8_base<float8_e5m2>;

  constexpr float8_e5m2(uint8_t rep, ConstructFromRepTag)
      : Base(rep, ConstructFromRepTag{}) {}

 public:
  constexpr float8_e5m2() = default;

  template <typename T,
            typename EnableIf = std::enable_if<std::is_arithmetic_v<T>>>
  explicit EIGEN_DEVICE_FUNC float8_e5m2(T f)
      : float8_e5m2(ConvertFrom(static_cast<float>(f))) {}
  explicit EIGEN_DEVICE_FUNC float8_e5m2(double f64)
      : float8_e5m2(ConvertFrom(f64)) {}
  explicit EIGEN_DEVICE_FUNC float8_e5m2(float f32)
      : float8_e5m2(ConvertFrom(f32)) {}
  explicit EIGEN_DEVICE_FUNC float8_e5m2(Eigen::bfloat16 bf16)
      : float8_e5m2(ConvertFrom(bf16)) {}
  explicit EIGEN_DEVICE_FUNC float8_e5m2(Eigen::half f16)
      : float8_e5m2(ConvertFrom(f16)) {}
  explicit EIGEN_DEVICE_FUNC float8_e5m2(float8_e4m3fn f8)
      : float8_e5m2(ConvertFrom(f8)) {}

  template <typename T,
            typename EnableIf = std::enable_if<std::is_arithmetic_v<T>>>
  explicit EIGEN_DEVICE_FUNC operator T() const {
    return static_cast<T>(static_cast<float>(*this));
  }
  explicit EIGEN_DEVICE_FUNC operator double() const {
    return ConvertTo<double>(*this);
  }
  explicit EIGEN_DEVICE_FUNC operator float() const {
    return ConvertTo<float>(*this);
  }
  explicit EIGEN_DEVICE_FUNC operator Eigen::bfloat16() const {
    return ConvertTo<Eigen::bfloat16>(*this);
  }
  explicit EIGEN_DEVICE_FUNC operator Eigen::half() const {
    return ConvertTo<Eigen::half>(*this);
  }
  explicit EIGEN_DEVICE_FUNC operator bool() const {
    return (rep() & 0x7F) != 0;
  }
};

// Structures for use in specializing std::numeric_limits.
struct numeric_limits_float8_base {
  // NOLINTBEGIN: these names must match std::numeric_limits.
  static inline constexpr const bool is_specialized = true;
  static inline constexpr const bool is_signed = true;
  static inline constexpr const bool is_integer = false;
  static inline constexpr const bool is_exact = false;
  static inline constexpr const bool has_quiet_NaN = true;
  static inline constexpr const std::float_denorm_style has_denorm =
      std::denorm_present;
  static inline constexpr const bool has_denorm_loss = false;
  static inline constexpr const std::float_round_style round_style =
      std::round_to_nearest;
  static inline constexpr const bool is_bounded = true;
  static inline constexpr const bool is_modulo = false;
  static inline constexpr const int radix = std::numeric_limits<float>::radix;
  static inline constexpr const bool traps = std::numeric_limits<float>::traps;
  static inline constexpr const bool tinyness_before =
      std::numeric_limits<float>::tinyness_before;
  // NOLINTEND
};

template <typename Derived>
struct numeric_limits_float8 {
  // NOLINTBEGIN: these names must match std::numeric_limits.
  static inline constexpr const int digits = 0;
  static inline constexpr const int digits10 = 0;
  static inline constexpr const int max_digits10 = 0;
  static inline constexpr const int min_exponent = 0;
  static inline constexpr const int min_exponent10 = 0;
  static inline constexpr const int max_exponent = 0;
  static inline constexpr const int max_exponent10 = 0;
  static inline constexpr const bool is_iec559 = false;
  static inline constexpr const bool has_infinity = false;
  static inline constexpr const bool has_signaling_NaN = false;
  // NOLINTEND
};

template <>
struct numeric_limits_float8<float8_e4m3fn>
    : public numeric_limits_float8_base {
  // NOLINTBEGIN: these names must match std::numeric_limits.
  static inline constexpr const int digits = 4;
  static inline constexpr const int digits10 = 0;      // floor(3 * log10(2));
  static inline constexpr const int max_digits10 = 3;  // ceil(4 * log10(2) + 1)
  static inline constexpr const int min_exponent = -5;
  static inline constexpr const int min_exponent10 = -1;
  static inline constexpr const int max_exponent = 9;  // Extended format.
  static inline constexpr const int max_exponent10 = 2;
  static inline constexpr const bool is_iec559 = false;
  static inline constexpr const bool has_infinity = false;
  static inline constexpr const bool has_signaling_NaN = false;
  // NOLINTEND

  static constexpr float8_e4m3fn min() { return float8_e4m3fn::FromRep(0x08); }
  static constexpr float8_e4m3fn lowest() {
    return float8_e4m3fn::FromRep(0xFE);
  }
  static constexpr float8_e4m3fn max() { return float8_e4m3fn::FromRep(0x7E); }
  static constexpr float8_e4m3fn epsilon() {
    return float8_e4m3fn::FromRep(0x20);
  }
  static constexpr float8_e4m3fn round_error() {
    return float8_e4m3fn::FromRep(0x30);
  }
  static constexpr float8_e4m3fn infinity() {
    return float8_e4m3fn::FromRep(0x7F);
  }  // NaN.
  static constexpr float8_e4m3fn quiet_NaN() {
    return float8_e4m3fn::FromRep(0x7F);
  }
  static constexpr float8_e4m3fn signaling_NaN() {
    return float8_e4m3fn::FromRep(0x7F);
  }
  static constexpr float8_e4m3fn denorm_min() {
    return float8_e4m3fn::FromRep(0x01);
  }
};

template <>
struct numeric_limits_float8<float8_e5m2> : public numeric_limits_float8_base {
  // NOLINTBEGIN: these names must match std::numeric_limits.
  static inline constexpr const int digits = 3;
  static inline constexpr const int digits10 = 0;      // floor(2 * log10(2))
  static inline constexpr const int max_digits10 = 2;  // ceil(3 * log10(2) + 1)
  static inline constexpr const int min_exponent = -13;
  static inline constexpr const int min_exponent10 = -4;
  static inline constexpr const int max_exponent = 16;
  static inline constexpr const int max_exponent10 = 4;
  static inline constexpr const bool is_iec559 = true;
  static inline constexpr const bool has_infinity = true;
  static inline constexpr const bool has_signaling_NaN = true;
  // NOLINTEND

  static constexpr float8_e5m2 min() { return float8_e5m2::FromRep(0x04); }
  static constexpr float8_e5m2 lowest() { return float8_e5m2::FromRep(0xFB); }
  static constexpr float8_e5m2 max() { return float8_e5m2::FromRep(0x7B); }
  static constexpr float8_e5m2 epsilon() { return float8_e5m2::FromRep(0x34); }
  static constexpr float8_e5m2 round_error() {
    return float8_e5m2::FromRep(0x38);
  }
  static constexpr float8_e5m2 infinity() { return float8_e5m2::FromRep(0x7C); }
  static constexpr float8_e5m2 quiet_NaN() {
    return float8_e5m2::FromRep(0x7F);
  }
  static constexpr float8_e5m2 signaling_NaN() {
    return float8_e5m2::FromRep(0x7D);
  }
  static constexpr float8_e5m2 denorm_min() {
    return float8_e5m2::FromRep(0x01);
  }
};

// Free-functions for use with ADL and in Eigen.
constexpr inline float8_e4m3fn abs(const float8_e4m3fn& a) {
  return float8_e4m3fn::FromRep(a.rep() & 0x7F);
}

constexpr inline bool isnan(const float8_e4m3fn& a) {
  return (a.rep() & 0x7F) == 0x7F;
}

constexpr inline bool isinf(const float8_e4m3fn& a) {
  return false;  // No inf representation.
}

constexpr inline float8_e5m2 abs(const float8_e5m2& a) {
  return float8_e5m2::FromRep(a.rep() & 0x7F);
}

constexpr inline bool isnan(const float8_e5m2& a) {
  return (a.rep() & 0x7F) > 0x7C;
}

constexpr inline bool isinf(const float8_e5m2& a) {
  return (a.rep() & 0x7F) == 0x7C;
}

template <typename Float8>
std::ostream& operator<<(std::ostream& os, const float8_base<Float8>& f8) {
  os << static_cast<float>(f8.derived());
  return os;
}

//==============================================================================
// Inline conversion routines between float8 and other types.
//==============================================================================

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
    } else if (from_bits == 0) {
      // Zeros.
      bits = 0;
    } else if ((from.rep() & kFromExponentMask) == 0) {
      // Subnormals.

      // All float8 subnormals become normalized when casting to a type
      // with a larger number of exponent bits.  We do this by setting the
      // normalized mantissa bits in the source type, shifting it up to the
      // destination type, then inserting the exponent bits.
      const int normalization_factor =
          absl::countl_zero(from_bits) - (kFromBits - kFromMantissaBits) + 1;
      // Shift the mantissa to account for the number of leading zeros.
      bits <<= normalization_factor;
      // Clear the hidden bit.
      bits &= ~(ToBits{1} << kToMantissaBits);
      // Insert the exponent bits.
      bits |= static_cast<ToBits>(kExponentOffset - normalization_factor + 1)
              << kToMantissaBits;
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
    const FromBits from_sign = from_bits & SetFromBit(kFromBits - 1);
    const ToBits sign = from_sign >> (kFromBits - kToBits);
    from_bits ^= from_sign;  // Zeros sign bit to obtain absolute value.

    // Special values, preserving sign.
    if (Eigen::numext::isinf(from)) {
      return sign != 0 ? -Eigen::NumTraits<To>::infinity()
                       : Eigen::NumTraits<To>::infinity();
    } else if (Eigen::numext::isnan(from)) {
      return Eigen::numext::bit_cast<To>(
          static_cast<uint8_t>(Eigen::NumTraits<To>::quiet_NaN().rep() | sign));
    }

    // Adjust mantissa.
    FromBits rounded_from_bits = from_bits;
    if constexpr (!kTruncate) {
      rounded_from_bits = RoundBitsToNearestEven(from_bits, kDigitShift);
    }
    // Zero-out tail bits.
    rounded_from_bits &= ~(SetFromBit(kDigitShift) - 1);

    // Check for overflows.
    if constexpr (kExponentOffset > 0) {
      // Shift up exponent and mantissa, add offset to adjust exponent to
      // source type.
      constexpr ToBits kToHighest = Eigen::NumTraits<To>::highest().rep();
      constexpr FromBits kHighest = ToFromBits(kToHighest);

      if (rounded_from_bits > kHighest) {
        ToBits bits =
            kSaturate ? kToHighest : Eigen::NumTraits<To>::infinity().rep();
        return Eigen::numext::bit_cast<To>(static_cast<ToBits>(bits | sign));
      }
    }

    // Subnormals and zero.
    constexpr FromBits kLowestNormal =
        ToFromBits(std::numeric_limits<To>::min().rep());
    if (rounded_from_bits < kLowestNormal) {
      // Round and shift mantissa down.
      constexpr FromBits kMantissaMask = SetFromBit(kFromMantissaBits) - 1;
      int exponent = ((from_bits >> kFromMantissaBits) - kFromExponentBias);
      int exponent_shift = kDigitShift - exponent - kToExponentBias + 1;

      // Insert the implicit leading 1 bit on the mantissa.  This assumes
      // the input is normalized.  If it is not, then the mantissa bits -
      // including the implicit one - will be shifted to zero.
      // NOTE: we need to round again from the original from_bits, otherwise
      // the lower precision bits may already be lost.  There is an edge-case
      // where rounding to a normalized value would normally round down,
      // but for a subnormal, we need to round up.
      rounded_from_bits =
          (SetFromBit(kFromMantissaBits) | (from_bits & kMantissaMask));
      ToBits bits = 0;
      // To avoid UB, limit rounding and shifting to the full mantissa plus
      // leading 1.
      if (exponent_shift <= kFromMantissaBits + 1) {
        if constexpr (!kTruncate) {
          rounded_from_bits =
              RoundBitsToNearestEven(rounded_from_bits, exponent_shift);
        }
        bits = (rounded_from_bits >> exponent_shift);
      }
      // Insert sign and return.
      return Eigen::numext::bit_cast<To>(static_cast<ToBits>(bits | sign));
    }

    // Adjust exponent.
    rounded_from_bits += static_cast<FromBits>(-kExponentOffset)
                         << kFromMantissaBits;

    // Shift bits and insert sign.
    ToBits bits =
        static_cast<ToBits>((rounded_from_bits >> kDigitShift) | sign);
    return Eigen::numext::bit_cast<To>(bits);
  }
};

template <bool kSaturate, bool kTruncate>
struct ConvertImpl<float8_e5m2, float8_e4m3fn, kSaturate, kTruncate> {
  static EIGEN_DEVICE_FUNC inline float8_e4m3fn run(const float8_e5m2& from) {
    uint8_t from_bits = from.rep();
    uint8_t sign = from_bits & 0x80;
    from_bits ^= sign;

    // Special values (NaN/Inf).
    if (from_bits > 0x7C) {
      return float8_e4m3fn::FromRep(sign | 0x7F);
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
      return float8_e4m3fn::FromRep(sign | bits);
    } else if (from_bits > 0x5F) {
      uint8_t bits = kSaturate ? 0x7E : 0x7F;
      return float8_e4m3fn::FromRep(sign | bits);
    }

    // Subtract exponent offset and shift.
    uint8_t bits = (from_bits - 0x20) << 1;
    return float8_e4m3fn::FromRep(sign | bits);
  }
};

template <bool kTruncate>
struct ConvertImpl<float8_e4m3fn, float8_e5m2, kTruncate, false> {
  static EIGEN_DEVICE_FUNC inline float8_e5m2 run(const float8_e4m3fn& from) {
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
struct ConvertImpl<float8_e4m3fn, float8_e5m2, kSaturate, kTruncate> {
  static EIGEN_DEVICE_FUNC inline float8_e5m2 run(const float8_e4m3fn& from) {
    return ConvertImpl<float8_e4m3fn, float8_e5m2, kTruncate, false>::run(from);
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

}  // namespace float8_internal

// Exported types.
using float8_e4m3fn = float8_internal::float8_e4m3fn;
using float8_e5m2 = float8_internal::float8_e5m2;

}  // namespace tsl

// Standard-library overrides.  Note that these are picked up by Eigen as well.
namespace std {
template <>
struct numeric_limits<tsl::float8_e4m3fn>
    : public tsl::float8_internal::numeric_limits_float8<tsl::float8_e4m3fn> {};

template <>
struct numeric_limits<tsl::float8_e5m2>
    : public tsl::float8_internal::numeric_limits_float8<tsl::float8_e5m2> {};

}  // namespace std

// Eigen-specific overrides.
namespace Eigen {
namespace numext {

template <>
EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC tsl::float8_e4m3fn
bit_cast<tsl::float8_e4m3fn, uint8_t>(const uint8_t& src) {
  return tsl::float8_e4m3fn::FromRep(src);
}

template <>
EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC uint8_t
bit_cast<uint8_t, tsl::float8_e4m3fn>(const tsl::float8_e4m3fn& src) {
  return src.rep();
}

template <>
EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC tsl::float8_e5m2
bit_cast<tsl::float8_e5m2, uint8_t>(const uint8_t& src) {
  return tsl::float8_e5m2::FromRep(src);
}

template <>
EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC uint8_t
bit_cast<uint8_t, tsl::float8_e5m2>(const tsl::float8_e5m2& src) {
  return src.rep();
}

}  // namespace numext

// Work-around for isinf/isnan issue on aarch64.
namespace internal {
template <>
EIGEN_DEVICE_FUNC inline bool isinf_impl<tsl::float8_e4m3fn>(
    const tsl::float8_e4m3fn& x) {
  return tsl::float8_internal::isinf(x);
}

template <>
EIGEN_DEVICE_FUNC inline bool isinf_impl<tsl::float8_e5m2>(
    const tsl::float8_e5m2& x) {
  return tsl::float8_internal::isinf(x);
}

template <>
EIGEN_DEVICE_FUNC inline bool isnan_impl<tsl::float8_e4m3fn>(
    const tsl::float8_e4m3fn& x) {
  return tsl::float8_internal::isnan(x);
}

template <>
EIGEN_DEVICE_FUNC inline bool isnan_impl<tsl::float8_e5m2>(
    const tsl::float8_e5m2& x) {
  return tsl::float8_internal::isnan(x);
}

}  // namespace internal
}  // namespace Eigen

#endif  // TENSORFLOW_TSL_PLATFORM_FLOAT8_H_
