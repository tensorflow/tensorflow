/* Copyright 2019 The OpenXLA Authors.

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

#ifndef XLA_TESTS_EXHAUSTIVE_EXHAUSTIVE_OP_TEST_UTILS_H_
#define XLA_TESTS_EXHAUSTIVE_EXHAUSTIVE_OP_TEST_UTILS_H_

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <initializer_list>
#include <iterator>
#include <limits>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/hlo/builder/xla_builder.h"
#include "xla/literal.h"
#include "xla/primitive_util.h"
#include "xla/tests/exhaustive/error_spec.h"
#include "xla/types.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace exhaustive_op_test {

// The primitive type used to compute the reference output.
constexpr PrimitiveType Ref(PrimitiveType T) {
  return (!primitive_util::IsFloatingPointType(T) || T == F64) ? T : F32;
}

// The primitive type of the component of T. If T is not complex, then
// ComponentT = T.
constexpr PrimitiveType Component(PrimitiveType T) {
  return primitive_util::IsComplexType(T)
             ? primitive_util::ComplexComponentType(T)
             : T;
}

// Associates constants and types with a PrimitiveType (T) and number of test
// arguments (N) for the exhaustive test infrastructure.
template <PrimitiveType T, size_t N>
class ExhaustiveOpTestTraits {
 public:
  static constexpr PrimitiveType kT = T;
  static constexpr size_t kN = N;

  static constexpr bool kIsComplex = primitive_util::IsComplexType(T);
  static constexpr PrimitiveType kRef = Ref(T);

  static constexpr PrimitiveType kComponent = Component(T);
  static constexpr PrimitiveType kComponentRef = Component(kRef);
  // The PrimitiveType of the associated unsigned integer to use T with
  // bitcasting.
  static constexpr PrimitiveType kComponentIntegral =
      primitive_util::UnsignedIntegralTypeForBitWidth(
          primitive_util::BitWidth(kComponent));
  static constexpr PrimitiveType kComponentIntegralRef =
      primitive_util::UnsignedIntegralTypeForBitWidth(
          primitive_util::BitWidth(kComponentRef));

  using NativeT = primitive_util::NativeTypeOf<T>;
  using NativeRefT = primitive_util::NativeTypeOf<kRef>;
  using ComponentNativeT = primitive_util::NativeTypeOf<kComponent>;
  using ComponentNativeRefT = primitive_util::NativeTypeOf<kComponentRef>;
  using ComponentIntegralNativeT =
      primitive_util::NativeTypeOf<kComponentIntegral>;
  using ComponentIntegralNativeRefT =
      primitive_util::NativeTypeOf<kComponentIntegralRef>;

  using NativeInputs = std::array<NativeT, N>;
  // N spans corresponding to the list of literal data values.
  using NativeListInputs = std::array<absl::Span<const NativeT>, N>;
  using NativeRefInputs = std::array<NativeRefT, N>;
  using LiteralInputs = std::array<Literal, N>;
  using XlaInputs = std::array<XlaOp, N>;

  using EnqueueOp = std::conditional_t<
      N == 1, std::function<XlaOp(XlaOp)>,
      std::conditional_t<N == 2, std::function<XlaOp(XlaOp, XlaOp)>,
                         std::enable_if_t<N == 1 || N == 2, void>>>;
  using EvaluateOp = std::conditional_t<
      N == 1, NativeRefT (*)(NativeRefT),
      std::conditional_t<N == 2, NativeRefT (*)(NativeRefT, NativeRefT),
                         std::enable_if_t<N == 1 || N == 2, void>>>;
  using OutputRangeCheck = std::function<bool(NativeInputs, NativeT)>;

  using ErrorSpecGen = std::conditional_t<
      N == 1, std::function<ErrorSpec(NativeT)>,
      std::conditional_t<N == 2, std::function<ErrorSpec(NativeT, NativeT)>,
                         std::enable_if_t<N == 1 || N == 2, void>>>;
  using ErrorSpecGenFnPtr = std::conditional_t<
      N == 1, ErrorSpec (*)(NativeT),
      std::conditional_t<N == 2, ErrorSpec (*)(NativeT, NativeT),
                         std::enable_if_t<N == 1 || N == 2, void>>>;

  // Returns an ErrorSpecGen that sets no error tolerances.
  //
  // The intention of this default is to force test writers to tighten bounds at
  // least somewhat and not rely on overly large default tolerances.
  static ErrorSpecGen FallbackErrorSpecGen();
};

template <PrimitiveType T, size_t N>
ErrorSpec DefaultSpecGenerator(typename ExhaustiveOpTestTraits<T, N>::NativeT);

template <PrimitiveType T, size_t N>
ErrorSpec DefaultSpecGenerator(typename ExhaustiveOpTestTraits<T, N>::NativeT,
                               typename ExhaustiveOpTestTraits<T, N>::NativeT);

// The following two constants set the default absolute and relative error
// tolerance in units of the smallest normalized value and the relative accuracy
// of the format, respectively. Notice that setting an absolute tolerance above
// the value of the smallest normalized float means that we effectively ignore
// relative errors in values at or below the subnormal boundary (e.g. for values
// less than ~1e-38 for FP32).
static constexpr float kDefaultAbsoluteToleranceSlackFactor = 2;
static constexpr float kDefaultRelativeToleranceSlackFactor = 20;

template <>
inline ErrorSpec DefaultSpecGenerator<C128, 1>(xla::complex128) {
  double atol = kDefaultAbsoluteToleranceSlackFactor *
                std::numeric_limits<double>::min();  // NOLINT
  double rtol = kDefaultRelativeToleranceSlackFactor *
                std::numeric_limits<double>::epsilon();
  return ErrorSpec::Builder().abs_err(atol).rel_err(rtol).build();
}

template <>
inline ErrorSpec DefaultSpecGenerator<C64, 1>(xla::complex64) {
  double atol = kDefaultAbsoluteToleranceSlackFactor *
                std::numeric_limits<float>::min();  // NOLINT
  double rtol = 40 * std::numeric_limits<float>::epsilon();
  return ErrorSpec::Builder().abs_err(atol).rel_err(rtol).build();
}

template <>
inline ErrorSpec DefaultSpecGenerator<F64, 1>(double) {
  double atol = kDefaultAbsoluteToleranceSlackFactor *
                std::numeric_limits<double>::min();  // NOLINT
  double rtol = kDefaultRelativeToleranceSlackFactor *
                std::numeric_limits<double>::epsilon();
  return ErrorSpec::Builder().abs_err(atol).rel_err(rtol).build();
}

template <>
inline ErrorSpec DefaultSpecGenerator<F32, 1>(float) {
  double atol = kDefaultAbsoluteToleranceSlackFactor *
                std::numeric_limits<float>::min();  // NOLINT
  double rtol = kDefaultRelativeToleranceSlackFactor *
                std::numeric_limits<float>::epsilon();
  return ErrorSpec::Builder().abs_err(atol).rel_err(rtol).build();
}

template <>
inline ErrorSpec DefaultSpecGenerator<F16, 1>(xla::half) {
  double atol = kDefaultAbsoluteToleranceSlackFactor *
                std::numeric_limits<Eigen::half>::min();  // NOLINT
  // epsilon for FP16 is quite large, so a slack factor of 5 suffices.
  double rtol = 5 * std::numeric_limits<Eigen::half>::epsilon();
  return ErrorSpec::Builder().abs_err(atol).rel_err(rtol).build();
}

template <>
inline ErrorSpec DefaultSpecGenerator<BF16, 1>(xla::bfloat16) {
  double atol = kDefaultAbsoluteToleranceSlackFactor *
                std::numeric_limits<bfloat16>::min();  // NOLINT
  // epsilon for BF16 is quite large, so a slack factor of 2 suffices.
  double rtol = 2 * std::numeric_limits<bfloat16>::epsilon();
  return ErrorSpec::Builder().abs_err(atol).rel_err(rtol).build();
}

template <>
inline ErrorSpec DefaultSpecGenerator<F8E4M3FN, 1>(tsl::float8_e4m3fn) {
  return ErrorSpec::Builder().strict_signed_zeros().build();
}

template <>
inline ErrorSpec DefaultSpecGenerator<F8E5M2, 1>(tsl::float8_e5m2) {
  return ErrorSpec::Builder().strict_signed_zeros().build();
}

template <>
inline ErrorSpec DefaultSpecGenerator<F64, 2>(double, double) {
  double atol = kDefaultAbsoluteToleranceSlackFactor *
                std::numeric_limits<double>::min();  // NOLINT
  double rtol = kDefaultRelativeToleranceSlackFactor *
                std::numeric_limits<double>::epsilon();
  return ErrorSpec::Builder().abs_err(atol).rel_err(rtol).build();
}

template <>
inline ErrorSpec DefaultSpecGenerator<F32, 2>(float, float) {
  double atol = kDefaultAbsoluteToleranceSlackFactor *
                std::numeric_limits<float>::min();  // NOLINT
  double rtol = kDefaultRelativeToleranceSlackFactor *
                std::numeric_limits<float>::epsilon();
  return ErrorSpec::Builder().abs_err(atol).rel_err(rtol).build();
}

template <>
inline ErrorSpec DefaultSpecGenerator<F16, 2>(xla::half, xla::half) {
  double atol = kDefaultAbsoluteToleranceSlackFactor *
                std::numeric_limits<Eigen::half>::min();  // NOLINT
  // epsilon for FP16 is quite large, so a slack factor of 5 suffices.
  double rtol = 5 * std::numeric_limits<Eigen::half>::epsilon();
  return ErrorSpec::Builder().abs_err(atol).rel_err(rtol).build();
}

template <>
inline ErrorSpec DefaultSpecGenerator<BF16, 2>(bfloat16, bfloat16) {
  double atol = kDefaultAbsoluteToleranceSlackFactor *
                std::numeric_limits<bfloat16>::min();  // NOLINT
  // epsilon for BF16 is quite large, so a slack factor of 5 suffices.
  double rtol = 2 * std::numeric_limits<bfloat16>::epsilon();
  return ErrorSpec::Builder().abs_err(atol).rel_err(rtol).build();
}

template <>
inline ErrorSpec DefaultSpecGenerator<F8E4M3FN, 2>(tsl::float8_e4m3fn,
                                                   tsl::float8_e4m3fn) {
  return ErrorSpec::Builder().strict_signed_zeros().build();
}

template <>
inline ErrorSpec DefaultSpecGenerator<F8E5M2, 2>(tsl::float8_e5m2,
                                                 tsl::float8_e5m2) {
  return ErrorSpec::Builder().strict_signed_zeros().build();
}

template <PrimitiveType T, size_t N>
typename ExhaustiveOpTestTraits<T, N>::ErrorSpecGen GetDefaultSpecGenerator() {
  // Select overload by casting to fn ptr type.
  return static_cast<typename ExhaustiveOpTestTraits<T, N>::ErrorSpecGenFnPtr>(
      DefaultSpecGenerator<T, N>);
}

template <typename Traits>
typename Traits::ErrorSpecGen PickFirstErrorSpecGenPresent(
    std::initializer_list<typename Traits::ErrorSpecGen> error_specs) {
  typename Traits::ErrorSpecGen ret = Traits::FallbackErrorSpecGen();
  for (auto it = error_specs.begin(); it != error_specs.end(); it++) {
    // Check if the ErrorSpecGen is nullptr to indicate it is not set. Replace
    // ret with the first non-nullptr ErrorSpecGen.
    if (*it != nullptr) {
      ret = *it;
      break;
    }
  }
  return ret;
}

// Determines if the real component of the complex number is subnormal (either
// sign).
//
// See also IsSubnormal to check if either component is subnormal.
bool IsSubnormalReal(xla::complex64);
bool IsSubnormalReal(xla::complex128);

// Determines if the real component of the complex number is the minimum
// normal floating point value (either sign).
//
// See also IsMinPositive to check if either component is the minimum normal
// floating point value.
bool IsMinNormalReal(xla::complex64);
bool IsMinNormalReal(xla::complex128);

// Determines if the imaginary component of the complex number is subnormal
// (either sign).
//
// See also IsSubnormal to check if either component is subnormal.
bool IsSubnormalImaginary(xla::complex64);
bool IsSubnormalImaginary(xla::complex128);

// Determines if the imaginary component of the complex number is the minimum
// normal floating point value (either sign).
//
// See also IsMinPositive to check if either component is the minimum normal
// floating point value.
bool IsMinNormalImaginary(xla::complex64);
bool IsMinNormalImaginary(xla::complex128);

// Determines if the NativeT is subnormal (either sign).
//
// For complex numbers, this will return true if either real or imaginary
// component is subnormal. See IsSubnormalReal and IsSubnormalImaginary if you
// only care about one component.
template <typename NativeT>
bool IsSubnormal(NativeT value) {
  if constexpr (std::is_same_v<NativeT, xla::complex64> ||
                std::is_same_v<NativeT, xla::complex128>) {
    return IsSubnormalReal(value) || IsSubnormalImaginary(value);
  } else {
    return std::fpclassify(value) == FP_SUBNORMAL;
  }
}

// Determines if the NativeT is the minimum normal floating point value
// (either sign).
//
// For complex numbers, this will return true if either real or imaginary
// component is the minimum normal floating point value. See IsMinPositiveReal
// and IsMinPositiveImaginary if you only care about one component.
template <typename NativeT>
bool IsMinNormal(NativeT value) {
  if constexpr (std::is_same_v<NativeT, xla::complex64> ||
                std::is_same_v<NativeT, xla::complex128>) {
    return IsMinNormalReal(value) || IsMinNormalImaginary(value);
  } else {
    return std::abs(value) == std::numeric_limits<NativeT>::min();  // NOLINT
  }
}

// Determines if the NativeT is subnormal or the minimum normal floating point
// value (either sign).
//
// For complex numbers, this will return true if either real or imaginary
// component is subnormal or the minimum normal floating point value.
template <typename NativeT>
bool IsSubnormalOrMinNormal(NativeT value) {
  return IsSubnormal(value) || IsMinNormal(value);
}

// Represents a set of 64 bit chunks by representing the starting bit chunk,
// the last bit chunk, and the spacing between two adjacent bit chunks, without
// actually storing all the bit chunks being generated. The bit chunk iterator
// is provided to retrieve all the bit chunks.
//
// This data structure is used to generate the bit representation to test
// operations that requires more than 64 bit input data. In this case,
// truly exhaustive testing is not possible and we want to test a value every
// n values, where n == spacing_.
//
// Currently, the iterator of BitChunks adds the `spacing_` to a bit chunk to
// compute the next bit chunk. We can change this to use values generated
// by a random number generator that can achieve the average spacing
// statistically, if we will find this is necessary.
class BitChunks {
 public:
  class iterator {
   public:
    using iterator_category = std::input_iterator_tag;
    using value_type = uint64_t;
    using difference_type = uint64_t;
    using pointer = const uint64_t*;
    using reference = uint64_t;

    iterator() = default;

    explicit iterator(const BitChunks* bit_chunks)
        : bit_chunks_(bit_chunks), next_bit_chunk_(bit_chunks->start_) {}

    iterator& operator++() {
      Next();
      return *this;
    }

    iterator operator++(int) {
      iterator retval = *this;
      Next();
      return retval;
    }

    bool operator==(iterator other) const {
      return bit_chunks_ == other.bit_chunks_ &&
             next_bit_chunk_ == other.next_bit_chunk_;
    }

    bool operator!=(iterator other) const { return !(*this == other); }

    iterator MoveToEnd() {
      MoveNextBitChunkToOnePassEnd();
      return *this;
    }

    reference operator*() const {
      CHECK(*this != this->bit_chunks_->end());
      return next_bit_chunk_;
    }

    const BitChunks* GetBitChunks() const { return bit_chunks_; }

    void Reset() { next_bit_chunk_ = bit_chunks_->start_; }

    void Next() {
      CHECK(*this != this->bit_chunks_->end());
      if (next_bit_chunk_ == bit_chunks_->end_) {
        MoveNextBitChunkToOnePassEnd();
      } else {
        next_bit_chunk_ += bit_chunks_->spacing_;
        if (next_bit_chunk_ > bit_chunks_->end_) {
          next_bit_chunk_ = bit_chunks_->end_;
        }
      }
    }

    std::string ToString() const {
      return absl::StrFormat("0x%08x", next_bit_chunk_);
    }

   private:
    // Move next_bit_chunk_ to 1 pass the bit_chunks_->end, to mark that the
    // iterator has reached the end. When spacing_ is not one, or if we will
    // change to use a random value instead of spacing_ in function Next(),
    // normalizing the representation of the iterator ending this way can
    // can simplify the checking for iterator ending.
    void MoveNextBitChunkToOnePassEnd() {
      next_bit_chunk_ = bit_chunks_->end_ + 1;
    }

    const BitChunks* bit_chunks_;
    uint64_t next_bit_chunk_;
  };

  iterator begin() const { return iterator(this); }
  iterator end() const {
    iterator end(this);
    return end.MoveToEnd();
  }

  explicit BitChunks(uint64_t start = 0, uint64_t end = 0, uint64_t spacing = 1)
      : start_(start), end_(end), spacing_(spacing) {
    CHECK_GE(end_, start_);
    CHECK_NE(spacing, 0) << ToString();
  }

  int64_t GetTotalBitChunks() const {
    if (start_ == end_) {
      return 1;
    }

    return 1 + (end_ - start_ + spacing_ - 1) / spacing_;
  }

  std::string ToString() const {
    return absl::StrFormat("(0x%08x, 0x%08x, 0x%08x)", start_, end_, spacing_);
  }

  uint64_t start_;
  uint64_t end_;
  uint64_t spacing_;
};

inline std::string StringifyNum(BitChunks c) { return c.ToString(); }

inline std::string StringifyNum(BitChunks::iterator c) { return c.ToString(); }

template <typename T>
void AppendStringifyNum(std::string* s, T x) {
  absl::StrAppend(s, StringifyNum(x));
}

// Represents a set of floating point values through the possible values for
// the three components: mantissa, exponent, and sign. Also implements an
// iterator for retrieving all the represented floating point values.
class FpValues {
 public:
  static constexpr int kTotalBitChunks = 3;

  class iterator {
   public:
    using iterator_category = std::input_iterator_tag;
    using value_type = uint64_t;
    using difference_type = uint64_t;
    using pointer = const uint64_t*;
    using reference = uint64_t;

    explicit iterator(const FpValues* fp_values) : fp_values_(fp_values) {
      for (int i = 0; i < FpValues::kTotalBitChunks; ++i) {
        iters_[i] = BitChunks::iterator(&fp_values->GetBitChunks(i));
      }
    }

    iterator& operator++() {
      Next();
      return *this;
    }

    iterator operator++(int) {
      iterator retval = *this;
      Next();
      return retval;
    }

    bool operator==(iterator other) const {
      for (int i = 0; i < FpValues::kTotalBitChunks; ++i) {
        if (iters_[i] != other.GetBitChunksIter(i)) {
          return false;
        }
      }
      return true;
    }

    bool operator!=(iterator other) const { return !(*this == other); }

    iterator MoveToEnd() {
      for (int i = 0; i < FpValues::kTotalBitChunks; ++i) {
        iters_[i].MoveToEnd();
      }
      return *this;
    }

    uint64_t operator*() const {
      uint64_t value = 0;
      for (int i = 0; i < FpValues::kTotalBitChunks; ++i) {
        value = value | (*iters_[i]) << fp_values_->offsets_[i];
      }
      return value;
    }

    const BitChunks::iterator& GetBitChunksIter(int i) { return iters_[i]; }

    std::string ToString() const {
      return absl::StrJoin(iters_, ",",
                           AppendStringifyNum<BitChunks::iterator>);
    }

   private:
    // Moves the iterator for the ith BitChunks to the next value, and
    // returns true if the new state is not the end of the iterator.
    bool Next(int i = 0) {
      iters_[i].Next();
      if (iters_[i] == iters_[i].GetBitChunks()->end()) {
        if (i == FpValues::kTotalBitChunks - 1) {
          return false;
        }
        if (Next(i + 1)) {
          iters_[i].Reset();
          return true;
        }
        return false;
      }
      return true;
    }

    std::array<BitChunks::iterator, FpValues::kTotalBitChunks> iters_;
    const FpValues* fp_values_;
  };

  FpValues() : bit_chunks_(), offsets_() {}
  FpValues(absl::Span<const BitChunks> chunks, absl::Span<const int> offsets) {
    CHECK_EQ(chunks.size(), offsets.size() - 1);
    CHECK_EQ(chunks.size(), kTotalBitChunks);
    std::copy_n(chunks.begin(), kTotalBitChunks, bit_chunks_.begin());
    std::copy_n(offsets.begin(), kTotalBitChunks, offsets_.begin());

    // The last value in `offsets` is the total number of bits.
    offsets_[kTotalBitChunks] = offsets[kTotalBitChunks];
    // Validate the input values.
    for (int i = 0; i < kTotalBitChunks; ++i) {
      int total_bits = offsets[i + 1] - offsets[i];
      if (total_bits < 64) {
        uint64_t bound = 1ull << total_bits;
        CHECK_LT(chunks[i].start_, bound);
        CHECK_LT(chunks[i].end_, bound);
      } else {
        CHECK_EQ(total_bits, 64);
      }
    }
  }

  iterator begin() const { return iterator(this); }

  iterator end() const {
    iterator end(this);
    return end.MoveToEnd();
  }

  int64_t GetTotalNumValues() const {
    int64_t total = 1;
    absl::c_for_each(bit_chunks_, [&](const BitChunks& chunks) {
      total *= chunks.GetTotalBitChunks();
    });
    return total;
  }

  const BitChunks& GetBitChunks(int i) const { return bit_chunks_[i]; }

  std::string ToString() const {
    return absl::StrCat(
        "[", absl::StrJoin(bit_chunks_, ",", AppendStringifyNum<BitChunks>),
        "]");
  }

  std::array<BitChunks, kTotalBitChunks> bit_chunks_;
  std::array<int, kTotalBitChunks + 1> offsets_;
};

template <typename T>
int GetMantissaTotalBits() {
  return std::numeric_limits<T>::digits - 1;
}

template <typename T>
int GetFpTotalBits() {
  return sizeof(T) * 8;
}

template <typename T>
int GetExponentTotalBits() {
  return GetFpTotalBits<T>() - GetMantissaTotalBits<T>() - 1;
}

template <typename T>
uint64_t GetAllOneMantissa() {
  return (1ull << GetMantissaTotalBits<T>()) - 1ull;
}

template <typename T>
uint64_t GetAllOneExponent() {
  return (1ull << GetExponentTotalBits<T>()) - 1ull;
}

template <typename T>
FpValues GetFpValues(BitChunks mantissa, BitChunks exponent, BitChunks sign) {
  int total_bits = GetFpTotalBits<T>();
  return FpValues({mantissa, exponent, sign},
                  {0, GetMantissaTotalBits<T>(), total_bits - 1, total_bits});
}

template <typename T>
FpValues GetZeros() {
  return GetFpValues<T>(BitChunks(0, 0, 1), BitChunks(0, 0, 1),
                        BitChunks(0, 1, 1));
}

template <typename T>
FpValues GetSubnormals(int approx_num_values) {
  int mantissa = GetMantissaTotalBits<T>();
  uint64_t mantissa_spacing = (1ull << mantissa) / (approx_num_values * 2);
  return GetFpValues<T>(
      BitChunks(0x1, GetAllOneMantissa<T>(), mantissa_spacing),
      BitChunks(0, 0, 1), BitChunks(0, 1, 1));
}

template <typename T>
FpValues GetInfinites() {
  uint64_t all_one_exp = GetAllOneExponent<T>();
  return GetFpValues<T>(BitChunks(0, 0, 1),
                        BitChunks(all_one_exp, all_one_exp, 1),
                        BitChunks(0, 1, 1));
}

template <typename T>
FpValues GetNans(int approx_num_values) {
  int mantissa = GetMantissaTotalBits<T>();
  uint64_t mantissa_spacing = (1ull << mantissa) / (approx_num_values * 2);
  uint64_t all_one_exp = GetAllOneExponent<T>();
  return GetFpValues<T>(
      BitChunks(0x1, GetAllOneMantissa<T>(), mantissa_spacing),
      BitChunks(all_one_exp, all_one_exp, 1), BitChunks(0, 1, 1));
}

template <typename T>
FpValues GetNormals(int approx_num_values) {
  float component_total = std::sqrt(static_cast<float>(approx_num_values));
  return GetFpValues<T>(
      BitChunks(0x1, GetAllOneMantissa<T>(),
                (1ull << (GetMantissaTotalBits<T>() + 1)) / component_total),
      BitChunks(0x1, GetAllOneExponent<T>() - 1,
                (1ull << (GetExponentTotalBits<T>() + 1)) / component_total),
      BitChunks(0, 1, 1));
}

// Returns a vector of FpValues, which together represent about
// `approx_num_values` floating point values of type `T`, with each FpValues
// represents about `num_values_per_group` floating point values.
template <typename T>
std::vector<FpValues> GetFpValuesWithExponents(uint64_t first_exponent,
                                               uint64_t exponent_spacing,
                                               uint64_t num_exponents,
                                               uint64_t approx_num_values,
                                               uint64_t num_values_per_group) {
  const uint64_t num_signs = 2;
  uint64_t approx_num_mantissa =
      approx_num_values / (num_exponents * num_signs);
  uint64_t num_mantissa_per_group =
      num_values_per_group / (num_exponents * num_signs);
  CHECK_GT(approx_num_mantissa, 0);
  CHECK_GT(num_mantissa_per_group, 0);

  CHECK_LT(first_exponent + num_exponents - 1ull, GetAllOneExponent<T>());
  int mantissa = GetMantissaTotalBits<T>();
  uint64_t mantissa_spacing = (1ull << mantissa) / approx_num_mantissa;

  std::vector<FpValues> result;
  for (uint64_t group_start = 0; group_start < GetAllOneMantissa<T>();
       group_start += mantissa_spacing * num_mantissa_per_group) {
    uint64_t group_end =
        group_start + (num_mantissa_per_group - 1) * mantissa_spacing;
    if (group_end > GetAllOneMantissa<T>()) {
      group_end = GetAllOneMantissa<T>();
    }
    result.push_back(GetFpValues<T>(
        BitChunks(group_start, group_end, mantissa_spacing),
        BitChunks(first_exponent, first_exponent + num_exponents - 1, 1),
        BitChunks(0, 1, 1)));
  }
  return result;
}

// Returns a vector of FpValues together represent about `approx_num_values`
// "very large" floating point values and `approx_num_values` "very small"
// floating point values of type `T`, which each FpValues represent about
// `num_values_per_group` floating point values. Because we use FpValues as
// a parameter for parameterized testing, the number of floating values
// represented by each FpValues affects the input size for each sub-test and
// the hence the peak memory usage of the test.
template <typename T>
std::vector<FpValues> GetFpValuesForMagnitudeExtremeNormals(
    uint64_t approx_num_values = 40000, uint64_t num_values_per_group = 4000) {
  std::vector<FpValues> large =
      GetFpValuesWithExponents<T>(GetAllOneExponent<T>() - 5, 1, 5,
                                  approx_num_values / 2, num_values_per_group);
  std::vector<FpValues> small = GetFpValuesWithExponents<T>(
      1, 1, 5, approx_num_values / 2, num_values_per_group);
  large.insert(large.end(), small.begin(), small.end());
  return large;
}

template <typename T>
std::vector<FpValues> CreateFpValuesForBoundaryTest() {
  return {GetZeros<T>(), GetSubnormals<T>(1000), GetInfinites<T>(),
          GetNans<T>(1000)};
}

// Creates ranges for exhaustively testing T pairs, (T1, T2), where T1 and T2
// both only range over the subnormal values for T.
//
// This is intended to be used for exhaustive binary tests where it is helpful
// to only look at how subnormals interact for both parameters.
//
// Ranges are encoded as a tuple of `int64_t`. Each `int64_t` tuple element is a
// packed pair of values equal to `(T1 << TotalBits(T)) | T2`. A range
// `((T1,T2),(T3,T4))` will test all binary pairs between `(T1,T2)`, inclusive,
// and `(T3,T4)`, exclusive.
//
// Any `T` supported by `std::numeric_limits<T>` is supported here.
template <typename T>
inline std::vector<std::pair<int64_t, int64_t>> CreateSubnormalStrictRanges() {
  std::vector<std::pair<int64_t, int64_t>> ret;
  // N.B.: Exclude 0.
  int subnormal_count = (1ull << GetMantissaTotalBits<T>()) - 1;
  // N.B.: subnormal_count / 2 is intended to provide mantissa spacing of 1.
  for (auto subnormal :
       GetSubnormals<T>((1ull << GetMantissaTotalBits<T>()) / 2)) {
    // | 1 avoids selecting 0 as the first right value.
    auto start = (subnormal << GetFpTotalBits<T>()) | 1;
    auto end = start + subnormal_count;
    ret.push_back({start, end});
  }
  return ret;
}

// Creates ranges for exhaustively testing T pairs, (T1, T2), where T1 only
// ranges over the subnormal values for T and T2 ranges over all values.
//
// This is intended to be used for exhaustive binary tests where it is helpful
// to only look at how subnormals interact for both parameters.
//
// Ranges are encoded as a tuple of `int64_t`. Each `int64_t` tuple element is a
// packed pair of values equal to `(T1 << TotalBits(T)) | T2`. A range
// `((T1,T2),(T3,T4))` will test all binary pairs between `(T1,T2)`, inclusive,
// and `(T3,T4)`, exclusive.
//
// Any `T` supported by `std::numeric_limits<T>` is supported here.
template <typename T>
inline std::vector<std::pair<int64_t, int64_t>>
CreateSubnormalExhaustiveRanges() {
  std::vector<std::pair<int64_t, int64_t>> ret;
  int entire_count = 1ull << GetFpTotalBits<T>();
  // N.B.: subnormal_count / 2 is intended to provide mantissa spacing of 1.
  for (auto subnormal :
       GetSubnormals<T>((1ull << GetMantissaTotalBits<T>()) / 2)) {
    auto start = subnormal << GetFpTotalBits<T>();
    auto end = start + entire_count;
    ret.push_back({start, end});
  }
  return ret;
}

inline std::vector<std::pair<int64_t, int64_t>> CreateExhaustiveU16Ranges() {
  // The entire U16 range is small enough that we don't need to do any
  // partitioning.
  return {{0, std::numeric_limits<uint16_t>::max()}};
}

inline std::vector<std::pair<int64_t, int64_t>> CreateExhaustiveU32Ranges() {
  // We break up the 2^32-element space into small-ish chunks to keep peak
  // memory usage low.
  std::vector<std::pair<int64_t, int64_t>> result;
  const int64_t step = 1 << 25;
  for (int64_t i = 0; i < (int64_t{1} << 32); i += step) {
    result.push_back({i, i + step});
  }
  return result;
}

template <typename T>
T ReferenceMax(T x, T y) {
  if (x != x) {
    return x;
  }
  if (y != y) {
    return y;
  }

  return ToSignMagnitude(x) < ToSignMagnitude(y) ? y : x;
}

template <typename T>
T ReferenceMin(T x, T y) {
  if (x != x) {
    return x;
  }
  if (y != y) {
    return y;
  }

  return ToSignMagnitude(x) < ToSignMagnitude(y) ? x : y;
}

// Returns a wrapper of the given build method, which build an HLO operation
// with an empty broadcast dimension.
inline std::function<XlaOp(XlaOp, XlaOp)> AddEmptyBroadcastDimension(
    std::function<XlaOp(XlaOp, XlaOp, absl::Span<const int64_t>)>
        build_method) {
  return [build_method](XlaOp src0, XlaOp src1) -> XlaOp {
    return build_method(src0, src1, {});
  };
}

}  // namespace exhaustive_op_test
}  // namespace xla
#endif  // XLA_TESTS_EXHAUSTIVE_EXHAUSTIVE_OP_TEST_UTILS_H_
