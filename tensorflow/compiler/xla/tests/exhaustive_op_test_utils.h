/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_XLA_TESTS_EXHAUSTIVE_OP_TEST_UTILS_H_
#define TENSORFLOW_COMPILER_XLA_TESTS_EXHAUSTIVE_OP_TEST_UTILS_H_

#include <cmath>
#include <iterator>

#include "tensorflow/compiler/xla/bit_cast.h"
#include "tensorflow/compiler/xla/client/lib/constants.h"
#include "tensorflow/compiler/xla/client/lib/math.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/tests/client_library_test_base.h"
#include "tensorflow/compiler/xla/tests/literal_test_util.h"
#include "tensorflow/compiler/xla/tests/test_macros.h"

namespace xla {
using Eigen::half;

class ExhaustiveOpTestBase : public ClientLibraryTestBase {
 public:
  struct ErrorSpec {
    float abs_err;
    float rel_err;

    // If true, will consider -0 not near to +0 and vice versa.  Note that
    // +epsilon may still be considered close to -0, depending on the error
    // spec; this only covers the case when both `expected` and `actual` are
    // equal to 0.
    bool strict_signed_zeros = false;
  };

  // `ty` is the primitive type being tested.
  explicit ExhaustiveOpTestBase(PrimitiveType ty)
      : ty_(ty), platform_(client_->platform()->Name()) {
    SetFastMathDisabled(true);

    // Run all HLO passes.  In particular, constant folding is disabled by
    // default for tests, but we need to run it in order to tickle some bugs.
    mutable_debug_options()->clear_xla_disable_hlo_passes();
  }

  // Builds and runs the computation using the LocalClient API, rather than the
  // plain Client API, which is used by ClientLibraryTestBase.  This is because
  // the plain Client API results does more memcpys to/from Literals, and that's
  // slow given that we're touching a lot of data here.
  StatusOr<Literal> RunComputation(
      const XlaComputation& computation,
      absl::Span<const Literal* const> input_literals) {
    // Copy debug options from ClientLibraryTestBase.  In particular, we're
    // interested in disabling constant folding.
    ExecutableBuildOptions build_opts;
    *build_opts.mutable_debug_options() = *mutable_debug_options();

    std::vector<const Shape*> input_shapes;
    absl::c_transform(
        input_literals, std::back_inserter(input_shapes),
        [&](const Literal* input_literal) { return &input_literal->shape(); });

    TF_ASSIGN_OR_RETURN(
        auto executable,
        client_->Compile(computation, input_shapes, build_opts));

    std::vector<ScopedShapedBuffer> input_buffers;
    absl::c_transform(input_literals, std::back_inserter(input_buffers),
                      [&](const Literal* input_literal) {
                        return client_
                            ->LiteralToShapedBuffer(*input_literal,
                                                    /*device_ordinal=*/0)
                            .ConsumeValueOrDie();
                      });

    std::vector<const ShapedBuffer*> input_buffer_pointers;
    absl::c_transform(
        input_buffers, std::back_inserter(input_buffer_pointers),
        [&](const ScopedShapedBuffer& buffer) { return &buffer; });

    ExecutableRunOptions run_opts;
    run_opts.set_allocator(client_->backend().memory_allocator());
    run_opts.set_intra_op_thread_pool(
        client_->backend().eigen_intra_op_thread_pool_device());
    TF_ASSIGN_OR_RETURN(ScopedShapedBuffer result,
                        executable->Run(input_buffer_pointers, run_opts));

    TF_ASSIGN_OR_RETURN(Literal result_literal,
                        client_->ShapedBufferToLiteral(result));
    return std::move(result_literal);
  }

  // Returns the number of elements in each input literal.
  virtual int64 GetInputSize() = 0;

  Literal CreateInputLiteral() {
    return LiteralUtil::CreateFromDimensions(ty_, {GetInputSize()});
  }

  // `T` is the type of the value being compared, which is float if ty_ is of 32
  // bits or less, and double otherwise.
  template <typename T>
  bool IsClose(T expected, T actual, ErrorSpec spec) {
    static_assert(
        std::is_same<T, float>::value || std::is_same<T, double>::value,
        "Only supports float and double.");
    // Replace Inf with Max when calculating absolute or relative errors. This
    // allows the test to pass when another value are close to Inf and the
    // specified absolute or relative errors are not zero.
    T abs_err =
        std::abs(ReplaceInfWithMax(expected) - ReplaceInfWithMax(actual));
    T rel_err = abs_err / std::abs(ReplaceInfWithMax(expected));
    if (spec.strict_signed_zeros && actual == T{0} && expected == T{0}) {
      // Check sign of zero.
      return std::signbit(actual) == std::signbit(expected);
    }
    return abs_err <= spec.abs_err || rel_err <= spec.rel_err ||
           (std::isnan(expected) && std::isnan(actual)) ||
           (std::isinf(expected) && std::isinf(actual) &&
            (expected > 0) == (actual > 0));
  }

  template <typename ErrorGenerator>
  void PrintMismatch(int64* mismatches, const ErrorGenerator& err_generator) {
    // We send a few mismatches to gunit so they show up nicely in test logs.
    // Then we send more to LOG(ERROR).  The remainder we squelch unless we're
    // at vlog level 2.
    constexpr int64 kMaxMismatchesLoggedToGunit = 10;
    constexpr int64 kMaxMismatchesLoggedToErr = 1000;

    (*mismatches)++;
    if (*mismatches < kMaxMismatchesLoggedToGunit) {
      FAIL() << err_generator();
    } else if (*mismatches < kMaxMismatchesLoggedToErr || VLOG_IS_ON(2)) {
      LOG(ERROR) << err_generator();
    } else if (*mismatches == kMaxMismatchesLoggedToErr) {
      LOG(ERROR) << "Not printing any more mismatches; pass "
                    "--vmodule=exhaustive_op_test=2 to see "
                    "all of them.";
    }
  }

  template <int N>
  struct IntegralTypeWithByteWidth {};

  template <>
  struct IntegralTypeWithByteWidth<2> {
    using type = uint16;
  };

  template <>
  struct IntegralTypeWithByteWidth<4> {
    using type = uint32;
  };

  template <>
  struct IntegralTypeWithByteWidth<8> {
    using type = uint64;
  };

  // Converts part or all bits in an uint64 to the value of the floating point
  // data type being tested.
  //
  // When trying to exhaustive test for an operation of data type T, we always
  // use an integral I with the same number of bits at T to exhaustive the input
  // bit patterns for T. This bit pattern is zero extended and stored as uint64.
  // This function is used to convert such a bit pattern stored as uint64 to
  // the input value for T.
  //
  // T is the type of the floating value represented by the `bits`.
  template <typename T>
  T ConvertValue(uint64 bits) {
    using I = typename IntegralTypeWithByteWidth<sizeof(T)>::type;
    I used_bits = static_cast<I>(bits);
    return BitCast<T>(used_bits);
  }

  template <typename T>
  T ConvertAndReplaceKnownIncorrectValueWith(uint64 bits,
                                             int replacement_value = 0) {
    if (known_incorrect_fn_ && known_incorrect_fn_(bits)) {
      return static_cast<T>(replacement_value);
    }
    return ConvertValue<T>(bits);
  }

  static string StringifyNum(float x);

  static string StringifyNum(half x);

  static string StringifyNum(bfloat16 x);

  template <typename T>
  static string StringifyNum(std::complex<T> x) {
    return absl::StrCat(StringifyNum(x.real()), " ", StringifyNum(x.imag()));
  }

  template <typename T>
  static void AppendStringifyNum(std::string* s, T x) {
    absl::StrAppend(s, StringifyNum(x));
  }

  static std::function<ErrorSpec(float)> GetDefaultSpecGenerator(
      PrimitiveType ty);

  static std::vector<std::pair<int64, int64>> CreateExhaustiveF32Ranges();

 private:
  template <typename T>
  T ReplaceInfWithMax(T value) {
    if (std::isinf(value)) {
      return std::copysign(std::numeric_limits<T>::max(), value);
    }

    return value;
  }

 protected:
  // The primitive type under test.
  const PrimitiveType ty_;

  // The platform under test.
  const string platform_;

  // Testing will ignore inputs for which known_incorect_fn_ returns true. The
  // argument to the function is the raw bits for the data being test, zero
  // extended to 64 bits if the data type is less than 64 bits.
  std::function<bool(int64)> known_incorrect_fn_;

  // If true, allows denormals to be flushed to non-sign-preserving 0.
  //
  // For example, normally we'd expect sqrt(-denormal) to be either nan (sqrt of
  // a negative number) or -inf (flush the denormal to sign-perserving zero,
  // then sqrt(-0)).  But with this as true, we'll also accept 0 (sqrt(0)).
  //
  // XLA:GPU preserves denormal signs, but other backends don't.
  bool relaxed_denormal_signs_ = platform_ != "CUDA";
};

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
  class iterator
      : public std::iterator<std::input_iterator_tag,  // iterator_category
                             uint64,                   // value_type
                             uint64,                   // difference_type
                             const uint64*,            // pointer
                             uint64                    // reference
                             > {
   public:
    iterator() {}

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
    uint64 next_bit_chunk_;
  };

  iterator begin() const { return iterator(this); }
  iterator end() const {
    iterator end(this);
    return end.MoveToEnd();
  }

  explicit BitChunks(uint64 start = 0, uint64 end = 0, uint64 spacing = 1)
      : start_(start), end_(end), spacing_(spacing) {
    CHECK_GE(end_, start_);
    CHECK_NE(spacing, 0) << ToString();
  }

  int64 GetTotalBitChunks() const {
    if (start_ == end_) {
      return 1;
    }

    return 1 + (end_ - start_ + spacing_ - 1) / spacing_;
  }

  std::string ToString() const {
    return absl::StrFormat("(0x%08x, 0x%08x, 0x%08x)", start_, end_, spacing_);
  }

  uint64 start_;
  uint64 end_;
  uint64 spacing_;
};

inline string StringifyNum(BitChunks c) { return c.ToString(); }

inline string StringifyNum(BitChunks::iterator c) { return c.ToString(); }

template <typename T>
void AppendStringifyNum(std::string* s, T x) {
  absl::StrAppend(s, StringifyNum(x));
}

// Represents a set of floating point values through the possible values for
// the three components: mantissa, exponent, and sign. Also implements an
// iterator for retrieving all the represented floating point values.
class FpValues {
 public:
  static constexpr uint kTotalBitChunks = 3;

  class iterator
      : public std::iterator<std::input_iterator_tag,  // iterator_category
                             uint64,                   // value_type
                             uint64,                   // difference_type
                             const uint64*,            // pointer
                             uint64                    // reference
                             > {
   public:
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

    uint64 operator*() const {
      uint64 value = 0;
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
        uint64 bound = 1ull << total_bits;
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

  int64 GetTotalNumValues() const {
    int64 total = 1;
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
  static_assert(std::is_same<T, float>::value || std::is_same<T, double>::value,
                "Only supports float and double.");
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
uint64 GetAllOneMantissa() {
  return (1ull << GetMantissaTotalBits<T>()) - 1ull;
}

template <typename T>
uint64 GetAllOneExponent() {
  return (1ull << GetExponentTotalBits<T>()) - 1ull;
}

template <typename T>
FpValues GetFpValues(BitChunks mantissa, BitChunks exponent, BitChunks sign) {
  static_assert(std::is_same<T, float>::value || std::is_same<T, double>::value,
                "Only supports float and double.");
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
  uint64 mantissa_spacing = (1ull << mantissa) / (approx_num_values * 2);
  return GetFpValues<T>(
      BitChunks(0x1, GetAllOneMantissa<T>(), mantissa_spacing),
      BitChunks(0, 0, 1), BitChunks(0, 1, 1));
}

template <typename T>
FpValues GetInfinites() {
  uint64 all_one_exp = GetAllOneExponent<T>();
  return GetFpValues<T>(BitChunks(0, 0, 1),
                        BitChunks(all_one_exp, all_one_exp, 1),
                        BitChunks(0, 1, 1));
}

template <typename T>
FpValues GetNans(int approx_num_values) {
  int mantissa = GetMantissaTotalBits<T>();
  uint64 mantissa_spacing = (1ull << mantissa) / (approx_num_values * 2);
  uint64 all_one_exp = GetAllOneExponent<T>();
  return GetFpValues<T>(
      BitChunks(0x1, GetAllOneMantissa<T>(), mantissa_spacing),
      BitChunks(all_one_exp, all_one_exp, 1), BitChunks(0, 1, 1));
}

template <typename T>
FpValues GetNormals(int approx_num_values) {
  float component_total = std::sqrtf(approx_num_values);
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
std::vector<FpValues> GetFpValuesWithExponents(uint64 first_exponent,
                                               uint64 exponent_spacing,
                                               uint64 num_exponents,
                                               uint64 approx_num_values,
                                               uint64 num_values_per_group) {
  const uint64 num_signs = 2;
  uint64 approx_num_mantissa = approx_num_values / (num_exponents * num_signs);
  uint64 num_mantissa_per_group =
      num_values_per_group / (num_exponents * num_signs);
  CHECK_GT(approx_num_mantissa, 0);
  CHECK_GT(num_mantissa_per_group, 0);

  CHECK_LT(first_exponent + num_exponents - 1ull, GetAllOneExponent<T>());
  int mantissa = GetMantissaTotalBits<T>();
  uint64 mantissa_spacing = (1ull << mantissa) / approx_num_mantissa;

  std::vector<FpValues> result;
  for (uint64 group_start = 0; group_start < GetAllOneMantissa<T>();
       group_start += mantissa_spacing * num_mantissa_per_group) {
    uint64 group_end =
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
    uint64 approx_num_values = 40000, uint64 num_values_per_group = 4000) {
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

}  // namespace xla
#endif  // TENSORFLOW_COMPILER_XLA_TESTS_EXHAUSTIVE_OP_TEST_UTILS_H_
