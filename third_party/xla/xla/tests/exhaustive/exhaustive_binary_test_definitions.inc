/* Copyright 2024 The OpenXLA Authors.

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

// Exhaustive test for binary operations for 8-bit floating point types,
// including float16 and bfloat.
//
// Test parameter is a pair of (begin, end) for range under test.
template <PrimitiveType T, bool kLeftToRightPacking = false>
class Exhaustive8BitBinaryTest
    : public ExhaustiveBinaryTest<T>,
      public ::testing::WithParamInterface<std::pair<int64_t, int64_t>> {
 public:
  int64_t GetInputSize() override {
    int64_t begin, end;
    std::tie(begin, end) = GetParam();
    return end - begin;
  }

  // Given a range of uint64_t representation, uses bits 7..0 and bits 15..8
  // for the values of src0 and src1 (see below for ordering) for the 8-bit
  // binary operation being tested, and generate the cartesian product of the
  // two sets as the two inputs for the test.
  //
  // If `kLeftToRightPacking == true`, then bits 15..8 are interpreted as src0
  // and bits 7..0 are interpreted as src1. If `kLeftToRightPacking == false`,
  // then bits 15..8 are interpreted as src1 and 7..0 are interpreted as src0.
  void FillInput(std::array<Literal, 2>* input_literals) override {
    int64_t input_size = GetInputSize();
    CHECK_EQ(input_size, (*input_literals)[0].element_count());
    CHECK_EQ(input_size, (*input_literals)[1].element_count());

    int64_t begin, end;
    std::tie(begin, end) = GetParam();

    if (VLOG_IS_ON(2)) {
      uint8_t left_begin, left_end, right_begin, right_end;
      if constexpr (kLeftToRightPacking) {
        left_begin = std::bit_cast<uint8_t>(static_cast<int8_t>(begin >> 8));
        left_end = std::bit_cast<uint8_t>(static_cast<int8_t>(end >> 8));
        right_begin = std::bit_cast<uint8_t>(static_cast<int8_t>(begin));
        right_end = std::bit_cast<uint8_t>(static_cast<int8_t>(end));
      } else {
        left_begin = std::bit_cast<uint8_t>(static_cast<int8_t>(begin));
        left_end = std::bit_cast<uint8_t>(static_cast<int8_t>(end));
        right_begin = std::bit_cast<uint8_t>(static_cast<int8_t>(begin >> 8));
        right_end = std::bit_cast<uint8_t>(static_cast<int8_t>(end >> 8));
      }

      LOG(INFO) << this->SuiteName() << this->TestName() << " Range:";
      // N.B.: Cast to u32 to avoid printing values as char.
      LOG(INFO) << "\tfrom=(" << static_cast<uint32_t>(left_begin) << ", "
                << static_cast<uint32_t>(right_begin) << "); hex=(" << std::hex
                << static_cast<uint32_t>(left_begin) << ", "
                << static_cast<uint32_t>(right_begin) << "); float=("
                << std::bit_cast<tsl::float8_e5m2>(left_begin) << ", "
                << std::bit_cast<tsl::float8_e5m2>(right_begin)
                << ") (inclusive)";
      LOG(INFO) << "\tto=(" << static_cast<uint32_t>(left_end) << ", "
                << static_cast<uint32_t>(right_end) << "); hex=(" << std::hex
                << static_cast<uint32_t>(left_end) << ", "
                << static_cast<uint32_t>(right_end) << "); float=("
                << std::bit_cast<tsl::float8_e5m2>(left_end) << ", "
                << std::bit_cast<tsl::float8_e5m2>(right_end)
                << ") (exclusive)";
      LOG(INFO) << "\ttotal values to test=" << (end - begin);
    }

    absl::Span<NativeT> input_arr_0 = (*input_literals)[0].data<NativeT>();
    absl::Span<NativeT> input_arr_1 = (*input_literals)[1].data<NativeT>();
    for (int64_t i = 0; i < input_size; i++) {
      uint32_t input_val = i + begin;
      // Convert the packed bits to a pair of NativeT and replace known
      // incorrect input values with 0.
      //
      // In either case, we only use 16 bits out of the 64 bits possible.
      if constexpr (kLeftToRightPacking) {
        input_arr_0[i] = this->ConvertValue(input_val >> 8);
        input_arr_1[i] = this->ConvertValue(input_val);
      } else {
        input_arr_0[i] = this->ConvertValue(input_val);
        input_arr_1[i] = this->ConvertValue(input_val >> 8);
      }
    }
  }

 protected:
  using typename ExhaustiveBinaryTest<T>::NativeT;
};

// Exhaustive test for binary operations for 16 bit floating point types,
// including float16 and bfloat.
//
// Test parameter is a pair of (begin, end) for range under test.
template <PrimitiveType T, bool kLeftToRightPacking = false>
class Exhaustive16BitBinaryTest
    : public ExhaustiveBinaryTest<T>,
      public ::testing::WithParamInterface<std::pair<int64_t, int64_t>> {
 protected:
  int64_t GetInputSize() override {
    int64_t begin, end;
    std::tie(begin, end) = GetParam();
    return end - begin;
  }

  // Given a range of uint64_t representation, uses bits 0..15 and bits 16..31
  // for the values of src0 and src1 (see below for ordering) for the 16 bit
  // binary operation being tested, and generates the cartesian product of the
  // two sets as the two inputs for the test.
  //
  // If `kLeftToRightPacking == true`, bit 31..16 become src0 and 15..0 becomes
  // src1. If `kLeftToRightPacking == false`, then bits 31..16 become src1
  // and 15..0 becomes src0.
  void FillInput(std::array<Literal, 2>* input_literals) override {
    using NativeT = typename ExhaustiveBinaryTest<T>::NativeT;

    int64_t input_size = GetInputSize();
    CHECK_EQ(input_size, (*input_literals)[0].element_count());
    CHECK_EQ(input_size, (*input_literals)[1].element_count());

    int64_t begin, end;
    std::tie(begin, end) = GetParam();
    if (VLOG_IS_ON(2)) {
      uint16_t left_begin, left_end, right_begin, right_end;
      if constexpr (kLeftToRightPacking) {
        left_begin = std::bit_cast<uint16_t>(static_cast<int16_t>(begin >> 16));
        left_end = std::bit_cast<uint16_t>(static_cast<int16_t>(end >> 16));
        right_begin = std::bit_cast<uint16_t>(static_cast<int16_t>(begin));
        right_end = std::bit_cast<uint16_t>(static_cast<int16_t>(end));
      } else {
        left_begin = std::bit_cast<uint16_t>(static_cast<int16_t>(begin));
        left_end = std::bit_cast<uint16_t>(static_cast<int16_t>(end));
        right_begin =
            std::bit_cast<uint16_t>(static_cast<int16_t>(begin >> 16));
        right_end = std::bit_cast<uint16_t>(static_cast<int16_t>(end >> 16));
      }

      // N.B.: Use INFO directly instead of doing another thread-safe VLOG
      // check.
      LOG(INFO) << this->SuiteName() << this->TestName() << " Range:";
      LOG(INFO) << "\tfrom=(" << left_begin << ", " << right_begin << "); hex=("
                << std::hex << left_begin << ", " << right_begin << "); float=("
                << *reinterpret_cast<xla::bfloat16*>(&left_begin) << ", "
                << *reinterpret_cast<xla::bfloat16*>(&right_begin)
                << ") (inclusive)";
      LOG(INFO) << "\tto=(" << left_end << ", " << right_end << "); hex=("
                << std::hex << left_end << ", " << right_end << "); float=("
                << *reinterpret_cast<xla::bfloat16*>(&left_end) << ", "
                << *reinterpret_cast<xla::bfloat16*>(&right_end)
                << ") (exclusive)";
      LOG(INFO) << "\ttotal values to test=" << (end - begin);
    }

    absl::Span<NativeT> input_arr_0 = (*input_literals)[0].data<NativeT>();
    absl::Span<NativeT> input_arr_1 = (*input_literals)[1].data<NativeT>();
    for (int64_t i = 0; i < input_size; i++) {
      uint32_t input_val = i + begin;
      // Convert the packed bits to a pair of NativeT and replace known
      // incorrect input values with 0.
      //
      // In either case, we only use 32 bits out of the 64 bits possible.
      if constexpr (kLeftToRightPacking) {
        // Left is stored at higher 16 bits.
        input_arr_0[i] = this->ConvertValue(input_val >> 16);
        input_arr_1[i] = this->ConvertValue(input_val);
      } else {
        // Left is stored at lower 16 bits.
        input_arr_0[i] = this->ConvertValue(input_val);
        input_arr_1[i] = this->ConvertValue(input_val >> 16);
      }
    }
  }
};

// Exhaustive test for binary operations for float and double.
//
// Test parameter is a tuple of (FpValues, FpValues) describing the possible
// values for each operand. The inputs for the test are the Cartesian product
// of the possible values for the two operands.
template <PrimitiveType T>
class Exhaustive32BitOrMoreBinaryTest
    : public ExhaustiveBinaryTest<T>,
      public ::testing::WithParamInterface<std::tuple<FpValues, FpValues>> {
 protected:
  int64_t GetInputSize() override {
    FpValues values_0;
    FpValues values_1;
    std::tie(values_0, values_1) = GetParam();
    return values_0.GetTotalNumValues() * values_1.GetTotalNumValues();
  }

  void FillInput(std::array<Literal, 2>* input_literals) override {
    using NativeT = typename ExhaustiveBinaryTest<T>::NativeT;

    int64_t input_size = GetInputSize();
    FpValues values_0;
    FpValues values_1;
    std::tie(values_0, values_1) = GetParam();
    if (VLOG_IS_ON(2)) {
      // N.B.: Use INFO directly instead of doing another thread-safe VLOG
      // check.
      LOG(INFO) << this->SuiteName() << this->TestName() << " Values:";
      LOG(INFO) << "\tleft values=" << values_0.ToString();
      LOG(INFO) << "\tright values=" << values_1.ToString();
      LOG(INFO) << "\ttotal values to test=" << input_size;
    }
    CHECK(input_size == (*input_literals)[0].element_count() &&
          input_size == (*input_literals)[1].element_count());

    absl::Span<NativeT> input_arr_0 = (*input_literals)[0].data<NativeT>();
    absl::Span<NativeT> input_arr_1 = (*input_literals)[1].data<NativeT>();

    uint64_t i = 0;
    for (auto src0 : values_0) {
      for (auto src1 : values_1) {
        input_arr_0[i] = this->ConvertValue(src0);
        input_arr_1[i] = this->ConvertValue(src1);
        ++i;
      }
    }
    CHECK_EQ(i, input_size);
  }
};

using ExhaustiveF8E4M3FNBinaryTest = Exhaustive8BitBinaryTest<F8E4M3FN>;
using ExhaustiveF8E5M2BinaryTest = Exhaustive8BitBinaryTest<F8E5M2>;
using ExhaustiveF16BinaryTest = Exhaustive16BitBinaryTest<F16>;
using ExhaustiveBF16BinaryTest = Exhaustive16BitBinaryTest<BF16>;
using ExhaustiveF32BinaryTest = Exhaustive32BitOrMoreBinaryTest<F32>;
using ExhaustiveF64BinaryTest = Exhaustive32BitOrMoreBinaryTest<F64>;

#if !defined(XLA_BACKEND_DOES_NOT_SUPPORT_F8E4M3FN)
#define BINARY_TEST_F8E4M3FN(test_name, ...)          \
  XLA_TEST_P(ExhaustiveF8E4M3FNBinaryTest, test_name) \
  __VA_ARGS__
#else
#define BINARY_TEST_E4M3FN(test_name, ...)
#endif

#if !defined(XLA_BACKEND_DOES_NOT_SUPPORT_F8E5M2)
#define BINARY_TEST_F8E5M2(test_name, ...)          \
  XLA_TEST_P(ExhaustiveF8E5M2BinaryTest, test_name) \
  __VA_ARGS__
#else
#define BINARY_TEST_E5M2(test_name, ...)
#endif

#if !defined(XLA_BACKEND_DOES_NOT_SUPPORT_FLOAT16)
#define BINARY_TEST_F16(test_name, ...)          \
  XLA_TEST_P(ExhaustiveF16BinaryTest, test_name) \
  __VA_ARGS__
#else
#define BINARY_TEST_F16(test_name, ...)
#endif

#define BINARY_TEST_BF16(test_name, ...)          \
  XLA_TEST_P(ExhaustiveBF16BinaryTest, test_name) \
  __VA_ARGS__

#define BINARY_TEST_F32(test_name, ...)          \
  XLA_TEST_P(ExhaustiveF32BinaryTest, test_name) \
  __VA_ARGS__

#if !defined(XLA_BACKEND_DOES_NOT_SUPPORT_FLOAT64)
#define BINARY_TEST_F64(test_name, ...)          \
  XLA_TEST_P(ExhaustiveF64BinaryTest, test_name) \
  __VA_ARGS__
#else
#define BINARY_TEST_F64(test_name, ...)
#endif

#define BINARY_TEST(test_name, ...)            \
  BINARY_TEST_F8E4M3FN(test_name, __VA_ARGS__) \
  BINARY_TEST_F8E5M2(test_name, __VA_ARGS__)   \
  BINARY_TEST_F16(test_name, __VA_ARGS__)      \
  BINARY_TEST_BF16(test_name, __VA_ARGS__)     \
  BINARY_TEST_F32(test_name, __VA_ARGS__)      \
  BINARY_TEST_F64(test_name, __VA_ARGS__)

#define BINARY_TEST_COMPLEX(test_name, ...) \
  BINARY_TEST_F32(test_name, __VA_ARGS__)   \
  BINARY_TEST_F64(test_name, __VA_ARGS__)