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

namespace test_util {
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
}

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

    ErrorSpec(float a, float r) : abs_err(a), rel_err(r) {}
  };

  // `ty` is the primitive type being tested.
  explicit ExhaustiveOpTestBase(PrimitiveType ty)
      : ty_(ty), platform_(client_->platform()->Name()) {}

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
    T abs_err = std::abs(expected - actual);
    T rel_err = abs_err / std::abs(expected);
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
    using I = typename test_util::IntegralTypeWithByteWidth<sizeof(T)>::type;
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

}  // namespace xla
#endif  // TENSORFLOW_COMPILER_XLA_TESTS_EXHAUSTIVE_OP_TEST_UTILS_H_
