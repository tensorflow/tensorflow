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

#include "tensorflow/compiler/xla/tests/exhaustive_op_test_utils.h"

namespace xla {

// For f64, f32, f16, and bf16, we need 17, 9, 5, and 4 decimal places of
// precision to be guaranteed that we're printing the full number.
//
// (The general formula is, given a floating-point number with S significand
// bits, the number of decimal digits needed to print it to full precision is
//
//   ceil(1 + S * log_10(2)) ~= ceil(1 + S * 0.30103).
//
// See https://people.eecs.berkeley.edu/~wkahan/Math128/BinDecBin.pdf.)
namespace {
template <typename T>
struct ComponentStringifyFormat {};

template <>
struct ComponentStringifyFormat<double> {
  static constexpr absl::string_view value = "%0.17g (0x%16x)";
};

template <>
struct ComponentStringifyFormat<float> {
  static constexpr absl::string_view value = "%0.9g (0x%08x)";
};

template <>
struct ComponentStringifyFormat<Eigen::half> {
  static constexpr absl::string_view value = "%0.5g (0x%04x)";
};

template <>
struct ComponentStringifyFormat<bfloat16> {
  static constexpr absl::string_view value = "%0.4g (0x%04x)";
};
}  // namespace

/*static*/
template <PrimitiveType T, size_t N>
string ExhaustiveOpTestBase<T, N>::StringifyNum(
    typename ExhaustiveOpTestBase<T, N>::ComponentNativeT x) {
  typedef typename ExhaustiveOpTestBase<T, N>::ComponentNativeT ComponentType;
  typedef typename ExhaustiveOpTestBase<T, N>::ComponentIntegralNativeT
      IntegralType;
  return absl::StrFormat(ComponentStringifyFormat<ComponentType>::value,
                         static_cast<double>(x), BitCast<IntegralType>(x));
}

template <PrimitiveType T, size_t N>
void ExhaustiveOpTestBase<T, N>::ExpectNear(const InputLiterals& input_literals,
                                            const Literal& result_literal,
                                            EvaluateOp evaluate_op,
                                            ErrorSpecGen error_spec_gen) {
  // Cache for when all components are subnormal testing values.
  std::vector<NativeRefT> pure_subnormal_cache;
  pure_subnormal_cache.reserve(GetMaxCacheSize());
  for (int i = 0; i < GetMaxCacheSize(); ++i) {
    pure_subnormal_cache.push_back(
        CallOperation(evaluate_op, FromCacheLocation(i)));
  }

  NativeInputsList inputs_arr;
  for (int i = 0; i < N; ++i) {
    const Literal& literal = input_literals[i];
    inputs_arr[i] = literal.data<NativeT>();
  }

  absl::Span<const NativeT> result_arr = result_literal.data<NativeT>();

  int64 mismatches = 0;

  for (int64 i = 0; i < result_arr.size(); ++i) {
    NativeInputs inputs;
    NativeRefInputs inputs_ref_ty;

    for (int j = 0; j < N; ++j) {
      inputs[j] = inputs_arr[j][i];
      inputs_ref_ty[j] = static_cast<NativeRefT>(inputs[j]);
    }

    NativeT actual = result_arr[i];
    NativeT expected =
        static_cast<NativeT>(CallOperation(evaluate_op, inputs_ref_ty));
    ErrorSpec error_spec = CallErrorSpec(error_spec_gen, inputs);

    if (IsClose(static_cast<NativeRefT>(expected),
                static_cast<NativeRefT>(actual), error_spec)) {
      continue;
    }

    std::vector<NativeRefInputs> subnormal_test_inputs =
        GetTestValuesWithSubnormalSubstitutions(inputs_ref_ty);

    // Easy case: If `input` is not subnormal and !IsClose(expected, actual,
    // error_spec), print an error.
    if (subnormal_test_inputs.size() == 1) {
      PrintMismatch(&mismatches, [&] {
        return absl::StrFormat("Mismatch on %s. Expected %s, but got %s.",
                               StringifyNum(inputs), StringifyNum(expected),
                               StringifyNum(actual));
      });
      continue;
    }

    // Otherwise, we need to test the additional subnormal test values.
    std::vector<NativeRefT> subnormal_test_results;
    subnormal_test_results.reserve(subnormal_test_inputs.size());
    bool passed_subnormal_test = false;

    for (NativeRefInputs test_value : subnormal_test_inputs) {
      NativeRefT result;
      int cache_loc = GetCacheLocation(test_value);
      if (cache_loc == kInvalidCacheIndex) {
        result = CallOperation(evaluate_op, test_value);
      } else {
        result = pure_subnormal_cache[cache_loc];
      }

      if (IsClose(result, static_cast<NativeRefT>(actual), error_spec)) {
        passed_subnormal_test = true;
        break;
      }
      subnormal_test_results.push_back(std::move(result));
    }

    if (passed_subnormal_test) {
      continue;
    }

    std::string mismatch = absl::StrFormat(
        "Mismatch on subnormal value %s.  Expected one of:\n"
        "  %10s (evaluated at full-precision value)\n",
        StringifyNum(inputs), StringifyNum(expected));

    CHECK_EQ(subnormal_test_inputs.size(), subnormal_test_results.size());
    for (int i = 0; i < subnormal_test_inputs.size(); ++i) {
      absl::StrAppend(
          &mismatch,
          absl::StrFormat("  %10s (evaluated at %s)\n",
                          StringifyNum(subnormal_test_results[i]),
                          GetSubnormalDescription(subnormal_test_inputs[i],
                                                  inputs_ref_ty)));
    }
    absl::StrAppend(&mismatch,
                    absl::StrFormat("but got %s", StringifyNum(actual)));

    PrintMismatch(&mismatches, [mismatch] { return mismatch; });
  }
  EXPECT_EQ(mismatches, 0);
}

namespace {
template <PrimitiveType T, size_t N>
inline typename ExhaustiveOpTestBase<T, N>::ErrorSpec DefaultSpecGenerator(
    typename ExhaustiveOpTestBase<T, N>::NativeT) {
  LOG(FATAL) << "Unhandled Type";
}

template <PrimitiveType T, size_t N>
inline typename ExhaustiveOpTestBase<T, N>::ErrorSpec DefaultSpecGenerator(
    typename ExhaustiveOpTestBase<T, N>::NativeT,
    typename ExhaustiveOpTestBase<T, N>::NativeT) {
  LOG(FATAL) << "Unhandled Type";
}

template <>
inline ExhaustiveOpTestBase<C128, 1>::ErrorSpec DefaultSpecGenerator<C128, 1>(
    complex128) {
  return ExhaustiveOpTestBase<C128, 1>::ErrorSpec{0.0001, 0.0001};
}

template <>
inline ExhaustiveOpTestBase<C64, 1>::ErrorSpec DefaultSpecGenerator<C64, 1>(
    complex64) {
  return ExhaustiveOpTestBase<C64, 1>::ErrorSpec{0.0001, 0.0001};
}

template <>
inline ExhaustiveOpTestBase<F64, 1>::ErrorSpec DefaultSpecGenerator<F64, 1>(
    double) {
  return ExhaustiveOpTestBase<F64, 1>::ErrorSpec{0.0001, 0.0001};
}

template <>
inline ExhaustiveOpTestBase<F32, 1>::ErrorSpec DefaultSpecGenerator<F32, 1>(
    float) {
  return ExhaustiveOpTestBase<F32, 1>::ErrorSpec{0.0001, 0.0001};
}

template <>
inline ExhaustiveOpTestBase<F16, 1>::ErrorSpec DefaultSpecGenerator<F16, 1>(
    Eigen::half) {
  return ExhaustiveOpTestBase<F16, 1>::ErrorSpec{0.001, 0.001};
}

template <>
inline ExhaustiveOpTestBase<BF16, 1>::ErrorSpec DefaultSpecGenerator<BF16, 1>(
    bfloat16) {
  return ExhaustiveOpTestBase<BF16, 1>::ErrorSpec{0.002, 0.02};
}

template <>
inline ExhaustiveOpTestBase<F64, 2>::ErrorSpec DefaultSpecGenerator<F64, 2>(
    double, double) {
  return ExhaustiveOpTestBase<F64, 2>::ErrorSpec{0.001, 0.001};
}

template <>
inline ExhaustiveOpTestBase<F32, 2>::ErrorSpec DefaultSpecGenerator<F32, 2>(
    float, float) {
  return ExhaustiveOpTestBase<F32, 2>::ErrorSpec{0.001, 0.001};
}

template <>
inline ExhaustiveOpTestBase<F16, 2>::ErrorSpec DefaultSpecGenerator<F16, 2>(
    Eigen::half, Eigen::half) {
  return ExhaustiveOpTestBase<F16, 2>::ErrorSpec{0.001, 0.001};
}

template <>
inline ExhaustiveOpTestBase<BF16, 2>::ErrorSpec DefaultSpecGenerator<BF16, 2>(
    bfloat16, bfloat16) {
  return ExhaustiveOpTestBase<BF16, 2>::ErrorSpec{0.002, 0.02};
}
}  // namespace

/*static*/
template <PrimitiveType T, size_t N>
typename ExhaustiveOpTestBase<T, N>::ErrorSpecGen
ExhaustiveOpTestBase<T, N>::GetDefaultSpecGenerator() {
  return DefaultSpecGenerator<T, N>;
}

template class ExhaustiveOpTestBase<C128, 1>;
template class ExhaustiveOpTestBase<C64, 1>;
template class ExhaustiveOpTestBase<F64, 1>;
template class ExhaustiveOpTestBase<F32, 1>;
template class ExhaustiveOpTestBase<F16, 1>;
template class ExhaustiveOpTestBase<BF16, 1>;

template class ExhaustiveOpTestBase<F64, 2>;
template class ExhaustiveOpTestBase<F32, 2>;
template class ExhaustiveOpTestBase<F16, 2>;
template class ExhaustiveOpTestBase<BF16, 2>;

}  // namespace xla
