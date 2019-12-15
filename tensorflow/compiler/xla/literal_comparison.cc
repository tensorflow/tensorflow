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

#include "tensorflow/compiler/xla/literal_comparison.h"

#include <unistd.h>

#include <cmath>
#include <vector>

#include "absl/base/casts.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/platform/env.h"

using absl::StrAppend;
using absl::StrAppendFormat;
using absl::StrCat;

namespace xla {
namespace literal_comparison {
namespace {

// Since Eigen::half doesn't satisfy the absl::bit_cast contract, we need to be
// able to transparently access the raw 16-bit value contained within.
template <typename T>
T GetRawValue(T val) {
  return val;
}
uint16 GetRawValue(Eigen::half val) { return val.x; }

// Helper function for comparing a floating point type, FloatT, bitwise equal
// between the left-hand-side and right-hand-side, by bit-casting to UnsignedT
// -- on miscompare, a nice error message is given in the AssertionFailure.
template <typename FloatT, typename UnsignedT>
bool CompareFloatsBitwiseEqual(FloatT lhs, FloatT rhs,
                               absl::Span<const int64> multi_index) {
  auto ulhs = absl::bit_cast<UnsignedT>(GetRawValue(lhs));
  auto urhs = absl::bit_cast<UnsignedT>(GetRawValue(rhs));
  return ulhs == urhs;
}

// Templated comparator that specializes for float equality comparison with the
// bitwise helper above (this is the un-specialized fallback, to just use the
// default gunit implementation).
template <typename NativeT>
bool CompareEqual(NativeT lhs, NativeT rhs,
                  absl::Span<const int64> multi_index) {
  return lhs == rhs;
}

// Specializations for floating types that do bitwise comparisons when equality
// comparison is requested.
template <>
bool CompareEqual<bfloat16>(bfloat16 lhs, bfloat16 rhs,
                            absl::Span<const int64> multi_index) {
  return CompareFloatsBitwiseEqual<bfloat16, uint16>(lhs, rhs, multi_index);
}
template <>
bool CompareEqual<Eigen::half>(Eigen::half lhs, Eigen::half rhs,
                               absl::Span<const int64> multi_index) {
  return CompareFloatsBitwiseEqual<Eigen::half, uint16>(lhs, rhs, multi_index);
}
template <>
bool CompareEqual<float>(float lhs, float rhs,
                         absl::Span<const int64> multi_index) {
  return CompareFloatsBitwiseEqual<float, uint32>(lhs, rhs, multi_index);
}
template <>
bool CompareEqual<double>(double lhs, double rhs,
                          absl::Span<const int64> multi_index) {
  return CompareFloatsBitwiseEqual<double, uint64>(lhs, rhs, multi_index);
}
template <>
bool CompareEqual<complex64>(complex64 lhs, complex64 rhs,
                             absl::Span<const int64> multi_index) {
  return CompareEqual<float>(lhs.real(), rhs.real(), multi_index) &&
         CompareEqual<float>(lhs.imag(), rhs.imag(), multi_index);
}
template <>
bool CompareEqual<complex128>(complex128 lhs, complex128 rhs,
                              absl::Span<const int64> multi_index) {
  return CompareEqual<double>(lhs.real(), rhs.real(), multi_index) &&
         CompareEqual<double>(lhs.imag(), rhs.imag(), multi_index);
}

template <typename NativeT, typename UnsignedT>
Status MakeBitwiseErrorStatus(NativeT lhs, NativeT rhs,
                              absl::Span<const int64> multi_index) {
  auto ulhs = absl::bit_cast<UnsignedT>(GetRawValue(lhs));
  auto urhs = absl::bit_cast<UnsignedT>(GetRawValue(rhs));
  auto lhs_double = static_cast<double>(lhs);
  auto rhs_double = static_cast<double>(rhs);
  return InvalidArgument(
      "floating values are not bitwise-equal; and equality testing "
      "was requested: %s=%s=%a vs %s=%s=%a at array index %s",
      StrCat(absl::Hex(ulhs)), RoundTripFpToString(lhs), lhs_double,
      StrCat(absl::Hex(urhs)), RoundTripFpToString(rhs), rhs_double,
      LiteralUtil::MultiIndexAsString(multi_index));
}

template <typename NativeT>
Status MakeErrorStatus(NativeT lhs, NativeT rhs,
                       absl::Span<const int64> multi_index) {
  return InvalidArgument(
      "first mismatch at array index %s:\n  expected value: %s\n  actual "
      "value:   %s",
      LiteralUtil::MultiIndexAsString(multi_index), StrCat(lhs), StrCat(rhs));
}

template <>
Status MakeErrorStatus(bfloat16 lhs, bfloat16 rhs,
                       absl::Span<const int64> multi_index) {
  return MakeBitwiseErrorStatus<bfloat16, uint16>(lhs, rhs, multi_index);
}
template <>
Status MakeErrorStatus(Eigen::half lhs, Eigen::half rhs,
                       absl::Span<const int64> multi_index) {
  return MakeBitwiseErrorStatus<Eigen::half, uint16>(lhs, rhs, multi_index);
}
template <>
Status MakeErrorStatus(float lhs, float rhs,
                       absl::Span<const int64> multi_index) {
  return MakeBitwiseErrorStatus<float, uint32>(lhs, rhs, multi_index);
}
template <>
Status MakeErrorStatus(double lhs, double rhs,
                       absl::Span<const int64> multi_index) {
  return MakeBitwiseErrorStatus<double, uint64>(lhs, rhs, multi_index);
}
template <>
Status MakeErrorStatus(complex64 lhs, complex64 rhs,
                       absl::Span<const int64> multi_index) {
  if (!CompareEqual<float>(lhs.real(), rhs.real(), multi_index)) {
    return MakeErrorStatus(lhs.real(), rhs.real(), multi_index);
  }
  return MakeErrorStatus(lhs.imag(), rhs.imag(), multi_index);
}
template <>
Status MakeErrorStatus(complex128 lhs, complex128 rhs,
                       absl::Span<const int64> multi_index) {
  if (!CompareEqual<double>(lhs.real(), rhs.real(), multi_index)) {
    return MakeErrorStatus(lhs.real(), rhs.real(), multi_index);
  }
  return MakeErrorStatus(lhs.imag(), rhs.imag(), multi_index);
}

// A recursive function which iterates through every index of expected and
// actual literal and compares their values elementwise. Returns true if all
// elements are equal. Mismatched must either be:
//    - a literal of booleans that has the same shape as expected and actual. In
//      this case, each index in mismatched will be set to true if expected does
//      not equal actual at that index and false if there are equal.
//    - nullptr. In this case, the function will return once any mismatch is
//      found between expected and actual.
template <typename NativeT>
Status Equal(LiteralSlice expected, LiteralSlice actual,
             absl::Span<int64> multi_index, int64 dimension,
             Literal* mismatched = nullptr) {
  if (dimension == expected.shape().dimensions_size()) {
    NativeT expected_value = expected.Get<NativeT>(multi_index);
    NativeT actual_value = actual.Get<NativeT>(multi_index);
    bool result =
        CompareEqual<NativeT>(expected_value, actual_value, multi_index);
    if (mismatched) {
      mismatched->Set<bool>(multi_index, !result);
    }
    return result ? Status::OK()
                  : MakeErrorStatus<NativeT>(expected_value, actual_value,
                                             multi_index);
  }

  Status result;
  for (int64 i = 0; i < expected.shape().dimensions(dimension); ++i) {
    multi_index[dimension] = i;
    if (mismatched != nullptr) {
      result.Update(Equal<NativeT>(expected, actual, multi_index, dimension + 1,
                                   mismatched));
    } else {
      TF_RETURN_IF_ERROR(Equal<NativeT>(expected, actual, multi_index,
                                        dimension + 1, mismatched));
    }
  }
  return result;
}

// Gets the total element count.  For tuples, this is not the count of tuple
// elements, but the sum of elements of each tuple element.
int64 RecursiveElementCount(const Shape& shape) {
  if (shape.IsTuple()) {
    const int64 tuple_elements = ShapeUtil::TupleElementCount(shape);
    int64 total = 0;
    for (int64 i = 0; i < tuple_elements; ++i) {
      total += RecursiveElementCount(ShapeUtil::GetTupleElementShape(shape, i));
    }
    return total;
  } else if (shape.IsArray()) {
    return ShapeUtil::ElementsIn(shape);
  } else {
    return 0;
  }
}

// Returns whether the given value is infinity.
template <typename NativeT>
bool IsInf(NativeT val) {
  return std::isinf(val);
}

template <>
bool IsInf<half>(half val) {
  return std::isinf(static_cast<float>(val));
}

// Returns whether the given value is nan.
template <typename NativeT>
float IsNan(NativeT value) {
  return std::isnan(value);
}

template <>
float IsNan(half value) {
  return IsNan<float>(static_cast<float>(value));
}

// Converts the given floating-point value to a string.
string FpValueToString(bfloat16 value) {
  return absl::StrFormat("%10.4g", static_cast<double>(value));
}

string FpValueToString(half value) {
  return absl::StrFormat("%11.5g", static_cast<double>(value));
}

string FpValueToString(float value) {
  return absl::StrFormat("%15.9g", static_cast<double>(value));
}

string FpValueToString(double value) {
  return absl::StrFormat("%24.17g", value);
}

string FpValueToString(complex64 value) {
  return absl::StrCat(FpValueToString(value.real()), " + ",
                      FpValueToString(value.imag()));
}

string FpValueToString(complex128 value) {
  return absl::StrCat(FpValueToString(value.real()), " + ",
                      FpValueToString(value.imag()));
}

// A wrapper of std::abs to include data types that are not supported by
// std::abs, in particular, bfloat16 and half.
template <typename NativeT>
double FpAbsoluteValue(NativeT value) {
  return std::abs(value);
}

template <>
double FpAbsoluteValue(bfloat16 value) {
  return FpAbsoluteValue<float>(static_cast<float>(value));
}

template <>
double FpAbsoluteValue(half value) {
  return FpAbsoluteValue<float>(static_cast<float>(value));
}

// Helper class for comparing floating-point literals within an error bound.
template <typename NativeT>
class NearComparator {
 public:
  // Compares the two array literals elementwise and returns a comparison
  // result. The comparison is ok() if all actual and expected elements are
  // within the given error bound. In case of error, the status contains a
  // detailed message about the discrepancy.
  static Status Compare(const LiteralSlice& expected,
                        const LiteralSlice& actual,
                        const ShapeIndex& shape_index, ErrorSpec error,
                        bool detailed_message,
                        const MiscompareCallback& miscompare_callback) {
    NearComparator<NativeT> comparator(expected, actual, shape_index, error,
                                       detailed_message, miscompare_callback);
    return comparator.Run();
  }

 private:
  // Data structure encapsulating metadata about a single element mismatch.
  struct Mismatch {
    NativeT actual;
    NativeT expected;
    double rel_error;
    double abs_error;

    // The linear index of the failure within the shape. This linear index is
    // from the 'actual' literal.
    int64 linear_index;

    bool operator<(const Mismatch& other) const {
      return rel_error < other.rel_error;
    }

    string ToString(const Shape& shape) const {
      return absl::StrFormat(
          "actual %s, expected %s, index %s, rel error %8.3g, abs error %8.3g",
          FpValueToString(actual), FpValueToString(expected),
          LiteralUtil::MultiIndexAsString(
              IndexUtil::LinearIndexToMultidimensionalIndex(shape,
                                                            linear_index)),
          rel_error, abs_error);
    }
  };

  NearComparator(const LiteralSlice& expected, const LiteralSlice& actual,
                 const ShapeIndex& shape_index, ErrorSpec error,
                 bool detailed_message,
                 const MiscompareCallback& miscompare_callback)
      : expected_(expected),
        actual_(actual),
        shape_index_(shape_index),
        error_(error),
        detailed_message_(detailed_message),
        miscompare_callback_(miscompare_callback),
        abs_value_buckets_(kAbsValueBucketBounds.size() - 1, {0, 0}),
        abs_error_buckets_(kErrorBucketBounds.size(), 0),
        rel_error_buckets_(kErrorBucketBounds.size(), 0) {}

  // Runs the comparison between expected and actual literals.
  Status Run() {
    // If the shapes mismatch, we simply fail the expectation instead of
    // printing out data, as it's a type error rather than a value error.
    TF_RETURN_IF_ERROR(EqualShapes(expected_.shape(), actual_.shape()));
    if (!expected_.shape().IsArray()) {
      return InvalidArgument("Expected array shape; got %s.",
                             ShapeUtil::HumanString(expected_.shape()));
    }

    mismatches_ = Literal(ShapeUtil::ChangeElementType(actual_.shape(), PRED));
    mismatches_.PopulateWithValue(false);

    CompareLiterals();

    if (num_mismatches_ == 0) {
      return Status::OK();
    } else if (!VLOG_IS_ON(1) && miscompare_callback_ != nullptr) {
      miscompare_callback_(expected_, actual_, mismatches_, shape_index_);
    }
    return InvalidArgument("%s", ErrorMessage());
  }

  // Insert the given absolute value into the absolute value bucket vector. The
  // bounds of the buckets are given by kAbsValueBucketBounds.
  void UpdateAbsValueBucket(NativeT value, bool is_mismatch) {
    // Adjust the bucket containing the absolute values of the 'actual'
    // elements.
    const double abs_value = FpAbsoluteValue(value);
    for (int i = 0; i < abs_value_buckets_.size(); ++i) {
      if (i == abs_value_buckets_.size() - 1 ||
          (abs_value >= kAbsValueBucketBounds[i] &&
           abs_value < kAbsValueBucketBounds[i + 1])) {
        // The first value of the pair is the count of elements in the bucket,
        // the second is the count of mismatches in the bucket.
        abs_value_buckets_[i].first++;
        if (is_mismatch) {
          abs_value_buckets_[i].second++;
        }
        return;
      }
    }
  }

  // Insert the given error into the given error bucket vector.
  void UpdateErrorBucket(double error, absl::Span<int64> error_buckets) {
    CHECK_EQ(error_buckets.size(), kErrorBucketBounds.size());
    for (int i = 0; i < error_buckets.size(); ++i) {
      if (error >= kErrorBucketBounds[i]) {
        error_buckets[i]++;
      }
    }
  }

  // Compares the two given elements from the expected and actual literals at
  // the given literal_index and keeps track of various mismatch statistics.
  template <typename T>
  void CompareValues(T expected, T actual, int64 linear_index) {
    double abs_error;
    double rel_error;
    if (CompareEqual<T>(expected, actual, {linear_index})) {
      abs_error = 0;
      rel_error = 0;
    } else if (IsNan(expected) || IsNan(actual)) {
      if ((!error_.relaxed_nans && IsNan(expected) != IsNan(actual)) ||
          (error_.relaxed_nans && !IsNan(expected) && IsNan(actual))) {
        num_nan_mismatches_++;
        // A nan mismatch is considered to have infinite error. rel_error is
        // used for sorting a std::set of the top mismatchs, and a nan value
        // here will result in undefined behavior because nan's do not satisfy
        // the strict weak ordering requirement of std containers.
        abs_error = std::numeric_limits<float>::infinity();
        rel_error = std::numeric_limits<float>::infinity();
      } else {
        abs_error = 0;
        rel_error = 0;
      }
    } else if (IsInf(actual) && !IsInf(expected) && error_.fewer_infs_ok) {
      // `fewer_infs_ok` gives us the option of comparing as though `actual`
      // were float_max/min rather than inf.
      T actual_finite = actual > T{0} ? std::numeric_limits<T>::max()
                                      : std::numeric_limits<T>::lowest();
      abs_error = FpAbsoluteValue(actual_finite - expected);

      // Avoid division by 0 even though it's well-defined because ubsan can be
      // configured to treat this as a fatal error.
      if (expected != T{0}) {
        rel_error = abs_error / FpAbsoluteValue(expected);
      } else {
        rel_error = std::numeric_limits<float>::infinity();
      }
    } else if (IsInf(expected) || IsInf(actual)) {
      // If either the expected or actual value is infinity but not both,
      // then both absolute and relative error are regarded as inifity.
      CHECK(!CompareEqual(expected, actual, {linear_index}));
      abs_error = std::numeric_limits<float>::infinity();
      rel_error = std::numeric_limits<float>::infinity();
    } else {
      abs_error = FpAbsoluteValue(actual - expected);

      // Avoid division by 0 even though it's well-defined because ubsan can be
      // configured to treat this as a fatal error.
      if (expected != T{0}) {
        rel_error = abs_error / FpAbsoluteValue(expected);
      } else {
        rel_error = std::numeric_limits<float>::infinity();
      }
    }
    const bool is_abs_mismatch = abs_error > error_.abs;
    const bool is_rel_mismatch = rel_error > error_.rel;
    const bool is_mismatch = is_abs_mismatch && is_rel_mismatch;

    // Update the error of the relative bucket only if the *absolute* error
    // bound is exceeded and vice versa.
    if (is_abs_mismatch) {
      num_abs_mismatches_++;
      UpdateErrorBucket(rel_error, absl::MakeSpan(rel_error_buckets_));
    }
    if (is_rel_mismatch) {
      num_rel_mismatches_++;
      UpdateErrorBucket(abs_error, absl::MakeSpan(abs_error_buckets_));
    }

    UpdateAbsValueBucket(actual, is_mismatch);

    if (!is_mismatch) {
      return;
    }

    num_mismatches_++;

    // Keep track of the kTopRelativeErrorCount relative error mismatches.
    if (top_rel_mismatches_.size() < kTopRelativeErrorCount ||
        rel_error > top_rel_mismatches_.begin()->rel_error) {
      Mismatch mismatch = {actual, expected, rel_error, abs_error,
                           linear_index};
      top_rel_mismatches_.insert(mismatch);
      if (top_rel_mismatches_.size() > kTopRelativeErrorCount) {
        top_rel_mismatches_.erase(top_rel_mismatches_.begin());
      }
    }

    mismatches_.data<bool>()[linear_index] = true;
  }

  // For complex types, we compare real and imaginary parts individually.
  void CompareValues(complex64 expected, complex64 actual, int64 linear_index) {
    const auto both_parts_mismatch = num_mismatches_ + 2;
    CompareValues<float>(expected.real(), actual.real(), linear_index);
    CompareValues<float>(expected.imag(), actual.imag(), linear_index);
    if (num_mismatches_ == both_parts_mismatch) {
      // The mismatch counter had been incremented by each CompareValues() call,
      // which means that both real and imaginary parts of the passed-in complex
      // values are different. However, the counter should reflect a single
      // mismatch between these complex values.
      num_mismatches_--;
    }
  }

  void CompareValues(complex128 expected, complex128 actual,
                     int64 linear_index) {
    const auto both_parts_mismatch = num_mismatches_ + 2;
    CompareValues<double>(expected.real(), actual.real(), linear_index);
    CompareValues<double>(expected.imag(), actual.imag(), linear_index);
    if (num_mismatches_ == both_parts_mismatch) {
      // The mismatch counter had been incremented by each CompareValues() call,
      // which means that both real and imaginary parts of the passed-in complex
      // values are different. However, the counter should reflect a single
      // mismatch between these complex values.
      num_mismatches_--;
    }
  }

  // Compares the two literals elementwise.
  void CompareLiterals() {
    // Fast path optimization for the case were layouts match.
    if (LayoutUtil::Equal(actual_.shape().layout(),
                          expected_.shape().layout())) {
      absl::Span<const NativeT> expected_data = expected_.data<NativeT>();
      absl::Span<const NativeT> actual_data = actual_.data<NativeT>();
      const int64 len = expected_data.size();
      for (int64 i = 0; i < len; ++i) {
        CompareValues(expected_data[i], actual_data[i], i);
      }
      return;
    }
    std::vector<int64> multi_index(actual_.shape().rank(), 0);
    CompareLiteralsSlow(0, &multi_index);
  }

  // Slow path for CompareLiterals when 'actual' and 'expected' literals have
  // different layouts. In this case, multidimensional indices are constructed
  // and indexed for each element.
  void CompareLiteralsSlow(int64 dimension, std::vector<int64>* multi_index) {
    if (dimension == multi_index->size()) {
      CompareValues(expected_.Get<NativeT>(*multi_index),
                    actual_.Get<NativeT>(*multi_index),
                    IndexUtil::MultidimensionalIndexToLinearIndex(
                        actual_.shape(), *multi_index));
    } else {
      for (int64 i = 0; i < expected_.shape().dimensions(dimension); ++i) {
        (*multi_index)[dimension] = i;
        CompareLiteralsSlow(dimension + 1, multi_index);
      }
    }
  }

  // Returns an error message string with a detailed breakdown of the
  // mismatches. Called after calling Run().
  string ErrorMessage() {
    string out;
    int64 element_count = ShapeUtil::ElementsIn(actual_.shape());

    auto percent_string = [](float a, float b) {
      float pct = b == 0.0 ? 0.0 : 100.0 * a / b;
      return absl::StrFormat("%0.4f%%", pct);
    };

    StrAppendFormat(
        &out,
        "\nMismatch count %d (%s) in shape %s (%d elements), abs bound "
        "%g, rel bound %g\n",
        num_mismatches_, percent_string(num_mismatches_, element_count),
        ShapeUtil::HumanString(actual_.shape()),
        ShapeUtil::ElementsIn(actual_.shape()), error_.abs, error_.rel);
    if (num_nan_mismatches_ > 0) {
      StrAppend(&out, "nan mismatches ", num_nan_mismatches_, "\n");
    }
    StrAppendFormat(&out, "Top relative error mismatches:\n");
    for (auto it = top_rel_mismatches_.rbegin();
         it != top_rel_mismatches_.rend(); ++it) {
      StrAppend(&out, "  ", it->ToString(actual_.shape()), "\n");
    }

    if (!detailed_message_) {
      return out;
    }

    StrAppend(&out, "Absolute magnitude breakdown of actual values:\n");
    CHECK_EQ(abs_value_buckets_.size() + 1, kAbsValueBucketBounds.size());
    for (int i = 0; i < abs_value_buckets_.size(); ++i) {
      const int64 bucket_size = abs_value_buckets_[i].first;
      const int64 bucket_mismatches = abs_value_buckets_[i].second;
      string mismatch_str =
          bucket_mismatches > 0
              ? absl::StrFormat(", mismatches %d", bucket_mismatches)
              : "";
      StrAppendFormat(&out, "  %-6g <= x < %-6g : %7d (%9s)%s\n",
                      kAbsValueBucketBounds[i], kAbsValueBucketBounds[i + 1],
                      bucket_size, percent_string(bucket_size, element_count),
                      mismatch_str);
    }

    auto print_accum_buckets = [&](const string& header, int64 total,
                                   absl::Span<const int64> buckets) {
      StrAppend(&out, header, ":\n");
      StrAppendFormat(&out, "  <  %-6g : %7d (%s)\n", kErrorBucketBounds[0],
                      total - buckets[0],
                      percent_string(total - buckets[0], total));
      CHECK_EQ(buckets.size(), kErrorBucketBounds.size());
      for (int i = 0; i < kErrorBucketBounds.size(); ++i) {
        StrAppendFormat(&out, "  >= %-6g : %7d (%s)\n", kErrorBucketBounds[i],
                        buckets[i], percent_string(buckets[i], total));
      }
    };
    StrAppendFormat(&out, "Elements exceeding abs error bound %g: %d (%s)\n",
                    error_.abs, num_abs_mismatches_,
                    percent_string(num_abs_mismatches_, element_count));
    print_accum_buckets(
        "Relative error breakdown of elements exceeding abs error bound",
        num_abs_mismatches_, rel_error_buckets_);
    StrAppendFormat(&out, "Elements exceeding rel error bound %g: %d (%s)\n",
                    error_.rel, num_rel_mismatches_,
                    percent_string(num_rel_mismatches_, element_count));
    print_accum_buckets(
        "Absolute error breakdown of elements exceeding rel error bound",
        num_rel_mismatches_, abs_error_buckets_);
    return out;
  }

  // 'actual' and 'expected' literals being compared.
  LiteralSlice expected_;
  LiteralSlice actual_;

  // The shape index of the LiteralSlice that is being compared.
  ShapeIndex shape_index_;

  // The error bounds of the comparison.
  ErrorSpec error_;

  // Whether to include detailed breakdown of mismatches in the error message.
  bool detailed_message_;

  // Callback to invoke on miscompare.
  MiscompareCallback miscompare_callback_;

  // Number of element element mismatches encountered so far.
  int64 num_mismatches_ = 0;

  // Number of elements with a nan mismatch.
  int64 num_nan_mismatches_ = 0;

  // Number of elements which exceed the absolute/relative error bound.
  int64 num_abs_mismatches_ = 0;
  int64 num_rel_mismatches_ = 0;

  // A Literal containing which elements did not match in the expected and
  // actual literals. mismatches_ contains PREDs and is of the same sizes as
  // the comparison literals.
  Literal mismatches_;

  // The number of mismatches to report in the output, sorted by relative error
  // magnitude.
  static constexpr int64 kTopRelativeErrorCount = 5;

  // The set of mismatches with the largest relative error. The size of this set
  // is bounded by kTopRelativeErrorCount.
  std::multiset<Mismatch> top_rel_mismatches_;

  // Actual values are bucketed by absolute value. kAbsValueBucketBounds is the
  // bounds of these buckets. abs_value_buckets_ contains a pair for each
  // bucket: the element count and failure count.
  static constexpr std::array<float, 7> kAbsValueBucketBounds = {
      0.0, 0.0001, 0.001, 0.01, 0.1, 1, std::numeric_limits<float>::infinity()};
  std::vector<std::pair<int64, int64>> abs_value_buckets_;

  // Buckets for relative and absolute errors. The relative error buckets only
  // contains those elements which exceed the *absolute* error bound, and vice
  // versa. This makes it easy to see the effect of adjusting the relative (or
  // absolute) error bound on the success of the comparison. kErrorBucketBounds
  // are the lower bounds of the buckets in both vectors. The error buckets are
  // a cumulative distribution so an error value may appear in more than one
  // bucket. For example an error value of 0.003 may appear in the buckets
  // bounded by 0.01, 0.1, and 1.0.
  static constexpr std::array<float, 5> kErrorBucketBounds = {0.0001, 0.001,
                                                              0.01, 0.1, 1};
  std::vector<int64> abs_error_buckets_;
  std::vector<int64> rel_error_buckets_;
};

template <typename NativeT>
constexpr std::array<float, 7> NearComparator<NativeT>::kAbsValueBucketBounds;
template <typename NativeT>
constexpr std::array<float, 5> NearComparator<NativeT>::kErrorBucketBounds;

Status EqualHelper(const LiteralSlice& expected, const LiteralSlice& actual,
                   const ShapeIndex& shape_index,
                   const MiscompareCallback& miscompare_callback) {
  TF_RETURN_IF_ERROR(EqualShapes(expected.shape(), actual.shape()));

  Status result;
  if (expected.shape().IsTuple()) {
    ShapeIndex next_index = shape_index;
    for (int i = 0; i < ShapeUtil::TupleElementCount(expected.shape()); ++i) {
      next_index.push_back(i);
      Status tuple_result =
          EqualHelper(LiteralSlice(expected, {i}), LiteralSlice(actual, {i}),
                      next_index, miscompare_callback);
      if (miscompare_callback) {
        result.Update(tuple_result);
      } else {
        TF_RETURN_IF_ERROR(tuple_result);
      }
      next_index.pop_back();
    }
  } else {
    std::vector<int64> multi_index(expected.shape().dimensions_size(), 0);
    auto index = absl::MakeSpan(multi_index);

    Shape unequal_shape = ShapeUtil::MakeShape(PrimitiveType::PRED,
                                               expected.shape().dimensions());
    Literal miscompared(unequal_shape);
    Literal* miscompared_ptr =
        (miscompare_callback == nullptr ? nullptr : &miscompared);

    switch (expected.shape().element_type()) {
      case PRED:
        result = Equal<bool>(expected, actual, index, 0, miscompared_ptr);
        break;
      case S8:
        result = Equal<int8>(expected, actual, index, 0, miscompared_ptr);
        break;
      case S16:
        result = Equal<int16>(expected, actual, index, 0, miscompared_ptr);
        break;
      case S32:
        result = Equal<int32>(expected, actual, index, 0, miscompared_ptr);
        break;
      case S64:
        result = Equal<int64>(expected, actual, index, 0, miscompared_ptr);
        break;
      case U8:
        result = Equal<uint8>(expected, actual, index, 0, miscompared_ptr);
        break;
      case U16:
        result = Equal<uint16>(expected, actual, index, 0, miscompared_ptr);
        break;
      case U32:
        result = Equal<uint32>(expected, actual, index, 0, miscompared_ptr);
        break;
      case U64:
        result = Equal<uint64>(expected, actual, index, 0, miscompared_ptr);
        break;
      case BF16:
        result = Equal<bfloat16>(expected, actual, index, 0, miscompared_ptr);
        break;
      case F16:
        result = Equal<half>(expected, actual, index, 0, miscompared_ptr);
        break;
      case F32:
        result = Equal<float>(expected, actual, index, 0, miscompared_ptr);
        break;
      case F64:
        result = Equal<double>(expected, actual, index, 0, miscompared_ptr);
        break;
      case C64:
        result = Equal<complex64>(expected, actual, index, 0, miscompared_ptr);
        break;
      case C128:
        result = Equal<complex128>(expected, actual, index, 0, miscompared_ptr);
        break;
      case TOKEN:
        // Tokens have no on-device representation and are trivially equal.
        return Status::OK();
      default:
        LOG(FATAL) << "Unsupported primitive type: "
                   << PrimitiveType_Name(expected.shape().element_type());
    }

    if (!result.ok() && miscompare_callback) {
      miscompare_callback(expected, actual, LiteralSlice(miscompared),
                          shape_index);
    }
  }

  return result;
}

// Helper function for comparing two literals for nearness. Handles tuple-shapes
// via recursion. shape_index is the ShapeIndex of expected (or actual)
// currently being compared.
Status NearHelper(const LiteralSlice& expected, const LiteralSlice& actual,
                  const ShapeIndex& shape_index, const ErrorSpec& error,
                  absl::optional<bool> detailed_message,
                  const MiscompareCallback& miscompare_callback) {
  TF_RETURN_IF_ERROR(EqualShapes(expected.shape(), actual.shape()));

  if (expected.shape().IsTuple()) {
    Status return_status;
    for (int64 i = 0; i < ShapeUtil::TupleElementCount(expected.shape()); ++i) {
      const auto expected_element = LiteralSlice(expected, {i});
      const auto actual_element = LiteralSlice(actual, {i});
      ShapeIndex element_index = shape_index;
      element_index.push_back(i);
      Status element_result =
          NearHelper(expected_element, actual_element, element_index, error,
                     detailed_message, miscompare_callback);
      if (!element_result.ok()) {
        element_result = InvalidArgument("Array at shape index %s, %s",
                                         element_index.ToString(),
                                         element_result.error_message());
        if (return_status.ok()) {
          return_status = element_result;
        } else {
          return_status =
              AppendStatus(return_status, element_result.error_message());
        }
      }
    }
    if (!return_status.ok() && shape_index.empty()) {
      // Emit a top-level error message containing the top-level shape in case
      // of mismatch.
      int64 total_elements = RecursiveElementCount(actual.shape());
      return_status =
          InvalidArgument("\nMismatches in shape %s (%d elements):\n%s",
                          ShapeUtil::HumanString(actual.shape()),
                          total_elements, return_status.error_message());
    }
    return return_status;
  }

  if (ShapeUtil::ElementIsFloating(expected.shape()) ||
      ShapeUtil::ElementIsComplex(expected.shape())) {
    bool use_detailed_message = detailed_message.value_or(
        ShapeUtil::ElementsIn(expected.shape()) >= 64);
    switch (expected.shape().element_type()) {
      case BF16:
        return NearComparator<bfloat16>::Compare(expected, actual, shape_index,
                                                 error, use_detailed_message,
                                                 miscompare_callback);
        break;
      case F16:
        return NearComparator<half>::Compare(expected, actual, shape_index,
                                             error, use_detailed_message,
                                             miscompare_callback);
        break;
      case F32:
        return NearComparator<float>::Compare(expected, actual, shape_index,
                                              error, use_detailed_message,
                                              miscompare_callback);
        break;
      case F64:
        return NearComparator<double>::Compare(expected, actual, shape_index,
                                               error, use_detailed_message,
                                               miscompare_callback);
        break;
      case C64:
        return NearComparator<complex64>::Compare(expected, actual, shape_index,
                                                  error, use_detailed_message,
                                                  miscompare_callback);
        break;
      case C128:
        return NearComparator<complex128>::Compare(
            expected, actual, shape_index, error, use_detailed_message,
            miscompare_callback);
        break;
      default:
        LOG(FATAL) << "Unsupported primitive type in near comparator: "
                   << PrimitiveType_Name(expected.shape().element_type())
                   << ". Must be floating-point type.";
    }
  }

  // Non-floating point, non-tuple literal.
  return EqualHelper(expected, actual, shape_index, miscompare_callback);
}

}  // namespace

Status EqualShapes(const Shape& expected, const Shape& actual) {
  if (expected.element_type() != actual.element_type()) {
    return InvalidArgument("element type mismatch, want: %s got %s",
                           ShapeUtil::HumanString(expected),
                           ShapeUtil::HumanString(actual));
  }
  if (expected.IsTuple()) {
    if (ShapeUtil::TupleElementCount(expected) !=
        ShapeUtil::TupleElementCount(actual)) {
      return InvalidArgument(
          "want tuple element count: %d got tuple element count: %d",
          ShapeUtil::TupleElementCount(expected),
          ShapeUtil::TupleElementCount(actual));
    }
    for (int i = 0; i < expected.tuple_shapes_size(); ++i) {
      Status result =
          EqualShapes(expected.tuple_shapes(i), actual.tuple_shapes(i));
      if (!result.ok()) {
        return AppendStatus(result, StrCat("mismatch in tuple index", i));
      }
    }
  } else if (expected.IsArray()) {
    if (expected.rank() != actual.rank()) {
      return InvalidArgument("want rank of %s got rank of %s",
                             ShapeUtil::HumanString(expected),
                             ShapeUtil::HumanString(actual));
    }
    if (expected.element_type() != actual.element_type()) {
      return InvalidArgument("mismatch in primitive type %s vs %s",
                             PrimitiveType_Name(expected.element_type()),
                             PrimitiveType_Name(actual.element_type()));
    }
    if (expected.dimensions_size() != actual.dimensions_size()) {
      return InvalidArgument("want dimensions_size %d got dimensions_size %d",
                             expected.dimensions_size(),
                             actual.dimensions_size());
    }
    for (int i = 0; i < expected.dimensions_size(); ++i) {
      if (expected.dimensions(i) != actual.dimensions(i)) {
        return InvalidArgument(
            "mismatch in dimension #%d expected: %s actual: %s", i,
            ShapeUtil::HumanString(expected), ShapeUtil::HumanString(actual));
      }
    }
  }
  // Non-array, non-tuple shapes are trivially equivalent.
  return Status::OK();
}

namespace {

// If result is an error, extend the error message with the expected and actual
// literals.
Status EmitLiteralsInErrorMessage(const Status& result,
                                  const LiteralSlice& expected,
                                  const LiteralSlice& actual) {
  if (result.ok()) {
    return result;
  }
  return InvalidArgument("%s\n\nExpected literal:\n%s\n\nActual literal:\n%s",
                         result.error_message(), ToStringTruncated(expected),
                         ToStringTruncated(actual));
}

}  // namespace

Status Equal(const LiteralSlice& expected, const LiteralSlice& actual) {
  VLOG(1) << "expected:";
  XLA_VLOG_LINES(1, expected.ToString());
  VLOG(1) << "actual:";
  XLA_VLOG_LINES(1, actual.ToString());
  Status result = EqualHelper(expected, actual, {}, nullptr);
  return EmitLiteralsInErrorMessage(result, expected, actual);
}

Status Near(const LiteralSlice& expected, const LiteralSlice& actual,
            const ErrorSpec& error, absl::optional<bool> detailed_message,
            const MiscompareCallback& miscompare_callback) {
  VLOG(1) << "Expected literal:";
  XLA_VLOG_LINES(1, expected.ToString());
  VLOG(1) << "Actual literal:";
  XLA_VLOG_LINES(1, actual.ToString());
  Status result = NearHelper(expected, actual, /*shape_index=*/{}, error,
                             detailed_message, miscompare_callback);
  return EmitLiteralsInErrorMessage(result, expected, actual);
}

string ToStringTruncated(const LiteralSlice& literal) {
  return RecursiveElementCount(literal.shape()) < 1000
             ? literal.ToString()
             : "[TRUNCATED, Literal with more than 1000 values]";
}

}  // namespace literal_comparison
}  // namespace xla
