/* Copyright 2018 The OpenXLA Authors.

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

#include "xla/literal_comparison.h"

#include <complex>

#ifndef _WIN32
#include <unistd.h>
#endif

#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "Eigen/Core"
#include "xla/error_spec.h"
#include "xla/fp_util.h"
#include "xla/index_util.h"
#include "xla/layout_util.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/primitive_util.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/types.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/logging.h"  // IWYU pragma: keep

using absl::StrAppend;
using absl::StrAppendFormat;
using absl::StrCat;

namespace xla {
namespace literal_comparison {
namespace {

// Helper function for comparing a floating point type, FloatT, bitwise equal
// between the left-hand-side and right-hand-side, by bit-casting to UnsignedT
// -- on miscompare, a nice error message is given in the AssertionFailure.
template <typename FloatT, typename UnsignedT>
bool CompareFloatsBitwiseEqual(FloatT lhs, FloatT rhs,
                               absl::Span<const int64_t> multi_index) {
  auto ulhs = Eigen::numext::bit_cast<UnsignedT>(lhs);
  auto urhs = Eigen::numext::bit_cast<UnsignedT>(rhs);
  return ulhs == urhs;
}

// Templated comparator that specializes for float equality comparison with the
// bitwise helper above (this is the un-specialized fallback, to just use the
// default gunit implementation).
template <typename NativeT>
bool CompareEqual(NativeT lhs, NativeT rhs,
                  absl::Span<const int64_t> multi_index) {
  // Specializations for floating types that do bitwise comparisons when
  // equality comparison is requested.
  if constexpr (is_specialized_floating_point_v<NativeT>) {
    using BitT = UnsignedIntegerTypeForSizeType<sizeof(NativeT)>;
    return CompareFloatsBitwiseEqual<NativeT, BitT>(lhs, rhs, multi_index);
  }
  if constexpr (is_complex_v<NativeT>) {
    using ComponentT = typename NativeT::value_type;
    return CompareEqual<ComponentT>(lhs.real(), rhs.real(), multi_index) &&
           CompareEqual<ComponentT>(lhs.imag(), rhs.imag(), multi_index);
  }
  return lhs == rhs;
}

template <typename NativeT, typename UnsignedT>
absl::Status MakeBitwiseErrorStatus(NativeT lhs, NativeT rhs,
                                    absl::Span<const int64_t> multi_index) {
  auto ulhs = Eigen::numext::bit_cast<UnsignedT>(lhs);
  auto urhs = Eigen::numext::bit_cast<UnsignedT>(rhs);
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
absl::Status MakeErrorStatus(NativeT lhs, NativeT rhs,
                             absl::Span<const int64_t> multi_index) {
  if constexpr (is_specialized_integral_v<NativeT>) {
    return InvalidArgument(
        "first mismatch at array index %s:\n  expected value: %s\n  actual "
        "value:   %s",
        LiteralUtil::MultiIndexAsString(multi_index), StrCat(lhs), StrCat(rhs));
  }
  if constexpr (is_specialized_floating_point_v<NativeT>) {
    using BitT = UnsignedIntegerTypeForSizeType<sizeof(NativeT)>;
    return MakeBitwiseErrorStatus<NativeT, BitT>(lhs, rhs, multi_index);
  }
  if constexpr (is_complex_v<NativeT>) {
    using ComponentT = typename NativeT::value_type;
    if (!CompareEqual<ComponentT>(lhs.real(), rhs.real(), multi_index)) {
      return MakeErrorStatus(lhs.real(), rhs.real(), multi_index);
    }
    return MakeErrorStatus(lhs.imag(), rhs.imag(), multi_index);
  }
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
absl::Status Equal(LiteralSlice expected, LiteralSlice actual,
                   absl::Span<int64_t> multi_index, int64_t dimension,
                   Literal* mismatched = nullptr) {
  if (dimension == expected.shape().dimensions_size()) {
    NativeT expected_value = expected.Get<NativeT>(multi_index);
    NativeT actual_value = actual.Get<NativeT>(multi_index);
    bool result =
        CompareEqual<NativeT>(expected_value, actual_value, multi_index);
    if (mismatched) {
      mismatched->Set<bool>(multi_index, !result);
    }
    return result ? absl::OkStatus()
                  : MakeErrorStatus<NativeT>(expected_value, actual_value,
                                             multi_index);
  }

  absl::Status result;
  int64_t upper_bound = expected.shape().dimensions(dimension);
  if (expected.shape().is_dynamic_dimension(dimension)) {
    // If the dimension is dynamic, we only want to check up until the actual
    // dynamic size specified by the literal.
    upper_bound = expected.GetDynamicSize(dimension);
  }

  for (int64_t i = 0; i < upper_bound; ++i) {
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
int64_t RecursiveElementCount(const Shape& shape) {
  if (shape.IsTuple()) {
    const int64_t tuple_elements = ShapeUtil::TupleElementCount(shape);
    int64_t total = 0;
    for (int64_t i = 0; i < tuple_elements; ++i) {
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
  return Eigen::numext::isinf(val);
}
// Returns whether the given value is nan.
template <typename NativeT>
bool IsNan(NativeT value) {
  return Eigen::numext::isnan(value);
}

// Converts the given floating-point value to a string.
template <typename NativeT>
std::string FpValueToString(NativeT value) {
  if constexpr (is_specialized_floating_point_v<NativeT>) {
    constexpr int kPrecisionDigits = std::numeric_limits<NativeT>::max_digits10;
    const int kExponentDigts = std::ceil(
        std::log10(std::max(std::numeric_limits<NativeT>::max_exponent10, 1)));
    constexpr int kExtraChars = 4;
    const int kTotalChars = kPrecisionDigits * kExponentDigts + kExtraChars;
    return absl::StrFormat("%*.*g", kTotalChars, kPrecisionDigits,
                           static_cast<double>(value));
  }
  if constexpr (is_complex_v<NativeT>) {
    return absl::StrCat(FpValueToString(value.real()), " + ",
                        FpValueToString(value.imag()));
  }
}

template <typename NativeT>
double FpAbsoluteValue(NativeT value) {
  return static_cast<double>(Eigen::numext::abs(value));
}

// Helper class for comparing floating-point literals within an error bound.
template <typename NativeT>
class NearComparator {
 public:
  // Compares the two array literals elementwise and returns a comparison
  // result. The comparison is ok() if all actual and expected elements are
  // within the given error bound. In case of error, the status contains a
  // detailed message about the discrepancy.
  static absl::Status Compare(const LiteralSlice& expected,
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
    int64_t linear_index;

    int float_distance = -1;

    bool operator<(const Mismatch& other) const {
      return rel_error < other.rel_error;
    }

    std::string ToString(const Shape& shape) const {
      auto s = absl::StrFormat(
          "actual %s, expected %s, index %s, rel error %8.3g, abs error "
          "%8.3g",
          FpValueToString(actual), FpValueToString(expected),
          LiteralUtil::MultiIndexAsString(
              IndexUtil::LinearIndexToMultidimensionalIndex(shape,
                                                            linear_index)),
          rel_error, abs_error);
      if (float_distance >= 0) {
        StrAppendFormat(&s, ", float distance %d", float_distance);
      }
      return s;
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
  absl::Status Run() {
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
      return absl::OkStatus();
    } else if (!VLOG_IS_ON(1) && miscompare_callback_ != nullptr) {
      miscompare_callback_(
          expected_, actual_, mismatches_, shape_index_,
          ErrorBuckets(abs_error_buckets_, rel_error_buckets_));
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
  void UpdateErrorBucket(double error, absl::Span<int64_t> error_buckets) {
    CHECK_EQ(error_buckets.size(), kErrorBucketBounds.size());
    for (int i = 0; i < error_buckets.size(); ++i) {
      if (error >= kErrorBucketBounds[i]) {
        error_buckets[i]++;
      }
    }
  }

  template <typename T>
  int CalculateFloatDistance(T expected, T actual) {
    if (error_.low_precision_fp_error_spec.type ==
        PrimitiveType::PRIMITIVE_TYPE_INVALID)
      return -1;
    return primitive_util::FloatingPointTypeSwitch<int>(
        [&](const auto kType) -> int {
          using NarrowNativeT = primitive_util::NativeTypeOf<kType>;
          // TODO(b/370786669): Once ml_dtypes is updated to include
          // https://github.com/jax-ml/ml_dtypes/pull/205, do not special-case
          // e3m4 by casting to half first.
          if constexpr (std::is_same_v<NarrowNativeT, tsl::float8_e3m4>) {
            return CalculateDistanceInFloats(NarrowNativeT(half(expected)),
                                             NarrowNativeT(half(actual)));
          } else {
            return CalculateDistanceInFloats(NarrowNativeT(expected),
                                             NarrowNativeT(actual));
          }
        },
        error_.low_precision_fp_error_spec.type);
  }

  // Compares the two given elements from the expected and actual literals at
  // the given literal_index and keeps track of various mismatch statistics.
  template <typename T>
  void CompareValues(T expected, T actual, int64_t linear_index) {
    double abs_error;
    double rel_error;
    int float_distance = -1;
    if (CompareEqual<T>(expected, actual, {linear_index})) {
      abs_error = 0;
      rel_error = 0;
    } else if (IsNan(expected) || IsNan(actual)) {
      if (error_.all_nans_are_equivalent && IsNan(expected) && IsNan(actual)) {
        abs_error = 0;
        rel_error = 0;
      } else if ((!error_.relaxed_nans && IsNan(expected) != IsNan(actual)) ||
                 (error_.relaxed_nans && !IsNan(expected) && IsNan(actual))) {
        num_nan_mismatches_++;
        // A nan mismatch is considered to have infinite error. rel_error is
        // used for sorting a std::set of the top mismatches, and a nan value
        // here will result in undefined behavior because nan's do not satisfy
        // the strict weak ordering requirement of std containers.
        abs_error = std::numeric_limits<double>::infinity();
        rel_error = std::numeric_limits<double>::infinity();
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
        rel_error = std::numeric_limits<double>::infinity();
      }
    } else if (IsInf(expected) || IsInf(actual)) {
      // If either the expected or actual value is infinity but not both,
      // then both absolute and relative error are regarded as infinity.
      CHECK(!CompareEqual(expected, actual, {linear_index}));
      abs_error = std::numeric_limits<double>::infinity();
      rel_error = std::numeric_limits<double>::infinity();
    } else {
      float_distance = CalculateFloatDistance<T>(expected, actual);
      abs_error = FpAbsoluteValue(actual - expected);
      if (!std::numeric_limits<T>::is_signed && IsNaN(abs_error)) {
        abs_error = FpAbsoluteValue(expected - actual);
      }

      // Avoid division by 0 even though it's well-defined because ubsan can be
      // configured to treat this as a fatal error.
      if (expected != T{0}) {
        rel_error = abs_error / FpAbsoluteValue(expected);
      } else {
        rel_error = std::numeric_limits<double>::infinity();
      }
    }
    bool is_within_n_floats = false;
    bool should_use_float_error_spec =
        error_.low_precision_fp_error_spec.type !=
        PrimitiveType::PRIMITIVE_TYPE_INVALID;
    if (should_use_float_error_spec &&
        error_.low_precision_fp_error_spec.within_n_values >= 0) {
      is_within_n_floats =
          float_distance <= error_.low_precision_fp_error_spec.within_n_values;
    }

    const bool is_abs_mismatch =
        (should_use_float_error_spec && is_within_n_floats)
            ? false
            : (abs_error > error_.abs);
    const bool is_rel_mismatch =
        (should_use_float_error_spec && is_within_n_floats)
            ? false
            : (rel_error > error_.rel);
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
      Mismatch mismatch = {/*actual=*/actual,
                           /*expected=*/expected,
                           /*rel_error=*/rel_error,
                           /*abs_error=*/abs_error,
                           /*linear_index=*/linear_index,
                           /*float_distance=*/float_distance};
      top_rel_mismatches_.insert(mismatch);
      if (top_rel_mismatches_.size() > kTopRelativeErrorCount) {
        top_rel_mismatches_.erase(top_rel_mismatches_.begin());
      }
    }

    mismatches_.data<bool>()[linear_index] = true;
  }

  // For complex types, we compare real and imaginary parts individually.
  template <typename T>
  void CompareValues(std::complex<T> expected, std::complex<T> actual,
                     int64_t linear_index) {
    const auto both_parts_mismatch = num_mismatches_ + 2;
    CompareValues<T>(expected.real(), actual.real(), linear_index);
    CompareValues<T>(expected.imag(), actual.imag(), linear_index);
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
    // Fast path optimization for the case were layouts match and the shapes are
    // static.
    if (LayoutUtil::Equal(actual_.shape().layout(),
                          expected_.shape().layout()) &&
        expected_.shape().is_static() && actual_.shape().is_static()) {
      absl::Span<const NativeT> expected_data = expected_.data<NativeT>();
      absl::Span<const NativeT> actual_data = actual_.data<NativeT>();
      const int64_t len = expected_data.size();
      for (int64_t i = 0; i < len; ++i) {
        CompareValues(expected_data[i], actual_data[i], i);
      }
      return;
    }
    std::vector<int64_t> multi_index(actual_.shape().rank(), 0);
    CompareLiteralsSlow(0, &multi_index);
  }

  // Slow path for CompareLiterals when 'actual' and 'expected' literals are
  // dynamic or have different layouts. In this case, multidimensional indices
  // are constructed and indexed for each element.
  void CompareLiteralsSlow(int64_t dimension,
                           std::vector<int64_t>* multi_index) {
    if (dimension == multi_index->size()) {
      CompareValues(expected_.Get<NativeT>(*multi_index),
                    actual_.Get<NativeT>(*multi_index),
                    IndexUtil::MultidimensionalIndexToLinearIndex(
                        actual_.shape(), *multi_index));
    } else {
      int64_t upper_bound = expected_.shape().dimensions(dimension);
      if (expected_.shape().is_dynamic_dimension(dimension)) {
        // If the dimension is dynamic, we only want to check up until the
        // actual dynamic size specified by the literal.
        upper_bound = expected_.GetDynamicSize(dimension);
      }
      for (int64_t i = 0; i < upper_bound; ++i) {
        (*multi_index)[dimension] = i;
        CompareLiteralsSlow(dimension + 1, multi_index);
      }
    }
  }

  // Returns an error message string with a detailed breakdown of the
  // mismatches. Called after calling Run().
  std::string ErrorMessage() {
    std::string out;
    int64_t element_count = ShapeUtil::ElementsIn(actual_.shape());

    auto percent_string = [](double a, double b) {
      double pct = b == 0.0 ? 0.0 : 100.0 * a / b;
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
      const int64_t bucket_size = abs_value_buckets_[i].first;
      const int64_t bucket_mismatches = abs_value_buckets_[i].second;
      std::string mismatch_str =
          bucket_mismatches > 0
              ? absl::StrFormat(", mismatches %d", bucket_mismatches)
              : "";
      StrAppendFormat(&out, "  %-6g <= x < %-6g : %7d (%9s)%s\n",
                      kAbsValueBucketBounds[i], kAbsValueBucketBounds[i + 1],
                      bucket_size, percent_string(bucket_size, element_count),
                      mismatch_str);
    }

    auto print_accum_buckets = [&](const std::string& header, int64_t total,
                                   absl::Span<const int64_t> buckets) {
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

  // Number of element mismatches encountered so far.
  int64_t num_mismatches_ = 0;

  // Number of elements with a nan mismatch.
  int64_t num_nan_mismatches_ = 0;

  // Number of elements which exceed the absolute/relative error bound.
  int64_t num_abs_mismatches_ = 0;
  int64_t num_rel_mismatches_ = 0;

  // A Literal containing which elements did not match in the expected and
  // actual literals. mismatches_ contains PREDs and is of the same sizes as
  // the comparison literals.
  Literal mismatches_;

  // The number of mismatches to report in the output, sorted by relative error
  // magnitude.
  static constexpr int64_t kTopRelativeErrorCount = 5;

  // The set of mismatches with the largest relative error. The size of this set
  // is bounded by kTopRelativeErrorCount.
  std::multiset<Mismatch> top_rel_mismatches_;

  // Actual values are bucketed by absolute value. kAbsValueBucketBounds is the
  // bounds of these buckets. abs_value_buckets_ contains a pair for each
  // bucket: the element count and failure count.
  static inline constexpr std::array<double, 7> kAbsValueBucketBounds = {
      0.0, 0.0001, 0.001, 0.01, 0.1, 1, std::numeric_limits<double>::infinity(),
  };
  std::vector<std::pair<int64_t, int64_t>> abs_value_buckets_;

  // Buckets for relative and absolute errors. The relative error buckets only
  // contains those elements which exceed the *absolute* error bound, and vice
  // versa. This makes it easy to see the effect of adjusting the relative (or
  // absolute) error bound on the success of the comparison. kErrorBucketBounds
  // are the lower bounds of the buckets in both vectors. The error buckets are
  // a cumulative distribution so an error value may appear in more than one
  // bucket. For example an error value of 0.003 may appear in the buckets
  // bounded by 0.01, 0.1, and 1.0.
  static inline constexpr std::array<double, 5> kErrorBucketBounds = {
      0.0001, 0.001, 0.01, 0.1, 1};
  std::vector<int64_t> abs_error_buckets_;
  std::vector<int64_t> rel_error_buckets_;
};

absl::Status EqualHelper(const LiteralSlice& expected,
                         const LiteralSlice& actual,
                         const ShapeIndex& shape_index,
                         const MiscompareCallback& miscompare_callback) {
  if (expected.shape().is_static() && actual.shape().is_static()) {
    TF_RETURN_IF_ERROR(EqualShapes(expected.shape(), actual.shape()));
  } else {
    TF_RETURN_IF_ERROR(EqualDynamicShapesAndDimensions(expected, actual));
  }

  absl::Status result;
  if (expected.shape().IsTuple()) {
    ShapeIndex next_index = shape_index;
    for (int i = 0; i < ShapeUtil::TupleElementCount(expected.shape()); ++i) {
      next_index.push_back(i);
      absl::Status tuple_result =
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
    std::vector<int64_t> multi_index(expected.shape().dimensions_size(), 0);
    auto index = absl::MakeSpan(multi_index);

    Shape unequal_shape = ShapeUtil::MakeShape(PrimitiveType::PRED,
                                               expected.shape().dimensions());
    Literal miscompared(unequal_shape);
    Literal* miscompared_ptr =
        (miscompare_callback == nullptr ? nullptr : &miscompared);

    primitive_util::PrimitiveTypeSwitch<void>(
        [&](auto primitive_type_constant) -> void {
          if constexpr (primitive_util::IsArrayType(primitive_type_constant)) {
            using NativeT =
                primitive_util::NativeTypeOf<primitive_type_constant>;
            result =
                Equal<NativeT>(expected, actual, index, 0, miscompared_ptr);
            return;
          }
          if constexpr (primitive_type_constant == TOKEN) {
            // Tokens have no on-device representation and are trivially equal.
            return;
          }
          LOG(FATAL) << "Unsupported primitive type: "
                     << PrimitiveType_Name(expected.shape().element_type());
        },
        expected.shape().element_type());

    if (!result.ok() && miscompare_callback) {
      miscompare_callback(expected, actual, LiteralSlice(miscompared),
                          shape_index, ErrorBuckets());
    }
  }

  return result;
}

// Helper function for comparing two literals for nearness. Handles tuple-shapes
// via recursion. shape_index is the ShapeIndex of expected (or actual)
// currently being compared.
absl::Status NearHelper(const LiteralSlice& expected,
                        const LiteralSlice& actual,
                        const ShapeIndex& shape_index, const ErrorSpec& error,
                        std::optional<bool> detailed_message,
                        const MiscompareCallback& miscompare_callback) {
  if (expected.shape().is_static() && actual.shape().is_static()) {
    TF_RETURN_IF_ERROR(EqualShapes(expected.shape(), actual.shape()));
  } else {
    TF_RETURN_IF_ERROR(EqualDynamicShapesAndDimensions(expected, actual));
  }

  if (expected.shape().IsTuple()) {
    absl::Status return_status;
    for (int64_t i = 0; i < ShapeUtil::TupleElementCount(expected.shape());
         ++i) {
      const auto expected_element = LiteralSlice(expected, {i});
      const auto actual_element = LiteralSlice(actual, {i});
      ShapeIndex element_index = shape_index;
      element_index.push_back(i);
      absl::Status element_result =
          NearHelper(expected_element, actual_element, element_index, error,
                     detailed_message, miscompare_callback);
      if (!element_result.ok()) {
        element_result =
            InvalidArgument("Array at shape index %s, %s",
                            element_index.ToString(), element_result.message());
        if (return_status.ok()) {
          return_status = element_result;
        } else {
          return_status = AppendStatus(return_status, element_result.message());
        }
      }
    }
    if (!return_status.ok() && shape_index.empty()) {
      // Emit a top-level error message containing the top-level shape in case
      // of mismatch.
      int64_t total_elements = RecursiveElementCount(actual.shape());
      return_status =
          InvalidArgument("\nMismatches in shape %s (%d elements):\n%s",
                          ShapeUtil::HumanString(actual.shape()),
                          total_elements, return_status.message());
    }
    return return_status;
  }

  if (ShapeUtil::ElementIsFloating(expected.shape()) ||
      ShapeUtil::ElementIsComplex(expected.shape())) {
    bool use_detailed_message = detailed_message.value_or(
        ShapeUtil::ElementsIn(expected.shape()) >= 64);
    return primitive_util::PrimitiveTypeSwitch<absl::Status>(
        [&](auto primitive_type) -> absl::Status {
          if constexpr (primitive_util::IsFloatingPointType(primitive_type) ||
                        primitive_util::IsComplexType(primitive_type)) {
            using NativeT = primitive_util::NativeTypeOf<primitive_type>;
            return NearComparator<NativeT>::Compare(
                expected, actual, shape_index, error, use_detailed_message,
                miscompare_callback);
          }
          LOG(FATAL) << "Unsupported primitive type in near comparator: "
                     << PrimitiveType_Name(expected.shape().element_type())
                     << ". Must be floating-point type.";
        },
        expected.shape().element_type());
  }

  // Non-floating point, non-tuple literal.
  return EqualHelper(expected, actual, shape_index, miscompare_callback);
}

}  // namespace

absl::Status EqualShapes(const Shape& expected, const Shape& actual) {
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
      absl::Status result =
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
  return absl::OkStatus();
}

absl::Status EqualDynamicShapesAndDimensions(const LiteralSlice& expected,
                                             const LiteralSlice& actual) {
  TF_RETURN_IF_ERROR(EqualShapes(expected.shape(), actual.shape()));
  return ShapeUtil::ForEachSubshapeWithStatus(
      expected.shape(),
      [&expected, &actual](const Shape& expected_shape,
                           const ShapeIndex& index) -> absl::Status {
        auto actual_shape = ShapeUtil::GetSubshape(actual.shape(), index);
        for (int i = 0; i < expected_shape.dimensions().size(); ++i) {
          if (!expected_shape.is_dynamic_dimension(i) &&
              !actual_shape.is_dynamic_dimension(i)) {
            // We're only interested in dynamic dimensions.
            continue;
          }
          if (expected_shape.is_dynamic_dimension(i) &&
              !actual_shape.is_dynamic_dimension(i)) {
            return InvalidArgument(
                "mismatch at dimension %d. the expected shape %s is dynamic "
                "while "
                "the actual shape %s is not.",
                i, ShapeUtil::HumanString(expected.shape()),
                ShapeUtil::HumanString(actual.shape()));
          }
          if (!expected_shape.is_dynamic_dimension(i) &&
              actual_shape.is_dynamic_dimension(i)) {
            return InvalidArgument(
                "mismatch at dimension %d. the expected shape %s is not "
                "dynamic "
                "while the actual shape %s is dynamic.",
                i, ShapeUtil::HumanString(expected.shape()),
                ShapeUtil::HumanString(actual.shape()));
          }
          // Both dimensions are dynamic. Check that they are equal.
          int64_t expected_dynamic_size = expected.GetDynamicSize(i, index);
          int64_t actual_dynamic_size = actual.GetDynamicSize(i, index);
          if (expected_dynamic_size != actual_dynamic_size) {
            return InvalidArgument(
                "mismatch at dimension %d. The expected dynamic size does not "
                "match "
                "the actual dynamic size. %d vs. %d",
                i, expected_dynamic_size, actual_dynamic_size);
          }
        }

        return absl::OkStatus();
      });
}

namespace {

// If result is an error, extend the error message with the expected and actual
// literals.
absl::Status EmitLiteralsInErrorMessage(const absl::Status& result,
                                        const LiteralSlice& expected,
                                        const LiteralSlice& actual) {
  if (result.ok()) {
    return result;
  }
  return InvalidArgument("%s\n\nExpected literal:\n%s\n\nActual literal:\n%s",
                         result.message(), ToStringTruncated(expected),
                         ToStringTruncated(actual));
}

}  // namespace

absl::Status Equal(const LiteralSlice& expected, const LiteralSlice& actual) {
  if (VLOG_IS_ON(1)) {
    LOG(INFO) << "expected:";
    XLA_LOG_LINES(INFO, expected.ToString());
    LOG(INFO) << "actual:";
    XLA_LOG_LINES(INFO, actual.ToString());
  }
  absl::Status result = EqualHelper(expected, actual, {}, nullptr);
  return EmitLiteralsInErrorMessage(result, expected, actual);
}

absl::Status Near(const LiteralSlice& expected, const LiteralSlice& actual,
                  const ErrorSpec& error, std::optional<bool> detailed_message,
                  const MiscompareCallback& miscompare_callback) {
  if (VLOG_IS_ON(1)) {
    LOG(INFO) << "Expected literal:";
    XLA_LOG_LINES(INFO, expected.ToString());
    LOG(INFO) << "Actual literal:";
    XLA_LOG_LINES(INFO, actual.ToString());
  }
  absl::Status result = NearHelper(expected, actual, /*shape_index=*/{}, error,
                                   detailed_message, miscompare_callback);
  return EmitLiteralsInErrorMessage(result, expected, actual);
}

std::string ToStringTruncated(const LiteralSlice& literal) {
  return RecursiveElementCount(literal.shape()) < 1000
             ? literal.ToString()
             : "[TRUNCATED, Literal with more than 1000 values]";
}

}  // namespace literal_comparison
}  // namespace xla
