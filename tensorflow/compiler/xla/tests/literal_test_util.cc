/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/tests/literal_test_util.h"

#include <unistd.h>
#include <cmath>
#include <vector>

#include "tensorflow/compiler/xla/index_util.h"
#include "tensorflow/compiler/xla/layout_util.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/ptr_util.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/core/lib/core/casts.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/types.h"

namespace xla {

/* static */ ::testing::AssertionResult LiteralTestUtil::EqualShapes(
    const Shape& expected, const Shape& actual) {
  if (ShapeUtil::IsTuple(expected) != ShapeUtil::IsTuple(actual)) {
    return ::testing::AssertionFailure()
           << "tupleness-mismatch! want: " << ShapeUtil::HumanString(expected)
           << " got: " << ShapeUtil::HumanString(actual);
  }
  if (ShapeUtil::IsTuple(expected)) {
    if (ShapeUtil::TupleElementCount(expected) !=
        ShapeUtil::TupleElementCount(actual)) {
      return ::testing::AssertionFailure()
             << "want tuple element count: "
             << ShapeUtil::TupleElementCount(expected)
             << " got tuple element count: "
             << ShapeUtil::TupleElementCount(actual);
    }
    for (int i = 0; i < expected.tuple_shapes_size(); ++i) {
      ::testing::AssertionResult result =
          EqualShapes(expected.tuple_shapes(i), actual.tuple_shapes(i))
          << "mismatch in tuple index " << i;
      if (!result) {
        return result;
      }
    }
  } else {
    if (ShapeUtil::Rank(expected) != ShapeUtil::Rank(actual)) {
      return ::testing::AssertionFailure()
             << "want rank of: " << ShapeUtil::HumanString(expected)
             << " got rank of: " << ShapeUtil::HumanString(actual);
    }
    if (expected.element_type() != actual.element_type()) {
      return ::testing::AssertionFailure()
             << PrimitiveType_Name(expected.element_type()) << " vs "
             << PrimitiveType_Name(actual.element_type());
    }
    if (expected.dimensions_size() != actual.dimensions_size()) {
      return ::testing::AssertionFailure()
             << "want dimensions_size " << expected.dimensions_size()
             << " got dimensions_size " << actual.dimensions_size();
    }
    for (int i = 0; i < expected.dimensions_size(); ++i) {
      if (expected.dimensions(i) != actual.dimensions(i)) {
        return ::testing::AssertionFailure()
               << "mismatch in dimension #" << i
               << " expected: " << ShapeUtil::HumanString(expected)
               << " actual: " << ShapeUtil::HumanString(actual);
      }
    }
  }
  return ::testing::AssertionSuccess();
}

/* static */ void LiteralTestUtil::AssertEqualShapes(const Shape& expected,
                                                     const Shape& actual) {
  ASSERT_TRUE(EqualShapes(expected, actual));
}

/* static */ void LiteralTestUtil::AssertEqualShapesAndLayouts(
    const Shape& expected, const Shape& actual) {
  ASSERT_EQ(expected.ShortDebugString(), actual.ShortDebugString());
}

namespace {

// Return a literal with all arrays of type FromNativeT converted to type
// ToNativeT in the given literal.
template <typename FromNativeT, typename ToNativeT>
std::unique_ptr<Literal> ConvertType(const Literal& literal) {
  // First construct shape of the result.
  Shape result_shape(literal.shape());
  ShapeUtil::ForEachMutableSubshape(
      &result_shape, [](Shape* subshape, const ShapeIndex&) {
        if (subshape->element_type() ==
            primitive_util::NativeToPrimitiveType<FromNativeT>()) {
          subshape->set_element_type(
              primitive_util::NativeToPrimitiveType<ToNativeT>());
        }
      });
  auto result = MakeUnique<Literal>(result_shape);

  // Then copy over the data from 'literal' converting FromNativeT values to
  // ToNativeT values as necessary.
  ShapeUtil::ForEachSubshape(
      literal.shape(),
      [&](const Shape& subshape, const ShapeIndex& shape_index) {
        if (ShapeUtil::IsArray(subshape)) {
          if (subshape.element_type() ==
              primitive_util::NativeToPrimitiveType<FromNativeT>()) {
            auto src = literal.data<FromNativeT>(shape_index);
            auto dest = result->data<ToNativeT>(shape_index);
            for (int64 i = 0; i < src.size(); ++i) {
              dest[i] = static_cast<ToNativeT>(src[i]);
            }
          } else {
            TF_CHECK_OK(result->CopyFrom(literal,
                                         /*dest_shape_index=*/shape_index,
                                         /*src_shape_index=*/shape_index));
          }
        }
      });
  return result;
}

}  // namespace

/* static */ std::unique_ptr<Literal> LiteralTestUtil::ConvertBF16ToF32(
    const Literal& literal) {
  return ConvertType<bfloat16, float>(literal);
}

/* static */ std::unique_ptr<Literal> LiteralTestUtil::ConvertF32ToBF16(
    const Literal& literal) {
  return ConvertType<float, bfloat16>(literal);
}

namespace {

string Hostname() {
  char hostname[1024];
  gethostname(hostname, sizeof hostname);
  hostname[sizeof hostname - 1] = 0;
  return string(hostname);
}

// Helper function for comparing a floating point type, FloatT, bitwise equal
// between the left-hand-side and right-hand-side, by bit-casting to UnsignedT
// -- on miscompare, a nice error message is given in the AssertionFailure.
template <typename FloatT, typename UnsignedT>
::testing::AssertionResult CompareFloatsBitwiseEqual(FloatT lhs, FloatT rhs) {
  auto ulhs = tensorflow::bit_cast<UnsignedT>(lhs);
  auto urhs = tensorflow::bit_cast<UnsignedT>(rhs);
  auto lhs_double = static_cast<double>(lhs);
  auto rhs_double = static_cast<double>(rhs);
  if (ulhs != urhs) {
    return ::testing::AssertionFailure() << tensorflow::strings::Printf(
               "floating values are not bitwise-equal; and equality testing "
               "was requested: %s=%g=%a vs %s=%g=%a",
               tensorflow::strings::StrCat(tensorflow::strings::Hex(ulhs))
                   .c_str(),
               lhs_double, lhs_double,
               tensorflow::strings::StrCat(tensorflow::strings::Hex(urhs))
                   .c_str(),
               rhs_double, rhs_double);
  }
  return ::testing::AssertionSuccess();
}

// Templated comparator that specializes for float equality comparison with the
// bitwise helper above (this is the un-specialized fallback, to just use the
// default gunit implementation).
template <typename NativeT>
::testing::AssertionResult CompareEqual(NativeT lhs, NativeT rhs) {
  if (lhs == rhs) {
    return ::testing::AssertionSuccess();
  }
  ::testing::Message msg;
  msg << "Expected equality of these values:";
  msg << "\n  " << lhs;
  msg << "\n  " << rhs;

  return ::testing::AssertionFailure() << msg;
}

// Specializations for floating types that do bitwise comparisons when equality
// comparison is requested.
template <>
::testing::AssertionResult CompareEqual<bfloat16>(bfloat16 lhs, bfloat16 rhs) {
  return CompareFloatsBitwiseEqual<bfloat16, uint16>(lhs, rhs);
}
template <>
::testing::AssertionResult CompareEqual<float>(float lhs, float rhs) {
  return CompareFloatsBitwiseEqual<float, uint32>(lhs, rhs);
}
template <>
::testing::AssertionResult CompareEqual<double>(double lhs, double rhs) {
  return CompareFloatsBitwiseEqual<double, uint64>(lhs, rhs);
}
template <>
::testing::AssertionResult CompareEqual<complex64>(complex64 lhs,
                                                   complex64 rhs) {
  auto res = CompareEqual<float>(lhs.real(), rhs.real());
  if (!res) {
    return res;
  }
  return CompareEqual<float>(lhs.imag(), rhs.imag());
}

// A recursive function which iterates through every index of expected and
// actual literal and compares their values elementwise. Returns true if all
// elements are equal.
template <typename NativeT>
bool ExpectLiteralsEqual(const Literal& expected, const Literal& actual,
                         tensorflow::gtl::MutableArraySlice<int64> multi_index,
                         int64 dimension) {
  if (dimension == expected.shape().dimensions_size()) {
    NativeT expected_value = expected.Get<NativeT>(multi_index);
    NativeT actual_value = actual.Get<NativeT>(multi_index);
    ::testing::AssertionResult result =
        CompareEqual<NativeT>(expected_value, actual_value);
    return result;  // Defines implicit coersion to bool.
  }

  bool all_match = true;
  for (int64 i = 0; i < expected.shape().dimensions(dimension); ++i) {
    multi_index[dimension] = i;
    all_match = all_match && ExpectLiteralsEqual<NativeT>(
                                 expected, actual, multi_index, dimension + 1);
  }
  return all_match;
}

}  // namespace

/* static */ void LiteralTestUtil::ExpectEqual(const Literal& expected,
                                               const Literal& actual,
                                               const string& message) {
  EXPECT_TRUE(Equal(expected, actual))
      << "expected:\n"
      << expected.ToString() << "\n\tvs actual:\n"
      << actual.ToString()
      << (message.empty()
              ? ""
              : tensorflow::strings::StrCat("\nmessage: ", message));
}

/* static */ void LiteralTestUtil::ExpectNotEqual(const Literal& expected,
                                                  const Literal& actual) {
  EXPECT_FALSE(Equal(expected, actual));
}

/* static */ ::testing::AssertionResult LiteralTestUtil::Equal(
    const Literal& expected, const Literal& actual) {
  VLOG(1) << "expected:";
  XLA_VLOG_LINES(1, expected.ToString());
  VLOG(1) << "actual:";
  XLA_VLOG_LINES(1, actual.ToString());

  AssertEqualShapes(expected.shape(), actual.shape());
  std::vector<int64> multi_index(expected.shape().dimensions_size(), 0);
  bool match = false;
  switch (expected.shape().element_type()) {
    case PRED:
      match = ExpectLiteralsEqual<bool>(expected, actual, &multi_index, 0);
      break;
    case U8:
      match = ExpectLiteralsEqual<uint8>(expected, actual, &multi_index, 0);
      break;
    case S32:
      match = ExpectLiteralsEqual<int32>(expected, actual, &multi_index, 0);
      break;
    case S64:
      match = ExpectLiteralsEqual<int64>(expected, actual, &multi_index, 0);
      break;
    case U32:
      match = ExpectLiteralsEqual<uint32>(expected, actual, &multi_index, 0);
      break;
    case U64:
      match = ExpectLiteralsEqual<uint64>(expected, actual, &multi_index, 0);
      break;
    case BF16:
      match = ExpectLiteralsEqual<bfloat16>(expected, actual, &multi_index, 0);
      break;
    case F16:
      match = ExpectLiteralsEqual<half>(expected, actual, &multi_index, 0);
      break;
    case F32:
      match = ExpectLiteralsEqual<float>(expected, actual, &multi_index, 0);
      break;
    case F64:
      match = ExpectLiteralsEqual<double>(expected, actual, &multi_index, 0);
      break;
    case C64:
      match = ExpectLiteralsEqual<complex64>(expected, actual, &multi_index, 0);
      break;
    case TUPLE: {
      bool tuple_match = true;
      for (int i = 0; i < ShapeUtil::TupleElementCount(expected.shape()); ++i) {
        SCOPED_TRACE(tensorflow::strings::StrCat(
            "Tuple index ", i, " in ",
            ShapeUtil::HumanString(expected.shape())));

        // Create LiteralViews of the expected and actual elements.
        auto result = Equal(LiteralView::Create(expected, {i}),
                            LiteralView::Create(actual, {i}));
        tuple_match = tuple_match ? !!result : false;
      }
      match = tuple_match;
      break;
    }
    default:
      LOG(FATAL)
          << "Unsupported primitive type in LiteralTestUtil::ExpectEqual: "
          << PrimitiveType_Name(expected.shape().element_type());
  }
  ::testing::AssertionResult result = ::testing::AssertionSuccess();
  if (!match) {
    result = ::testing::AssertionFailure()
             << "expected: " << expected.ToString()
             << "\nactual:   " << actual.ToString();
    VLOG(1) << result.message();
  }
  return result;
}

namespace {

// Helper class for comparing floating-point literals within an error bound.
class NearComparator {
 public:
  explicit NearComparator(ErrorSpec error) : error_(error) {}

  // Compares the two literals elementwise. EXPECTs each pair of elements to be
  // within the error bound. Emits useful log messages and dumps literals to
  // temporary files on failure. Returns true if  literals match.
  bool ExpectNear(const Literal& expected, const Literal& actual) {
    VLOG(1) << "expected:";
    XLA_VLOG_LINES(1, expected.ToString());
    VLOG(1) << "actual:";
    XLA_VLOG_LINES(1, actual.ToString());

    // If the shapes mismatch, we simply fail the expectation instead of
    // printing out data, as it's a type error rather than a value error.
    ::testing::AssertionResult equal_shapes =
        LiteralTestUtil::EqualShapes(expected.shape(), actual.shape());
    if (!equal_shapes) {
      EXPECT_TRUE(equal_shapes);
      return false;
    }

    // Set up members used during the comparison.
    num_miscompares_ = 0;
    abs_diff_sum_ = 0.0;
    abs_expected_sum_ = 0.0;
    abs_diff_miscompare_sum_ = 0.0;
    abs_expected_miscompare_sum_ = 0.0;
    max_rel_err_ = 0.0;
    max_abs_err_ = 0.0;
    miscompares_ = Literal(ShapeUtil::ChangeElementType(actual.shape(), PRED));
    multi_index_.resize(expected.shape().dimensions_size(), 0);

    switch (expected.shape().element_type()) {
      case BF16:
        ExpectLiteralsNear<bfloat16>(expected, actual, 0);
        break;
      case F16:
        ExpectLiteralsNear<half>(expected, actual, 0);
        break;
      case F32:
        ExpectLiteralsNear<float>(expected, actual, 0);
        break;
      case F64:
        ExpectLiteralsNear<double>(expected, actual, 0);
        break;
      case C64:
        ExpectLiteralsNear<complex64>(expected, actual, 0);
        break;
      default:
        LOG(FATAL) << "Unsupported primitive type in near comparator: "
                   << PrimitiveType_Name(expected.shape().element_type())
                   << ". Must be floating-point type.";
    }

    if (num_miscompares_ > 0) {
      if (!VLOG_IS_ON(1)) {
        LOG(INFO) << "expected: " << ShapeUtil::HumanString(expected.shape())
                  << " " << expected.ToString();
        LOG(INFO) << "actual:   " << ShapeUtil::HumanString(actual.shape())
                  << " " << actual.ToString();
      }
      EXPECT_TRUE(num_miscompares_ == 0)
          << "\nmax relative mismatch at index "
          << LiteralTestUtil::MultiIndexAsString(max_rel_multi_index_)
          << "\nmaximum relative error " << max_rel_err_
          << "\nmax absolute mismatch at index "
          << LiteralTestUtil::MultiIndexAsString(max_abs_multi_index_)
          << "\nmaximum absolute error " << max_abs_err_
          << "\nfirst mismatch at index "
          << LiteralTestUtil::MultiIndexAsString(first_multi_index_)
          << "\nlast mismatch at index "
          << LiteralTestUtil::MultiIndexAsString(last_multi_index_)
          << "\ntotal absolute error " << abs_diff_sum_
          << "\ntotal absolute error of miscompares "
          << abs_diff_miscompare_sum_ << "\ntotal relative error "
          << (abs_diff_sum_ / abs_expected_sum_)
          << "\ntotal relative error of miscompares "
          << (abs_diff_miscompare_sum_ / abs_expected_miscompare_sum_)
          << "\nfailure count " << num_miscompares_;

      WriteLiteralToTempFile(expected, "expected");
      WriteLiteralToTempFile(actual, "actual");
      WriteLiteralToTempFile(miscompares_, "miscompares");
    }
    return num_miscompares_ == 0;
  }

 private:
  template <typename NativeT>
  bool NanMismatch(NativeT lhs, NativeT rhs) {
    return std::isnan(lhs) != std::isnan(rhs);
  }

  template <typename NativeT>
  void ExpectNear(NativeT expected, NativeT actual,
                  const ::testing::Message& message) {
    EXPECT_NEAR(expected, actual, error_.abs)
        << "expected:\n  " << expected << "\n\tvs actual:\n  " << actual << "\n"
        << message;
  }

  // EXPECTs that the two given scalar values are within the error bound. Keeps
  // track of how many mismatches have occurred to keep the size of the output
  // manageable.
  template <typename NativeT>
  bool ExpectValuesNear(NativeT expected, NativeT actual) {
    if (expected == actual) {
      return true;
    }

    float abs_diff = std::abs(actual - expected);
    float rel_err = abs_diff / std::abs(expected);
    abs_diff_sum_ += abs_diff;
    abs_expected_sum_ += std::abs(expected);
    if (rel_err > max_rel_err_) {
      max_rel_err_ = rel_err;
      max_rel_multi_index_ = multi_index_;
    }
    if (abs_diff > max_abs_err_) {
      max_abs_err_ = abs_diff;
      max_abs_multi_index_ = multi_index_;
    }
    VLOG(10) << tensorflow::strings::Printf(
        "index %s abs_diff %f rel_err %f",
        LiteralTestUtil::MultiIndexAsString(multi_index_).c_str(), abs_diff,
        rel_err);
    bool nan_mismatch = NanMismatch<NativeT>(expected, actual);
    bool mismatch =
        (nan_mismatch || (abs_diff >= error_.abs && rel_err >= error_.rel));
    if (mismatch) {
      abs_diff_miscompare_sum_ += abs_diff;
      abs_expected_miscompare_sum_ += std::abs(expected);
      const int64 kMaxFailures = 2;
      if (num_miscompares_ < kMaxFailures) {
        ::testing::Message msg;
        msg << "mismatch at index "
            << LiteralTestUtil::MultiIndexAsString(multi_index_) << " abs diff "
            << abs_diff << " rel err " << rel_err << " failure #"
            << num_miscompares_;
        ExpectNear<NativeT>(expected, actual, msg);
      } else if (num_miscompares_ == kMaxFailures) {
        LOG(ERROR)
            << "reached max 'loud' failure count; silently proceeding...";
      }
      if (num_miscompares_ == 0) {
        first_multi_index_ = multi_index_;
      }
      num_miscompares_++;
      last_multi_index_ = multi_index_;
    }
    return !mismatch;
  }

  // Recursive function which compares the two given literals elementwise.
  template <typename NativeT>
  void ExpectLiteralsNear(const Literal& expected, const Literal& actual,
                          int64 dimension) {
    if (dimension == expected.shape().dimensions_size()) {
      bool near = ExpectValuesNear(expected.Get<NativeT>(multi_index_),
                                   actual.Get<NativeT>(multi_index_));
      miscompares_.Set<bool>(multi_index_, !near);
    } else {
      for (int64 i = 0; i < expected.shape().dimensions(dimension); ++i) {
        multi_index_[dimension] = i;
        ExpectLiteralsNear<NativeT>(expected, actual, dimension + 1);
      }
    }
  }

  // Writes the given literal to a file in the test temporary directory.
  void WriteLiteralToTempFile(const Literal& literal, const string& name) {
    int64 now_usec = tensorflow::Env::Default()->NowMicros();
    string filename = tensorflow::io::JoinPath(
        tensorflow::testing::TmpDir(),
        tensorflow::strings::Printf("tempfile-%s-%llx-%s", Hostname().c_str(),
                                    now_usec, name.c_str()));
    TF_CHECK_OK(tensorflow::WriteBinaryProto(tensorflow::Env::Default(),
                                             filename, literal.ToProto()));
    LOG(ERROR) << "wrote to " << name << " file: " << filename;
  }

  ErrorSpec error_;

  // Number of element miscomparisons encountered so far.
  int64 num_miscompares_;

  // A Literal containing which elements did not match in the expected and
  // actual literals. miscompares_ contains PREDs and is of the same sizes as
  // the comparison literals.
  Literal miscompares_;

  // A multidimensional index used when performing the recursive comparison.
  std::vector<int64> multi_index_;

  // Aggregated Statistics on input.
  double abs_diff_sum_;
  double abs_expected_sum_;
  double abs_diff_miscompare_sum_;
  double abs_expected_miscompare_sum_;
  float max_rel_err_;
  float max_abs_err_;
  std::vector<int64> first_multi_index_;
  std::vector<int64> last_multi_index_;
  std::vector<int64> max_rel_multi_index_;
  std::vector<int64> max_abs_multi_index_;
};

template <>
bool NearComparator::NanMismatch<complex64>(complex64 lhs, complex64 rhs) {
  return std::isnan(lhs.real()) != std::isnan(rhs.real()) ||
         std::isnan(lhs.imag()) != std::isnan(rhs.imag());
}

template <>
void NearComparator::ExpectNear<complex64>(complex64 expected, complex64 actual,
                                           const ::testing::Message& message) {
  EXPECT_NEAR(expected.real(), actual.real(), error_.abs)
      << "expected:\n  " << expected << "\n\tvs actual:\n  " << actual << "\n"
      << message;
  EXPECT_NEAR(expected.imag(), actual.imag(), error_.abs)
      << "expected:\n  " << expected << "\n\tvs actual:\n  " << actual << "\n"
      << message;
}

template <>
bool NearComparator::ExpectValuesNear<bfloat16>(bfloat16 expected,
                                                bfloat16 actual) {
  return ExpectValuesNear(static_cast<float>(expected),
                          static_cast<float>(actual));
}

template <>
bool NearComparator::ExpectValuesNear<half>(half expected, half actual) {
  return ExpectValuesNear(static_cast<float>(std::move(expected)),
                          static_cast<float>(std::move(actual)));
}

}  // namespace

/* static */ ::testing::AssertionResult LiteralTestUtil::Near(
    const Literal& expected, const Literal& actual, const ErrorSpec& error) {
  ::testing::AssertionResult err =
      EqualShapes(expected.shape(), actual.shape());
  if (!err) {
    return err;
  }

  if (ShapeUtil::IsTuple(expected.shape())) {
    for (int64 i = 0; i < ShapeUtil::TupleElementCount(expected.shape()); ++i) {
      SCOPED_TRACE(tensorflow::strings::StrCat(
          "Tuple index ", i, " in ", ShapeUtil::HumanString(expected.shape())));
      const auto expected_element = LiteralView::Create(expected, {i});
      const auto actual_element = LiteralView::Create(actual, {i});

      ::testing::AssertionResult res =
          Near(expected_element, actual_element, error);
      if (err && !res) {
        err = res;
      }
    }
    return err;
  }

  if (ShapeUtil::ElementIsFloating(expected.shape()) ||
      ShapeUtil::ElementIsComplex(expected.shape())) {
    NearComparator comparator(error);
    return comparator.ExpectNear(expected, actual)
               ? ::testing::AssertionSuccess()
               : ::testing::AssertionFailure() << "values were not near";
  }

  return Equal(expected, actual);
}

/* static */ void LiteralTestUtil::ExpectNear(const Literal& expected,
                                              const Literal& actual,
                                              const ErrorSpec& error,
                                              const string& message) {
  EXPECT_TRUE(Near(expected, actual, error))
      << (message.empty()
              ? ""
              : tensorflow::strings::StrCat("\nmessage: ", message));
}

/*static*/ ::testing::AssertionResult LiteralTestUtil::NearOrEqual(
    const Literal& expected, const Literal& actual,
    const tensorflow::gtl::optional<ErrorSpec>& error) {
  if (error.has_value()) {
    VLOG(1) << "Expects near";
    return Near(expected, actual, *error);
  }
  VLOG(1) << "Expects equal";
  return Equal(expected, actual);
}

/*static*/ void LiteralTestUtil::ExpectNearOrEqual(
    const Literal& expected, const Literal& actual,
    const tensorflow::gtl::optional<ErrorSpec>& error) {
  EXPECT_TRUE(NearOrEqual(expected, actual, error));
}

/* static */ string LiteralTestUtil::MultiIndexAsString(
    tensorflow::gtl::ArraySlice<int64> multi_index) {
  return tensorflow::strings::StrCat(
      "{", tensorflow::str_util::Join(multi_index, ","), "}");
}

/* static */ std::unique_ptr<Literal> LiteralTestUtil::Reshape(
    tensorflow::gtl::ArraySlice<int64> new_dimensions,
    tensorflow::gtl::ArraySlice<int64> minor_to_major, const Literal& literal) {
  int64 new_num_elements = 1;
  for (int64 i = 0; i < new_dimensions.size(); ++i) {
    new_num_elements *= new_dimensions[i];
  }
  CHECK_EQ(ShapeUtil::ElementsIn(literal.shape()), new_num_elements);
  CHECK_EQ(new_dimensions.size(), minor_to_major.size());

  auto new_literal = MakeUnique<Literal>(
      ShapeUtil::MakeShape(literal.shape().element_type(), new_dimensions));

  // Create a new shape with the given minor-to-major layout. This shape is used
  // solely for converting linear address to multi-dimensional addresses when
  // writing elements to the new literal.
  Shape shape_with_layout = new_literal->shape();
  *shape_with_layout.mutable_layout() = LayoutUtil::MakeLayout(minor_to_major);

  // Copy data into new literal, element-by-element.
  for (int64 i = 0; i < ShapeUtil::ElementsIn(literal.shape()); ++i) {
    std::vector<int64> from_multi_index =
        IndexUtil::LinearIndexToMultidimensionalIndex(literal.shape(), i);
    std::vector<int64> to_multi_index =
        IndexUtil::LinearIndexToMultidimensionalIndex(shape_with_layout, i);
    switch (literal.shape().element_type()) {
      case PRED:
        new_literal->Set<bool>(to_multi_index,
                               literal.Get<bool>(from_multi_index));
        break;
      case U8:
        new_literal->Set<uint8>(to_multi_index,
                                literal.Get<uint8>(from_multi_index));
        break;
      case U32:
        new_literal->Set<uint32>(to_multi_index,
                                 literal.Get<uint32>(from_multi_index));
        break;
      case S32:
        new_literal->Set<int32>(to_multi_index,
                                literal.Get<int32>(from_multi_index));
        break;
      case U64:
        new_literal->Set<uint64>(to_multi_index,
                                 literal.Get<uint64>(from_multi_index));
        break;
      case S64:
        new_literal->Set<int64>(to_multi_index,
                                literal.Get<int64>(from_multi_index));
        break;
      case F32:
        new_literal->Set<float>(to_multi_index,
                                literal.Get<float>(from_multi_index));
        break;
      case F64:
        new_literal->Set<double>(to_multi_index,
                                 literal.Get<double>(from_multi_index));
        break;
      case C64:
        new_literal->Set<complex64>(to_multi_index,
                                    literal.Get<complex64>(from_multi_index));
        break;
      default:
        LOG(FATAL) << "Unhandled primitive element type: "
                   << PrimitiveType_Name(literal.shape().element_type());
    }
  }

  return new_literal;
}

}  // namespace xla
