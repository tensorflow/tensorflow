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

/* static */ void LiteralTestUtil::AssertEqualShapes(const Shape& expected,
                                                     const Shape& actual) {
  ASSERT_EQ(ShapeUtil::Rank(expected), ShapeUtil::Rank(actual));
  ASSERT_EQ(expected.element_type(), actual.element_type())
      << PrimitiveType_Name(expected.element_type()) << " vs "
      << PrimitiveType_Name(actual.element_type());
  ASSERT_EQ(expected.dimensions_size(), actual.dimensions_size());
  for (int i = 0; i < expected.dimensions_size(); ++i) {
    ASSERT_EQ(expected.dimensions(i), actual.dimensions(i))
        << "mismatch in dimension #" << i
        << " expected: " << ShapeUtil::HumanString(expected)
        << " actual: " << ShapeUtil::HumanString(actual);
  }
  ASSERT_EQ(expected.tuple_shapes_size(), actual.tuple_shapes_size());
  for (int i = 0; i < expected.tuple_shapes_size(); ++i) {
    AssertEqualShapes(expected.tuple_shapes(i), actual.tuple_shapes(i));
  }
}

/* static */ void LiteralTestUtil::AssertEqualShapesAndLayouts(
    const Shape& expected, const Shape& actual) {
  ASSERT_EQ(expected.ShortDebugString(), actual.ShortDebugString());
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
  if (ulhs != urhs) {
    return ::testing::AssertionFailure() << tensorflow::strings::Printf(
               "floating values are not bitwise-equal; and equality testing "
               "was requested: %s=%g=%a vs %s=%g=%a",
               tensorflow::strings::StrCat(tensorflow::strings::Hex(ulhs))
                   .c_str(),
               lhs, lhs,
               tensorflow::strings::StrCat(tensorflow::strings::Hex(urhs))
                   .c_str(),
               rhs, rhs);
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
::testing::AssertionResult CompareEqual<float>(float lhs, float rhs) {
  return CompareFloatsBitwiseEqual<float, uint32>(lhs, rhs);
}
template <>
::testing::AssertionResult CompareEqual<double>(double lhs, double rhs) {
  return CompareFloatsBitwiseEqual<double, uint64>(lhs, rhs);
}

// A recursive function which iterates through every index of expected and
// actual literal and compares their values elementwise. Returns true if all
// elements are equal.
template <typename NativeT>
bool ExpectLiteralsEqual(const Literal& expected, const Literal& actual,
                         tensorflow::gtl::MutableArraySlice<int64> multi_index,
                         int64 dimension) {
  if (dimension == expected.shape().dimensions_size()) {
    NativeT expected_value = LiteralUtil::Get<NativeT>(expected, multi_index);
    NativeT actual_value = LiteralUtil::Get<NativeT>(actual, multi_index);
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
                                               const Literal& actual) {
  EXPECT_TRUE(Equal(expected, actual)) << "expected:\n"
                                       << LiteralUtil::ToString(expected)
                                       << "\n\tvs actual:\n"
                                       << LiteralUtil::ToString(actual);
}

/* static */ void LiteralTestUtil::ExpectNotEqual(const Literal& expected,
                                                  const Literal& actual) {
  EXPECT_FALSE(Equal(expected, actual));
}

/* static */ ::testing::AssertionResult LiteralTestUtil::Equal(
    const Literal& expected, const Literal& actual) {
  VLOG(1) << "expected: " << LiteralUtil::ToString(expected);
  VLOG(1) << "actual:   " << LiteralUtil::ToString(actual);

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
    case F32:
      match = ExpectLiteralsEqual<float>(expected, actual, &multi_index, 0);
      break;
    case F64:
      match = ExpectLiteralsEqual<double>(expected, actual, &multi_index, 0);
      break;
    case TUPLE: {
      bool tuple_match = true;
      for (int i = 0; i < actual.tuple_literals_size(); ++i) {
        auto result =
            Equal(expected.tuple_literals(i), actual.tuple_literals(i));
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
             << "expected: " << LiteralUtil::ToString(expected)
             << "\nactual:   " << LiteralUtil::ToString(actual);
    VLOG(1) << result.message();
  }
  return result;
}

/* static */ void LiteralTestUtil::ExpectEqualTuple(const Literal& expected,
                                                    const Literal& actual) {
  VLOG(1) << "expected: " << LiteralUtil::ToString(expected);
  VLOG(1) << "actual:   " << LiteralUtil::ToString(actual);

  ASSERT_TRUE(ShapeUtil::IsTuple(expected.shape()));
  ASSERT_TRUE(ShapeUtil::IsTuple(actual.shape()));
  AssertEqualShapes(expected.shape(), actual.shape());
  for (uint64 i = 0; i < expected.tuple_literals_size(); ++i) {
    const auto& expected_element = expected.tuple_literals(i);
    const auto& actual_element = actual.tuple_literals(i);
    if (ShapeUtil::IsTuple(expected_element.shape())) {
      ExpectEqualTuple(expected_element, actual_element);
    } else {
      ExpectEqual(expected_element, actual_element);
    }
  }
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
    VLOG(1) << "expected: " << LiteralUtil::ToString(expected);
    VLOG(1) << "actual:   " << LiteralUtil::ToString(actual);

    LiteralTestUtil::AssertEqualShapes(expected.shape(), actual.shape());

    // Set up members used during the comparison.
    num_miscompares_ = 0;
    abs_diff_sum_ = 0.0;
    abs_expected_sum_ = 0.0;
    abs_diff_miscompare_sum_ = 0.0;
    abs_expected_miscompare_sum_ = 0.0;
    max_rel_err_ = 0.0;
    max_abs_err_ = 0.0;
    *miscompares_.mutable_shape() =
        ShapeUtil::ChangeElementType(actual.shape(), PRED);
    miscompares_.mutable_preds()->Resize(
        ShapeUtil::ElementsIn(miscompares_.shape()), false);
    multi_index_.resize(expected.shape().dimensions_size(), 0);

    switch (expected.shape().element_type()) {
      case F32:
        ExpectLiteralsNear<float>(expected, actual, 0);
        break;
      case F64:
        ExpectLiteralsNear<double>(expected, actual, 0);
        break;
      default:
        LOG(FATAL) << "Unsupported primitive type in near comparator: "
                   << PrimitiveType_Name(expected.shape().element_type())
                   << ". Must be floating-point type.";
    }

    if (num_miscompares_ > 0) {
      if (!VLOG_IS_ON(1)) {
        LOG(INFO) << "expected: " << ShapeUtil::HumanString(expected.shape())
                  << " " << LiteralUtil::ToString(expected);
        LOG(INFO) << "actual:   " << ShapeUtil::HumanString(actual.shape())
                  << " " << LiteralUtil::ToString(actual);
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
  // EXPECTs that the two given scalar values are within the error bound. Keeps
  // track of how many mismatches have occured to keep the size of the output
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
    bool nan_mismatch = std::isnan(actual) != std::isnan(expected);
    bool mismatch =
        (nan_mismatch || (abs_diff >= error_.abs && rel_err >= error_.rel));
    if (mismatch) {
      abs_diff_miscompare_sum_ += abs_diff;
      abs_expected_miscompare_sum_ += std::abs(expected);
      const int64 kMaxFailures = 2;
      if (num_miscompares_ < kMaxFailures) {
        EXPECT_NEAR(expected, actual, error_.abs)
            << "mismatch at index "
            << LiteralTestUtil::MultiIndexAsString(multi_index_) << " abs diff "
            << abs_diff << " rel err " << rel_err << " failure #"
            << num_miscompares_;
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
      bool near =
          ExpectValuesNear(LiteralUtil::Get<NativeT>(expected, multi_index_),
                           LiteralUtil::Get<NativeT>(actual, multi_index_));
      LiteralUtil::Set<bool>(&miscompares_, multi_index_, !near);
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
                                             filename, literal));
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

}  // namespace

/* static */ ::testing::AssertionResult LiteralTestUtil::Near(
    const Literal& expected, const Literal& actual, const ErrorSpec& error) {
  NearComparator comparator(error);
  return comparator.ExpectNear(expected, actual)
             ? ::testing::AssertionSuccess()
             : ::testing::AssertionFailure() << "values were not near";
}

/* static */ void LiteralTestUtil::ExpectNear(const Literal& expected,
                                              const Literal& actual,
                                              const ErrorSpec& error) {
  EXPECT_TRUE(Near(expected, actual, error));
}

/* static */ ::testing::AssertionResult LiteralTestUtil::NearTuple(
    const Literal& expected, const Literal& actual, const ErrorSpec& error) {
  VLOG(1) << "expected: " << LiteralUtil::ToString(expected);
  VLOG(1) << "actual:   " << LiteralUtil::ToString(actual);

  if (!ShapeUtil::IsTuple(expected.shape()) ||
      !ShapeUtil::IsTuple(actual.shape())) {
    return ::testing::AssertionFailure()
           << "tuples expected expected shape = "
           << expected.shape().ShortDebugString()
           << " actual shape = " << actual.shape().ShortDebugString();
  }
  AssertEqualShapes(expected.shape(), actual.shape());
  for (uint64 i = 0; i < expected.tuple_literals_size(); ++i) {
    const auto& expected_element = expected.tuple_literals(i);
    const auto& actual_element = actual.tuple_literals(i);
    if (ShapeUtil::IsTuple(expected_element.shape())) {
      auto ret = NearTuple(expected_element, actual_element, error);
      if (!ret) {
        return ret;
      }
    } else if (ShapeUtil::ElementIsFloating(expected_element.shape())) {
      auto ret = Near(expected_element, actual_element, error);
      if (!ret) {
        return ret;
      }
    } else {
      auto ret = Equal(expected_element, actual_element);
      if (!ret) {
        return ret;
      }
    }
  }

  return ::testing::AssertionSuccess();
}

/* static */ void LiteralTestUtil::ExpectNearTuple(const Literal& expected,
                                                   const Literal& actual,
                                                   const ErrorSpec& error) {
  EXPECT_TRUE(NearTuple(expected, actual, error));
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

  auto new_literal = MakeUnique<Literal>();
  *new_literal->mutable_shape() =
      ShapeUtil::MakeShape(literal.shape().element_type(), new_dimensions);

  // Create a new shape with the given minor-to-major layout. This shape is used
  // solely for converting linear address to multi-dimensional addresses when
  // writing elements to the new literal.
  Shape shape_with_layout = new_literal->shape();
  *shape_with_layout.mutable_layout() = LayoutUtil::MakeLayout(minor_to_major);

  // Allocate space in the new literal.
  LiteralUtil::Reserve(ShapeUtil::ElementsIn(literal.shape()),
                       new_literal.get());

  // Copy data into new literal, element-by-element.
  for (int64 i = 0; i < ShapeUtil::ElementsIn(literal.shape()); ++i) {
    std::vector<int64> from_multi_index =
        IndexUtil::LinearIndexToMultidimensionalIndex(literal.shape(), i);
    std::vector<int64> to_multi_index =
        IndexUtil::LinearIndexToMultidimensionalIndex(shape_with_layout, i);
    switch (literal.shape().element_type()) {
      case PRED:
        LiteralUtil::Set<bool>(
            new_literal.get(), to_multi_index,
            LiteralUtil::Get<bool>(literal, from_multi_index));
        break;
      case U8:
        LiteralUtil::Set<uint8>(
            new_literal.get(), to_multi_index,
            LiteralUtil::Get<uint8>(literal, from_multi_index));
        break;
      case U32:
        LiteralUtil::Set<uint32>(
            new_literal.get(), to_multi_index,
            LiteralUtil::Get<uint32>(literal, from_multi_index));
        break;
      case S32:
        LiteralUtil::Set<int32>(
            new_literal.get(), to_multi_index,
            LiteralUtil::Get<int32>(literal, from_multi_index));
        break;
      case U64:
        LiteralUtil::Set<uint64>(
            new_literal.get(), to_multi_index,
            LiteralUtil::Get<uint64>(literal, from_multi_index));
        break;
      case S64:
        LiteralUtil::Set<int64>(
            new_literal.get(), to_multi_index,
            LiteralUtil::Get<int64>(literal, from_multi_index));
        break;
      case F32:
        LiteralUtil::Set<float>(
            new_literal.get(), to_multi_index,
            LiteralUtil::Get<float>(literal, from_multi_index));
        break;
      case F64:
        LiteralUtil::Set<double>(
            new_literal.get(), to_multi_index,
            LiteralUtil::Get<double>(literal, from_multi_index));
        break;
      default:
        LOG(FATAL) << "Unhandled primitive element type: "
                   << PrimitiveType_Name(literal.shape().element_type());
    }
  }

  return new_literal;
}

}  // namespace xla
