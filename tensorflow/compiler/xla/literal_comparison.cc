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

#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/lib/core/casts.h"
#include "tensorflow/core/lib/strings/strcat.h"

using tensorflow::strings::StrCat;

namespace xla {
namespace literal_comparison {
namespace {

// Helper function for comparing a floating point type, FloatT, bitwise equal
// between the left-hand-side and right-hand-side, by bit-casting to UnsignedT
// -- on miscompare, a nice error message is given in the AssertionFailure.
template <typename FloatT, typename UnsignedT>
Status CompareFloatsBitwiseEqual(FloatT lhs, FloatT rhs) {
  auto ulhs = tensorflow::bit_cast<UnsignedT>(lhs);
  auto urhs = tensorflow::bit_cast<UnsignedT>(rhs);
  auto lhs_double = static_cast<double>(lhs);
  auto rhs_double = static_cast<double>(rhs);
  if (ulhs != urhs) {
    return InvalidArgument(
        "floating values are not bitwise-equal; and equality testing "
        "was requested: %s=%g=%a vs %s=%g=%a",
        StrCat(tensorflow::strings::Hex(ulhs)).c_str(), lhs_double, lhs_double,
        StrCat(tensorflow::strings::Hex(urhs)).c_str(), rhs_double, rhs_double);
  }
  return Status::OK();
}

// Templated comparator that specializes for float equality comparison with the
// bitwise helper above (this is the un-specialized fallback, to just use the
// default gunit implementation).
template <typename NativeT>
Status CompareEqual(NativeT lhs, NativeT rhs) {
  if (lhs == rhs) {
    return Status::OK();
  }
  return InvalidArgument("Expected equality of these values:\n  %s\n  %s",
                         StrCat(lhs).c_str(), StrCat(rhs).c_str());
}

// Specializations for floating types that do bitwise comparisons when equality
// comparison is requested.
template <>
Status CompareEqual<bfloat16>(bfloat16 lhs, bfloat16 rhs) {
  return CompareFloatsBitwiseEqual<bfloat16, uint16>(lhs, rhs);
}
template <>
Status CompareEqual<Eigen::half>(Eigen::half lhs, Eigen::half rhs) {
  return CompareFloatsBitwiseEqual<Eigen::half, uint16>(lhs, rhs);
}
template <>
Status CompareEqual<float>(float lhs, float rhs) {
  return CompareFloatsBitwiseEqual<float, uint32>(lhs, rhs);
}
template <>
Status CompareEqual<double>(double lhs, double rhs) {
  return CompareFloatsBitwiseEqual<double, uint64>(lhs, rhs);
}
template <>
Status CompareEqual<complex64>(complex64 lhs, complex64 rhs) {
  auto res = CompareEqual<float>(lhs.real(), rhs.real());
  if (!res.ok()) {
    return res;
  }
  return CompareEqual<float>(lhs.imag(), rhs.imag());
}

// A recursive function which iterates through every index of expected and
// actual literal and compares their values elementwise. Returns true if all
// elements are equal.
template <typename NativeT>
Status Equal(LiteralSlice expected, LiteralSlice actual,
             tensorflow::gtl::MutableArraySlice<int64> multi_index,
             int64 dimension) {
  if (dimension == expected.shape().dimensions_size()) {
    NativeT expected_value = expected.Get<NativeT>(multi_index);
    NativeT actual_value = actual.Get<NativeT>(multi_index);
    return CompareEqual<NativeT>(expected_value, actual_value);
  }

  Status result;
  for (int64 i = 0; i < expected.shape().dimensions(dimension); ++i) {
    multi_index[dimension] = i;
    result.Update(Equal<NativeT>(expected, actual, multi_index, dimension + 1));
  }
  return result;
}

}  // namespace

Status EqualShapes(const Shape& expected, const Shape& actual) {
  if (ShapeUtil::IsTuple(expected) != ShapeUtil::IsTuple(actual)) {
    return InvalidArgument("tupleness-mismatch! want: %s got %s",
                           ShapeUtil::HumanString(expected).c_str(),
                           ShapeUtil::HumanString(actual).c_str());
  }
  if (ShapeUtil::IsTuple(expected)) {
    if (ShapeUtil::TupleElementCount(expected) !=
        ShapeUtil::TupleElementCount(actual)) {
      return InvalidArgument(
          "want tuple element count: %lld got tuple element count: %lld",
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
  } else {
    if (ShapeUtil::Rank(expected) != ShapeUtil::Rank(actual)) {
      return InvalidArgument("want rank of %s got rank of %s",
                             ShapeUtil::HumanString(expected).c_str(),
                             ShapeUtil::HumanString(actual).c_str());
    }
    if (expected.element_type() != actual.element_type()) {
      return InvalidArgument(
          "mismatch in primitive type %s vs %s",
          PrimitiveType_Name(expected.element_type()).c_str(),
          PrimitiveType_Name(actual.element_type()).c_str());
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
            ShapeUtil::HumanString(expected).c_str(),
            ShapeUtil::HumanString(actual).c_str());
      }
    }
  }
  return Status::OK();
}

Status Equal(const LiteralSlice& expected, const LiteralSlice& actual) {
  VLOG(1) << "expected:";
  XLA_VLOG_LINES(1, expected.ToString());
  VLOG(1) << "actual:";
  XLA_VLOG_LINES(1, actual.ToString());

  TF_RETURN_IF_ERROR(EqualShapes(expected.shape(), actual.shape()));
  std::vector<int64> multi_index(expected.shape().dimensions_size(), 0);
  Status result;
  switch (expected.shape().element_type()) {
    case PRED:
      result = Equal<bool>(expected, actual, &multi_index, 0);
      break;
    case U8:
      result = Equal<uint8>(expected, actual, &multi_index, 0);
      break;
    case S32:
      result = Equal<int32>(expected, actual, &multi_index, 0);
      break;
    case S64:
      result = Equal<int64>(expected, actual, &multi_index, 0);
      break;
    case U32:
      result = Equal<uint32>(expected, actual, &multi_index, 0);
      break;
    case U64:
      result = Equal<uint64>(expected, actual, &multi_index, 0);
      break;
    case BF16:
      result = Equal<bfloat16>(expected, actual, &multi_index, 0);
      break;
    case F16:
      result = Equal<half>(expected, actual, &multi_index, 0);
      break;
    case F32:
      result = Equal<float>(expected, actual, &multi_index, 0);
      break;
    case F64:
      result = Equal<double>(expected, actual, &multi_index, 0);
      break;
    case C64:
      result = Equal<complex64>(expected, actual, &multi_index, 0);
      break;
    case TUPLE: {
      for (int i = 0; i < ShapeUtil::TupleElementCount(expected.shape()); ++i) {
        result.Update(
            Equal(LiteralSlice(expected, {i}), LiteralSlice(actual, {i})));
      }
      break;
    }
    default:
      LOG(FATAL)
          << "Unsupported primitive type in LiteralTestUtil::ExpectEqual: "
          << PrimitiveType_Name(expected.shape().element_type());
  }

  if (result.ok()) {
    return Status::OK();
  }

  return AppendStatus(result,
                      tensorflow::strings::Printf("expected: %s\nactual:   %s",
                                                  expected.ToString().c_str(),
                                                  actual.ToString().c_str()));
}

}  // namespace literal_comparison
}  // namespace xla
