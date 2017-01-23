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

#include "tensorflow/compiler/xla/shape_util.h"

#include <algorithm>
#include <functional>
#include <numeric>
#include <vector>

#include "tensorflow/compiler/xla/index_util.h"
#include "tensorflow/compiler/xla/layout_util.h"
#include "tensorflow/compiler/xla/primitive_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/strings/numbers.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/platform/regexp.h"

namespace xla {

namespace {

// Recursive helper for comparing the equality of two shapes. Returns true if
// the shapes are the same. If compare_layouts is true, then layouts must also
// match.
bool CompareShapes(const Shape& lhs, const Shape& rhs, bool compare_layouts) {
  if (ShapeUtil::IsTuple(lhs)) {
    if (!ShapeUtil::IsTuple(rhs)) {
      VLOG(3) << "CompareShapes: lhs is a tuple, rhs not a tuple";
      return false;
    }

    if (!ContainersEqual(lhs.tuple_shapes(), rhs.tuple_shapes(),
                         [=](const Shape& l, const Shape& r) {
                           return CompareShapes(l, r, compare_layouts);
                         })) {
      VLOG(3) << "CompareShapes: tuples on lhs and rhs not equal";
      return false;
    }
  }
  // Explicitly compare the fields rather than using MessageDifferencer because
  // we want empty layouts to be treated identically to missing layouts.
  if (compare_layouts) {
    if (!ContainersEqual(lhs.layout().minor_to_major(),
                         rhs.layout().minor_to_major())) {
      VLOG(3) << "CompareShapes: lhs layout != rhs layout";
      return false;
    }
    if (!ContainersEqual(lhs.layout().padded_dimensions(),
                         rhs.layout().padded_dimensions())) {
      VLOG(3)
          << "CompareShapes: lhs padded_dimensions != rhs padded_dimensions";
      return false;
    }
    if (lhs.layout().padding_value() != rhs.layout().padding_value()) {
      VLOG(3) << "CompareShapes: lhs padding value != rhs padding_value";
      return false;
    }
  }

  if (!ShapeUtil::SameDimensions(lhs, rhs)) {
    VLOG(3) << "CompareShapes: lhs dimensions != rhs dimensions";
    return false;
  }
  if (!ShapeUtil::SameElementType(lhs, rhs)) {
    VLOG(3) << "CompareShapes: lhs element type != rhs element type";
    return false;
  }
  return true;
}

}  // namespace

/* static */ bool ShapeUtil::Equal(const Shape& lhs, const Shape& rhs) {
  bool equal = CompareShapes(lhs, rhs, /*compare_layouts=*/true);
  if (!equal && VLOG_IS_ON(3)) {
    VLOG(3) << "ShapeUtil::Equal differ: lhs = " << lhs.ShortDebugString()
            << ", rhs = " << rhs.ShortDebugString();
  }

  return equal;
}

/* static */ int64 ShapeUtil::TrueRank(const Shape& shape) {
  int64 accum = 0;
  for (int64 dimension : shape.dimensions()) {
    // We do not count zero dimensions.
    if (dimension != 1) {
      accum += 1;
    }
  }
  return accum;
}

/* static */ ProgramShape ShapeUtil::MakeProgramShape(
    std::initializer_list<Shape> parameters, Shape result) {
  ProgramShape program_shape;
  for (const auto& shape : parameters) {
    *program_shape.add_parameters() = shape;
  }
  *program_shape.mutable_result() = result;
  return program_shape;
}

/* static */ Shape ShapeUtil::MakeShape(
    PrimitiveType element_type, tensorflow::gtl::ArraySlice<int64> dimensions) {
  DCHECK_NE(TUPLE, element_type);
  DCHECK_NE(OPAQUE, element_type);
  Shape result;
  PopulateShape(element_type, dimensions, &result);
  return result;
}

/* static */ Shape ShapeUtil::MakeShapeWithLayout(
    PrimitiveType element_type, tensorflow::gtl::ArraySlice<int64> dimensions,
    tensorflow::gtl::ArraySlice<int64> minor_to_major) {
  CHECK_EQ(dimensions.size(), minor_to_major.size());
  Shape shape = MakeShape(element_type, dimensions);
  auto min2maj = shape.mutable_layout()->mutable_minor_to_major();
  min2maj->Clear();
  for (int64 value : minor_to_major) {
    min2maj->Add(value);
  }
  DCHECK(shape.has_layout());
  TF_DCHECK_OK(ValidateShape(shape));
  return shape;
}

/* static */ Shape ShapeUtil::MakeShapeWithMonotonicDim0MajorLayout(
    PrimitiveType element_type, tensorflow::gtl::ArraySlice<int64> dimensions) {
  std::vector<int64> layout(dimensions.size());
  std::iota(layout.rbegin(), layout.rend(), static_cast<int64>(0));
  return MakeShapeWithLayout(element_type, dimensions, layout);
}

/* static */ Shape ShapeUtil::NormalizeShapeToMonotonicDim0MajorLayout(
    const Shape& shape) {
  std::vector<int64> dims(shape.dimensions_size());
  for (int i = 0; i < shape.dimensions_size(); ++i) {
    dims[i] = shape.dimensions(LayoutUtil::Major(shape.layout(), i));
  }
  return MakeShapeWithMonotonicDim0MajorLayout(shape.element_type(), dims);
}
/* static */ void ShapeUtil::PopulateShape(
    PrimitiveType element_type, tensorflow::gtl::ArraySlice<int64> dimensions,
    Shape* shape) {
  shape->Clear();
  shape->set_element_type(element_type);
  for (int64 dimension : dimensions) {
    shape->add_dimensions(dimension);
  }
  LayoutUtil::SetToDefaultLayout(shape);
  TF_DCHECK_OK(ValidateShape(*shape));
}

/* static */ Shape ShapeUtil::MakeTupleShape(
    tensorflow::gtl::ArraySlice<Shape> shapes) {
  Shape result;
  result.set_element_type(TUPLE);
  for (const auto& shape : shapes) {
    AppendShapeToTuple(shape, &result);
  }
  TF_DCHECK_OK(ValidateShapeWithOptionalLayout(result));
  return result;
}

/* static */ Shape ShapeUtil::MakeOpaqueShape() {
  Shape result;
  result.set_element_type(OPAQUE);
  TF_DCHECK_OK(ValidateShapeWithOptionalLayout(result));
  return result;
}

/* static */ void ShapeUtil::AppendShapeToTuple(const Shape& shape,
                                                Shape* tuple_shape) {
  TF_DCHECK_OK(ValidateShapeWithOptionalLayout(shape));
  *tuple_shape->add_tuple_shapes() = shape;
}

/* static */ void ShapeUtil::AppendMajorDimension(int bound, Shape* shape) {
  shape->mutable_layout()->add_minor_to_major(ShapeUtil::Rank(*shape));
  shape->add_dimensions(bound);
  TF_DCHECK_OK(ValidateShape(*shape));
}

/* static */ bool ShapeUtil::ElementIsIntegral(const Shape& shape) {
  return primitive_util::IsIntegralType(shape.element_type());
}

/* static */ bool ShapeUtil::ElementIsIntegralWithBits(const Shape& shape,
                                                       int32 bits) {
  return ElementIsIntegral(shape) && ElementHasBitWidth(shape, bits);
}

/* static */ bool ShapeUtil::ElementHasBitWidth(const Shape& shape, int bits) {
  if (shape.element_type() == TUPLE || shape.element_type() == OPAQUE) {
    return false;
  }
  return primitive_util::BitWidth(shape.element_type()) == bits;
}

/* static */ bool ShapeUtil::ElementIsSigned(const Shape& shape) {
  switch (shape.element_type()) {
    case S8:
    case S16:
    case S32:
    case S64:
    case F16:
    case F32:
    case F64:
      return true;

    case PRED:
    case U8:
    case U16:
    case U32:
    case U64:
    case TUPLE:
    case OPAQUE:
      return false;

    default:
      LOG(FATAL) << "Unhandled element type " << shape.element_type();
  }
}

/* static */ bool ShapeUtil::ElementIsFloating(const Shape& shape) {
  return primitive_util::IsFloatingPointType(shape.element_type());
}

/* static */ bool ShapeUtil::IsTuple(const Shape& shape) {
  return shape.element_type() == TUPLE;
}

/* static */ bool ShapeUtil::IsArray(const Shape& shape) {
  return !IsTuple(shape) && !IsOpaque(shape);
}

/* static */ bool ShapeUtil::IsNestedTuple(const Shape& shape) {
  return IsTuple(shape) && std::any_of(shape.tuple_shapes().begin(),
                                       shape.tuple_shapes().end(), IsTuple);
}

/* static */ bool ShapeUtil::IsEmptyTuple(const Shape& shape) {
  return IsTuple(shape) && TupleElementCount(shape) == 0;
}

/* static */ bool ShapeUtil::IsNil(const Shape& shape) {
  return IsEmptyTuple(shape) || HasZeroElements(shape);
}

/* static */ int64 ShapeUtil::TupleElementCount(const Shape& shape) {
  CHECK(IsTuple(shape));
  return shape.tuple_shapes_size();
}

/* static */ const Shape& ShapeUtil::GetTupleElementShape(const Shape& shape,
                                                          int64 index) {
  CHECK(IsTuple(shape));
  CHECK_GT(TupleElementCount(shape), index);
  TF_DCHECK_OK(ValidateShapeWithOptionalLayout(shape.tuple_shapes(index)));
  return shape.tuple_shapes(index);
}

/* static */ Shape ShapeUtil::SliceTuple(const Shape& tuple, int64 start,
                                         int64 limit) {
  TF_DCHECK_OK(ValidateShapeWithOptionalLayout(tuple));
  CHECK(IsTuple(tuple));
  CHECK_LE(start, TupleElementCount(tuple));
  CHECK_LE(limit, TupleElementCount(tuple));

  std::vector<Shape> new_elements(tuple.tuple_shapes().begin() + start,
                                  tuple.tuple_shapes().begin() + limit);
  return ShapeUtil::MakeTupleShape(new_elements);
}

/* static */ bool ShapeUtil::IsOpaque(const Shape& shape) {
  return shape.element_type() == OPAQUE;
}

/* static */ bool ShapeUtil::ShapeIs(const Shape& shape,
                                     PrimitiveType element_type,
                                     std::initializer_list<int64> dimensions) {
  TF_DCHECK_OK(ValidateShapeWithOptionalLayout(shape));
  if (shape.element_type() != element_type) {
    return false;
  }
  if (shape.dimensions_size() != ShapeUtil::Rank(shape)) {
    return false;
  }
  int64 i = 0;
  for (int64 dimension : dimensions) {
    if (shape.dimensions(i) != dimension) {
      return false;
    }
    i += 1;
  }
  return true;
}

/* static */ int64 ShapeUtil::ElementsIn(const Shape& shape) {
  CHECK_EQ(shape.dimensions_size(), ShapeUtil::Rank(shape));
  return std::accumulate<decltype(shape.dimensions().begin()), int64>(
      shape.dimensions().begin(), shape.dimensions().end(), 1LL,
      std::multiplies<int64>());
}

/* static */ bool ShapeUtil::HasZeroElements(const Shape& shape) {
  return ElementsIn(shape) == 0;
}

/* static */ bool ShapeUtil::IsScalarF32(const Shape& shape) {
  return shape.element_type() == F32 && ShapeUtil::Rank(shape) == 0;
}

/* static */ string ShapeUtil::HumanString(const Shape& shape) {
  if (shape.element_type() == TUPLE) {
    string text = "(";
    const char* prefix = "";
    for (const Shape& elem_shape : shape.tuple_shapes()) {
      tensorflow::strings::StrAppend(&text, prefix, HumanString(elem_shape));
      prefix = ", ";
    }
    text += ")";
    return text;
  } else {
    return tensorflow::strings::StrCat(
        tensorflow::str_util::Lowercase(
            PrimitiveType_Name(shape.element_type())),
        "[", tensorflow::str_util::Join(shape.dimensions(), ","), "]");
  }
}

/* static */ string ShapeUtil::HumanStringWithLayout(const Shape& shape) {
  if (shape.element_type() == TUPLE) {
    string text = "(";
    const char* prefix = "";
    for (const Shape& elem_shape : shape.tuple_shapes()) {
      tensorflow::strings::StrAppend(&text, prefix,
                                     HumanStringWithLayout(elem_shape));
      prefix = ", ";
    }
    text += ")";
    return text;
  } else {
    string layout;
    if (!IsScalar(shape) && !IsOpaque(shape)) {
      if (LayoutUtil::HasLayout(shape)) {
        layout = tensorflow::strings::StrCat(
            " ", LayoutUtil::HumanString(shape.layout()));
      } else {
        layout = " (no layout)";
      }
    }
    return tensorflow::strings::StrCat(
        tensorflow::str_util::Lowercase(
            PrimitiveType_Name(shape.element_type())),
        "[", tensorflow::str_util::Join(shape.dimensions(), ","), "]", layout);
  }
}

/* static */ string ShapeUtil::HumanString(const ProgramShape& program_shape) {
  std::vector<string> parameters;
  for (auto& shape : program_shape.parameters()) {
    const int i = parameters.size();
    parameters.push_back(
        tensorflow::strings::StrCat(i < program_shape.parameter_names_size()
                                        ? program_shape.parameter_names(i)
                                        : "(unknown)",
                                    ": ", HumanString(shape)));
  }
  return tensorflow::strings::StrCat(
      "(", tensorflow::str_util::Join(parameters, ", "), ") -> ",
      HumanString(program_shape.result()));
}

/* static */ StatusOr<Shape> ShapeUtil::ParseShapeString(const string& s) {
  string element_type_string;
  string dimensions_string;
  string layout_string;
  if (RE2::FullMatch(s, "([fsu]32)\\[([\\d,]*)\\](?: {([\\d,]*)})?",
                     &element_type_string, &dimensions_string,
                     &layout_string)) {
    auto comma_list_to_int64s =
        [&s](const string& input) -> StatusOr<std::vector<int64>> {
      std::vector<int64> results;
      for (const string& piece : tensorflow::str_util::Split(input, ',')) {
        int64 element;
        if (!tensorflow::strings::safe_strto64(piece.c_str(), &element)) {
          return InvalidArgument(
              "invalid value in parsed shape string: \"%s\" in \"%s\"",
              piece.c_str(), s.c_str());
        }
        results.push_back(element);
      }
      return results;
    };
    TF_ASSIGN_OR_RETURN(std::vector<int64> dimensions,
                        comma_list_to_int64s(dimensions_string));
    PrimitiveType primitive_type;
    if (element_type_string == "f32") {
      primitive_type = F32;
    } else if (element_type_string == "s32") {
      primitive_type = S32;
    } else if (element_type_string == "u32") {
      primitive_type = U32;
    } else {
      LOG(FATAL) << "unhandled element type string: " << element_type_string;
    }
    Shape result;
    if (layout_string.empty()) {
      result = ShapeUtil::MakeShape(primitive_type, dimensions);
    } else {
      TF_ASSIGN_OR_RETURN(std::vector<int64> min2maj,
                          comma_list_to_int64s(layout_string));
      TF_RET_CHECK(dimensions.size() == min2maj.size());
      result =
          ShapeUtil::MakeShapeWithLayout(primitive_type, dimensions, min2maj);
    }
    TF_DCHECK_OK(ValidateShape(result));
    return result;
  }

  return InvalidArgument("invalid shape string to parse: \"%s\"", s.c_str());
}

/* static */ bool ShapeUtil::SameDimensions(const Shape& lhs,
                                            const Shape& rhs) {
  return ContainersEqual(lhs.dimensions(), rhs.dimensions());
}

/* static */ bool ShapeUtil::Compatible(const Shape& lhs, const Shape& rhs) {
  if (lhs.element_type() == TUPLE) {
    return rhs.element_type() == TUPLE &&
           ContainersEqual(lhs.tuple_shapes(), rhs.tuple_shapes(), Compatible);
  }
  return SameDimensions(lhs, rhs) && SameElementType(lhs, rhs);
}

/* static */ int64 ShapeUtil::GetDimension(const Shape& shape,
                                           int64 dimension_number) {
  return shape.dimensions(GetDimensionNumber(shape, dimension_number));
}

/* static */ int64 ShapeUtil::GetDimensionNumber(const Shape& shape,
                                                 int64 dimension_number) {
  if (dimension_number < 0) {
    dimension_number += ShapeUtil::Rank(shape);
  }
  CHECK_GE(dimension_number, 0);
  return dimension_number;
}

/* static */ int64 ShapeUtil::ByteSizeOfPrimitiveType(
    PrimitiveType primitive_type) {
  switch (primitive_type) {
    case PRED:
      return sizeof(int8);
    case TUPLE:
      LOG(FATAL) << "tuples have no definitive size";
    case OPAQUE:
      LOG(FATAL) << "opaque have no definitive size";
    case S8:
      return sizeof(int8);
    case S16:
      return sizeof(int16);
    case S32:
      return sizeof(int32);
    case S64:
      return sizeof(int64);
    case U8:
      return sizeof(uint8);
    case U16:
      return sizeof(uint16);
    case U32:
      return sizeof(uint32);
    case U64:
      return sizeof(uint64);
    case F16:
      return sizeof(float) / 2;
    case F32:
      return sizeof(float);
    case F64:
      return sizeof(double);
    default:
      LOG(FATAL) << "Unhandled primitive type " << primitive_type;
  }
}

/* static */ int64 ShapeUtil::ByteSizeOf(const Shape& shape,
                                         int64 pointer_size) {
  TF_DCHECK_OK(ValidateShape(shape));
  DCHECK_NE(OPAQUE, shape.element_type());
  if (shape.element_type() == TUPLE) {
    return pointer_size * shape.tuple_shapes_size();
  }
  int64 allocated_element_count;
  if (shape.layout().padded_dimensions_size() > 0) {
    CHECK_EQ(ShapeUtil::Rank(shape), shape.layout().padded_dimensions_size());
    allocated_element_count = 1;
    for (int64 dimension_size : shape.layout().padded_dimensions()) {
      allocated_element_count *= dimension_size;
    }
  } else {
    allocated_element_count = ElementsIn(shape);
  }
  return allocated_element_count *
         ByteSizeOfPrimitiveType(shape.element_type());
}

/* static */ int64 ShapeUtil::ByteSizeOf(const Shape& shape) {
  return ShapeUtil::ByteSizeOf(shape, /*pointer_size=*/sizeof(void*));
}

/* static */ Status ShapeUtil::ValidateShapeWithOptionalLayoutInternal(
    const Shape& shape) {
  if (shape.element_type() == TUPLE) {
    // Tuple shape.
    if (ShapeUtil::Rank(shape) != 0) {
      return InvalidArgument("tuples must be rank-0; got rank %lld",
                             ShapeUtil::Rank(shape));
    }
    if (shape.dimensions_size() != 0) {
      return InvalidArgument("tuples must not have dimensions specified");
    }
    for (auto& element_shape : shape.tuple_shapes()) {
      TF_RETURN_IF_ERROR(
          ValidateShapeWithOptionalLayoutInternal(element_shape));
    }
    return Status::OK();
  }

  // Non-tuple shape.
  if (shape.tuple_shapes_size() > 0) {
    return InvalidArgument("non-tuple shape has tuple_shapes field");
  }
  if (shape.element_type() == PRIMITIVE_TYPE_INVALID) {
    return InvalidArgument("shape has invalid element type: %s",
                           shape.ShortDebugString().c_str());
  }
  if (ShapeUtil::Rank(shape) != shape.dimensions_size()) {
    return InvalidArgument(
        "shape's rank is mismatched with dimension count; rank=%lld "
        "dimensions_size=%d",
        ShapeUtil::Rank(shape), shape.dimensions_size());
  }
  for (int64 i = 0; i < ShapeUtil::Rank(shape); ++i) {
    int64 dimension = shape.dimensions(i);
    if (dimension < 0) {
      return InvalidArgument(
          "shape's dimensions must not be < 0; dimension at index %lld was "
          "%lld",
          i, dimension);
    }
  }

  return Status::OK();
}

/* static */ Status ShapeUtil::ValidateShapeWithOptionalLayout(
    const Shape& shape) {
  if (LayoutUtil::HasLayout(shape)) {
    // Since a layout is present, upgrade to the full set of invariant checks.
    return ValidateShape(shape);
  }
  return ValidateShapeWithOptionalLayoutInternal(shape);
}

/* static */ Status ShapeUtil::ValidateShape(const Shape& shape) {
  TF_RETURN_IF_ERROR(ValidateShapeWithOptionalLayoutInternal(shape));

  return LayoutUtil::ValidateLayoutInShape(shape);
}

/* static */ Shape ShapeUtil::ChangeElementType(const Shape& shape,
                                                PrimitiveType type) {
  Shape new_shape = shape;
  new_shape.set_element_type(type);
  return new_shape;
}

/* static */ const Shape& ShapeUtil::GetSubshape(const Shape& shape,
                                                 const ShapeIndex& index) {
  const Shape* return_shape = &shape;
  for (auto i : index) {
    CHECK(IsTuple(*return_shape));
    return_shape = &return_shape->tuple_shapes(i);
  }
  return *return_shape;
}

/* static */ Shape* ShapeUtil::GetMutableSubshape(Shape* shape,
                                                  const ShapeIndex& index) {
  Shape* return_shape = shape;
  for (auto i : index) {
    CHECK(IsTuple(*return_shape));
    return_shape = return_shape->mutable_tuple_shapes(i);
  }
  return return_shape;
}

/* static */ Shape ShapeUtil::StripDegenerateDimensions(const Shape& shape) {
  std::vector<int64> dimension_sizes;
  std::vector<int64> degenerate_dimensions;
  for (int64 i = 0; i < shape.dimensions_size(); ++i) {
    if (shape.dimensions(i) == 1) {
      degenerate_dimensions.push_back(i);
    } else {
      dimension_sizes.push_back(shape.dimensions(i));
    }
  }

  // Construct minor_to_major of stripped shape. The order of the non-degenerate
  // dimensions should be preserved from the original shape. First, create
  // vector of the non-degenerate dimensions from the original minor_to_major
  // array.
  std::vector<int64> minor_to_major;
  for (int64 i : shape.layout().minor_to_major()) {
    if (std::find(degenerate_dimensions.begin(), degenerate_dimensions.end(),
                  i) == degenerate_dimensions.end()) {
      minor_to_major.push_back(i);
    }
  }

  // The dimensions in minor_to_major need to be renumbered to account for the
  // degenerate dimensions which have removed. Decrement each dimension number
  // once for each degenerate dimension which has a smaller number.
  for (int i = 0; i < minor_to_major.size(); ++i) {
    int adjustment = 0;
    for (int64 dim : degenerate_dimensions) {
      if (minor_to_major[i] > dim) {
        adjustment++;
      }
    }
    minor_to_major[i] -= adjustment;
  }

  {
    std::vector<int64> dims(minor_to_major.size());
    std::iota(dims.begin(), dims.end(), 0);
    DCHECK(minor_to_major.size() == dims.size() &&
           std::is_permutation(minor_to_major.begin(), minor_to_major.end(),
                               dims.begin()));
  }
  Shape stripped_shape =
      shape.has_layout() ? MakeShapeWithLayout(shape.element_type(),
                                               dimension_sizes, minor_to_major)
                         : MakeShape(shape.element_type(), dimension_sizes);

  VLOG(10) << "Original_shape: " << HumanStringWithLayout(shape);
  VLOG(10) << "Stripped_shape: " << HumanStringWithLayout(stripped_shape);
  return stripped_shape;
}

namespace {

// Helper for ForEachSubshape which visits the subshapes of the given shape in
// DFS pre-order starting with the index.
Status ForEachSubshapeHelper(const Shape& shape,
                             const ShapeUtil::VisitorFunction func,
                             ShapeIndex* index) {
  TF_RETURN_IF_ERROR(func(shape, *index));
  if (ShapeUtil::IsTuple(shape)) {
    for (int64 i = 0; i < ShapeUtil::TupleElementCount(shape); ++i) {
      index->push_back(i);
      TF_RETURN_IF_ERROR(ForEachSubshapeHelper(
          ShapeUtil::GetTupleElementShape(shape, i), func, index));
      index->pop_back();
    }
  }
  return Status::OK();
}

// Helper for ForEachMutableSubshape which visits the subshapes of the given
// shape in DFS pre-order starting with the index.
Status ForEachMutableSubshapeHelper(
    Shape* shape, const ShapeUtil::MutatingVisitorFunction func,
    ShapeIndex* index) {
  TF_RETURN_IF_ERROR(func(shape, *index));
  if (ShapeUtil::IsTuple(*shape)) {
    for (int64 i = 0; i < ShapeUtil::TupleElementCount(*shape); ++i) {
      index->push_back(i);
      TF_RETURN_IF_ERROR(ForEachMutableSubshapeHelper(
          shape->mutable_tuple_shapes(i), func, index));
      index->pop_back();
    }
  }
  return Status::OK();
}

}  // namespace

/* static */ Status ShapeUtil::ForEachSubshape(const Shape& shape,
                                               VisitorFunction func) {
  ShapeIndex index;
  return ForEachSubshapeHelper(shape, func, &index);
}

/* static */ Status ShapeUtil::ForEachMutableSubshape(
    Shape* shape, MutatingVisitorFunction func) {
  ShapeIndex index;
  return ForEachMutableSubshapeHelper(shape, func, &index);
}

/* static */ Shape ShapeUtil::PermuteDimensions(
    tensorflow::gtl::ArraySlice<int64> permutation, const Shape& shape) {
  Shape new_shape = shape;
  new_shape.clear_dimensions();
  for (auto dim : Permute(permutation, shape.dimensions())) {
    new_shape.add_dimensions(dim);
  }
  if (shape.has_layout()) {
    new_shape.mutable_layout()->clear_minor_to_major();
    for (auto index : Permute(permutation, shape.layout().minor_to_major())) {
      new_shape.mutable_layout()->add_minor_to_major(index);
    }
  }
  return new_shape;
}

/* static */ std::tuple<bool, std::vector<int64>, std::vector<int64>>
ShapeUtil::InsertedOrDeleted1SizedDimensions(const Shape& shape_pre,
                                             const Shape& shape_post) {
  auto nil = std::make_tuple(false, std::vector<int64>(), std::vector<int64>());

  std::vector<int64> deleted_indices;
  std::vector<int64> inserted_indices;
  // Returns false if any input/output index between prior_unmodified_dim_pair
  // and unmodified_dim_pair have size >1. Otherwise, returns true and appends
  // the degerenate input/output dimensions in the gap to
  // deleted_indices/inserted_indices respectively.
  auto check_modified_dims = [&shape_pre, &shape_post, &deleted_indices,
                              &inserted_indices](
      std::pair<int64, int64> prior_unmodified_dim_pair,
      std::pair<int64, int64> unmodified_dim_pair) {
    for (int64 modified_input_dim = prior_unmodified_dim_pair.first + 1;
         modified_input_dim < unmodified_dim_pair.first; ++modified_input_dim) {
      if (shape_pre.dimensions(modified_input_dim) > 1) {
        return false;
      }
      deleted_indices.push_back(modified_input_dim);
    }
    for (int64 modified_output_dim = prior_unmodified_dim_pair.second + 1;
         modified_output_dim < unmodified_dim_pair.second;
         ++modified_output_dim) {
      if (shape_post.dimensions(modified_output_dim) > 1) {
        return false;
      }
      inserted_indices.push_back(modified_output_dim);
    }
    return true;
  };

  std::vector<std::pair<int64, int64>> unmodified_dims =
      DimensionsUnmodifiedByReshape(shape_pre, shape_post);
  // Returns nil if the reshape modifies any non-degenerate input/output
  // dimension. DimensionsUnmodifiedByReshape gives us all unmodified
  // dimensions, so we only need to check whether dimensions in the gaps (thus
  // modified) have size >1.
  for (size_t i = 0; i <= unmodified_dims.size(); ++i) {
    // Check (modified) dimensions between unmodified_dims[i-1] and
    // unmodified_dims[i].
    auto prior_unmodified_dim_pair =
        i > 0 ? unmodified_dims[i - 1] : std::make_pair(-1LL, -1LL);
    auto unmodified_dim_pair =
        i < unmodified_dims.size()
            ? unmodified_dims[i]
            : std::make_pair(ShapeUtil::Rank(shape_pre),
                             ShapeUtil::Rank(shape_post));
    if (!check_modified_dims(prior_unmodified_dim_pair, unmodified_dim_pair)) {
      return nil;
    }
  }

  return std::make_tuple(true, deleted_indices, inserted_indices);
}

/* static */ std::vector<std::pair<int64, int64>>
ShapeUtil::DimensionsUnmodifiedByReshape(const Shape& input_shape,
                                         const Shape& output_shape) {
  // Unmodified dimensions are merely common factors of rank 1.
  auto common_factors = CommonFactors(AsInt64Slice(input_shape.dimensions()),
                                      AsInt64Slice(output_shape.dimensions()));
  for (size_t i = 0; i < common_factors.size() - 1;) {
    if (1 != common_factors[i + 1].first - common_factors[i].first ||
        1 != common_factors[i + 1].second - common_factors[i].second) {
      common_factors.erase(common_factors.begin() + i);
    } else {
      ++i;
    }
  }
  // `CommonFactors(a, b).back() == (a.rank, b.rank)` so we must pop it.
  common_factors.pop_back();
  return common_factors;
}

/* static */ bool ShapeUtil::TransposeIsBitcast(
    const Shape& input_shape, const Shape& output_shape,
    tensorflow::gtl::ArraySlice<int64> dimension_mapping) {
  // Can't insert bitcasts without layout information.
  if (!LayoutUtil::HasLayout(input_shape) &&
      !LayoutUtil::HasLayout(output_shape)) {
    return false;
  }

  // Padding is not handled.
  if (LayoutUtil::IsPadded(input_shape) && LayoutUtil::IsPadded(output_shape)) {
    return false;
  }

  // Check the reshape permutes the positions of each dimension in the
  // minor-to-major order. positions[i]=k means dimension `i` is k-th minor.
  //   input_positions = apply(dimension_mapping, output_positions)
  //
  // Because the positions of each dimension are the inverse permutation of the
  // minor-to-major order, the above check is equivalent to
  //   inverse(input_dimensions) =
  //       apply(dimension_mapping, inverse(output_dimensions))
  //   # `I` indicates identity permutation.
  //   apply(input_dimensions, I) =
  //       apply(dimension_mapping, apply(output_dimensions, I))
  //   apply(input_dimensions, I) =
  //       apply((dimension_mapping * output_dimensions), I)
  //   input_dimensions = dimension_mapping * output_dimensions
  return ContainersEqual(
      ComposePermutations(dimension_mapping,
                          AsInt64Slice(output_shape.layout().minor_to_major())),
      input_shape.layout().minor_to_major());
}

/* static */ bool ShapeUtil::ReshapeIsBitcast(const Shape& input_shape,
                                              const Shape& output_shape) {
  // Can't convert reshapes into bitcasts without layout information.
  if (!LayoutUtil::HasLayout(input_shape) ||
      !LayoutUtil::HasLayout(output_shape)) {
    return false;
  }

  // Padding is not handled.
  if (LayoutUtil::IsPadded(input_shape) || LayoutUtil::IsPadded(output_shape)) {
    return false;
  }

  CHECK_EQ(ShapeUtil::ElementsIn(input_shape),
           ShapeUtil::ElementsIn(output_shape));
  if (ShapeUtil::ElementsIn(input_shape) == 0) {
    return true;
  }

  // TL;DR: The rest of the method checks that the reshape does not change the
  // physical location of any unit input or output index. Unit indices have
  // exactly one dimension that equals 1 and other dimensions 0. This condition
  // is necessary for the reshape to be a bitcast, because a bitcast-equivalent
  // reshape shouldn't change the physical location of any element. It is also a
  // sufficient condition as is proved below (note: many details are omitted for
  // space).
  //
  // Definitions:
  //
  // * Denote the input shape by IS and output shape by OS. IS[i] or OS[i] means
  // the size of i-th least significant dimension of IS or OS (this is opposite
  // to how we define the index of Shape::dimensions()).
  //
  // * Given an input or output index I, denote by p(I) I's physical linear
  // index (or physical index for short) and l(I) I's logical linear index (or
  // logical index for short).
  //
  // * Given a logical index k, denote by II(k) the input index whose linear
  // index is k, and OI(k) the corresponding output index.
  //
  // * Denote by IT[i] the increment of physical index if i-th dimension of the
  // input index is increased by 1. Similarly, OT[i] means the increment if i-th
  // dimension of the output index is increased by 1. Note that IT[i] or OT[i]
  // is a function of IS or OS and the layout, and not dependent on the specific
  // input or output index.
  //
  // To prove the reshape from IS to OS is a bitcast, it is sufficient to prove
  // that, for any linear index k, p(II(k))=p(OI(k)). We prove this by
  // induction. We know p(II(0))=p(OI(0)) is trivially true, so what's left is
  // to prove, with every increment on k, the above formula still holds.
  //
  // First, suppose reshaping from IS to OS is non-factorizable (we discuss
  // refactorizable reshapes later). A reshape from IS to OS is factorizable, if
  // there exists (i,j) such that
  //
  //   0<=i<=|IS|
  //   0<=j<=|OS|
  //   |IS|-i+|OS|-j > 0 (i.e., i,j mustn't both point to the end)
  //   product(IS[i], IS[i+1], ..., IS[|IS|-1])
  //     = product(OS[j], OS[j+1], ..., OS[|OS|-1])
  //
  // p(II(k))=p(OI(k)) is trivially true for k=0 because p(II(0)) and p(OI(0))
  // are both 0. It's also trivially true for k=1, because II(1) and OI(1) are
  // unit indices which are already tested. This also means IT[0]=OT[0]
  // because p(II(1))=IT[0] and p(OI(1))=OT[0].
  //
  // Furthermore, p(II(k))=p(OI(k)) for k<min(IS[0],OS[0]), because each
  // increment of k adds IT[0] to the input physical and OT[0] (same as IT[0])
  // to the output physical.
  //
  // When k=min(IS[0],OS[0]), the first wrap happens. Without losing generality,
  // suppose IS[0]<OS[0] and thus k=IS[0]. Similar proof applies to IS[0]>OS[0].
  // Note that IS[0]!=OS[0] because the reshape is non-factorizable. From
  // logical index k-1 to logical index k, dimension 1 of the input index
  // is increased by 1 and dimension 0 is reset to 0 thus decreased by
  // IS[0]-1. Therefore, the physical input index is increased by
  //
  //   p(II(k)) - p(II(k-1)) = IT[1] - (IS[0]-1) * IT[0]
  //
  // Because IS[0]<OS[0], the only change to the output index is that its
  // dimension 0 is increased by one. Therefore,
  //
  //   p(OI(k)) - p(OI(k-1)) = OT[0] = IT[0]
  //
  // Because II(k) is an unit index -- (0,..,0,1,0), we already tested that
  // p(II(k))=p(OI(k)). Therefore,
  //   IT[1] - (IS[0]-1) * IT[0] = IT[0]
  //   IT[1] = IS[0] * IT[0]
  // In other words, input dimension 1 is immediately more major than input
  // dimension 0. We can now conceptually collapse these two dimensions because
  // an increment in the logical index affecting only these two dimensions maps
  // to IT[0] in the physical index.
  //
  // By induction (omitted here), we can prove IT[i]=IS[i-1]*IT[i-1] and
  // OT[i]=OS[i-1]*OT[i-1]. Therefore, both IS and OS are row-major and bitwise
  // identical.
  //
  // A factorizable reshape can be factorized into a list of non-factorizable
  // sub-reshapes, each of which can be handled similarly to the proof above.
  // For example,
  //
  //   [7x9x2x15] -> [63x6x5]
  //
  // can be factorized into
  //
  //   [7x9] -> [63] and [2x15] -> [6x5].
  //
  // Suppose input index I=(x3,x2,x1,x0) and output index O=(y2,y1,y0) have the
  // same logical linear index. According to the factorization, we know
  // l(x3,x2,0,0)=l(y2,0,0) and l(0,0,x1,x0)=l(0,y1,y0). Using the proof for
  // non-factorizable reshapes, we can prove p(0,0,x1,x0)=p(0,y1,y0). Using a
  // similar proof, with the increment of the logical index set to
  // IS[1]*IS[0]=OS[1]*OS[0]=30 instead of 1, we can prove
  // p(x3,x2,0,0)=p(y2,0,0) too. Therefore,
  //
  //   p(x3,x2,x1,x0) = p(x3,x2,0,0) + p(0,0,x1,x0)
  //                  = p(y2,0,0) + p(0,0,y1,y0)
  //                  = p(y2,y1,y0)
  //
  // check_input_unit_indices checks one way of the condition: each input unit
  // index is mapped to an output index with the same physical location. This
  // lambda will be called again with input_shape and output_shape reversed to
  // check the other way.
  auto check_input_unit_indices = [](const Shape& input_shape,
                                     const Shape& output_shape) {
    // input_shape_dim0_major/output_shape_dim0_major has the same "dimensions"
    // as input_shape/output_shape and the dimension-0-major layout. These two
    // shapes are used for conversion between logical linear indices and
    // multi-dimensional indices.
    Shape input_shape_dim0_major =
        ShapeUtil::MakeShapeWithMonotonicDim0MajorLayout(
            input_shape.element_type(), AsInt64Slice(input_shape.dimensions()));
    Shape output_shape_dim0_major =
        ShapeUtil::MakeShapeWithMonotonicDim0MajorLayout(
            output_shape.element_type(),
            AsInt64Slice(output_shape.dimensions()));

    for (int64 input_dim = 0; input_dim < ShapeUtil::Rank(input_shape);
         ++input_dim) {
      if (input_shape.dimensions(input_dim) <= 1) {
        continue;
      }

      std::vector<int64> input_unit_index(ShapeUtil::Rank(input_shape), 0);
      input_unit_index[input_dim] = 1;
      int64 logical_linear_index =
          IndexUtil::MultidimensionalIndexToLinearIndex(input_shape_dim0_major,
                                                        input_unit_index);
      // output_index has the same logical linear index as input_unit_index.
      std::vector<int64> output_index =
          IndexUtil::LinearIndexToMultidimensionalIndex(output_shape_dim0_major,
                                                        logical_linear_index);
      // Check input_unit_index and output_index have the same physical linear
      // index.
      if (IndexUtil::MultidimensionalIndexToLinearIndex(input_shape,
                                                        input_unit_index) !=
          IndexUtil::MultidimensionalIndexToLinearIndex(output_shape,
                                                        output_index)) {
        return false;
      }
    }
    return true;
  };
  return check_input_unit_indices(input_shape, output_shape) &&
         check_input_unit_indices(output_shape, input_shape);
}

/* static */ Shape ShapeUtil::DeleteDimension(int64 dim_to_delete,
                                              Shape shape) {
  shape.mutable_dimensions()->erase(shape.dimensions().begin() + dim_to_delete);
  if (LayoutUtil::HasLayout(shape)) {
    Layout* layout = shape.mutable_layout();
    for (size_t i = 0; i < layout->minor_to_major().size();) {
      if (layout->minor_to_major(i) == dim_to_delete) {
        layout->mutable_minor_to_major()->erase(
            layout->minor_to_major().begin() + i);
        continue;
      }
      if (layout->minor_to_major(i) > dim_to_delete) {
        (*layout->mutable_minor_to_major())[i] -= 1;
      }
      ++i;
    }
  }
  return shape;
}

/* static */ Shape ShapeUtil::FilterDimensions(
    const std::function<bool(int64)>& p, Shape shape) {
  std::vector<int64> dims_to_delete;
  for (int64 i = shape.dimensions().size() - 1; i >= 0; --i) {
    if (!p(i)) {
      dims_to_delete.push_back(i);
    }
  }
  for (int64 dim : dims_to_delete) {
    shape = DeleteDimension(dim, shape);
  }
  return shape;
}

}  // namespace xla
