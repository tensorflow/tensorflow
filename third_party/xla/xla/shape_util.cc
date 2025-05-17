/* Copyright 2017 The OpenXLA Authors.

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

#include "xla/shape_util.h"

#include <algorithm>
#include <climits>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <initializer_list>
#include <numeric>
#include <optional>
#include <ostream>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/base/optimization.h"
#include "absl/container/inlined_vector.h"
#include "absl/functional/function_ref.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "xla/index_util.h"
#include "xla/layout.h"
#include "xla/layout_util.h"
#include "xla/overflow_util.h"
#include "xla/permutation_util.h"
#include "xla/primitive_util.h"
#include "xla/printer.h"
#include "xla/shape.h"
#include "xla/shape_partition.h"
#include "xla/status_macros.h"
#include "xla/tsl/lib/math/math_util.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/logging.h"  // IWYU pragma: keep
#include "xla/tsl/platform/macros.h"
#include "xla/tsl/platform/status.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/threadpool.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/cpu_info.h"

namespace xla {

using absl::StrCat;

namespace {

constexpr int64_t kAnnotationPrintInterval = 5;

inline absl::Status ShapeError(const Shape& shape, absl::string_view message) {
  return absl::InvalidArgumentError(absl::StrFormat(
      "Shape Error: %s Shape(%s): %s", message,
      PrimitiveType_IsValid(shape.element_type())
          ? primitive_util::LowercasePrimitiveTypeName(shape.element_type())
          : absl::StrCat(static_cast<int>(shape.element_type())),
      shape.ToString()));
}

template <bool kPrintLayout>
void PrintShape(Printer* printer, const Shape& shape) {
  if constexpr (kPrintLayout) {
    ShapeUtil::PrintHumanStringWithLayout(printer, shape);
  } else {
    ShapeUtil::PrintHumanString(printer, shape);
  }
}

template <bool kPrintLayout>
void PrintTupleShapes(Printer* printer, absl::Span<const Shape> tuple_shapes) {
  if (ABSL_PREDICT_FALSE(tuple_shapes.empty())) {
    printer->Append("()");
    return;
  }
  printer->Append("(");
  PrintShape<kPrintLayout>(printer, tuple_shapes[0]);
  for (int64_t i = 1; i < tuple_shapes.size(); ++i) {
    if (i % kAnnotationPrintInterval == 0) {
      // Faster than printer->Append(absl::StrFormat(", /*index=%lld*/", i));
      printer->Append(", /*index=");
      printer->Append(i);
      printer->Append("*/");
    } else {
      printer->Append(", ");
    }
    PrintShape<kPrintLayout>(printer, tuple_shapes[i]);
  }
  printer->Append(")");
}

template <bool kPrintLayout>
void PrintBufferShape(Printer* printer, const Shape& shape) {
  printer->Append("b(");
  PrintShape<kPrintLayout>(printer, shape.buffer_shape());
  printer->Append(")");
}

// Constructs and returns the new shape with the given minor_to_major order in
// its Layout.
absl::StatusOr<Shape> MakeShapeWithLayoutInternal(
    PrimitiveType element_type, absl::Span<const int64_t> dimensions,
    absl::Span<const int64_t> minor_to_major,
    absl::Span<const Tile> tiles, int64_t tail_padding_alignment_in_elements,
    PrimitiveType index_primitive_type, PrimitiveType pointer_primitive_type,
    int64_t element_size_in_bits, int64_t memory_space,
    absl::Span<const SplitConfig> split_configs,
    std::optional<Shape> physical_shape) {
  if (dimensions.size() != minor_to_major.size()) {
    return InvalidArgument("Dimensions size is %ld, but layout size is %ld.",
                           dimensions.size(), minor_to_major.size());
  }
  if (element_type == OPAQUE_TYPE || element_type == TUPLE ||
      element_type == TOKEN) {
    return InvalidArgument("Unsupported element type: %s",
                           PrimitiveType_Name(element_type));
  }
  TF_ASSIGN_OR_RETURN(Shape shape,
                      ShapeUtil::MakeValidatedShape(element_type, dimensions));
  if (element_size_in_bits ==
      ShapeUtil::ByteSizeOfPrimitiveType(element_type) * 8) {
    // Only set element_size_in_bits if it's different from the default value.
    element_size_in_bits = 0;
  }
  *shape.mutable_layout() = LayoutUtil::MakeLayout(
      minor_to_major, tiles, tail_padding_alignment_in_elements,
      index_primitive_type, pointer_primitive_type, element_size_in_bits,
      memory_space, split_configs, std::move(physical_shape));
  TF_RETURN_IF_ERROR(ShapeUtil::ValidateShape(shape));
  return shape;
}

template <typename T>
const T& Deref(const T* ptr) {
  DCHECK(ptr != nullptr);
  return *ptr;
}

template <typename T>
const T& Deref(const T& ref) {
  return ref;
}

template <typename ShapePtrOrRef>
Shape MakeTupleShapeImpl(absl::Span<ShapePtrOrRef> shapes) {
  Shape result(std::vector<Shape>{});
  result.mutable_tuple_shapes()->reserve(shapes.size());
  for (const auto& shape : shapes) {
    ShapeUtil::AppendShapeToTuple(Deref(shape), &result);
  }
  TF_DCHECK_OK(ShapeUtil::ValidateShapeWithOptionalLayout(result));
  return result;
}

}  // namespace

std::string ShapeIndex::ToString() const {
  return StrCat("{", absl::StrJoin(*this, ","), "}");
}

std::ostream& operator<<(std::ostream& out, const ShapeIndex& shape_index) {
  out << shape_index.ToString();
  return out;
}

/* static */ bool ShapeUtil::Equal(const Shape& lhs, const Shape& rhs) {
  bool equal = Shape::Equal()(lhs, rhs);

  if (!equal && VLOG_IS_ON(3)) {
    VLOG(3) << "ShapeUtil::Equal differ: lhs = " << lhs.ToString()
            << ", rhs = " << rhs.ToString();
  }

  return equal;
}

/* static */ bool ShapeUtil::EqualIgnoringElementType(const Shape& lhs,
                                                      const Shape& rhs) {
  bool equal = Shape::Equal().IgnoreElementType()(lhs, rhs);
  if (!equal && VLOG_IS_ON(3)) {
    VLOG(3) << "ShapeUtil::EqualIgnoringElementType differ: lhs = "
            << lhs.ToString() << ", rhs = " << rhs.ToString();
  }

  return equal;
}

/* static */ bool ShapeUtil::EqualIgnoringFpPrecision(const Shape& lhs,
                                                      const Shape& rhs) {
  bool equal = Shape::Equal().IgnoreFpPrecision()(lhs, rhs);
  if (!equal && VLOG_IS_ON(3)) {
    VLOG(3) << "ShapeUtil::EqualIgnoringFpPrecision differ: lhs = "
            << lhs.ToString() << ", rhs = " << rhs.ToString();
  }

  return equal;
}

/* static */ bool ShapeUtil::EqualStructure(const Shape& lhs,
                                            const Shape& rhs) {
  bool equal = true;
  ForEachSubshape(lhs, [&](const Shape& /*subshape*/, const ShapeIndex& index) {
    equal = equal && IndexIsValid(rhs, index);
  });
  ForEachSubshape(rhs, [&](const Shape& /*subshape*/, const ShapeIndex& index) {
    equal = equal && IndexIsValid(lhs, index);
  });

  return equal;
}

/* static */ int64_t ShapeUtil::TrueNumDimensions(const Shape& array_shape) {
  CHECK(array_shape.IsArray())
      << "TrueNumDimensions called on non-array shape: "
      << array_shape.ToString();

  int64_t accum = 0;
  for (const int64_t dimension : array_shape.dimensions()) {
    // We do not count unit dimensions.
    if (dimension != 1) {
      accum += 1;
    }
  }
  return accum;
}

/* static */ ProgramShape ShapeUtil::MakeProgramShape(
    std::initializer_list<Shape> parameters, Shape result) {
  ProgramShape program_shape;
  for (const Shape& shape : parameters) {
    program_shape.AddParameter(shape, "");
  }
  *program_shape.mutable_result() = std::move(result);
  return program_shape;
}

static std::vector<bool> MakeDynamicDimensions(
    absl::Span<const int64_t> dimensions) {
  std::vector<bool> dynamic_dimensions;
  dynamic_dimensions.reserve(dimensions.size());
  for (int64_t dimension : dimensions) {
    dynamic_dimensions.push_back(dimension == Shape::kUnboundedSize);
  }
  return dynamic_dimensions;
}

/* static */ Shape ShapeUtil::MakeShape(PrimitiveType element_type,
                                        absl::Span<const int64_t> dimensions) {
  return MakeValidatedShape(element_type, dimensions).value();
}

/* static */ Shape ShapeUtil::MakeScalarShape(PrimitiveType element_type) {
  return MakeShape(element_type, {});
}

/* static */ Shape ShapeUtil::MakeShape(
    PrimitiveType element_type, absl::Span<const int64_t> dimensions,
    const std::vector<bool>& dynamic_dimensions) {
  return MakeValidatedShape(element_type, dimensions, dynamic_dimensions)
      .value();
}

/* static */ Shape ShapeUtil::MakeBufferShape(
    PrimitiveType element_type, absl::Span<const int64_t> dimensions) {
  return Shape::MakeBufferShape(MakeShape(element_type, dimensions));
}

/* static */ Shape ShapeUtil::MakeShapeWithStaticDimensions(
    const Shape& shape) {
  Shape output = shape;
  output.clear_dynamic_dimensions();
  return output;
}

/* static */ absl::StatusOr<Shape> ShapeUtil::MakeValidatedShape(
    PrimitiveType element_type, absl::Span<const int64_t> dimensions) {
  return MakeValidatedShape(element_type, dimensions,
                            MakeDynamicDimensions(dimensions));
}

/* static */ absl::StatusOr<Shape> ShapeUtil::MakeValidatedShape(
    PrimitiveType element_type, absl::Span<const int64_t> dimensions,
    const std::vector<bool>& dynamic_dimensions) {
  if (dynamic_dimensions.size() != dimensions.size()) {
    return InvalidArgument(
        "dynamic dimensions size %d did not match number of dimensions %d",
        dynamic_dimensions.size(), dimensions.size());
  }

  Shape shape;
  int64_t dense_shape_size = primitive_util::IsArrayType(element_type)
                                 ? primitive_util::ByteWidth(element_type)
                                 : -1;

  // Verify that array-based lookup is consistent with public API.
  DCHECK_EQ(dense_shape_size, ByteSizeOfPrimitiveType(element_type))
      << element_type;

  shape.set_element_type(element_type);
  const int ndims = dimensions.size();
  auto layout = shape.mutable_layout();
  auto* minor_to_major = layout->mutable_minor_to_major();
  int64_t static_extent_product = dense_shape_size;
  bool any_overflows = false;
  for (int i = 0; i < ndims; i++) {
    const int64_t d = dimensions[i];
    const bool is_dynamic = dynamic_dimensions[i];
    if (!Shape::IsValidDimensionSize(d, is_dynamic)) {
      return InvalidArgument("Invalid dimension size %d, is_dynamic=%s", d,
                             is_dynamic ? "true" : "false");
    }
    if (d != Shape::kUnboundedSize) {
      bool overflow;
      std::tie(static_extent_product, overflow) =
          OverflowSafeMultiply(static_extent_product, d);
      any_overflows |= overflow;
    }

    shape.add_dimensions(d, is_dynamic);
    minor_to_major->push_back(ndims - 1 - i);
  }

  if (any_overflows) {
    return InvalidArgument("overflow in static extent product: dimes=[%s]",
                           absl::StrJoin(dimensions, ","));
  }
  return shape;
}

/* static */ Shape ShapeUtil::MakeShapeWithDenseLayout(
    PrimitiveType element_type, absl::Span<const int64_t> dimensions,
    absl::Span<const int64_t> minor_to_major, absl::Span<const Tile> tiles,
    int64_t tail_padding_alignment_in_elements, int64_t element_size_in_bits,
    int64_t memory_space, absl::Span<const SplitConfig> split_configs) {
  auto ret = MakeShapeWithLayoutInternal(
      element_type, dimensions, minor_to_major, tiles,
      tail_padding_alignment_in_elements,
      /*index_primitive_type=*/PRIMITIVE_TYPE_INVALID,
      /*pointer_primitive_type=*/PRIMITIVE_TYPE_INVALID, element_size_in_bits,
      memory_space, split_configs,
      /*physical_shape=*/std::nullopt);
  TF_CHECK_OK(ret.status());
  return *ret;
}

/* static */ Shape ShapeUtil::MakeShapeWithSparseLayout(
    PrimitiveType element_type, absl::Span<const int64_t> dimensions,
    absl::Span<const int64_t> minor_to_major,
    PrimitiveType index_primitive_type, PrimitiveType pointer_primitive_type,
    int64_t tail_padding_alignment_in_elements, int64_t element_size_in_bits,
    int64_t memory_space, std::optional<Shape> physical_shape) {
  auto ret = MakeShapeWithLayoutInternal(
      element_type, dimensions, minor_to_major,
      /*tiles=*/{}, tail_padding_alignment_in_elements, index_primitive_type,
      pointer_primitive_type, element_size_in_bits, memory_space,
      /*split_configs=*/{}, std::move(physical_shape));
  TF_CHECK_OK(ret.status());
  return *ret;
}

/* static */ Shape ShapeUtil::MoveDimToMajor(const Shape& shape, int64_t dim) {
  if (shape.IsTuple()) {
    std::vector<Shape> result_shapes;
    result_shapes.reserve(shape.tuple_shapes().size());
    for (const Shape& s : shape.tuple_shapes()) {
      result_shapes.push_back(MoveDimToMajor(s, dim));
    }
    return ShapeUtil::MakeTupleShape(result_shapes);
  }

  Shape ret = shape;
  if (!ret.has_layout()) {
    LayoutUtil::SetToDefaultLayout(&ret);
  }
  *ret.mutable_layout() = LayoutUtil::MoveDimToMajor(ret.layout(), dim);
  DimensionVector minor_to_major;
  for (int64_t d : LayoutUtil::MinorToMajor(ret)) {
    if (d != dim) {
      minor_to_major.push_back(d);
    }
  }
  minor_to_major.push_back(dim);
  *ret.mutable_layout() = LayoutUtil::MakeLayout(minor_to_major);
  return ret;
}

/* static */ Shape ShapeUtil::MakeShapeWithDescendingLayout(
    PrimitiveType element_type, absl::Span<const int64_t> dimensions) {
  std::vector<int64_t> layout(dimensions.size());
  std::iota(layout.rbegin(), layout.rend(), static_cast<int64_t>(0));
  return MakeShapeWithDenseLayout(element_type, dimensions, layout);
}

/* static */ Shape
ShapeUtil::MakeShapeWithDescendingLayoutAndSamePhysicalLayout(
    const Shape& shape) {
  std::vector<int64_t> dims(shape.dimensions().size());
  for (int i = 0; i < shape.dimensions().size(); ++i) {
    int dim = i;
    if (shape.has_layout()) {
      dim = LayoutUtil::Major(shape.layout(), dim);
    }
    dims[i] = shape.dimensions(dim);
  }
  Shape new_shape = MakeShapeWithDescendingLayout(shape.element_type(), dims);
  // Since the physical layout is kept the same, the tiles and element size are
  // the same also.
  if (shape.has_layout()) {
    new_shape.mutable_layout()->mutable_tiles()->assign(
        shape.layout().tiles().begin(), shape.layout().tiles().end());
    new_shape.mutable_layout()->set_element_size_in_bits(
        shape.layout().element_size_in_bits());
    new_shape.mutable_layout()->set_tail_padding_alignment_in_elements(
        shape.layout().tail_padding_alignment_in_elements());
  }
  for (int i = 0; i < shape.dimensions().size(); ++i) {
    int dim = i;
    if (shape.has_layout()) {
      dim = LayoutUtil::Major(shape.layout(), dim);
    }
    new_shape.set_dynamic_dimension(i, shape.is_dynamic_dimension(dim));
  }
  new_shape.mutable_layout()->set_memory_space(shape.layout().memory_space());
  return new_shape;
}

/* static */ absl::Status ShapeUtil::PopulateShape(
    PrimitiveType element_type, absl::Span<const int64_t> dimensions,
    Shape* shape) {
  shape->Clear();
  shape->set_element_type(element_type);
  if (shape->IsArray()) {
    for (int64_t dimension : dimensions) {
      shape->add_dimensions(dimension);
    }
    LayoutUtil::SetToDefaultLayout(shape);
  } else {
    CHECK(dimensions.empty()) << "Non-array shape " << shape->ToString()
                              << " cannot have dimensions.";
  }
  return ValidateShape(*shape);
}

/* static */ Shape ShapeUtil::MakeStaticShape(const Shape& original) {
  Shape result = original;
  result.clear_dynamic_dimensions();
  if (result.has_layout()) {
    result.mutable_layout()->set_dynamic_shape_metadata_prefix_bytes(0);
  }
  return result;
}

/* static */ Shape ShapeUtil::MakeTupleShape(absl::Span<const Shape> shapes) {
  return MakeTupleShapeImpl(shapes);
}

/* static */ Shape ShapeUtil::MakeTupleShapeWithPtrs(
    absl::Span<const Shape* const> shapes) {
  return MakeTupleShapeImpl(shapes);
}

/* static */ Shape ShapeUtil::MakeMaybeTupleShape(
    absl::Span<const Shape> shapes) {
  if (shapes.size() == 1) {
    return shapes[0];
  }
  return MakeTupleShape(shapes);
}

/* static */ Shape ShapeUtil::MakeOpaqueShape() { return Shape(OPAQUE_TYPE); }

/* static */ Shape ShapeUtil::MakeTokenShape() { return Shape(TOKEN); }

/* static */ void ShapeUtil::AppendShapeToTuple(const Shape& shape,
                                                Shape* tuple_shape) {
  TF_DCHECK_OK(ValidateShapeWithOptionalLayout(shape));
  *tuple_shape->add_tuple_shapes() = shape;
}

/* static */ void ShapeUtil::UpdateTupleShape(const Shape& shape, int64_t index,
                                              Shape* tuple_shape) {
  CHECK_LT(index, tuple_shape->tuple_shapes().size());
  *tuple_shape->mutable_tuple_shapes(index) = shape;
}

/* static */ void ShapeUtil::UpdateDynamicDimension(Shape* shape,
                                                    ShapeIndexView index,
                                                    int64_t dim,
                                                    bool is_dynamic) {
  if (index.empty()) {
    CHECK(!shape->IsTuple());
    shape->set_dynamic_dimension(dim, is_dynamic);
    return;
  }

  UpdateDynamicDimension(shape->mutable_tuple_shapes(index.front()),
                         index.subspan(1), dim, is_dynamic);
}

/* static */ void ShapeUtil::AppendMajorDimension(int bound, Shape* shape) {
  CHECK(LayoutUtil::IsDenseArray(*shape));
  if (shape->has_layout()) {
    shape->mutable_layout()->add_minor_to_major(shape->dimensions().size());
  }
  shape->add_dimensions(bound);
  TF_DCHECK_OK(ValidateShape(*shape));
}

// Prepend new major-most dimension sized `bound` to the shape.
Shape ShapeUtil::PrependMajorDimension(int64_t bound, Shape shape) {
  Shape new_shape(shape.element_type(), {}, {});
  new_shape.add_dimensions(bound);
  for (const int64_t dim : shape.dimensions()) {
    new_shape.add_dimensions(dim);
  }
  if (shape.has_layout()) {
    for (const int64_t dim : shape.layout().minor_to_major()) {
      new_shape.mutable_layout()->add_minor_to_major(dim + 1);
    }
    new_shape.mutable_layout()->add_minor_to_major(0);
  }
  return new_shape;
}

/* static */ void ShapeUtil::AppendMinorDimension(int bound, Shape* shape) {
  CHECK(LayoutUtil::IsDenseArray(*shape));
  shape->add_dimensions(bound);
  if (shape->has_layout()) {
    // Append an empty field to the layout.
    shape->mutable_layout()->add_minor_to_major(0);
    // Shift by one position all values in the layout in the major direction.
    for (int dim_idx = shape->layout().minor_to_major().size() - 2;
         dim_idx >= 0; --dim_idx) {
      int layout_idx = shape->layout().minor_to_major(dim_idx);
      shape->mutable_layout()->set_minor_to_major(dim_idx + 1, layout_idx);
    }
    // Insert the newly added dimension at the minor-most position.
    shape->mutable_layout()->set_minor_to_major(0,
                                                shape->dimensions().size() - 1);
  }
  TF_DCHECK_OK(ValidateShape(*shape));
}

/* static */ void ShapeUtil::CopyDynamicDimensions(Shape* to,
                                                   const Shape& from) {
  CHECK_EQ(to->dimensions().size(), from.dimensions().size());
  for (int64_t i = 0; i < from.dimensions().size(); ++i) {
    to->set_dynamic_dimension(i, from.is_dynamic_dimension(i));
  }
  TF_DCHECK_OK(ValidateShape(*to));
}

/* static */ bool ShapeUtil::IsEffectivelyMostMajorDimension(
    const Shape& shape, int64_t dimension) {
  // Check if the dimension is most major as returned by LayoutUtil::Major(0).
  // If not, and the most major dimension's size is 1, then we can repeat the
  // same check for next most major dimension as returned by
  // LayoutUtil::Major(1) and so on.
  for (int64_t i = 0; i < shape.dimensions().size(); ++i) {
    int64_t major_dimension = LayoutUtil::Major(shape.layout(), i);
    if (major_dimension == dimension) {
      return true;
    }
    if (shape.dimensions(major_dimension) != 1) {
      return false;
    }
  }
  return false;
}

/* static */ bool ShapeUtil::ElementIsIntegral(const Shape& shape) {
  return primitive_util::IsIntegralType(shape.element_type());
}

/* static */ bool ShapeUtil::ElementIsIntegralWithBits(const Shape& shape,
                                                       int32_t bits) {
  return ElementIsIntegral(shape) && ElementHasBitWidth(shape, bits);
}

/* static */ bool ShapeUtil::ElementHasBitWidth(const Shape& shape, int bits) {
  if (!shape.IsArray()) {
    return false;
  }
  return primitive_util::BitWidth(shape.element_type()) == bits;
}

/* static */ bool ShapeUtil::ElementIsSigned(const Shape& shape) {
  return primitive_util::IsSignedIntegralType(shape.element_type()) ||
         primitive_util::IsFloatingPointType(shape.element_type());
}

/* static */ bool ShapeUtil::ElementIsComplex(const Shape& shape) {
  return primitive_util::IsComplexType(shape.element_type());
}

/* static */ bool ShapeUtil::ElementIsFloating(const Shape& shape) {
  return primitive_util::IsFloatingPointType(shape.element_type());
}

/* static */ bool ShapeUtil::IsNestedTuple(const Shape& shape) {
  return shape.IsTuple() &&
         absl::c_any_of(shape.tuple_shapes(),
                        [](const Shape& s) { return s.IsTuple(); });
}

/* static */ bool ShapeUtil::IsEmptyTuple(const Shape& shape) {
  return shape.IsTuple() && shape.tuple_shapes().empty();
}

/* static */ int64_t ShapeUtil::TupleElementCount(const Shape& shape) {
  return shape.tuple_shapes().size();
}

/* static */ const Shape& ShapeUtil::GetTupleElementShape(const Shape& shape,
                                                          int64_t index) {
  CHECK_GT(TupleElementCount(shape), index);
  TF_DCHECK_OK(ValidateShapeWithOptionalLayout(shape.tuple_shapes(index)));
  return shape.tuple_shapes(index);
}

/* static */ int64_t ShapeUtil::SubshapeCount(const Shape& shape) {
  int64_t n = 0;
  ForEachSubshape(shape, [&](const Shape& literal_subshape,
                             const ShapeIndex& index) { ++n; });
  return n;
}

/* static */ Shape ShapeUtil::SliceTuple(const Shape& tuple, int64_t start,
                                         int64_t limit) {
  TF_DCHECK_OK(ValidateShapeWithOptionalLayout(tuple));
  CHECK(tuple.IsTuple());
  CHECK_LE(start, tuple.tuple_shapes().size());
  CHECK_LE(limit, tuple.tuple_shapes().size());

  std::vector<Shape> new_elements(tuple.tuple_shapes().begin() + start,
                                  tuple.tuple_shapes().begin() + limit);
  return MakeTupleShape(new_elements);
}

// Returns the shape of a real or imaginary component.
/* static */ Shape ShapeUtil::ComplexComponentShape(
    const Shape& complex_shape) {
  CHECK(ElementIsComplex(complex_shape)) << HumanString(complex_shape);
  return ChangeElementType(complex_shape, primitive_util::ComplexComponentType(
                                              complex_shape.element_type()));
}

/* static */ int64_t ShapeUtil::ElementsInRecursive(const Shape& shape) {
  CHECK(shape.IsArray() || shape.IsTuple());
  if (shape.IsArray()) {
    return ElementsIn(shape);
  }
  int64_t count = 0;
  for (const Shape& element_shape : shape.tuple_shapes()) {
    count += ElementsInRecursive(element_shape);
  }
  return count;
}

/* static */ bool ShapeUtil::HasPrimitiveType(const Shape& shape,
                                              PrimitiveType primitive_type) {
  if (shape.element_type() == primitive_type) {
    return true;
  }
  if (shape.IsTuple()) {
    for (const Shape& element_shape : shape.tuple_shapes()) {
      if (HasPrimitiveType(element_shape, primitive_type)) {
        return true;
      }
    }
  }
  return false;
}

/* static */ bool ShapeUtil::IsZeroElementArray(const Shape& shape) {
  return shape.IsArray() && absl::c_linear_search(shape.dimensions(), 0);
}

/* static */ bool ShapeUtil::IsScalarWithElementType(
    const Shape& shape, PrimitiveType element_type) {
  return IsScalar(shape) && shape.element_type() == element_type;
}

/* static */ void ShapeUtil::PrintHumanString(xla::Printer* printer,
                                              const Shape& shape) {
  if (shape.IsTuple()) {
    PrintTupleShapes</*kPrintLayout=*/false>(printer, shape.tuple_shapes());
    return;
  }
  if (shape.IsBuffer()) {
    PrintBufferShape</*kPrintLayout=*/false>(printer, shape);
    return;
  }
  printer->Append(
      primitive_util::LowercasePrimitiveTypeName(shape.element_type()));
  if (!shape.IsArray() || shape.dimensions().empty()) {
    printer->Append("[]");
    return;
  }
  // Now we are in array shape with at least one dimension.
  printer->Append("[");
  // Prints the i-th dimension of the array shape.
  auto print_dimension = [&](int i) {
    if (shape.is_dynamic_dimension(i)) {
      if (shape.dimensions(i) != Shape::kUnboundedSize) {
        printer->Append(StrCat("<=", shape.dimensions(i)));
      } else {
        printer->Append("?");
      }
    } else {
      printer->Append(shape.dimensions(i));
    }
  };
  print_dimension(0);
  for (int i = 1, n = shape.dimensions().size(); i < n; ++i) {
    printer->Append(",");
    print_dimension(i);
  }
  printer->Append("]");
}

/* static */ void ShapeUtil::PrintHumanStringWithLayout(xla::Printer* printer,
                                                        const Shape& shape) {
  if (shape.IsTuple()) {
    PrintTupleShapes</*kPrintLayout=*/true>(printer, shape.tuple_shapes());
    return;
  }
  if (shape.IsBuffer()) {
    PrintBufferShape</*kPrintLayout=*/true>(printer, shape);
    return;
  }
  PrintHumanString(printer, shape);
  if (!shape.IsArray()) return;
  if (!shape.has_layout()) return;
  if (IsScalar(shape)) {
    std::string layout_str = LayoutUtil::HumanString(shape.layout());
    // Don't print "{}" as layout for scalars.
    if (layout_str != "{}") {
      printer->Append(layout_str);
    }
  } else {
    LayoutUtil::PrintHumanString(printer, shape.layout());
  }
}

/* static */ void ShapeUtil::PrintHumanString(
    xla::Printer* printer, const ProgramShape& program_shape) {
  printer->Append("(");
  const auto& shape_parameters = program_shape.parameters();
  if (!shape_parameters.empty()) {
    auto print_one = [&](int i) {
      printer->Append(program_shape.parameter_names(i).empty()
                          ? "(unknown)"
                          : program_shape.parameter_names(i));
      printer->Append(": ");
      PrintHumanString(printer, shape_parameters[i]);
    };
    print_one(0);
    for (int i = 1; i < shape_parameters.size(); ++i) {
      printer->Append(", ");
      print_one(i);
    }
  }
  printer->Append(") -> ");
  PrintHumanString(printer, program_shape.result());
}

/* static */ std::string ShapeUtil::HumanString(const Shape& shape) {
  StringPrinter printer;
  PrintHumanString(&printer, shape);
  return std::move(printer).ToString();
}

/* static */ std::string ShapeUtil::HumanStringWithLayout(const Shape& shape) {
  StringPrinter printer;
  PrintHumanStringWithLayout(&printer, shape);
  return std::move(printer).ToString();
}

/* static */ std::string ShapeUtil::HumanString(
    const ProgramShape& program_shape) {
  StringPrinter printer;
  PrintHumanString(&printer, program_shape);
  return std::move(printer).ToString();
}

/* static */ bool ShapeUtil::SameDimensions(const Shape& lhs,
                                            const Shape& rhs) {
  if (!SameRank(lhs, rhs)) return false;
  for (int i = 0; i < lhs.dimensions().size(); ++i) {
    if (!lhs.is_unbounded_dynamic_dimension(i) &&
        !rhs.is_unbounded_dynamic_dimension(i) &&
        lhs.dimensions(i) != rhs.dimensions(i)) {
      return false;
    }
  }

  return true;
}

/* static */ bool ShapeUtil::SameRank(const Shape& lhs, const Shape& rhs) {
  return lhs.dimensions().size() == rhs.dimensions().size();
}

/* static */ bool ShapeUtil::Compatible(const Shape& lhs, const Shape& rhs) {
  return Shape::Equal().IgnoreDynamicDimension().IgnoreLayout()(lhs, rhs);
}

/* static */ bool ShapeUtil::CompatibleIgnoringElementType(const Shape& lhs,
                                                           const Shape& rhs) {
  return Shape::Equal()
      .IgnoreDynamicDimension()
      .IgnoreElementType()
      .IgnoreLayout()(lhs, rhs);
}

/* static */ bool ShapeUtil::CompatibleKind(const Shape& lhs,
                                            const Shape& rhs) {
  return Shape::Equal()
      .IgnoreElementType()
      .IgnoreLayout()
      .IgnoreDimensions()
      .IgnoreDynamicDimension()(lhs, rhs);
}

/* static */ bool ShapeUtil::CompatibleIgnoringFpPrecision(const Shape& lhs,
                                                           const Shape& rhs) {
  return Shape::Equal()
      .IgnoreDynamicDimension()
      .IgnoreFpPrecision()
      .IgnoreLayout()(lhs, rhs);
}

/* static */ DimensionVector ShapeUtil::CreateDimensionVectorFromShape(
    const Shape& shape) {
  DimensionVector dimensions;
  if (shape.IsArray()) {
    dimensions.reserve(shape.dimensions().size());
    for (int i = 0; i < shape.dimensions().size(); ++i) {
      dimensions.push_back(shape.dimensions(i));
    }
  }
  return dimensions;
}

/* static */ int64_t ShapeUtil::GetDimension(const Shape& shape,
                                             int64_t dimension_number) {
  return shape.dimensions(GetDimensionNumber(shape, dimension_number));
}

/* static */ int64_t ShapeUtil::GetDimensionNumber(const Shape& shape,
                                                   int64_t dimension_number) {
  if (dimension_number < 0) {
    dimension_number += shape.dimensions().size();
  }
  CHECK_GE(dimension_number, 0);
  return dimension_number;
}

/* static */ int64_t ShapeUtil::ByteSizeOfPrimitiveType(
    PrimitiveType primitive_type) {
  return primitive_util::ByteWidth(primitive_type);
}

/* static */ int64_t ShapeUtil::ByteSizeOf(const Shape& shape,
                                           int64_t pointer_size) {
  TF_DCHECK_OK(ValidateShapeWithOptionalLayout(shape));
  if (shape.element_type() == TUPLE) {
    return ByteSizeOfTupleIndexTable(shape, pointer_size);
  } else if (shape.IsArray()) {
    return ByteSizeOfElements(shape);
  } else if (shape.element_type() == TOKEN) {
    return 0;
  } else if (shape.element_type() == OPAQUE_TYPE) {
    CHECK_GT(pointer_size, 0);
    return pointer_size;
  }
  LOG(FATAL) << PrimitiveType_Name(shape.element_type())
             << " primitive type has no definitive size";
}

/* static */ int64_t ShapeUtil::ByteSizeOfTupleIndexTable(
    const Shape& shape, int64_t pointer_size) {
  TF_DCHECK_OK(ValidateShape(shape));
  CHECK_EQ(TUPLE, shape.element_type());
  CHECK_GT(pointer_size, 0);
  return pointer_size * shape.tuple_shapes().size();
}

/* static */ int64_t ShapeUtil::ByteSizeOfElements(const Shape& shape) {
  TF_DCHECK_OK(ValidateShapeWithOptionalLayout(shape));
  int64_t allocated_element_count;

  CHECK(LayoutUtil::IsDenseArray(shape)) << shape.ToString();
  allocated_element_count = ElementsIn(shape);

  if (shape.has_layout() && shape.layout().element_size_in_bits() != 0) {
    const int64_t num_bits =
        allocated_element_count * shape.layout().element_size_in_bits();
    return CeilOfRatio<int64_t>(num_bits, CHAR_BIT);
  }
  return allocated_element_count *
         ByteSizeOfPrimitiveType(shape.element_type());
}

/* static */ absl::StatusOr<int64_t> ShapeUtil::SerializedSize(
    const Shape& shape) {
  return SerializedSizeWithProto(shape, shape.ToProto());
}

/* static */ absl::StatusOr<int64_t> ShapeUtil::SerializedSizeWithProto(
    const Shape& shape, const ShapeProto& proto) {
  // The size computed here must be kept in sync with the serialized format as
  // described in the comments for LiteralBase::SerializeWithShapeProto in
  // literal.h.
  TF_RETURN_IF_ERROR(ValidateShapeWithOptionalLayout(shape));
  int64_t size = sizeof(int64_t) + proto.ByteSizeLong();

  TF_RETURN_IF_ERROR(ShapeUtil::ForEachSubshapeWithStatus(
      shape,
      [&](const Shape& subshape, const ShapeIndex& index) -> absl::Status {
        if (subshape.IsTuple()) {
          return absl::OkStatus();
        }
        if (!subshape.IsArray()) {
          return ShapeError(shape, "Shape cannot be serialiized.");
        }
        if (subshape.is_dynamic()) {
          size += sizeof(DynamicSizeType) * subshape.dimensions().size();
        }
        if (subshape.element_type() == PRED) {
          // PRED is packed 8 elements per byte.
          size += CeilOfRatio<int64_t>(ElementsIn(subshape), 8);
        } else if (primitive_util::IsSubByteNonPredType(
                       subshape.element_type())) {
          // 4-bit types are packed 2 elements per byte.
          size += CeilOfRatio<int64_t>(
              ElementsIn(subshape),
              8 / primitive_util::BitWidth(subshape.element_type()));
        } else {
          size += ByteSizeOfElements(subshape);
        }
        return absl::OkStatus();
      }));

  return size;
}

namespace {

// Validates the shape size is sane. This makes sure it's safe to do
// calculations in int64_t without overflowing.
absl::Status ValidateShapeSize(const Shape& shape) {
  if (!shape.IsArray()) {
    return absl::OkStatus();
  }

  auto [extent_product, extent_overflow] =
      ShapeUtil::ExtentProduct</*kBoundedDynamicOk=*/true>(shape);
  auto [dense_shape_size, byte_width_overflow] = OverflowSafeMultiply(
      extent_product, ShapeUtil::ByteSizeOfPrimitiveType(shape.element_type()));

  if (extent_overflow || byte_width_overflow) {
    return InvalidArgument("Shape %s size may overflow int64_t.",
                           ShapeUtil::HumanString(shape));
  }
  return absl::OkStatus();
}

absl::Status ValidateDimensions(const Shape& shape) {
  bool any_overflows = false;
  int64_t product = 1;
  for (int64_t i = 0; i < shape.dimensions().size(); ++i) {
    int64_t dimension = shape.dimensions(i);
    if (dimension == Shape::kUnboundedSize) {
      continue;
    }
    if (dimension < 0) {
      return ShapeError(
          shape,
          absl::StrFormat("Negative dimension at index %d: %d.", i, dimension));
    }
    bool overflow;
    std::tie(product, overflow) = OverflowSafeMultiply(product, dimension);
    any_overflows |= overflow;
  }
  if (any_overflows) {
    return ShapeError(shape, "Dimensions overflow.");
  }
  return absl::OkStatus();
}
}  // namespace

// Validates all of the non-layout properties of the shape -- this is a helper
// used by both the layout-optional and layout-required public method.
absl::Status ValidateNonLayoutProperties(const Shape& shape) {
  // Make sure the element type is valid.
  if (shape.element_type() == PRIMITIVE_TYPE_INVALID ||
      !PrimitiveType_IsValid(shape.element_type())) {
    return ShapeError(shape, "Invalid element type.");
  }

  // Validate tuple shapes.
  if (shape.element_type() == TUPLE) {
    if (!shape.if_tuple_state()) {
      return ShapeError(shape, "This type must have a tuple state.");
    }
    for (auto& element_shape : shape.tuple_shapes()) {
      TF_RETURN_IF_ERROR(ValidateNonLayoutProperties(element_shape));
    }
    return absl::OkStatus();
  }

  if (shape.element_type() == BUFFER) {
    if (!shape.if_buffer_state()) {
      return ShapeError(shape, "This type must have a buffer state.");
    }
    return ValidateNonLayoutProperties(shape.buffer_shape());
  }

  // Validate token shapes.
  if (shape.element_type() == TOKEN) {
    if (!shape.if_token_state()) {
      return ShapeError(shape, "This type must have a token state.");
    }
    return absl::OkStatus();
  }

  // Validate opaque shapes.
  if (shape.element_type() == OPAQUE_TYPE) {
    if (!shape.if_opaque_state()) {
      return ShapeError(shape, "This type must have an opaque state.");
    }
    return absl::OkStatus();
  }

  // Validate array shapes.
  if (primitive_util::IsArrayType(shape.element_type())) {
    if (!shape.if_array_state()) {
      return ShapeError(shape, "This type must have an array state.");
    }
    TF_RETURN_IF_ERROR(ValidateDimensions(shape));
    TF_RETURN_IF_ERROR(ValidateShapeSize(shape));
    return absl::OkStatus();
  }

  return ShapeError(shape, "Unsupported element type.");
}

/* static */ absl::Status ShapeUtil::ValidateShapeWithOptionalLayout(
    const Shape& shape) {
  TF_RETURN_IF_ERROR(ValidateNonLayoutProperties(shape));

  return LayoutUtil::ValidateLayoutInShape(shape,
                                           /*allow_missing_layouts=*/true);
}

/* static */ absl::Status ShapeUtil::ValidateShape(const Shape& shape) {
  TF_RETURN_IF_ERROR(ValidateNonLayoutProperties(shape));

  return LayoutUtil::ValidateLayoutInShape(shape);
}

/* static */ Shape ShapeUtil::ChangeElementType(const Shape& original,
                                                PrimitiveType type) {
  if (original.IsTuple()) {
    std::vector<Shape> new_operands;
    new_operands.reserve(original.tuple_shapes().size());
    for (const Shape& operand : original.tuple_shapes()) {
      new_operands.push_back(ChangeElementType(operand, type));
    }
    return MakeTupleShape(new_operands);
  } else {
    Shape new_shape = original;
    new_shape.set_element_type(type);
    if (new_shape.has_layout() && !primitive_util::IsSubByteNonPredType(type)) {
      new_shape.mutable_layout()->set_element_size_in_bits(0);
    }
    return new_shape;
  }
}

/* static */ bool ShapeUtil::IndexIsValid(const Shape& shape,
                                          ShapeIndexView index) {
  const Shape* subshape = &shape;
  for (auto i : index) {
    if (!subshape->IsTuple() || i >= subshape->tuple_shapes().size() || i < 0) {
      return false;
    }
    subshape = &subshape->tuple_shapes(i);
  }
  return true;
}

/* static */ const Shape& ShapeUtil::GetSubshape(const Shape& shape,
                                                 ShapeIndexView index) {
  const Shape* return_shape = &shape;
  for (auto i : index) {
    CHECK(return_shape->IsTuple())
        << "Invalid index " << ShapeIndex(index) << " for shape " << shape;
    return_shape = &return_shape->tuple_shapes(i);
  }
  return *return_shape;
}

/* static */ absl::StatusOr<const Shape*> ShapeUtil::TryGetSubshape(
    const Shape& shape, ShapeIndexView index) {
  const Shape* return_shape = &shape;
  for (auto i : index) {
    if (!return_shape->IsTuple() || i < 0 ||
        i >= return_shape->tuple_shapes().size()) {
      return InvalidArgument(
          "Shape index %s not a valid subshape index for tuple with shape %s",
          ShapeIndex(index).ToString(), shape.ToString());
    }
    return_shape = &return_shape->tuple_shapes(i);
  }
  return return_shape;
}

/* static */ Shape* ShapeUtil::GetMutableSubshape(Shape* shape,
                                                  ShapeIndexView index) {
  Shape* return_shape = shape;
  for (auto i : index) {
    CHECK(return_shape->IsTuple());
    return_shape = return_shape->mutable_tuple_shapes(i);
  }
  return return_shape;
}

/* static */
bool ShapeUtil::IsLeafIndex(const Shape& shape, const ShapeIndex& index) {
  return !GetSubshape(shape, index).IsTuple();
}

/* static */ int64_t ShapeUtil::GetLeafCountTuple(const Shape& shape) {
  DCHECK(shape.IsTuple());
  int64_t count = 0;
  for (const Shape& subshape : shape.tuple_shapes()) {
    if (subshape.IsTuple()) {
      count += GetLeafCount(subshape);
    } else {
      ++count;
    }
  }
  return count;
}

/* static */ int64_t ShapeUtil::GetLeafCount(const Shape& shape) {
  if (!shape.IsTuple()) {
    return 1;
  }
  return GetLeafCountTuple(shape);
}

/* static */ std::vector<ShapeUtil::IndexedShape> ShapeUtil::GetLeafShapes(
    const Shape& shape) {
  std::vector<IndexedShape> leaves;
  ForEachSubshape(shape, [&](const Shape& sub_shape, const ShapeIndex& index) {
    if (IsLeafIndex(shape, index)) {
      leaves.emplace_back(index, sub_shape);
    }
  });
  return leaves;
}

/* static */ bool ShapeUtil::HasDegenerateDimensions(const Shape& shape) {
  CHECK(shape.IsArray());
  return absl::c_linear_search(shape.dimensions(), 1);
}

/* static */ absl::StatusOr<int64_t>
ShapeUtil::PackedFactorFor1DInterleavedArray(const Shape& shape) {
  if (shape.dimensions().size() == 1 && shape.layout().tiles().size() == 3 &&
      shape.layout().tiles()[2].dimensions().size() == 2) {
    return shape.layout().tiles()[2].dimension(0);
  }
  return InvalidArgument("Shape %s is not a 1D interleaved array.",
                         ShapeUtil::HumanStringWithLayout(shape));
}

/* static */ Shape ShapeUtil::DropDegenerateDimensions(const Shape& shape) {
  return FilterDimensions(
      [&](int64_t dim) -> bool { return shape.dimensions()[dim] != 1; }, shape);
}

/* static */ Shape ShapeUtil::PermuteDimensions(
    absl::Span<const int64_t> permutation, const Shape& shape) {
  Shape new_shape = shape;
  new_shape.clear_dimensions();
  const auto permuted_dims = Permute(shape.dimensions(), permutation);
  const auto permuted_dynamic_dims =
      Permute(shape.dynamic_dimensions(), permutation);
  for (int i = 0; i < permuted_dims.size(); ++i) {
    new_shape.add_dimensions(permuted_dims[i], permuted_dynamic_dims[i]);
  }

  // If `shape` has a layout, by contract we choose a new layout such that the
  // transpose defined by this permutation is a bitcast.
  //
  // Some formalism helps to understand the correct way to do this.  We're going
  // to do algebra in the group of permutations of the dimensions of `shape`.
  //
  // Since the order of `shape`'s dimensions is not permuted relative to itself,
  // `shape`'s list of dimensions is isomorphic to the identity I.
  //
  // Let `shape`'s layout be L.  A layout is a permutation which maps a
  // minor-to-major physical dimension ordering to a shape's logical dimension
  // ordering.  Therefore the inverse of a layout maps from logical to physical
  // dims, and so the physical ordering of I is simply L'.I = L', where L' is
  // the inverse of L.
  //
  // Let the argument `permutation` be P.  This is a permutation over `shape`'s
  // dimensions, so our return value will be a shape with dims P.I = P.  Our
  // goal is to construct a layout permutation L* for this shape. The physical
  // dimension ordering of this returned shape must be the same as that of the
  // original shape, namely L'.
  //
  // Our returned shape has dims P and layout L*, so its in-memory ordering is
  // L*'.P.  Setting this equal to L' and solving for L*, we get:
  //
  //   L*'.P = L'    =>
  //   L*'   = L'P'  =>
  //   L*    = P.L
  //
  if (shape.has_layout()) {
    CHECK(LayoutUtil::IsDenseArray(shape));
    Layout* new_layout = new_shape.mutable_layout();
    new_layout->clear_minor_to_major();
    for (auto index : ComposePermutations(InversePermutation(permutation),
                                          shape.layout().minor_to_major())) {
      new_layout->add_minor_to_major(index);
    }
    // The permutation accepted by TransposeIsBitcast is the inverse of the
    // permutation here.
    CHECK(TransposeIsBitcast(shape, new_shape, permutation))
        << "shape=" << HumanStringWithLayout(shape)
        << ", new_shape=" << HumanStringWithLayout(new_shape)
        << ", permutation={" << absl::StrJoin(permutation, ",") << "}";
  }
  return new_shape;
}

/* static */ std::optional<ShapeUtil::ShapeEqualityDescriptor>
ShapeUtil::InsertedOrDeleted1SizedDimensions(const Shape& shape_pre,
                                             const Shape& shape_post) {
  CHECK(shape_pre.IsArray());
  CHECK(shape_post.IsArray());

  std::vector<int64_t> deleted_indices;
  std::vector<int64_t> inserted_indices;
  // Returns false if any input/output index between prior_unmodified_dim_pair
  // and unmodified_dim_pair have size >1. Otherwise, returns true and appends
  // the degenerate input/output dimensions in the gap to
  // deleted_indices/inserted_indices respectively.
  auto check_modified_dims =
      [&shape_pre, &shape_post, &deleted_indices, &inserted_indices](
          std::pair<int64_t, int64_t> prior_unmodified_dim_pair,
          std::pair<int64_t, int64_t> unmodified_dim_pair) {
        for (int64_t modified_input_dim = prior_unmodified_dim_pair.first + 1;
             modified_input_dim < unmodified_dim_pair.first;
             ++modified_input_dim) {
          if (shape_pre.dimensions(modified_input_dim) > 1) {
            return false;
          }
          deleted_indices.push_back(modified_input_dim);
        }
        for (int64_t modified_output_dim = prior_unmodified_dim_pair.second + 1;
             modified_output_dim < unmodified_dim_pair.second;
             ++modified_output_dim) {
          if (shape_post.dimensions(modified_output_dim) > 1) {
            return false;
          }
          inserted_indices.push_back(modified_output_dim);
        }
        return true;
      };

  std::vector<std::pair<int64_t, int64_t>> unmodified_dims =
      DimensionsUnmodifiedByReshape(shape_pre, shape_post);
  // Returns nil if the reshape modifies any non-degenerate input/output
  // dimension. DimensionsUnmodifiedByReshape gives us all unmodified
  // dimensions, so we only need to check whether dimensions in the gaps (thus
  // modified) have size >1.
  for (size_t i = 0; i <= unmodified_dims.size(); ++i) {
    // Check (modified) dimensions between unmodified_dims[i-1] and
    // unmodified_dims[i].
    auto prior_unmodified_dim_pair =
        i > 0 ? unmodified_dims[i - 1] : std::pair<int64_t, int64_t>(-1, -1);
    auto unmodified_dim_pair =
        i < unmodified_dims.size()
            ? unmodified_dims[i]
            : std::make_pair(
                  static_cast<int64_t>(shape_pre.dimensions().size()),
                  static_cast<int64_t>(shape_post.dimensions().size()));
    if (!check_modified_dims(prior_unmodified_dim_pair, unmodified_dim_pair)) {
      return std::nullopt;
    }
  }

  return ShapeEqualityDescriptor{deleted_indices, inserted_indices};
}

/* static */ std::vector<std::pair<int64_t, int64_t>>
ShapeUtil::DimensionsUnmodifiedByReshape(const Shape& input_shape,
                                         const Shape& output_shape) {
  CHECK(input_shape.IsArray());
  CHECK(output_shape.IsArray());

  // Unmodified dimensions are merely common factors of rank 1.
  auto common_factors =
      CommonFactors(input_shape.dimensions(), output_shape.dimensions());
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
  return std::vector<std::pair<int64_t, int64_t>>(common_factors.begin(),
                                                  common_factors.end());
}

/* static */ std::optional<std::vector<int64_t>>
ShapeUtil::ReshapeLeavesDimensionsUnmodified(
    const Shape& from_shape, const Shape& to_shape,
    absl::Span<const int64_t> input_dim_indices) {
  if (!std::is_sorted(input_dim_indices.begin(), input_dim_indices.end())) {
    return std::nullopt;
  }

  std::vector<int64_t> output_dim_indices;
  std::vector<std::pair<int64_t, int64_t>> unmodified_dims =
      ShapeUtil::DimensionsUnmodifiedByReshape(from_shape, to_shape);
  size_t i = 0;  // index to unmodified_dims
  for (int64_t input_dim_index : input_dim_indices) {
    // Search unmodified_dims for input_dim_index. We can search from the last
    // matching position because input_dim_indices is guaranteed to be sorted.
    while (i < unmodified_dims.size() &&
           unmodified_dims[i].first < input_dim_index) {
      ++i;
    }
    if (i >= unmodified_dims.size() ||
        unmodified_dims[i].first != input_dim_index) {
      return std::nullopt;
    }
    output_dim_indices.push_back(unmodified_dims[i].second);
  }
  return output_dim_indices;
}

/* static */ bool ShapeUtil::TransposeIsBitcast(
    const Shape& input_shape, const Shape& output_shape,
    absl::Span<const int64_t> dimension_mapping, bool ignore_element_type) {
  CHECK(LayoutUtil::IsDenseArray(input_shape)) << input_shape.ToString(true);
  CHECK(LayoutUtil::IsDenseArray(output_shape)) << output_shape.ToString(true);
  CHECK(input_shape.has_layout()) << input_shape.ToString(true);
  CHECK(output_shape.has_layout()) << output_shape.ToString(true);

  if (!ignore_element_type && !SameElementType(input_shape, output_shape)) {
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
  return absl::c_equal(
      ComposePermutations(dimension_mapping,
                          output_shape.layout().minor_to_major()),
      input_shape.layout().minor_to_major());
}

/* static */ bool ShapeUtil::IsReshapeOrTransposeBitcast(
    const Shape& a, const Shape& b, bool ignore_element_type) {
  if (!ignore_element_type && !SameElementType(a, b)) {
    return false;
  }
  if (ShapeUtil::EqualIgnoringElementType(a, b)) {
    return true;
  }
  if (ReshapeIsBitcast(a, b, /*ignore_element_type=*/true)) {
    return true;
  }
  if (std::optional<std::vector<int64_t>> dimensions =
          ShapeUtil::DeduceTransposeDimensionsForBitcast(a, b)) {
    return TransposeIsBitcast(b, a, *dimensions,
                              /*ignore_element_type=*/true);
  }
  return false;
}

/* static */ bool ShapeUtil::ReshapeIsBitcast(const Shape& input_shape,
                                              const Shape& output_shape,
                                              bool ignore_element_type) {
  CHECK(LayoutUtil::IsDenseArray(input_shape)) << input_shape.ToString(true);
  CHECK(LayoutUtil::IsDenseArray(output_shape)) << output_shape.ToString(true);
  CHECK(input_shape.has_layout()) << input_shape.ToString(true);
  CHECK(output_shape.has_layout()) << output_shape.ToString(true);

  if (!ignore_element_type && !SameElementType(input_shape, output_shape)) {
    return false;
  }

  if (ElementsIn(input_shape) != ElementsIn(output_shape)) {
    VLOG(3) << "input_shape=" << input_shape.ToString()
            << ", output_shape=" << output_shape.ToString();
    return false;
  }
  if (ElementsIn(input_shape) == 0) {
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
    Shape input_shape_dim0_major = MakeShapeWithDescendingLayout(
        input_shape.element_type(), input_shape.dimensions());
    Shape output_shape_dim0_major = MakeShapeWithDescendingLayout(
        output_shape.element_type(), output_shape.dimensions());

    for (int64_t input_dim = 0; input_dim < input_shape.dimensions().size();
         ++input_dim) {
      if (input_shape.dimensions(input_dim) <= 1) {
        continue;
      }

      std::vector<int64_t> input_unit_index(input_shape.dimensions().size(), 0);
      input_unit_index[input_dim] = 1;
      int64_t logical_linear_index =
          IndexUtil::MultidimensionalIndexToLinearIndex(input_shape_dim0_major,
                                                        input_unit_index);
      // output_index has the same logical linear index as input_unit_index.
      auto output_index = IndexUtil::LinearIndexToMultidimensionalIndex(
          output_shape_dim0_major, logical_linear_index);
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

static absl::Span<const int64_t> LayoutPerm(const Shape& s) {
  return s.layout().minor_to_major();
}

/* static */ std::optional<std::vector<int64_t>>
ShapeUtil::DeduceTransposeDimensionsForBitcast(const Shape& input_shape,
                                               const Shape& output_shape) {
  if (output_shape.dimensions().size() != input_shape.dimensions().size()) {
    return std::nullopt;
  }

  std::vector<int64_t> transpose_perm = ComposePermutations(
      LayoutPerm(input_shape), InversePermutation(LayoutPerm(output_shape)));

  std::vector<int64_t> new_dims =
      ComposePermutations(input_shape.dimensions(), transpose_perm);
  if (!absl::c_equal(output_shape.dimensions(), new_dims)) {
    return std::nullopt;
  }
  CHECK(TransposeIsBitcast(
      input_shape, ChangeElementType(output_shape, input_shape.element_type()),
      transpose_perm));
  return transpose_perm;
}

namespace {

static absl::InlinedVector<int64_t, 8> ReverseIota(int64_t n) {
  absl::InlinedVector<int64_t, 8> ret(n);
  absl::c_generate(ret, [n = ret.size()]() mutable { return --n; });
  return ret;
}

}  // namespace

bool ShapeUtil::BitcastDecompositionTrt::IsTranspose1Identity() const {
  return absl::c_is_sorted(transpose1_dims);
}

bool ShapeUtil::BitcastDecompositionTrt::IsTranspose2Identity() const {
  return absl::c_is_sorted(transpose2_dims);
}

/* static */ ShapeUtil::BitcastDecompositionTrt
ShapeUtil::DecomposeBitcastToTrt(const Shape& input_shape,
                                 const Shape& output_shape) {
  CHECK(input_shape.has_layout()) << input_shape.ToString();
  CHECK(output_shape.has_layout()) << output_shape.ToString();

  BitcastDecompositionTrt decomposition;
  decomposition.transpose1_shape =
      MakeShapeWithDescendingLayoutAndSamePhysicalLayout(input_shape);
  decomposition.reshape_shape =
      MakeShapeWithDescendingLayoutAndSamePhysicalLayout(output_shape);
  CHECK(ReshapeIsBitcast(decomposition.transpose1_shape,
                         decomposition.reshape_shape,
                         /*ignore_element_type=*/true));

  // Let a * b denote Permute(a, perm=b).
  //
  // (input_dims * transpose1_dims) * R = input_dims * input_layout
  // transpose1_dims * R = input_layout  | * R, knowing R * R = I
  // transpose1_dims = input_layout * R
  decomposition.transpose1_dims = ComposePermutations(
      LayoutPerm(input_shape), ReverseIota(input_shape.dimensions().size()));
  CHECK(TransposeIsBitcast(input_shape, decomposition.transpose1_shape,
                           decomposition.transpose1_dims,
                           /*ignore_element_type=*/false));

  // (reshape_dims * transpose2_dims) * output_layout = reshape_dims * R
  // transpose2_dims * output_layout = R  | * inv(output_layout)
  // transpose2_dims = R * inv(output_layout)
  decomposition.transpose2_dims =
      ComposePermutations(ReverseIota(output_shape.dimensions().size()),
                          InversePermutation(LayoutPerm(output_shape)));
  CHECK(TransposeIsBitcast(decomposition.reshape_shape, output_shape,
                           decomposition.transpose2_dims,
                           /*ignore_element_type=*/false));

  return decomposition;
}

/* static */ ShapeUtil::BitcastDecomposition ShapeUtil::DecomposeBitcast(
    const Shape& input_shape, const Shape& output_shape) {
  CHECK(input_shape.has_layout()) << input_shape.ToString();
  CHECK(output_shape.has_layout()) << output_shape.ToString();

  if (ShapeUtil::ReshapeIsBitcast(input_shape, output_shape,
                                  /*ignore_element_type=*/true)) {
    return BitcastDecompositionReshape{};
  }

  if (std::optional<std::vector<int64_t>> transpose_dims =
          DeduceTransposeDimensionsForBitcast(input_shape, output_shape)) {
    return BitcastDecompositionTranspose{transpose_dims.value()};
  }

  return DecomposeBitcastToTrt(input_shape, output_shape);
}

/* static */ std::optional<Shape> ShapeUtil::AlignLayouts(
    const Shape& input_shape, const Shape& output_shape) {
  CHECK(input_shape.IsArray());
  CHECK(output_shape.IsArray());
  // Removing trivial dimensions from the shape simplifies the alignment
  // algorithm since ones can go in any position.
  if (HasDegenerateDimensions(input_shape) ||
      HasDegenerateDimensions(output_shape)) {
    auto simple_output_shape =
        AlignLayouts(DropDegenerateDimensions(input_shape),
                     DropDegenerateDimensions(output_shape));
    if (!simple_output_shape) {
      return std::nullopt;
    }

    std::vector<int64_t> layout =
        SpanToVector(simple_output_shape->layout().minor_to_major());
    // For each one sized dimension in the output, increment the dimension
    // numbers in layout that are more minor than the one.
    absl::InlinedVector<int64_t, 8> dim_map;
    dim_map.reserve(simple_output_shape->dimensions().size());
    for (int64_t i = 0; i < output_shape.dimensions().size(); ++i) {
      if (output_shape.dimensions(i) != 1) {
        dim_map.push_back(i);
      }
    }
    for (int64_t& d : layout) {
      d = dim_map[d];
    }

    // Add the ones in descending order to the layout. Descending layouts tend
    // to reduce the number of copies inserted in layout assignment.
    for (int64_t i = output_shape.dimensions().size() - 1; i >= 0; --i) {
      if (output_shape.dimensions(i) == 1) {
        layout.push_back(i);
      }
    }
    Shape output_shape_with_layout = output_shape;
    *output_shape_with_layout.mutable_layout() = Layout{layout};
    return output_shape_with_layout;
  }

  auto common_factors =
      CommonFactors(input_shape.dimensions(), output_shape.dimensions());
  const int64_t input_rank = input_shape.dimensions().size();
  DimensionVector input_to_factor(input_rank);
  for (int64_t pos = 0; pos < common_factors.size() - 1; ++pos) {
    const int64_t input_start = common_factors[pos].first;
    const int64_t input_end = common_factors[pos + 1].first;
    int64_t input_physical =
        PositionInContainer(input_shape.layout().minor_to_major(), input_start);
    input_to_factor[input_start] = pos;
    for (int64_t i = input_start + 1; i < input_end; ++i) {
      --input_physical;
      if (input_physical < 0 ||
          input_shape.layout().minor_to_major(input_physical) != i) {
        return std::nullopt;
      }
      input_to_factor[i] = pos;
    }
  }

  int64_t output_rank = output_shape.dimensions().size();
  DimensionVector output_layout;
  output_layout.reserve(output_rank);
  int64_t input_minor = 0;
  while (output_layout.size() < output_rank) {
    const int64_t input_dim = input_shape.layout().minor_to_major(input_minor);
    const int64_t common_factor = input_to_factor[input_dim];
    const auto start_factor = common_factors[common_factor];
    const auto end_factor = common_factors[common_factor + 1];
    for (int64_t dim = end_factor.second - 1; dim >= start_factor.second;
         --dim) {
      output_layout.push_back(dim);
    }
    input_minor += end_factor.first - start_factor.first;
  }

  Shape output_shape_with_layout = MakeShapeWithDenseLayout(
      output_shape.element_type(), output_shape.dimensions(), output_layout);
  CHECK(ReshapeIsBitcast(input_shape, output_shape_with_layout))
      << "reshape is not a bitcast for input_shape: "
      << ShapeUtil::HumanStringWithLayout(input_shape)
      << " and output_shape_with_layout: "
      << ShapeUtil::HumanStringWithLayout(output_shape_with_layout);
  return output_shape_with_layout;
}

/* static */ Shape ShapeUtil::ReorderLogicalDimensions(
    const Shape& shape, absl::Span<const int64_t> permutation) {
  CHECK(shape.IsArray());
  const std::vector<bool> dynamic_dimensions =
      Permute(shape.dynamic_dimensions(), permutation);

  Shape new_shape(shape.element_type(),
                  Permute(shape.dimensions(), permutation),
                  absl::InlinedVector<bool, 8>(dynamic_dimensions.begin(),
                                               dynamic_dimensions.end()));
  if (shape.has_layout()) {
    *new_shape.mutable_layout() = shape.layout();
    for (int64_t& dim : *new_shape.mutable_layout()->mutable_minor_to_major()) {
      dim = permutation[dim];
    }
  }
  return new_shape;
}

/* static */ Shape ShapeUtil::DeleteDimension(int64_t dim_to_delete,
                                              Shape shape) {
  CHECK(shape.IsArray());
  shape.DeleteDimension(dim_to_delete);
  return shape;
}

/* static */ bool ShapeUtil::DynamicArrayShapeIsCompatible(
    const xla::Shape& dynamic_shape, const xla::Shape& bounded_shape) {
  if (dynamic_shape.dimensions().size() != bounded_shape.dimensions().size()) {
    return false;
  }
  for (int64_t i = 0; i < dynamic_shape.dimensions().size(); ++i) {
    if (dynamic_shape.dimensions(i) > bounded_shape.dimensions(i)) {
      return false;
    }
  }
  return true;
}

/* static */ bool ShapeUtil::DynamicShapeIsCompatible(
    const xla::Shape& dynamic_shape, const xla::Shape& bounded_shape) {
  bool compatible = true;
  xla::ShapeUtil::ForEachSubshape(
      dynamic_shape, [&](const Shape& sub_shape, const ShapeIndex& index) {
        if (compatible) {
          auto subshape_result = TryGetSubshape(bounded_shape, index);
          if (subshape_result.ok()) {
            const Shape* bounded_sub_shape = std::move(subshape_result).value();
            if (sub_shape.IsTuple()) {
              if (!bounded_sub_shape->IsTuple()) {
                compatible = false;
              }
            } else {
              if (bounded_sub_shape->IsTuple()) {
                compatible = false;
              } else if (!sub_shape.is_static() &&
                         !DynamicArrayShapeIsCompatible(sub_shape,
                                                        *bounded_sub_shape)) {
                compatible = false;
              }
            }
          } else {
            compatible = false;
          }
        }
      });
  return compatible;
}

/* static */ absl::Status ShapeUtil::ForEachIndexWithStatus(
    const Shape& shape, absl::Span<const int64_t> base,
    absl::Span<const int64_t> count, absl::Span<const int64_t> incr,
    const ForEachVisitorFunction& visitor_function) {
  return ForEachIndexInternal(shape, base, count, incr, visitor_function);
}

/* static */ void ShapeUtil::ForEachIndex(
    const Shape& shape, absl::Span<const int64_t> base,
    absl::Span<const int64_t> count, absl::Span<const int64_t> incr,
    const ForEachVisitorFunction& visitor_function) {
  ForEachIndexWithStatus(shape, base, count, incr, visitor_function)
      .IgnoreError();
}

/* static */ void ShapeUtil::ForEachIndexNoStatus(
    const Shape& shape, absl::Span<const int64_t> base,
    absl::Span<const int64_t> count, absl::Span<const int64_t> incr,
    const ForEachVisitorFunctionNoStatus& visitor_function) {
  ForEachIndexInternalNoStatus(shape, base, count, incr, visitor_function);
}

/* static */ void ShapeUtil::ForEachIndexParallel(
    const Shape& shape, absl::Span<const int64_t> base,
    absl::Span<const int64_t> count, absl::Span<const int64_t> incr,
    const ForEachParallelVisitorFunction& visitor_function) {
  // The parallel version of ForEachIndexInternal can never fail.
  TF_CHECK_OK(ForEachIndexParallelWithStatus(shape, base, count, incr,
                                             visitor_function));
}

/* static */ absl::Status ShapeUtil::ForEachIndexParallelWithStatus(
    const Shape& shape, absl::Span<const int64_t> base,
    absl::Span<const int64_t> count, absl::Span<const int64_t> incr,
    const ForEachParallelVisitorFunction& visitor_function) {
  // The parallel version of ForEachIndexInternal can never fail.
  return ForEachIndexInternalParallel(shape, base, count, incr,
                                      visitor_function);
}

/* static */ void ShapeUtil::ForEachIndexParallel(
    const Shape& shape,
    const ForEachParallelVisitorFunction& visitor_function) {
  TF_CHECK_OK(ForEachIndexParallelWithStatus(shape, visitor_function));
}

/* static */ absl::Status ShapeUtil::ForEachIndexParallelWithStatus(
    const Shape& shape,
    const ForEachParallelVisitorFunction& visitor_function) {
  std::vector<int64_t> base(shape.dimensions().size());
  std::vector<int64_t> incr(shape.dimensions().size(), 1);
  return ForEachIndexParallelWithStatus(shape, base,
                                        /*count=*/shape.dimensions(), incr,
                                        visitor_function);
}

/* static */ absl::Status ShapeUtil::ForEachIndexInternal(
    const Shape& shape, absl::Span<const int64_t> base,
    absl::Span<const int64_t> count, absl::Span<const int64_t> incr,
    const ForEachVisitorFunction& visitor_function) {
  ForEachState s(shape, base, count, incr);
  if (s.IsZeroElementArray()) {
    return absl::OkStatus();
  }
  // Allows handling R0 arrays, such that the visitor function will be called
  // once with the proper empty indexes.
  int64_t n = -1;
  int64_t rank = s.rank;
  while (n < rank) {
    TF_ASSIGN_OR_RETURN(bool should_continue, visitor_function(s.indexes_span));
    if (TF_PREDICT_FALSE(!should_continue)) {
      break;
    }
    // Increments dimensions in minor to major order.
    n = s.IncrementDim();
  }
  return absl::OkStatus();
}

/* static */ void ShapeUtil::ForEachIndexInternalNoStatus(
    const Shape& shape, absl::Span<const int64_t> base,
    absl::Span<const int64_t> count, absl::Span<const int64_t> incr,
    const ForEachVisitorFunctionNoStatus& visitor_function) {
  ForEachState s(shape, base, count, incr);
  if (s.IsZeroElementArray()) {
    return;
  }
  // Allows handling R0 arrays, such that the visitor function will be called
  // once with the proper empty indexes.
  int64_t n = -1;
  int64_t rank = s.rank;
  while (n < rank) {
    bool should_continue = visitor_function(s.indexes_span);
    if (TF_PREDICT_FALSE(!should_continue)) {
      break;
    }
    // Increments dimensions in minor to major order.
    n = s.IncrementDim();
  }
}

namespace {

struct ParallelState {
  explicit ParallelState(int64_t task_count) {
    // If this method is changed, please remember to change
    // GetForEachIndexParallelThreadCount() as well.
    static auto* const global_pool = new tsl::thread::ThreadPool(
        tsl::Env::Default(), "foreach", tsl::port::MaxParallelism());
    pool = global_pool;
  }
  ~ParallelState() = default;

  absl::Mutex mu;
  tsl::thread::ThreadPool* pool;
  absl::Status status;  // Guarded by mu
};

}  // anonymous namespace

/* static */ absl::Status ShapeUtil::ForEachIndexInternalParallel(
    const Shape& shape, absl::Span<const int64_t> base,
    absl::Span<const int64_t> count, absl::Span<const int64_t> incr,
    const ForEachParallelVisitorFunction& visitor_function) {
  // Short-circuit if there is no work to do.
  ForEachState s(shape, base, count, incr);
  if (s.IsZeroElementArray()) return absl::OkStatus();

  // Execute the visitor function inline in the caller thread.
  auto execute_inline = [&] {
    return ForEachIndexInternal(shape, base, count, incr,
                                [&](absl::Span<const int64_t> index) {
                                  return visitor_function(index, 0);
                                });
  };

  // Don't try to handle special cases with 0 counts or increments in parallel.
  auto is_zero = [](int64_t value) { return value == 0; };
  if (absl::c_any_of(count, is_zero) || absl::c_any_of(incr, is_zero)) {
    return execute_inline();
  }

  // Don't try to handle scalar shapes in parallel.
  if (ElementsIn(shape) == 1) {
    return execute_inline();
  }

  // Compute the dimensions of the "work" which are defined by the count of
  // elements in each dimension and the increment.
  std::vector<int64_t> work_dims(shape.dimensions().size());
  for (size_t d = 0; d < shape.dimensions().size(); ++d) {
    work_dims[d] = tsl::MathUtil::CeilOfRatio(count[d], incr[d]);
  }

  // Create the shape of the "work" which has same layout as the original shape.
  Shape work_shape = ShapeUtil::MakeShape(shape.element_type(), work_dims);
  *work_shape.mutable_layout() = shape.layout();

  // We target one task (partition) per available thread.
  size_t target_partition_count = GetForEachIndexParallelThreadCount();

  // Compute major-to-minor partition counts for parallelizing the work.
  ShapePartitionAssigner assigner(work_shape);
  std::vector<int64_t> partition_counts = assigner.Run(target_partition_count);
  int64_t partition_count = assigner.GetTotalPartitionCount(partition_counts);

  // For a single partition compute everything in the caller thread.
  if (partition_count == 1) return execute_inline();

  // Iterate over all partitions in parallel.
  ShapePartitionIterator iterator(work_shape, partition_counts);
  ParallelState pstate(partition_count);

  // Process a single partition using bounds defined by the partition iterator.
  auto process_partition = [&](size_t i) {
    auto partition = iterator.GetPartition(i);

    // Adjust base and count for the `i`-th partition.
    std::vector<int64_t> partition_base(base.begin(), base.end());
    std::vector<int64_t> partition_count(count.begin(), count.end());

    // Iterate over all dimension bounds starting from the outermost dimension.
    for (int64_t d = 0; d < partition.size(); ++d) {
      // Size and offset in the `work_shape` for the `i`-th partition.
      auto [partition_start, partition_size] = partition[d];

      // Update base and count for the `i`-th partition.
      size_t dim = LayoutUtil::Major(shape.layout(), d);
      partition_base[dim] += partition_start * incr[dim];
      partition_count[dim] = std::min(count[dim] - partition_start * incr[dim],
                                      partition_size * incr[dim]);
    }

    ForEachState s(shape, partition_base, partition_count, incr);
    const int thread_id = pstate.pool->CurrentThreadId();

    // Allows handling R0 arrays, such that the visitor function will be
    // called once with the proper empty indexes.
    int64_t n = -1;
    while (n < s.rank) {
      absl::StatusOr<bool> result = visitor_function(s.indexes, thread_id);
      if (!result.ok()) {
        absl::MutexLock lock(&pstate.mu);
        if (pstate.status.ok()) {
          pstate.status = result.status();
        }
      }
      // Increments dimensions in minor to major order.
      n = s.IncrementDim();
    }
  };

  // Launch parallel for loop to process all partitions.
  pstate.pool->ParallelFor(partition_count, /*cost_per_unit=*/1000000,
                           [&](int64_t p_begin, int64_t p_end) {
                             for (size_t i = p_begin; i < p_end; ++i) {
                               process_partition(i);
                             }
                           });
  return pstate.status;
}

/* static */ int ShapeUtil::GetForEachIndexParallelThreadCount() {
  ParallelState pstate(/*task_count=*/0);
  return pstate.pool->NumThreads();
}

/* static */ Shape ShapeUtil::DeleteDimensions(
    absl::Span<int64_t const> dims_to_delete, Shape shape) {
  shape.DeleteDimensions(dims_to_delete);
  return shape;
}

/* static */ Shape ShapeUtil::FilterDimensions(
    absl::FunctionRef<bool(int64_t)> p, Shape shape) {
  CHECK(shape.IsArray());
  std::vector<int64_t> dims_to_delete;
  for (int64_t i = 0; i < shape.dimensions().size(); ++i) {
    if (!p(i)) {
      dims_to_delete.push_back(i);
    }
  }
  shape.DeleteDimensions(dims_to_delete);
  return shape;
}

Shape ShapeUtil::DeviceShapeToHostShape(Shape s) {
  ForEachMutableSubshape(&s, [](Shape* subshape, const ShapeIndex& index) {
    if (subshape->IsArray() && subshape->has_layout()) {
      subshape->mutable_layout()->clear_tiles();
      subshape->mutable_layout()->set_memory_space(Layout::kDefaultMemorySpace);
      subshape->mutable_layout()->clear_physical_shape();
      subshape->mutable_layout()->set_element_size_in_bits(0);
      subshape->mutable_layout()->set_tail_padding_alignment_in_elements(1);
      subshape->mutable_layout()->set_dynamic_shape_metadata_prefix_bytes(0);
    }
  });
  return s;
}

/*static*/ bool ShapeUtil::ElementCanUpcast(const Shape& from,
                                            const Shape& to) {
  return HigherPrecisionElementType(from, to) == to.element_type();
}

/*static*/
absl::Status ShapeUtil::ByteStrides(const Shape& shape,
                                    absl::Span<int64_t> strides) {
  TF_RET_CHECK(shape.IsArray());
  TF_RET_CHECK(shape.has_layout());
  TF_RET_CHECK(shape.dimensions().size() == strides.size());

  int64_t stride = ByteSizeOfPrimitiveType(shape.element_type());
  for (int i : shape.layout().minor_to_major()) {
    strides.at(i) = stride;
    stride *= shape.dimensions(i);
  }
  return absl::OkStatus();
}

/*static*/
std::optional<absl::InlinedVector<int64_t, 4>> ShapeUtil::ByteStrides(
    const Shape& shape) {
  absl::InlinedVector<int64_t, 4> strides(shape.dimensions().size());
  if (!ByteStrides(shape, absl::MakeSpan(strides)).ok()) {
    return std::nullopt;
  }
  return strides;
}

/*static*/ int64_t ShapeUtil::ElementSizeInBits(const Shape& shape) {
  if (shape.has_layout() && shape.layout().element_size_in_bits() != 0) {
    return shape.layout().element_size_in_bits();
  }
  return ShapeUtil::ByteSizeOfPrimitiveType(shape.element_type()) * CHAR_BIT;
}

/*static*/ int64_t ShapeUtil::ArraySize(const Shape& shape) {
  CHECK(LayoutUtil::IsDenseArray(shape));
  if (shape.layout().tiles().empty()) {
    return ByteSizeOfElements(shape);
  }

  auto tile_dimensions = shape.layout().tiles(0).dimensions();
  auto minor_to_major = shape.layout().minor_to_major();
  int64_t shape_dim_size = shape.dimensions().size();
  int64_t tile_dim_size = tile_dimensions.size();

  // Use the top-level tile for shape size calculation. We assume the
  // sub-tiles won't cause additional padding.
  int64_t num_of_elements = 1;
  int64_t dim = 0;
  for (dim = 0; dim < tile_dim_size; dim++) {
    int64_t dim_size = dim < shape_dim_size ? LayoutUtil::MaxSplitSize(
                                                  shape, minor_to_major[dim])
                                            : 1;
    num_of_elements *=
        RoundUpTo(dim_size, tile_dimensions[tile_dim_size - dim - 1]);
  }
  for (; dim < shape_dim_size; dim++) {
    int64_t dim_size = LayoutUtil::MaxSplitSize(shape, minor_to_major[dim]);
    num_of_elements *= dim_size;
  }

  if (shape.layout().tail_padding_alignment_in_elements() != 1) {
    num_of_elements = RoundUpTo(
        num_of_elements, shape.layout().tail_padding_alignment_in_elements());
  }

  if (shape.layout().element_size_in_bits() != 0) {
    const int64_t num_bits =
        num_of_elements * shape.layout().element_size_in_bits();
    return CeilOfRatio<int64_t>(num_bits, CHAR_BIT);
  }

  return num_of_elements * ByteSizeOfPrimitiveType(shape.element_type());
}

/*static*/ int64_t ShapeUtil::ArrayDataSize(const Shape& shape) {
  CHECK(LayoutUtil::IsDenseArray(shape));
  absl::InlinedVector<int64_t, 4> indices;
  for (int64_t dim : shape.dimensions()) {
    indices.push_back(dim - 1);
  }
  int64_t size = LayoutUtil::LinearIndex(shape, indices) + 1;

  if (shape.layout().element_size_in_bits() != 0) {
    const int64_t num_bits = size * shape.layout().element_size_in_bits();
    return CeilOfRatio<int64_t>(num_bits, CHAR_BIT);
  }
  return size * ByteSizeOfPrimitiveType(shape.element_type());
}

int64_t ShapeUtil::ForEachState::CalculateNumSteps() const {
  if (IsZeroElementArray()) return 0;

  int64_t size = 1;
  // This works for rank = 0 as well.
  for (int64_t i = 0; i < rank; ++i) {
    // When the count is zero, it can mean that the given dimension is fixed,
    // but we still iterate over the others.
    if (count[i] == 0) {
      continue;
    }

    CHECK_NE(incr[i], 0);
    int64_t dim = 1 + (count[i] - 1) / incr[i];
    size *= dim;
  }
  return size;
}

/*static*/ void ShapeUtil::UpdateElementSizeInBits(Shape* s,
                                                   bool pack_subbyte_types) {
  ForEachMutableSubshape(s, [pack_subbyte_types](Shape* subshape,
                                                 const ShapeIndex& index) {
    if (subshape->has_layout()) {
      int element_size =
          pack_subbyte_types &&
                  primitive_util::IsSubByteNonPredType(subshape->element_type())
              ? primitive_util::BitWidth(subshape->element_type())
              : 0;
      subshape->mutable_layout()->set_element_size_in_bits(element_size);
    }
  });
}

/*static*/ void ShapeUtil::FlattenTupleShape(
    const Shape& shape, std::vector<const Shape*>& flattened) {
  if (shape.IsTuple()) {
    for (const Shape& subshape : shape.tuple_shapes()) {
      FlattenTupleShape(subshape, flattened);
    }
  } else {
    flattened.push_back(&shape);
  }
}

/*static*/ std::vector<const Shape*> ShapeUtil::FlattenTupleShape(
    const Shape& shape) {
  std::vector<const Shape*> flattened;
  FlattenTupleShape(shape, flattened);
  return flattened;
}

}  // namespace xla
