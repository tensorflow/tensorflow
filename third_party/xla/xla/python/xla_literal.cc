/* Copyright 2025 The OpenXLA Authors.

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

#include "xla/python/xla_literal.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "absl/hash/hash.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "nanobind/nanobind.h"
#include "nanobind/ndarray.h"
#include "nanobind/stl/optional.h"  // IWYU pragma: keep
#include "nanobind/stl/pair.h"  // IWYU pragma: keep
#include "nanobind/stl/shared_ptr.h"  // IWYU pragma: keep
#include "nanobind/stl/string.h"  // IWYU pragma: keep
#include "nanobind/stl/string_view.h"  // IWYU pragma: keep
#include "nanobind/stl/variant.h"  // IWYU pragma: keep
#include "nanobind/stl/vector.h"  // IWYU pragma: keep
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/layout.h"
#include "xla/layout_util.h"
#include "xla/literal.h"
#include "xla/pjrt/exceptions.h"
#include "xla/pjrt/status_casters.h"
#include "xla/python/dlpack_types.h"
#include "xla/python/nb_numpy.h"
#include "xla/python/types.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/tsl/lib/strings/proto_serialization.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"

namespace xla {
namespace {

namespace nb = nanobind;

// Safe version of ShapeUtil::MakeShapeWithDenseLayout that fails gracefully on
// invalid input.
absl::StatusOr<Shape> MakeShapeWithDenseLayout(
    PrimitiveType element_type, absl::Span<const int64_t> dims,
    std::optional<absl::Span<const int64_t>> minor_to_major,
    std::optional<const std::vector<bool>> dynamic_dimensions) {
  Shape shape;
  if (dynamic_dimensions) {
    TF_ASSIGN_OR_RETURN(
        shape, ShapeUtil::MakeValidatedShape(element_type, dims,
                                             dynamic_dimensions.value()));
  } else {
    TF_ASSIGN_OR_RETURN(shape,
                        ShapeUtil::MakeValidatedShape(element_type, dims));
  }
  if (minor_to_major) {
    *shape.mutable_layout() = LayoutUtil::MakeLayout(*minor_to_major);
    TF_RETURN_IF_ERROR(
        LayoutUtil::ValidateLayoutForShape(shape.layout(), shape));
  }

  return shape;
}

// Helper to convert an xla::Literal to a nanobind::ndarray.
// By default, LiteralToNdarray creates a view into the Literal's data buffer,
// avoiding a copy. A copy is only made if the caller of np.array() explicitly
// requests one via the copy argument.
nb::ndarray<> LiteralToNdarray(Literal& obj) {
  const Shape& shape = obj.shape();

  if (!shape.has_layout()) {
    throw XlaRuntimeError(
        "Creating an array is only supported for Literals with a layout.");
  }

  const Layout& layout = shape.layout();

  if (!layout.tiles().empty()) {
    throw XlaRuntimeError(
        "Creating an array from a tiled Literal is not supported.");
  }

  if (!shape.IsArray()) {
    throw XlaRuntimeError(
        "Creating an array is only supported for dense Literals.");
  }

  xla::PrimitiveType primitive_type = shape.element_type();
  nb::dlpack::dtype dtype =
      ValueOrThrow(xla::PrimitiveTypeToNbDLDataType(primitive_type));

  absl::Span<const int64_t> dimensions = shape.dimensions();
  std::vector<size_t> unsigned_dimensions(dimensions.begin(), dimensions.end());
  auto strides = xla::StridesForShape(primitive_type, dimensions, layout);

  return nb::ndarray<>(obj.untyped_data(), unsigned_dimensions.size(),
                       unsigned_dimensions.data(), {}, strides.data(), dtype,
                       nb::device::cpu::value, 0);
}

}  // namespace

void BuildLiteral(nb::module_& m) {
  // Types
  nb::enum_<PrimitiveType>(m, "PrimitiveType", nb::is_arithmetic())
      .value("PRIMITIVE_TYPE_INVALID", PRIMITIVE_TYPE_INVALID)
      .value("PRED", PRED)
      .value("S4", S4)
      .value("S8", S8)
      .value("S16", S16)
      .value("S32", S32)
      .value("S64", S64)
      .value("U4", U4)
      .value("U8", U8)
      .value("U16", U16)
      .value("U32", U32)
      .value("U64", U64)
      .value("F16", F16)
      .value("F4E2M1FN", F4E2M1FN)
      .value("F8E3M4", F8E3M4)
      .value("F8E4M3", F8E4M3)
      .value("F8E4M3FN", F8E4M3FN)
      .value("F8E4M3B11FNUZ", F8E4M3B11FNUZ)
      .value("F8E4M3FNUZ", F8E4M3FNUZ)
      .value("F8E5M2", F8E5M2)
      .value("F8E5M2FNUZ", F8E5M2FNUZ)
      .value("F8E8M0FNU", F8E8M0FNU)
      .value("BF16", BF16)
      .value("F32", F32)
      .value("F64", F64)
      .value("C64", C64)
      .value("C128", C128)
      .value("TUPLE", TUPLE)
      .value("OPAQUE_TYPE", OPAQUE_TYPE)
      .value("TOKEN", TOKEN);

  // Shapes
  nb::class_<Layout> layout_class(m, "Layout");
  layout_class.def(nb::init<absl::Span<const int64_t>>())
      .def("__init__",
           [](Layout* self, nb::typed<nb::sequence, int> minor_to_major,
              nb::typed<nb::sequence, nb::typed<nb::tuple, int, nb::ellipsis>>
                  tiling,
              int64_t element_size_in_bits) {
             std::vector<Tile> xla_tiles;
             xla_tiles.reserve(nb::len(tiling.ptr()));
             for (auto tile : tiling) {
               xla_tiles.push_back(Tile(
                   SequenceToVector<int64_t>(nb::cast<nb::sequence>(tile))));
             }
             std::vector<int64_t> xla_minor_to_major =
                 SequenceToVector<int64_t>(minor_to_major);
             new (self)
                 Layout(xla_minor_to_major, xla_tiles, element_size_in_bits);
           })
      .def("minor_to_major",
           [](Layout layout) { return SpanToNbTuple(layout.minor_to_major()); })
      .def("element_size_in_bits", &Layout::element_size_in_bits)
      .def("tiling",
           [](Layout layout) {
             std::vector<nb::typed<nb::tuple, int, nb::ellipsis>> result;
             result.reserve(layout.tiles().size());
             for (auto& t : layout.tiles()) {
               result.push_back(SpanToNbTuple(t.dimensions()));
             }
             return result;
           })
      .def(
          "__eq__",
          [](const Layout& layout, const Layout& other) {
            return layout == other;
          },
          nb::is_operator(),
          nb::sig("def __eq__(self, other: object, /) -> bool"))
      .def(
          "__ne__",
          [](const Layout& layout, const Layout& other) {
            return layout != other;
          },
          nb::is_operator(),
          nb::sig("def __ne__(self, other: object, /) -> bool"))
      .def("__str__", &Layout::ToString)
      .def("__hash__",
           [](const Layout& layout) { return absl::HashOf(layout); })
      .def("to_string", &Layout::ToString)
      .def("__getstate__",
           [](const Layout& self) -> nb::tuple {
             auto proto = self.ToProto();
             std::string result;
             if (!tsl::SerializeToStringDeterministic(proto, &result)) {
               // throw converted by PyBind to a Python RuntimeError.
               throw XlaRuntimeError(
                   absl::StrCat("Layout.py_pickle: ",
                                "SerializeToStringDeterministic failed"));
             }
             return nb::make_tuple(nb::bytes(result.data(), result.size()));
           })
      .def("__setstate__", [](Layout* self, nb::tuple t) {
        LayoutProto result;
        nb::bytes serialized = nb::cast<nb::bytes>(t[0]);
        result.ParseFromArray(serialized.c_str(), serialized.size());
        new (self) Layout(ValueOrThrow(Layout::FromProto(result)));
      });

  nb::class_<Shape> shape_class(m, "Shape");
  shape_class
      .def("__init__",
           [](Shape* self, const std::string& s) {
             new (self) Shape(ValueOrThrow(ParseShape(s)));
           })
      .def_static(
          "tuple_shape",
          [](std::vector<Shape> shapes) -> Shape {
            return ShapeUtil::MakeTupleShape(shapes);
          },
          "Constructs a tuple shape.")
      .def_static(
          "array_shape",
          xla::ValueOrThrowWrapper(
              [](PrimitiveType type, nb::typed<nb::sequence, int> dims_seq,
                 std::optional<nb::typed<nb::sequence, int>> layout_seq,
                 std::optional<std::vector<bool>> dynamic_dimensions)
                  -> absl::StatusOr<Shape> {
                std::vector<int64_t> dims = SequenceToVector<int64_t>(dims_seq);
                if (layout_seq) {
                  std::vector<int64_t> layout =
                      SequenceToVector<int64_t>(*layout_seq);
                  return MakeShapeWithDenseLayout(type, dims, layout,
                                                  dynamic_dimensions);
                } else {
                  return MakeShapeWithDenseLayout(type, dims, std::nullopt,
                                                  dynamic_dimensions);
                }
              }),
          "Constructs an array shape.", nb::arg("type"), nb::arg("dims"),
          nb::arg("layout").none() = std::nullopt,
          nb::arg("dynamic_dimensions").none() = std::nullopt)
      .def_static(
          "array_shape",
          xla::ValueOrThrowWrapper(
              [](nb_dtype dtype, nb::typed<nb::sequence, int> dims_seq,
                 std::optional<nb::typed<nb::sequence, int>> layout_seq,
                 std::optional<std::vector<bool>> dynamic_dimensions)
                  -> absl::StatusOr<Shape> {
                PrimitiveType type = ValueOrThrow(DtypeToPrimitiveType(dtype));
                std::vector<int64_t> dims = SequenceToVector<int64_t>(dims_seq);
                if (layout_seq) {
                  std::vector<int64_t> layout =
                      SequenceToVector<int64_t>(*layout_seq);
                  return MakeShapeWithDenseLayout(type, dims, layout,
                                                  dynamic_dimensions);
                } else {
                  return MakeShapeWithDenseLayout(type, dims, std::nullopt,
                                                  dynamic_dimensions);
                }
              }),
          "Constructs an array shape.", nb::arg("type"), nb::arg("dims"),
          nb::arg("layout").none() = std::nullopt,
          nb::arg("dynamic_dimensions").none() = std::nullopt)
      .def_static("token_shape", []() { return ShapeUtil::MakeTokenShape(); })
      .def_static(
          "scalar_shape",
          [](PrimitiveType type) -> Shape {
            return ShapeUtil::MakeScalarShape(type);
          },
          "Constructs a scalar shape.", nb::arg("type"))
      .def_static(
          "scalar_shape",
          [](nb_dtype dtype) -> Shape {
            PrimitiveType type = xla::ValueOrThrow(DtypeToPrimitiveType(dtype));
            return ShapeUtil::MakeScalarShape(type);
          },
          "Constructs a scalar shape.", nb::arg("type"))
      .def("dimensions",
           [](const Shape& shape) { return SpanToNbTuple(shape.dimensions()); })
      .def("layout",
           [](const Shape& shape) -> Layout { return shape.layout(); })
      .def("xla_element_type", &Shape::element_type)
      .def("element_type",
           [](const Shape& shape) {
             return xla::ValueOrThrow(
                 PrimitiveTypeToNbDtype(shape.element_type()));
           })
      .def("numpy_dtype",
           [](const Shape& shape) {
             if (shape.IsTuple()) {
               return nb_dtype("O");
             }
             return xla::ValueOrThrow(
                 PrimitiveTypeToNbDtype(shape.element_type()));
           })
      .def("is_tuple", &Shape::IsTuple)
      .def("is_array", &Shape::IsArray)
      .def("is_token", &Shape::IsToken)
      .def("is_static", &Shape::is_static)
      .def("is_dynamic", &Shape::is_dynamic)
      .def("is_dynamic_dimension", &Shape::is_dynamic_dimension,
           nb::arg("dimension"))
      .def("set_dynamic_dimension", &Shape::set_dynamic_dimension,
           nb::arg("dimension"), nb::arg("is_dynamic"))
      .def("rank", &Shape::dimensions_size)
      .def("to_serialized_proto",
           [](const Shape& shape) {
             ShapeProto proto = shape.ToProto();
             std::string s = proto.SerializeAsString();
             return nb::bytes(s.data(), s.size());
           })
      .def("tuple_shapes",
           [](const Shape& shape) {
             return std::vector<Shape>(shape.tuple_shapes());
           })
      .def("leaf_count",
           [](const Shape& shape) { return ShapeUtil::GetLeafCount(shape); })
      .def(
          "with_major_to_minor_layout_if_absent",
          [](const Shape& shape) {
            Shape out = shape;
            ShapeUtil::ForEachMutableSubshape(
                &out, [](Shape* subshape, const ShapeIndex&) {
                  if (!subshape->has_layout()) {
                    LayoutUtil::SetToDefaultLayout(subshape);
                  }
                });
            return out;
          },
          "Returns a copy of a shape with missing layouts set to "
          "major-to-minor.")
      .def(
          "__eq__",
          [](const Shape& shape, const Shape& other) { return shape == other; },
          nb::is_operator(),
          nb::sig("def __eq__(self, other: object, /) -> bool"))
      .def(
          "__ne__",
          [](const Shape& shape, const Shape& other) { return shape != other; },
          nb::is_operator(),
          nb::sig("def __ne__(self, other: object, /) -> bool"))
      .def("__hash__", [](const Shape& shape) { return absl::HashOf(shape); })
      .def("__repr__", [](const Shape& shape) {
        return shape.ToString(/*print_layout=*/true);
      });
  nb::class_<Literal>(m, "Literal")
      .def(nb::init<const Shape&>())
      .def("__repr__", &Literal::ToString)
      .def(
          "__array__",
          [](std::shared_ptr<Literal> obj, std::optional<nb::object> dtype,
             std::optional<bool> copy) {
            // Provides the interface required by numpy to create a np.ndarray.
            nb::ndarray<nb::numpy> np_array(LiteralToNdarray(*obj));

            if (dtype.has_value()) {
              throw XlaRuntimeError(
                  "Passing of dtype to __array__ not currently supported.");
            }

            if (copy.has_value() && *copy) {
              // when a copy is requested we _must_ return a copy:
              // https://numpy.org/doc/2.1/reference/generated/numpy.ndarray.__array__.html
              return np_array.cast(nb::rv_policy::copy);
            }

            return np_array.cast(nb::rv_policy::reference_internal,
                                 nb::cast(obj));
          },
          nb::arg("dtype") = nb::none(), nb::arg("copy") = nb::none())
      .def("shape", &Literal::shape);
}

}  // namespace xla
