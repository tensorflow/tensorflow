/* Copyright 2020 The OpenXLA Authors.

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

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/inlined_vector.h"
#include "absl/hash/hash.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "mlir/Support/LLVM.h"
#include "nanobind/nanobind.h"
#include "nanobind/ndarray.h"
#include "nanobind/stl/optional.h"  // IWYU pragma: keep
#include "nanobind/stl/pair.h"  // IWYU pragma: keep
#include "nanobind/stl/shared_ptr.h"  // IWYU pragma: keep
#include "nanobind/stl/string.h"  // IWYU pragma: keep
#include "nanobind/stl/string_view.h"  // IWYU pragma: keep
#include "nanobind/stl/variant.h"  // IWYU pragma: keep
#include "nanobind/stl/vector.h"  // IWYU pragma: keep
#include "xla/array.h"
#include "xla/client/executable_build_options.h"
#include "xla/debug_options_flags.h"
#include "xla/hlo/builder/xla_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_print_options.h"
#include "xla/hlo/ir/hlo_sharding.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/layout.h"
#include "xla/layout_util.h"
#include "xla/literal.h"
#include "xla/pjrt/exceptions.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/pjrt/proto/compile_options.pb.h"
#include "xla/pjrt/status_casters.h"
#include "xla/python/dlpack_types.h"
#include "xla/python/nb_absl_span.h"  // IWYU pragma: keep
#include "xla/python/nb_numpy.h"
#include "xla/python/types.h"
#include "xla/service/computation_placer.h"
#include "xla/service/hlo.pb.h"
#include "xla/service/hlo_graph_dumper.h"
#include "xla/service/hlo_module_config.h"
#include "xla/service/spmd/shardy/stablehlo_round_trip/stablehlo_import.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/tsl/lib/strings/proto_serialization.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/logging.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"
#include "xla/xla.pb.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace {

namespace nb = nanobind;

// Converts a computation to a serialized HloModuleProto.
absl::StatusOr<nb::bytes> GetComputationSerializedProto(
    const XlaComputation& computation) {
  std::string result;
  if (!tsl::SerializeToStringDeterministic(computation.proto(), &result)) {
    return Unknown("Failed to serialize the HloModuleProto.");
  }
  return nb::bytes(result.data(), result.size());
}

// Converts a hlo module to a serialized HloModuleProto.
absl::StatusOr<nb::bytes> GetHloModuleSerializedProto(const HloModule& module) {
  std::string result;
  if (!tsl::SerializeToStringDeterministic(module.ToProto(), &result)) {
    return Unknown("Failed to serialize the HloModuleProto.");
  }
  return nb::bytes(result.data(), result.size());
}

// Converts a serialized HloModuleProto into a HloModule.
absl::StatusOr<std::shared_ptr<HloModule>> HloModuleFromSerializedProto(
    const nb::bytes& bytes) {
  HloModuleProto proto;
  proto.ParseFromArray(bytes.c_str(), bytes.size());
  TF_ASSIGN_OR_RETURN(const HloModuleConfig module_config,
                      HloModule::CreateModuleConfigFromProto(
                          proto, GetDebugOptionsFromFlags()));
  TF_ASSIGN_OR_RETURN(std::unique_ptr<HloModule> module,
                      HloModule::CreateFromProto(proto, module_config));
  return std::shared_ptr<HloModule>(std::move(module));
}

absl::StatusOr<std::shared_ptr<HloModule>> GetHloModule(
    const XlaComputation& computation) {
  TF_ASSIGN_OR_RETURN(const HloModuleConfig module_config,
                      HloModule::CreateModuleConfigFromProto(
                          computation.proto(), GetDebugOptionsFromFlags()));
  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<HloModule> module,
      HloModule::CreateFromProto(computation.proto(), module_config));
  return std::shared_ptr<HloModule>(std::move(module));
}

// Converts a computation to textual HLO form.
absl::StatusOr<std::string> GetComputationHloText(
    const XlaComputation& computation, bool print_large_constants = false) {
  TF_ASSIGN_OR_RETURN(std::shared_ptr<HloModule> hlo_module,
                      GetHloModule(computation));
  HloPrintOptions options;
  options = HloPrintOptions::ShortParsable();
  options.set_print_large_constants(print_large_constants);
  return hlo_module->ToString(options);
}

// Converts a computation to HLO dot graph form.
absl::StatusOr<std::string> GetComputationHloDotGraph(
    const XlaComputation& computation) {
  TF_ASSIGN_OR_RETURN(std::shared_ptr<HloModule> hlo_module,
                      GetHloModule(computation));
  return RenderGraph(*hlo_module->entry_computation(), /*label=*/"",
                     hlo_module->config().debug_options(),
                     RenderedGraphFormat::kDot);
}

// Hashes the HLO module.
absl::StatusOr<uint64_t> HashComputation(const XlaComputation& computation) {
  TF_ASSIGN_OR_RETURN(std::shared_ptr<HloModule> hlo_module,
                      GetHloModule(computation));
  return absl::HashOf(*hlo_module);
}
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

// Pybind function for HloSharding.iota_tile, which is a non-crashing factory
// that produces a HloSharding instance backed by tile assignment of a
// transposed and reshaped iota array of device ids. More specifically the tile
// assignment array is as if it is produced by the following numpy code:
// numpy.arange(math.prod(dims)).reshape(reshape_dims)
//      .transpose(transpose_perm).reshape(math.prod(dims))
// where:
// `dims`: is the dimensions of the tile assignment array, which corresponds to
//   OpSharding.tile_assignment_dimensions.
// `reshape_dims`: is the dimensions the 1D iota array is reshaped to.
// `transpose_perm`: is the dimension permutation to transpose `reshape_dims`.
// `subgroup_types`: indicates the subgroups of the last `subgroup_types.size()`
//   dimensions in `dims`.
//
// In practice, `reshape_dims` often maps to the axes of user defined device
// mesh, and `transpose_perm` often maps to the user specification of how a
// tensor is partitioned based on the axes defined in the mesh, e.g. for a mesh
// of size 4x2x2 as AxBxC:
// PartitionSpec('A', 'B', 'C') corresponds to reshape_dims=[4,2,2],
// transpose_perm=[0,1,2] (no transpose)
// PartitionSpec('B', 'A', 'C') corresponds to reshape_dims=[4,2,2],
// transpose_perm=[1,0,2] (swap A and B)
absl::StatusOr<HloSharding> IotaTileHelper(
    absl::Span<const int64_t> dims, absl::Span<const int64_t> reshape_dims,
    absl::Span<const int> transpose_perm,
    absl::Span<const OpSharding::Type> subgroup_types) {
  if (dims.empty()) {
    return InvalidArgument("`dims` should not be empty.");
  }
  if (reshape_dims.size() != transpose_perm.size()) {
    return InvalidArgument(
        "`reshape_dims` and `transpose_perm` should have the same size, saw "
        "[%s] v.s. [%s]",
        absl::StrJoin(reshape_dims, ","), absl::StrJoin(transpose_perm, ","));
  }
  if (!reshape_dims.empty() && Product(dims) != Product(reshape_dims)) {
    return InvalidArgument(
        "Cannot reshape from `dims` [%s] to `reshape_dims` [%s].",
        absl::StrJoin(dims, ","), absl::StrJoin(reshape_dims, ","));
  }
  if (subgroup_types.size() > dims.size()) {
    return InvalidArgument(
        "`subgroup_types`(%lld) should not have more dimensions than "
        "`dims`(%lld).",
        subgroup_types.size(), dims.size());
  }
  if (reshape_dims.empty()) {
    return subgroup_types.empty()
               ? HloSharding::IotaTile(dims)
               : HloSharding::Subgroup(TileAssignment(dims), subgroup_types);
  }
  return subgroup_types.empty()
             ? HloSharding::IotaTile(dims, reshape_dims, transpose_perm)
             : HloSharding::Subgroup(
                   TileAssignment(dims, reshape_dims, transpose_perm),
                   subgroup_types);
}

template <typename T, typename Container>
void DefRepeatedProperty(nb::class_<T>& cls, const char* name,
                         Container* (T::*getter)()) {
  cls.def_prop_rw(
      name,
      [getter](T& obj) {
        Container* elems = (obj.*getter)();
        std::vector<typename Container::value_type> result;
        result.reserve(elems->size());
        std::copy(elems->begin(), elems->end(), std::back_inserter(result));
        return result;
      },
      [getter](T& obj, std::vector<typename Container::value_type> new_elems) {
        Container* elems = (obj.*getter)();
        elems->Clear();
        elems->Reserve(new_elems.size());
        for (typename Container::value_type& e : new_elems) {
          elems->Add(std::move(e));
        }
      });
}

template <typename T, typename Container>
void DefRepeatedEnumProperty(nb::class_<T>& cls, const char* name,
                             Container* (T::*getter)()) {
  cls.def_prop_rw(
      name,
      [getter](T& obj) {
        Container* elems = (obj.*getter)();
        std::vector<typename Container::value_type> result;
        result.reserve(elems->size());
        std::copy(elems->begin(), elems->end(), std::back_inserter(result));
        return result;
      },
      [getter](
          T& obj,
          nb::typed<nb::sequence, typename Container::value_type> new_elems) {
        Container* elems = (obj.*getter)();
        elems->Clear();
        for (nb::handle e : new_elems) {
          elems->Add(nb::cast<int>(e.attr("value")));
        }
      });
}

template <typename T>
Array<T> NDArrayToArray(nb::ndarray<T, nb::c_contig> ndarray) {
  std::vector<int64_t> shapes;
  shapes.reserve(ndarray.ndim());
  for (int i = 0; i < ndarray.ndim(); ++i) {
    shapes.push_back(ndarray.shape(i));
  }
  xla::Array<int64_t> array(shapes);
  array.Each([&](absl::Span<const int64_t> indices, int64_t* val) {
    int64_t offset = indices.back();
    int64_t multiplier = 1;
    for (int i = ndarray.ndim() - 1; i > 0; --i) {
      multiplier *= ndarray.shape(i);
      offset += indices[i - 1] * multiplier;
    }
    *val = *(ndarray.data() + offset);
  });
  return array;
}

absl::StatusOr<HloSharding> SubgroupWithTileAssignmentHelper(
    nb::ndarray<int64_t, nb::c_contig> tile_assignment,
    absl::Span<const OpSharding::Type> subgroup_types) {
  return HloSharding::Subgroup(NDArrayToArray(tile_assignment), subgroup_types);
}

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
      ValueOrThrow(PrimitiveTypeToNbDLDataType(primitive_type));

  absl::Span<const int64_t> dimensions = shape.dimensions();
  std::vector<size_t> unsigned_dimensions(dimensions.begin(), dimensions.end());
  auto strides = StridesForShape(primitive_type, dimensions, layout);

  return nb::ndarray<>(obj.untyped_data(), unsigned_dimensions.size(),
                       unsigned_dimensions.data(), {}, strides.data(), dtype,
                       nb::device::cpu::value, 0);
}

struct Descriptor {};

}  // namespace

NB_MODULE(_xla, m) {
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

  nb::class_<ProgramShape>(m, "ProgramShape")
      .def(
          "__init__",
          [](ProgramShape* self, absl::Span<const Shape> params, Shape result) {
            new (self) ProgramShape();
            for (const Shape& param : params) {
              self->AddParameter(param, "");
            }
            *self->mutable_result() = result;
          })
      .def("parameter_shapes",
           static_cast<const std::vector<Shape>& (ProgramShape::*)() const>(
               &ProgramShape::parameters))
      .def("result_shape", &ProgramShape::result)
      .def("__repr__", &ProgramShape::ToString);

  // Literals
  nb::class_<Literal>(m, "Literal")
      .def(nb::init<const Shape&>())
      .def("__repr__", &Literal::ToString)
      .def(
          "__array__",
          [](std::shared_ptr<Literal> obj, std::optional<nb::object> dtype,
             std::optional<bool> copy) {
            // Provides the interface required by numpy to create a np.ndarray.
            // Currently don't support the __dl_pack__ interface but can be
            // added with very little effort it if needed.

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
          nb::arg("dtype").none() = nb::none(),
          nb::arg("copy").none() = nb::none())
      .def("shape", &Literal::shape);

  nb::class_<XlaComputation>(m, "XlaComputation")
      .def("__init__",
           [](XlaComputation* self,
              const nb::bytes& serialized_hlo_module_proto) {
             HloModuleProto proto;
             proto.ParseFromArray(serialized_hlo_module_proto.c_str(),
                                  serialized_hlo_module_proto.size());
             new (self) XlaComputation(proto);
           })
      .def("get_hlo_module", xla::ValueOrThrowWrapper(GetHloModule))
      .def("program_shape",
           xla::ValueOrThrowWrapper(&XlaComputation::GetProgramShape))
      .def("name", &XlaComputation::name)
      .def("as_serialized_hlo_module_proto",
           xla::ValueOrThrowWrapper(GetComputationSerializedProto))
      .def("as_hlo_text", xla::ValueOrThrowWrapper(GetComputationHloText),
           nb::arg("print_large_constants") = false)
      .def("as_hlo_dot_graph",
           xla::ValueOrThrowWrapper(GetComputationHloDotGraph))
      .def("hash", xla::ValueOrThrowWrapper(HashComputation))
      .def("as_hlo_module", xla::ValueOrThrowWrapper(GetHloModule));

  nb::class_<HloPrintOptions> hlo_print_options_class(m, "HloPrintOptions");
  hlo_print_options_class.def(nb::init<>())
      .def_static("short_parsable", &HloPrintOptions::ShortParsable)
      .def_static("canonical", &HloPrintOptions::Canonical)
      .def_static("fingerprint", &HloPrintOptions::Fingerprint)
      .def_prop_rw("print_large_constants",
                   &HloPrintOptions::print_large_constants,
                   &HloPrintOptions::set_print_large_constants)
      .def_prop_rw("print_metadata", &HloPrintOptions::print_metadata,
                   &HloPrintOptions::set_print_metadata)
      .def_prop_rw("print_backend_config",
                   &HloPrintOptions::print_backend_config,
                   &HloPrintOptions::set_print_backend_config)
      .def_prop_rw("print_result_shape", &HloPrintOptions::print_result_shape,
                   &HloPrintOptions::set_print_result_shape)
      .def_prop_rw("print_operand_shape", &HloPrintOptions::print_operand_shape,
                   &HloPrintOptions::set_print_operand_shape)
      .def_prop_rw("print_operand_names", &HloPrintOptions::print_operand_names,
                   &HloPrintOptions::set_print_operand_names)
      .def_prop_rw("print_ids", &HloPrintOptions::print_ids,
                   &HloPrintOptions::set_print_ids)
      .def_prop_rw("print_extra_attributes",
                   &HloPrintOptions::print_extra_attributes,
                   &HloPrintOptions::set_print_extra_attributes)
      .def_prop_rw("print_program_shape", &HloPrintOptions::print_program_shape,
                   &HloPrintOptions::set_print_program_shape)
      .def_prop_rw("print_percent", &HloPrintOptions::print_percent,
                   &HloPrintOptions::set_print_percent)
      .def_prop_rw("print_control_dependencies",
                   &HloPrintOptions::print_control_dependencies,
                   &HloPrintOptions::set_print_control_dependencies)
      .def_prop_rw("compact_operands", &HloPrintOptions::compact_operands,
                   &HloPrintOptions::set_compact_operands)
      .def_prop_rw("include_layout_in_shapes",
                   &HloPrintOptions::include_layout_in_shapes,
                   &HloPrintOptions::set_include_layout_in_shapes)
      .def_prop_rw("canonicalize_instruction_names",
                   &HloPrintOptions::canonicalize_instruction_names,
                   &HloPrintOptions::set_canonicalize_instruction_names)
      .def_prop_rw("canonicalize_computations",
                   &HloPrintOptions::canonicalize_computations,
                   &HloPrintOptions::set_canonicalize_computations)
      .def_prop_rw("indent_amount", &HloPrintOptions::indent_amount,
                   &HloPrintOptions::set_indent_amount)
      .def_prop_rw("is_in_nested_computation",
                   &HloPrintOptions::is_in_nested_computation,
                   &HloPrintOptions::set_is_in_nested_computation);

  // HloModule.computations() returns raw pointers.
  // pybind seems to prefer smart pointers.
  // We give pybind a smart pointer to a wrapper around a raw pointer to satisfy
  // pybind and avoid double frees.
  class ComputationWrapper {
   public:
    ComputationWrapper(const HloComputation* comp,
                       const std::shared_ptr<HloModule> module)
        : comp_(comp), module_(module) {}
    absl::string_view name() const { return comp_->name(); }
    void render_html(const std::string& filename) {
      std::string html = xla::ValueOrThrow(RenderGraph(
          *comp_, /*label=*/"", comp_->parent()->config().debug_options(),
          RenderedGraphFormat::kHtml, HloRenderOptions()));
      xla::ThrowIfError(tsl::WriteStringToFile(
          tsl::Env::Default(), absl::StrCat(filename, ".html"), html));
    }

   private:
    const HloComputation* comp_;
    // The module owns the computations: if its destructor is called, the
    // computations are freed. To prevent that from happening in cases where the
    // module Python object goes out of scope and gets garbage collected before
    // the computations, we keep a shared_ptr to the module that originated the
    // computation.
    const std::shared_ptr<HloModule> module_;
  };

  nb::class_<ComputationWrapper> hlo_computation_class(m, "HloComputation");

  hlo_computation_class.def_prop_ro("name", &ComputationWrapper::name)
      .def("render_html", &ComputationWrapper::render_html);

  nb::class_<HloModule> hlo_module_class(m, "HloModule");
  hlo_module_class.def_prop_ro("name", &HloModule::name)
      .def(
          "to_string",
          static_cast<std::string (HloModule::*)(const HloPrintOptions&) const>(
              &HloModule::ToString),
          nb::arg("options") = HloPrintOptions())
      .def("as_serialized_hlo_module_proto",
           xla::ValueOrThrowWrapper(GetHloModuleSerializedProto))
      .def("from_serialized_hlo_module_proto",
           xla::ValueOrThrowWrapper(HloModuleFromSerializedProto))
      .def("computations",
           [](const std::shared_ptr<HloModule> m)
               -> std::vector<std::shared_ptr<ComputationWrapper>> {
             std::vector<std::shared_ptr<ComputationWrapper>> computations;
             for (HloComputation* comp : m->computations())
               computations.push_back(
                   std::make_shared<ComputationWrapper>(comp, m));
             return computations;
           })
      .def_prop_ro("spmd_output_sharding",
                   [](const HloModule& m) -> std::optional<xla::OpSharding> {
                     if (!m.has_spmd_output_sharding()) return std::nullopt;
                     return m.spmd_output_sharding().ToProto();
                   })
      .def_prop_ro("spmd_parameters_shardings",
                   [](const HloModule& m)
                       -> std::optional<std::vector<xla::OpSharding>> {
                     if (!m.has_spmd_parameters_shardings())
                       return std::nullopt;
                     std::vector<xla::OpSharding> param_shardings;
                     for (const auto& parameter_sharding :
                          m.spmd_parameters_shardings()) {
                       param_shardings.push_back(parameter_sharding.ToProto());
                     }
                     return param_shardings;
                   });

  m.def("hlo_module_to_dot_graph",
        [](const HloModule& hlo_module) -> std::string {
          return xla::ValueOrThrow(RenderGraph(
              *hlo_module.entry_computation(), /*label=*/"",
              hlo_module.config().debug_options(), RenderedGraphFormat::kDot));
        });
  m.def("hlo_module_from_text",
        xla::ValueOrThrowWrapper(
            [](const std::string& hlo_module_text)
                -> absl::StatusOr<std::shared_ptr<HloModule>> {
              auto hlo_module =
                  xla::ParseAndReturnUnverifiedModule(hlo_module_text);
              TF_RETURN_IF_ERROR(hlo_module.status());
              std::shared_ptr<HloModule> result(std::move(*hlo_module));
              return result;
            }));

  // Device assignments
  nb::class_<DeviceAssignment>(m, "DeviceAssignment")
      .def_static(
          "create",
          xla::ValueOrThrowWrapper([](nb::ndarray<int, nb::ndim<2>> array)
                                       -> absl::StatusOr<DeviceAssignment> {
            if (array.ndim() != 2) {
              return InvalidArgument(
                  "Argument to DeviceAssignment constructor must be a "
                  "2D array, received an %dD array.",
                  array.ndim());
            }
            DeviceAssignment result(array.shape(0), array.shape(1));
            for (int i = 0; i < array.shape(0); ++i) {
              for (int j = 0; j < array.shape(1); ++j) {
                result(i, j) = array(i, j);
              }
            }
            return result;
          }))
      .def("replica_count", &DeviceAssignment::replica_count)
      .def("computation_count", &DeviceAssignment::computation_count)
      .def("__repr__", &DeviceAssignment::ToString)
      .def("serialize",
           xla::ValueOrThrowWrapper(
               [](const DeviceAssignment& da) -> absl::StatusOr<nb::bytes> {
                 DeviceAssignmentProto proto;
                 da.Serialize(&proto);
                 std::string result;
                 if (!tsl::SerializeToStringDeterministic(proto, &result)) {
                   return Unknown(
                       "Failed to serialize the DeviceAssignmentProto.");
                 }
                 return nb::bytes(result.data(), result.size());
               }));

  nb::class_<CompileOptions> compile_options(m, "CompileOptions");
  compile_options
      .def("__init__",
           [](CompileOptions* self) {
             new (self) CompileOptions();
             DebugOptions* debug_options =
                 self->executable_build_options.mutable_debug_options();
             // Sets fast-math-disabling default options expected by JAX.
             debug_options->set_xla_cpu_enable_fast_min_max(false);
             debug_options->set_xla_gpu_enable_fast_min_max(false);
           })
      .def("__getstate__",
           [](const CompileOptions& self) -> nb::tuple {
             auto proto = ValueOrThrow(self.ToProto());
             std::string result;
             if (!tsl::SerializeToStringDeterministic(proto, &result)) {
               // throw converted by PyBind to a Python RuntimeError.
               throw XlaRuntimeError(
                   absl::StrCat("CompileOptions.py_pickle: ",
                                "SerializeToStringDeterministic failed"));
             }
             return nb::make_tuple(nb::bytes(result.data(), result.size()));
           })
      .def("__setstate__",
           [](CompileOptions* self, nb::tuple t) {
             CompileOptionsProto result;
             nb::bytes serialized = nb::cast<nb::bytes>(t[0]);
             result.ParseFromArray(serialized.c_str(), serialized.size());
             new (self) CompileOptions(
                 ValueOrThrow(CompileOptions::FromProto(result)));
           })
      .def("SerializeAsString",
           [](const CompileOptions& self) -> nb::bytes {
             auto proto = ValueOrThrow(self.ToProto());
             std::string result;
             if (!tsl::SerializeToStringDeterministic(proto, &result)) {
               // throw converted by PyBind to a Python RuntimeError.
               throw XlaRuntimeError(
                   absl::StrCat("CompileOptions.SerializeAsString: ",
                                "SerializeToStringDeterministic failed"));
             }
             return nb::bytes(result.data(), result.size());
           })
      .def_static("ParseFromString",
                  [](nb::bytes s) {
                    CompileOptionsProto result;
                    result.ParseFromArray(s.c_str(), s.size());
                    return ValueOrThrow(CompileOptions::FromProto(result));
                  })
      .def_rw("argument_layouts", &CompileOptions::argument_layouts)
      .def_rw("parameter_is_tupled_arguments",
              &CompileOptions::parameter_is_tupled_arguments)
      .def_rw("compile_portable_executable",
              &CompileOptions::compile_portable_executable)
      .def_ro("executable_build_options",
              &CompileOptions::executable_build_options)
      .def_rw("env_option_overrides", &CompileOptions::env_option_overrides)
      .def_prop_rw(
          "num_replicas",
          [](const CompileOptions& options) {
            return options.executable_build_options.num_replicas();
          },
          [](CompileOptions& options, int num_replicas) {
            options.executable_build_options.set_num_replicas(num_replicas);
          })
      .def_prop_rw(
          "num_partitions",
          [](const CompileOptions& options) {
            return options.executable_build_options.num_partitions();
          },
          [](CompileOptions& options, int num_partitions) {
            options.executable_build_options.set_num_partitions(num_partitions);
          })
      .def_prop_rw(
          "profile_version",
          [](const CompileOptions& options) { return options.profile_version; },
          [](CompileOptions& options, int64_t profile_version) {
            options.profile_version = profile_version;
          })
      .def_prop_rw(
          "device_assignment",
          [](const CompileOptions& options) -> std::optional<DeviceAssignment> {
            return options.executable_build_options.has_device_assignment()
                       ? std::optional<DeviceAssignment>(
                             options.executable_build_options
                                 .device_assignment())
                       : std::nullopt;
          },
          [](CompileOptions& options,
             const DeviceAssignment& device_assignment) {
            options.executable_build_options.set_device_assignment(
                device_assignment);
          });

  nb::enum_<DebugOptions::AutotuneCacheMode>(m, "AutotuneCacheMode")
      .value("UNSPECIFIED", DebugOptions::AUTOTUNE_CACHE_MODE_UNSPECIFIED)
      .value("UPDATE", DebugOptions::AUTOTUNE_CACHE_MODE_UPDATE)
      .value("READ", DebugOptions::AUTOTUNE_CACHE_MODE_READ);

  nb::class_<DebugOptions>(m, "DebugOptions")
      .def("__repr__", &DebugOptions::DebugString)
      .def_prop_rw("xla_backend_optimization_level",
                   &DebugOptions::xla_backend_optimization_level,
                   &DebugOptions::set_xla_backend_optimization_level)
      .def_prop_rw("xla_cpu_enable_fast_math",
                   &DebugOptions::xla_cpu_enable_fast_math,
                   &DebugOptions::set_xla_cpu_enable_fast_math)
      .def_prop_rw("xla_cpu_enable_xprof_traceme",
                   &DebugOptions::xla_cpu_enable_xprof_traceme,
                   &DebugOptions::set_xla_cpu_enable_xprof_traceme)
      .def_prop_rw("xla_cpu_fast_math_honor_infs",
                   &DebugOptions::xla_cpu_fast_math_honor_infs,
                   &DebugOptions::set_xla_cpu_fast_math_honor_infs)
      .def_prop_rw("xla_cpu_fast_math_honor_nans",
                   &DebugOptions::xla_cpu_fast_math_honor_nans,
                   &DebugOptions::set_xla_cpu_fast_math_honor_nans)
      .def_prop_rw("xla_cpu_fast_math_honor_division",
                   &DebugOptions::xla_cpu_fast_math_honor_division,
                   &DebugOptions::set_xla_cpu_fast_math_honor_division)
      .def_prop_rw("xla_cpu_fast_math_honor_functions",
                   &DebugOptions::xla_cpu_fast_math_honor_functions,
                   &DebugOptions::set_xla_cpu_fast_math_honor_functions)
      .def_prop_rw("xla_detailed_logging", &DebugOptions::xla_detailed_logging,
                   &DebugOptions::set_xla_detailed_logging)
      .def_prop_rw("xla_enable_dumping", &DebugOptions::xla_enable_dumping,
                   &DebugOptions::set_xla_enable_dumping)
      .def_prop_rw("xla_gpu_enable_fast_min_max",
                   &DebugOptions::xla_gpu_enable_fast_min_max,
                   &DebugOptions::set_xla_gpu_enable_fast_min_max)
      .def_prop_rw("xla_gpu_dump_autotune_results_to",
                   &DebugOptions::xla_gpu_dump_autotune_results_to,
                   [](DebugOptions* self, std::string value) {
                     self->set_xla_gpu_dump_autotune_results_to(value);
                   })
      .def_prop_rw("xla_gpu_load_autotune_results_from",
                   &DebugOptions::xla_gpu_load_autotune_results_from,
                   [](DebugOptions* self, std::string value) {
                     self->set_xla_gpu_load_autotune_results_from(value);
                   })
      .def_prop_rw("xla_gpu_cuda_data_dir",
                   &DebugOptions::xla_gpu_cuda_data_dir,
                   [](DebugOptions* self, std::string value) {
                     self->set_xla_gpu_cuda_data_dir(value);
                   })
      .def_prop_rw("xla_llvm_disable_expensive_passes",
                   &DebugOptions::xla_llvm_disable_expensive_passes,
                   &DebugOptions::set_xla_llvm_disable_expensive_passes)
      .def_prop_rw(
          "xla_disable_hlo_passes",
          [](DebugOptions* self) {
            return absl::StrJoin(self->xla_disable_hlo_passes(), ",");
          },
          [](DebugOptions* self, std::string value) {
            self->clear_xla_disable_hlo_passes();
            for (const auto& passname :
                 std::vector<std::string>(absl::StrSplit(value, ','))) {
              self->add_xla_disable_hlo_passes(passname);
            }
          })
      .def_prop_rw(
          "xla_enable_hlo_passes_only",
          [](DebugOptions* self) {
            return absl::StrJoin(self->xla_enable_hlo_passes_only(), ",");
          },
          [](DebugOptions* self, std::string value) {
            self->clear_xla_enable_hlo_passes_only();
            for (const auto& passname :
                 std::vector<std::string>(absl::StrSplit(value, ','))) {
              self->add_xla_enable_hlo_passes_only(passname);
            }
          })
      .def_prop_rw("xla_test_all_input_layouts",
                   &DebugOptions::xla_test_all_input_layouts,
                   &DebugOptions::set_xla_test_all_input_layouts)
      .def_prop_rw("xla_force_host_platform_device_count",
                   &DebugOptions::xla_force_host_platform_device_count,
                   &DebugOptions::set_xla_force_host_platform_device_count)
      .def_prop_rw("xla_dump_to", &DebugOptions::xla_dump_to,
                   [](DebugOptions* self, std::string value) {
                     self->set_xla_dump_to(value);
                   })
      .def_prop_rw("xla_dump_hlo_module_re",
                   &DebugOptions::xla_dump_hlo_module_re,
                   [](DebugOptions* self, std::string value) {
                     self->set_xla_dump_hlo_module_re(value);
                   })
      .def_prop_rw("xla_dump_hlo_pass_re", &DebugOptions::xla_dump_hlo_pass_re,
                   [](DebugOptions* self, std::string value) {
                     self->set_xla_dump_hlo_pass_re(value);
                   })
      .def_prop_rw("xla_dump_hlo_as_text", &DebugOptions::xla_dump_hlo_as_text,
                   &DebugOptions::set_xla_dump_hlo_as_text)
      .def_prop_rw("xla_dump_hlo_as_proto",
                   &DebugOptions::xla_dump_hlo_as_proto,
                   &DebugOptions::set_xla_dump_hlo_as_proto)
      .def_prop_rw("xla_dump_hlo_as_dot", &DebugOptions::xla_dump_hlo_as_dot,
                   &DebugOptions::set_xla_dump_hlo_as_dot)
      .def_prop_rw("xla_dump_hlo_as_url", &DebugOptions::xla_dump_hlo_as_url,
                   &DebugOptions::set_xla_dump_hlo_as_url)
      .def_prop_rw("xla_dump_hlo_as_html", &DebugOptions::xla_dump_hlo_as_html,
                   &DebugOptions::set_xla_dump_hlo_as_html)
      .def_prop_rw("xla_dump_fusion_visualization",
                   &DebugOptions::xla_dump_fusion_visualization,
                   &DebugOptions::set_xla_dump_fusion_visualization)
      .def_prop_rw("xla_dump_hlo_snapshots",
                   &DebugOptions::xla_dump_hlo_snapshots,
                   &DebugOptions::set_xla_dump_hlo_snapshots)
      .def_prop_rw("xla_dump_max_hlo_modules",
                   &DebugOptions::xla_dump_max_hlo_modules,
                   &DebugOptions::set_xla_dump_max_hlo_modules)
      .def_prop_rw("xla_dump_module_metadata",
                   &DebugOptions::xla_dump_module_metadata,
                   &DebugOptions::set_xla_dump_module_metadata)
      .def_prop_rw("xla_dump_compress_protos",
                   &DebugOptions::xla_dump_compress_protos,
                   &DebugOptions::set_xla_dump_compress_protos)
      .def_prop_rw("xla_dump_hlo_as_long_text",
                   &DebugOptions::xla_dump_hlo_as_long_text,
                   &DebugOptions::set_xla_dump_hlo_as_long_text)
      .def_prop_rw("xla_dump_disable_metadata",
                   &DebugOptions::xla_dump_disable_metadata,
                   &DebugOptions::set_xla_dump_disable_metadata)
      .def_prop_rw("xla_dump_hlo_pipeline_re",
                   &DebugOptions::xla_dump_hlo_pipeline_re,
                   [](DebugOptions* self, std::string value) {
                     self->set_xla_dump_hlo_pipeline_re(value);
                   })
      .def_prop_rw("xla_gpu_dump_autotune_logs_to",
                   &DebugOptions::xla_gpu_dump_autotune_logs_to,
                   [](DebugOptions* self, std::string value) {
                     self->set_xla_gpu_dump_autotune_logs_to(value);
                   })
      .def_prop_rw("xla_gpu_kernel_cache_file",
                   &DebugOptions::xla_gpu_kernel_cache_file,
                   [](DebugOptions* self, std::string value) {
                     self->set_xla_gpu_kernel_cache_file(value);
                   })
      .def_prop_rw(
          "xla_gpu_enable_llvm_module_compilation_parallelism",
          &DebugOptions::xla_gpu_enable_llvm_module_compilation_parallelism,
          &DebugOptions::set_xla_gpu_enable_llvm_module_compilation_parallelism)
      .def_prop_rw("xla_gpu_per_fusion_autotune_cache_dir",
                   &DebugOptions::xla_gpu_per_fusion_autotune_cache_dir,
                   [](DebugOptions* self, std::string value) {
                     self->set_xla_gpu_per_fusion_autotune_cache_dir(value);
                   })
      .def_prop_rw("xla_gpu_experimental_autotune_cache_mode",
                   &DebugOptions::xla_gpu_experimental_autotune_cache_mode,
                   &DebugOptions::set_xla_gpu_experimental_autotune_cache_mode);

  nb::class_<ExecutableBuildOptions>(m, "ExecutableBuildOptions")
      .def(nb::init<>())
      .def("__repr__", &ExecutableBuildOptions::ToString)
      .def_prop_rw(
          "fdo_profile",
          [](const ExecutableBuildOptions& options) {
            return nb::bytes(options.fdo_profile().data(),
                             options.fdo_profile().size());
          },
          [](ExecutableBuildOptions& options, nb::bytes fdo_profile) {
            options.set_fdo_profile(
                std::string(fdo_profile.c_str(), fdo_profile.size()));
          })
      .def_prop_rw(
          "result_layout",
          [](const ExecutableBuildOptions& options) -> std::optional<Shape> {
            return options.result_layout()
                       ? std::optional<Shape>(*options.result_layout())
                       : std::nullopt;
          },
          &ExecutableBuildOptions::set_result_layout)
      .def_prop_rw("num_replicas", &ExecutableBuildOptions::num_replicas,
                   &ExecutableBuildOptions::set_num_replicas)
      .def_prop_rw("num_partitions", &ExecutableBuildOptions::num_partitions,
                   &ExecutableBuildOptions::set_num_partitions)
      .def_prop_ro("debug_options",
                   &ExecutableBuildOptions::mutable_debug_options,
                   nb::rv_policy::reference, nb::keep_alive<1, 0>())
      .def_prop_rw(
          "device_assignment",
          [](const ExecutableBuildOptions& options)
              -> std::optional<DeviceAssignment> {
            return options.has_device_assignment()
                       ? std::optional<DeviceAssignment>(
                             options.device_assignment())
                       : std::nullopt;
          },
          &ExecutableBuildOptions::set_device_assignment)
      .def("compilation_environments_from_serialized_proto",
           [](ExecutableBuildOptions& options,
              const nb::bytes& serialized_proto) {
             xla::CompilationEnvironmentsProto env_proto;
             env_proto.ParseFromArray(serialized_proto.c_str(),
                                      serialized_proto.size());
             auto comp_envs = xla::ValueOrThrow(
                 xla::CompilationEnvironments::CreateFromProto(env_proto));
             *options.mutable_comp_envs() = std::move(*comp_envs);
           })
      .def_prop_rw("exec_time_optimization_effort",
                   &ExecutableBuildOptions::exec_time_optimization_effort,
                   &ExecutableBuildOptions::set_exec_time_optimization_effort)
      .def_prop_rw("memory_fitting_effort",
                   &ExecutableBuildOptions::memory_fitting_effort,
                   &ExecutableBuildOptions::set_memory_fitting_effort)
      .def_prop_rw(
          "optimization_level",
          [](ExecutableBuildOptions& options) {
            return static_cast<int>(options.optimization_level());
          },
          [](ExecutableBuildOptions& options, int value) {
            options.set_optimization_level(
                static_cast<xla::ExecutionOptions::EffortLevel>(value));
          })
      .def_prop_rw(
          "memory_fitting_level",
          [](ExecutableBuildOptions& options) {
            return static_cast<int>(options.memory_fitting_level());
          },
          [](ExecutableBuildOptions& options, int value) {
            options.set_memory_fitting_level(
                static_cast<xla::ExecutionOptions::EffortLevel>(value));
          })
      .def_prop_rw("use_spmd_partitioning",
                   &ExecutableBuildOptions::use_spmd_partitioning,
                   &ExecutableBuildOptions::set_use_spmd_partitioning)
      .def_prop_rw("use_auto_spmd_partitioning",
                   &ExecutableBuildOptions::use_auto_spmd_partitioning,
                   &ExecutableBuildOptions::set_use_auto_spmd_partitioning)
      .def_prop_rw(
          "auto_spmd_partitioning_mesh_shape",
          &ExecutableBuildOptions::auto_spmd_partitioning_mesh_shape,
          &ExecutableBuildOptions::set_auto_spmd_partitioning_mesh_shape)
      .def_prop_rw("auto_spmd_partitioning_mesh_ids",
                   &ExecutableBuildOptions::auto_spmd_partitioning_mesh_ids,
                   &ExecutableBuildOptions::set_auto_spmd_partitioning_mesh_ids)
      .def_prop_rw(
          "allow_spmd_sharding_propagation_to_parameters",
          [](const ExecutableBuildOptions& options) -> std::vector<bool> {
            return std::vector<bool>(
                options.allow_spmd_sharding_propagation_to_parameters().begin(),
                options.allow_spmd_sharding_propagation_to_parameters().end());
          },
          [](ExecutableBuildOptions& options, std::vector<bool> values) {
            absl::InlinedVector<bool, 1> v(values.begin(), values.end());
            options.set_allow_spmd_sharding_propagation_to_parameters(v);
          })
      .def_prop_rw(
          "allow_spmd_sharding_propagation_to_output",
          [](const ExecutableBuildOptions& options) -> std::vector<bool> {
            return std::vector<bool>(
                options.allow_spmd_sharding_propagation_to_output().begin(),
                options.allow_spmd_sharding_propagation_to_output().end());
          },
          [](ExecutableBuildOptions& options, std::vector<bool> values) {
            absl::InlinedVector<bool, 1> v(values.begin(), values.end());
            options.set_allow_spmd_sharding_propagation_to_output(v);
          })
      .def_prop_rw("use_shardy_partitioner",
                   &ExecutableBuildOptions::use_shardy_partitioner,
                   &ExecutableBuildOptions::set_use_shardy_partitioner);

  nb::enum_<OpSharding::Type> op_sharding_type(m, "OpSharding_Type",
                                               nb::is_arithmetic());
  op_sharding_type.value("REPLICATED", OpSharding::REPLICATED)
      .value("MAXIMAL", OpSharding::MAXIMAL)
      .value("MANUAL", OpSharding::MANUAL)
      .value("UNREDUCED", OpSharding::UNREDUCED)
      .value("TUPLE", OpSharding::TUPLE)
      .value("OTHER", OpSharding::OTHER)
      .value("UNKNOWN", OpSharding::UNKNOWN);

  nb::enum_<OpSharding::ShardGroupType> op_sharding_shard_group_type(
      m, "OpSharding_ShardGroupType");
  op_sharding_shard_group_type.value("AS", OpSharding::AS)
      .value("LIKE", OpSharding::LIKE);

  nb::class_<OpSharding> op_sharding(m, "OpSharding");
  op_sharding.attr("Type") = op_sharding_type;
  op_sharding.attr("ShardGroupType") = op_sharding_shard_group_type;
  op_sharding.def(nb::init<>())
      .def("__getstate__",
           [](const OpSharding& self) {
             std::string serialized = self.SerializeAsString();
             return nb::make_tuple(
                 nb::bytes(serialized.data(), serialized.size()));
           })
      .def("__setstate__",
           [](OpSharding* self, nb::tuple t) {
             new (self) OpSharding();
             nb::bytes serialized = nb::cast<nb::bytes>(t[0]);
             self->ParseFromArray(serialized.c_str(), serialized.size());
           })
      .def_prop_rw("type", &xla::OpSharding::type, &xla::OpSharding::set_type)
      .def_prop_rw("replicate_on_last_tile_dim",
                   &xla::OpSharding::replicate_on_last_tile_dim,
                   &xla::OpSharding::set_replicate_on_last_tile_dim)
      .def_prop_rw("is_shard_group", &xla::OpSharding::is_shard_group,
                   &xla::OpSharding::set_is_shard_group)
      .def_prop_rw("shard_group_id", &xla::OpSharding::shard_group_id,
                   &xla::OpSharding::set_shard_group_id)
      .def_prop_rw("shard_group_type", &xla::OpSharding::shard_group_type,
                   &xla::OpSharding::set_shard_group_type)
      .def("__repr__",
           [](const xla::OpSharding& self) { return self.DebugString(); })
      .def("ParseFromString",
           [](OpSharding& sharding, const nb::bytes& s) {
             sharding.ParseFromArray(s.c_str(), s.size());
           })
      .def("SerializeToString",
           [](const OpSharding& sharding) {
             std::string serialized = sharding.SerializeAsString();
             return nb::bytes(serialized.data(), serialized.size());
           })
      .def("clone",
           [](const OpSharding& sharding) { return OpSharding(sharding); });
  DefRepeatedProperty(op_sharding, "tile_assignment_dimensions",
                      &xla::OpSharding::mutable_tile_assignment_dimensions);
  DefRepeatedProperty(op_sharding, "tile_assignment_devices",
                      &xla::OpSharding::mutable_tile_assignment_devices);
  DefRepeatedProperty(op_sharding, "iota_reshape_dims",
                      &xla::OpSharding::mutable_iota_reshape_dims);
  DefRepeatedProperty(op_sharding, "iota_transpose_perm",
                      &xla::OpSharding::mutable_iota_transpose_perm);
  DefRepeatedProperty(op_sharding, "tuple_shardings",
                      &xla::OpSharding::mutable_tuple_shardings);
  DefRepeatedEnumProperty(op_sharding, "last_tile_dims",
                          &xla::OpSharding::mutable_last_tile_dims);

  nb::class_<HloSharding> hlo_sharding(m, "HloSharding");
  hlo_sharding
      .def_static("from_proto",
                  xla::ValueOrThrowWrapper(xla::HloSharding::FromProto))
      .def_static("from_string", xla::ValueOrThrowWrapper(xla::ParseSharding))
      .def_static(
          "tuple_sharding",
          [](xla::Shape shape,
             std::vector<xla::HloSharding> shardings) -> xla::HloSharding {
            return HloSharding::Tuple(shape, shardings);
          },
          "Constructs a tuple sharding.")
      .def_static(
          "iota_tile", xla::ValueOrThrowWrapper(IotaTileHelper),
          nb::arg("dims"),
          nb::arg("reshape_dims") = absl::Span<const int64_t>(),
          nb::arg("transpose_perm") = absl::Span<const int>(),
          nb::arg("subgroup_types") = absl::Span<const xla::OpSharding::Type>())
      .def_static("manual", [] { return HloSharding::Manual(); })
      .def_static("replicate", [] { return HloSharding::Replicate(); })
      .def_static("unreduced", [] { return HloSharding::Unreduced(); })
      .def_static("unknown", [] { return HloSharding::Unknown(); })
      .def_static(
          "subgroup_with_device_ordering",
          xla::ValueOrThrowWrapper(SubgroupWithTileAssignmentHelper),
          nb::arg("tile_assignment"),
          nb::arg("subgroup_types") = absl::Span<const xla::OpSharding::Type>())
      .def(
          "__eq__",
          [](const xla::HloSharding& a, const xla::HloSharding& b) {
            return a == b;
          },
          nb::is_operator(),
          nb::sig("def __eq__(self, other: object, /) -> bool"))
      .def(
          "__ne__",
          [](const xla::HloSharding& a, const xla::HloSharding& b) {
            return a != b;
          },
          nb::is_operator(),
          nb::sig("def __ne__(self, other: object, /) -> bool"))
      .def("__hash__",
           [](const xla::HloSharding& self) { return absl::HashOf(self); })
      .def("is_replicated", &xla::HloSharding::IsReplicated)
      .def("is_manual", &xla::HloSharding::IsManual)
      .def("is_unreduced", &xla::HloSharding::IsUnreduced)
      .def("is_unknown", &xla::HloSharding::IsUnknown)
      .def("is_tiled", &xla::HloSharding::IsTiled)
      .def("is_maximal", &xla::HloSharding::IsTileMaximal)
      .def("tile", [](const xla::HloSharding& self,
                      xla::Shape shape) { return self.TileShape(shape); })
      // tile_assignment.array() is computed using an internal cache,
      // which is why nb::lock_self() is required. It may be preferable to move
      // this locking into the TileAssignment class if we find it to race with
      // non-Python users of that class.
      .def(
          "tuple_elements",
          [](const xla::HloSharding& self) { return self.tuple_elements(); },
          nb::lock_self())
      .def(
          "num_devices",
          [](const xla::HloSharding& self) {
            return self.tile_assignment().num_elements();
          },
          nb::lock_self())
      .def(
          "num_dimensions",
          [](const xla::HloSharding& self) {
            return self.tile_assignment().num_dimensions();
          },
          nb::lock_self())
      .def("is_tile_assignment_iota",
           [](const xla::HloSharding& self) {
             return self.tile_assignment().iota().has_value();
           })
      .def(
          "tile_assignment_dimensions",
          [](const xla::HloSharding& self) {
            absl::Span<int64_t const> span =
                self.tile_assignment().dimensions();
            CHECK(span.data());
            return span;
          },
          nb::lock_self())
      .def(
          "tile_assignment_devices",
          [](const xla::HloSharding& self) {
            auto span =
                absl::MakeConstSpan(self.tile_assignment().array().data(),
                                    self.tile_assignment().num_elements());
            CHECK(span.data());
            return span;
          },
          nb::lock_self())
      .def("replicate_on_last_tile_dim",
           &xla::HloSharding::ReplicateOnLastTileDim)
      .def("subgroup_types", &xla::HloSharding::subgroup_types)
      .def("__repr__",
           [](const xla::HloSharding& self) { return self.ToString(); })
      .def("to_proto", &xla::HloSharding::ToProto)
      .def("get_axis_sizes", [](const xla::HloSharding& self) {
        // If returning the SmallVector, we encounter the error "unable to
        // convert function return value to a Python type!".
        mlir::SmallVector<int64_t> mesh_shape =
            xla::sdy::getAxisSizes(self.tile_assignment());
        return std::vector<int64_t>(mesh_shape.begin(), mesh_shape.end());
      });
}  // NOLINT(readability/fn_size)
}  // namespace xla
