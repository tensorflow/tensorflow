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

#include "xla/python/xla_compiler.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/hash/hash.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "third_party/nanobind/include/nanobind/nanobind.h"
#include "third_party/nanobind/include/nanobind/ndarray.h"
#include "third_party/nanobind/include/nanobind/stl/optional.h"  // IWYU pragma: keep
#include "third_party/nanobind/include/nanobind/stl/pair.h"  // IWYU pragma: keep
#include "third_party/nanobind/include/nanobind/stl/shared_ptr.h"  // IWYU pragma: keep
#include "third_party/nanobind/include/nanobind/stl/string.h"  // IWYU pragma: keep
#include "third_party/nanobind/include/nanobind/stl/string_view.h"  // IWYU pragma: keep
#include "third_party/nanobind/include/nanobind/stl/variant.h"  // IWYU pragma: keep
#include "third_party/nanobind/include/nanobind/stl/vector.h"  // IWYU pragma: keep
#include "xla/array.h"
#include "xla/client/executable_build_options.h"
#include "xla/client/xla_builder.h"
#include "xla/client/xla_computation.h"
#include "xla/debug_options_flags.h"
#include "xla/ffi/api/c_api.h"
#include "xla/ffi/ffi.h"
#include "xla/ffi/ffi_api.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_module_group.h"
#include "xla/hlo/ir/hlo_sharding.h"
#include "xla/layout.h"
#include "xla/layout_util.h"
#include "xla/literal.h"
#include "xla/pjrt/exceptions.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/pjrt/status_casters.h"
#include "xla/python/nb_absl_span.h"  // IWYU pragma: keep
#include "xla/python/nb_helpers.h"
#include "xla/python/nb_numpy.h"
#include "xla/python/py_client.h"
#include "xla/python/types.h"
#include "xla/service/call_inliner.h"
#include "xla/service/computation_placer.h"
#include "xla/service/custom_call_target_registry.h"
#include "xla/service/flatten_call_graph.h"
#include "xla/service/hlo.pb.h"
#include "xla/service/hlo_dce.h"
#include "xla/service/hlo_graph_dumper.h"
#include "xla/service/hlo_module_config.h"
#include "xla/service/hlo_parser.h"
#include "xla/service/hlo_pass_interface.h"
#include "xla/service/name_uniquer.h"
#include "xla/service/tuple_simplifier.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/statusor.h"
#include "xla/util.h"
#include "xla/xla.pb.h"
#include "xla/xla_data.pb.h"
#include "tsl/lib/strings/proto_serialization.h"
#include "tsl/platform/logging.h"

namespace nanobind {
namespace detail {

template <>
struct type_caster<xla::OpMetadata> {
 public:
  NB_TYPE_CASTER_FROM_PYTHON_ONLY(xla::OpMetadata,
                                  const_name("xla::OpMetadata"));

  bool from_python(handle h, uint8_t, cleanup_list*) {
    handle op_type = getattr(h, "op_type");
    if (!op_type.is_none()) {
      value.set_op_type(cast<std::string>(op_type));
    }
    handle op_name = getattr(h, "op_name");
    if (!op_name.is_none()) {
      value.set_op_name(cast<std::string>(op_name));
    }
    handle source_file = getattr(h, "source_file");
    if (!source_file.is_none()) {
      value.set_source_file(cast<std::string>(source_file));
    }
    handle source_line = getattr(h, "source_line");
    if (!source_line.is_none()) {
      value.set_source_line(cast<int32_t>(source_line));
    }
    return true;
  }
};

}  // namespace detail
}  // namespace nanobind

namespace xla {
namespace {

namespace nb = nanobind;

struct Uniquer {
  absl::Mutex mu;
  NameUniquer name_uniquer ABSL_GUARDED_BY(mu);
};

Uniquer* GetUniquer() {
  static Uniquer* uniquer = new Uniquer;
  return uniquer;
}

static std::string UniquifyName(const std::string& name) {
  Uniquer* uniquer = GetUniquer();
  absl::MutexLock lock(&uniquer->mu);
  return uniquer->name_uniquer.GetUniqueName(name);
}

// Converts a computation to a serialized HloModuleProto.
StatusOr<nb::bytes> GetComputationSerializedProto(
    const XlaComputation& computation) {
  std::string result;
  if (!tsl::SerializeToStringDeterministic(computation.proto(), &result)) {
    return Unknown("Failed to serialize the HloModuleProto.");
  }
  return nb::bytes(result.data(), result.size());
}

// Converts a hlo module to a serialized HloModuleProto.
StatusOr<nb::bytes> GetHloModuleSerializedProto(const HloModule& module) {
  std::string result;
  if (!tsl::SerializeToStringDeterministic(module.ToProto(), &result)) {
    return Unknown("Failed to serialize the HloModuleProto.");
  }
  return nb::bytes(result.data(), result.size());
}

// Converts a serialized HloModuleProto into a HloModule.
StatusOr<std::shared_ptr<HloModule>> HloModuleFromSerializedProto(
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

StatusOr<std::shared_ptr<HloModule>> GetHloModule(
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
StatusOr<std::string> GetComputationHloText(
    const XlaComputation& computation, bool print_large_constants = false) {
  TF_ASSIGN_OR_RETURN(std::shared_ptr<HloModule> hlo_module,
                      GetHloModule(computation));
  HloPrintOptions options;
  options = HloPrintOptions::ShortParsable();
  options.set_print_large_constants(print_large_constants);
  return hlo_module->ToString(options);
}

// Converts a computation to HLO dot graph form.
StatusOr<std::string> GetComputationHloDotGraph(
    const XlaComputation& computation) {
  TF_ASSIGN_OR_RETURN(std::shared_ptr<HloModule> hlo_module,
                      GetHloModule(computation));
  return RenderGraph(*hlo_module->entry_computation(), /*label=*/"",
                     hlo_module->config().debug_options(),
                     RenderedGraphFormat::kDot);
}

// Hashes the HLO module.
StatusOr<uint64_t> HashComputation(const XlaComputation& computation) {
  TF_ASSIGN_OR_RETURN(std::shared_ptr<HloModule> hlo_module,
                      GetHloModule(computation));
  return absl::HashOf(*hlo_module);
}
// Safe version of ShapeUtil::MakeShapeWithDenseLayout that fails gracefully on
// invalid input.
StatusOr<Shape> MakeShapeWithDenseLayout(
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
  } else {
    shape.clear_layout();
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
// In practice, `reshape_dims` often maps to the axises of user defined device
// mesh, and `transpose_perm` often maps to the user specification of how a
// tensor is partitioned based on the axes defined in the mesh, e.g. for a mesh
// of size 4x2x2 as AxBxC:
// PartitionSpec('A', 'B', 'C') corresponds to reshape_dims=[4,2,2],
// transpose_perm=[0,1,2] (no transpose)
// PartitionSpec('B', 'A', 'C') corresponds to reshape_dims=[4,2,2],
// transpose_perm=[1,0,2] (swap A and B)
StatusOr<HloSharding> IotaTileHelper(
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

// Registers a 'fn_capsule' as a custom call target.
// 'fn_capsule' must be a void* pointer encapsulated in a PyCapsule object.
// 'platform' is an XLA platform name, e.g., "Host" or "CUDA".
absl::Status PyRegisterCustomCallTarget(const std::string& fn_name,
                                        nb::capsule capsule,
                                        const std::string& platform,
                                        int api_version) {
  switch (api_version) {
    case 0:
      CustomCallTargetRegistry::Global()->Register(
          fn_name, static_cast<void*>(capsule.data()), platform);
      return absl::OkStatus();
    case 1:
      ffi::Ffi::RegisterStaticHandler(xla::ffi::GetXlaFfiApi(), fn_name,
                                      platform,
                                      reinterpret_cast<XLA_FFI_Handler*>(
                                          static_cast<void*>(capsule.data())));
      return absl::OkStatus();
    default:
      return absl::UnimplementedError(absl::StrFormat(
          "API version %d is not supported by RegisterCustomCallTarget. "
          "Supported versions are 0 and 1.",
          api_version));
  }
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

}  // namespace

void BuildXlaCompilerSubmodule(nb::module_& m) {
  // Shapes
  nb::class_<Layout> layout_class(m, "Layout");
  layout_class.def(nb::init<absl::Span<const int64_t>>())
      .def("minor_to_major",
           [](Layout layout) { return SpanToNbTuple(layout.minor_to_major()); })
      .def("__eq__", [](const Layout& layout,
                        const Layout& other) { return layout == other; })
      .def("__ne__", [](const Layout& layout,
                        const Layout& other) { return layout != other; })
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
        new (self) Layout(Layout::CreateFromProto(result));
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
      .def_static("array_shape",
                  xla::ValueOrThrowWrapper(
                      [](PrimitiveType type, nb::sequence dims_seq,
                         std::optional<nb::sequence> layout_seq,
                         std::optional<std::vector<bool>> dynamic_dimensions)
                          -> StatusOr<Shape> {
                        std::vector<int64_t> dims =
                            SequenceToVector<int64_t>(dims_seq);
                        if (layout_seq) {
                          std::vector<int64_t> layout =
                              SequenceToVector<int64_t>(*layout_seq);
                          return MakeShapeWithDenseLayout(type, dims, layout,
                                                          dynamic_dimensions);
                        } else {
                          return MakeShapeWithDenseLayout(
                              type, dims, std::nullopt, dynamic_dimensions);
                        }
                      }),
                  "Constructs an array shape.", nb::arg("type"),
                  nb::arg("dims"), nb::arg("layout").none() = std::nullopt,
                  nb::arg("dynamic_dimensions").none() = std::nullopt)
      .def_static(
          "array_shape",
          xla::ValueOrThrowWrapper(
              [](nb_dtype dtype, nb::sequence dims_seq,
                 std::optional<nb::sequence> layout_seq,
                 std::optional<std::vector<bool>> dynamic_dimensions)
                  -> StatusOr<Shape> {
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
           [](const Shape& shape) -> nb::tuple {
             return SpanToNbTuple(shape.dimensions());
           })
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
      .def("rank", &Shape::rank)
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
      .def("__eq__", [](const Shape& shape,
                        const Shape& other) { return shape == other; })
      .def("__ne__", [](const Shape& shape,
                        const Shape& other) { return shape != other; })
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
              *self->add_parameters() = param;
            }
            *self->mutable_result() = result;
          })
      .def("parameter_shapes",
           static_cast<const std::vector<Shape>& (ProgramShape::*)() const>(
               &ProgramShape::parameters))
      .def("result_shape", &ProgramShape::result)
      .def("__repr__", &ProgramShape::ToString);

  nb::class_<ShapeIndex>(m, "ShapeIndex")
      .def("__init__",
           [](ShapeIndex* self, const std::vector<int64_t>& v) {
             new (self) ShapeIndex(v.begin(), v.end());
           })
      .def("__repr__", &ShapeIndex::ToString)
      .def("__eq__", [](const ShapeIndex& shape_ind,
                        const ShapeIndex& other) { return shape_ind == other; })
      .def("__ne__", [](const ShapeIndex& shape_ind,
                        const ShapeIndex& other) { return shape_ind != other; })
      .def("__hash__",
           [](const ShapeIndex& shape_ind) { return absl::HashOf(shape_ind); });

  // Literals
  nb::class_<Literal>(m, "Literal").def("__repr__", &Literal::ToString);

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
        : comp{comp}, module{module} {}
    absl::string_view name() const { return comp->name(); }
    void render_html(const std::string& filename) {
      std::string html = xla::ValueOrThrow(RenderGraph(
          *comp, /*label=*/"", comp->parent()->config().debug_options(),
          RenderedGraphFormat::kHtml, HloRenderOptions()));
      xla::ThrowIfError(tsl::WriteStringToFile(
          tsl::Env::Default(), absl::StrCat(filename, ".html"), html));
    }

   private:
    const HloComputation* comp;
    // The module owns the computations: if its destructor is called, the
    // computations are freed. To prevent that from happening in cases where the
    // module Python object goes out of scope and gets garbage collected before
    // the computations, we keep a shared_ptr to the module that originated the
    // computation.
    const std::shared_ptr<HloModule> module;
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

  nb::class_<HloModuleGroup> hlo_module_group_class(m, "HloModuleGroup");
  hlo_module_group_class
      .def("__init__",
           [](HloModuleGroup* self, const std::string& name,
              const std::vector<std::shared_ptr<HloModule>>& hlo_modules) {
             std::vector<std::unique_ptr<HloModule>> modules;
             modules.reserve(hlo_modules.size());
             for (const auto& m : hlo_modules) {
               modules.push_back(m->Clone(/*suffix=*/""));
             }
             new (self) HloModuleGroup(name, std::move(modules));
           })
      .def_prop_ro("name", &HloModuleGroup::name)
      .def("to_string", &HloModuleGroup::ToString)
      .def("to_modules",
           [](HloModuleGroup& m) -> std::vector<std::shared_ptr<HloModule>> {
             std::vector<std::unique_ptr<HloModule>> modules =
                 m.ConsumeModules();
             std::vector<std::shared_ptr<HloModule>> shared_modules;
             shared_modules.reserve(modules.size());
             for (auto& module : modules) {
               shared_modules.push_back(std::move(module));
             }
             return shared_modules;
           });

  m.def("hlo_module_to_dot_graph",
        [](const HloModule& hlo_module) -> std::string {
          return xla::ValueOrThrow(RenderGraph(
              *hlo_module.entry_computation(), /*label=*/"",
              hlo_module.config().debug_options(), RenderedGraphFormat::kDot));
        });
  m.def(
      "hlo_module_cost_analysis",
      xla::ValueOrThrowWrapper([](PyClient* client, const HloModule& module)
                                   -> StatusOr<nb::dict> {
        TF_ASSIGN_OR_RETURN(auto analysis,
                            client->pjrt_client()->GetHloCostAnalysis());
        TF_RETURN_IF_ERROR(module.entry_computation()->Accept(analysis.get()));

        // Convert from HloCostAnalysis::Properties to a standard map.
        nb::dict ret;
        analysis->properties().ForEach([&](absl::string_view key, float val) {
          ret[nb::str(key.data(), key.size())] = nb::cast(val);
        });
        return ret;
      }));
  m.def("hlo_module_from_text",
        xla::ValueOrThrowWrapper([](const std::string& hlo_module_text)
                                     -> StatusOr<std::shared_ptr<HloModule>> {
          auto hlo_module =
              xla::ParseAndReturnUnverifiedModule(hlo_module_text);
          TF_RETURN_IF_ERROR(hlo_module.status());
          std::shared_ptr<HloModule> result(std::move(*hlo_module));
          return result;
        }));

  nb::class_<XlaOp> xla_op_class(m, "XlaOp");

  nb::class_<XlaBuilder>(m, "XlaBuilder")
      .def("__init__",
           [](XlaBuilder* self, const std::string& name) {
             new (self) XlaBuilder(UniquifyName(name));
           })
      // TODO(phawkins): delete capitalized names after updating callers.
      .def("Build",
           xla::ValueOrThrowWrapper(
               [](XlaBuilder& builder, std::optional<XlaOp> root) {
                 return root ? builder.Build(*root) : builder.Build();
               }),
           "Builds a computation from the contents of the builder.",
           nb::arg("root") = std::nullopt)
      .def("GetShape", xla::ValueOrThrowWrapper(&XlaBuilder::GetShape))
      .def("build",
           xla::ValueOrThrowWrapper(
               [](XlaBuilder& builder, std::optional<XlaOp> root) {
                 return root ? builder.Build(*root) : builder.Build();
               }),
           "Builds a computation from the contents of the builder.",
           nb::arg("root") = std::nullopt)
      .def("clear_op_metadata", &XlaBuilder::ClearOpMetadata)
      .def("get_shape", xla::ValueOrThrowWrapper(&XlaBuilder::GetShape))
      .def(
          "get_program_shape",
          [](const XlaBuilder& builder,
             std::optional<XlaOp> root) -> StatusOr<ProgramShape> {
            return root ? builder.GetProgramShape(*root)
                        : builder.GetProgramShape();
          },
          nb::arg("root") = std::nullopt)
      .def("is_constant", xla::ValueOrThrowWrapper(&XlaBuilder::IsConstant))
      .def("set_op_metadata", &XlaBuilder::SetOpMetadata)
      .def("set_sharding", &XlaBuilder::SetSharding)
      .def("clear_sharding", &XlaBuilder::ClearSharding)
      .def("set_frontend_attributes", &XlaBuilder::SetFrontendAttributes)
      .def("clear_frontend_attributes", &XlaBuilder::ClearFrontendAttributes)
      .def("setup_alias",
           [](XlaBuilder& builder, const std::vector<int64_t>& output_index,
              int64_t param_number, const std::vector<int64_t>& param_index) {
             builder.SetUpAlias(
                 ShapeIndex(output_index.begin(), output_index.end()),
                 param_number,
                 ShapeIndex(param_index.begin(), param_index.end()));
           });

  // Device assignments
  nb::class_<DeviceAssignment>(m, "DeviceAssignment")
      .def_static(
          "create",
          xla::ValueOrThrowWrapper([](nb::ndarray<int, nb::ndim<2>> array)
                                       -> StatusOr<DeviceAssignment> {
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
      .def("serialize", xla::ValueOrThrowWrapper([](const DeviceAssignment& da)
                                                     -> StatusOr<nb::bytes> {
             DeviceAssignmentProto proto;
             TF_RETURN_IF_ERROR(da.Serialize(&proto));
             std::string result;
             if (!tsl::SerializeToStringDeterministic(proto, &result)) {
               return Unknown("Failed to serialize the DeviceAssignmentProto.");
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
      // TODO(phawkins): the following fields exist for backward compatibility.
      // Remove them after JAX has been updated not to use them.
      .def_rw("tuple_arguments", &CompileOptions::parameter_is_tupled_arguments)
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

  // Custom-call targets.
  m.def(
      "register_custom_call_target",
      [](nb::object fn_name_py, nb::capsule capsule,
         const std::string& platform, const int api_version) {
        std::string fn_name;
        if (!nb::try_cast<std::string>(fn_name_py, fn_name)) {
          nb::bytes bytes = nb::cast<nb::bytes>(fn_name_py);
          fn_name = std::string(bytes.c_str(), bytes.size());
        }
        xla::ThrowIfError(PyRegisterCustomCallTarget(
            fn_name, std::move(capsule), platform, api_version));
      },
      nb::arg("fn_name"), nb::arg("capsule"), nb::arg("platform"),
      nb::arg("api_version") = 0);

  m.def(
      "custom_call_targets",
      [](const std::string& platform) -> nb::dict {
        nb::dict targets;
        for (const auto& [name, target] :
             CustomCallTargetRegistry::Global()->registered_symbols(platform)) {
          targets[nb::str(name.data(), name.size())] = nb::capsule(target);
        }

        for (const auto& [name, registration] :
             ffi::StaticRegisteredHandlers(platform)) {
          targets[nb::str(name.data(), name.size())] =
              nb::capsule(reinterpret_cast<void*>(registration.handler));
        }
        return targets;
      },
      nb::arg("platform"));

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
      .def_prop_rw("xla_gpu_enable_async_all_reduce",
                   &DebugOptions::xla_gpu_enable_async_all_reduce,
                   &DebugOptions::set_xla_gpu_enable_async_all_reduce)
      .def_prop_rw("xla_gpu_enable_async_all_gather",
                   &DebugOptions::xla_gpu_enable_async_all_gather,
                   &DebugOptions::set_xla_gpu_enable_async_all_gather)
      .def_prop_rw("xla_gpu_enable_async_collective_broadcast",
                   &DebugOptions::xla_gpu_enable_async_collective_broadcast,
                   &DebugOptions::set_xla_gpu_enable_async_collective_broadcast)
      .def_prop_rw("xla_gpu_enable_async_collective_permute",
                   &DebugOptions::xla_gpu_enable_async_collective_permute,
                   &DebugOptions::set_xla_gpu_enable_async_collective_permute)
      .def_prop_rw("xla_gpu_enable_async_all_to_all",
                   &DebugOptions::xla_gpu_enable_async_all_to_all,
                   &DebugOptions::set_xla_gpu_enable_async_all_to_all)
      .def_prop_rw("xla_gpu_enable_async_reduce_scatter",
                   &DebugOptions::xla_gpu_enable_async_reduce_scatter,
                   &DebugOptions::set_xla_gpu_enable_async_reduce_scatter);

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
          });

  nb::enum_<OpSharding::Type> op_sharding_type(m, "OpSharding_Type");
  op_sharding_type.value("REPLICATED", OpSharding::REPLICATED)
      .value("MAXIMAL", OpSharding::MAXIMAL)
      .value("MANUAL", OpSharding::MANUAL)
      .value("TUPLE", OpSharding::TUPLE)
      .value("OTHER", OpSharding::OTHER)
      .value("UNKNOWN", OpSharding::UNKNOWN);

  nb::enum_<OpSharding::ShardGroupType> op_sharding_shard_group_type(
      m, "OpSharding_ShardGroupType");
  op_sharding_shard_group_type.value("AS", OpSharding::AS)
      .value("LIKE", OpSharding::LIKE);

  nb::class_<OpSharding> op_sharding(m, "OpSharding");
  op_sharding
      .def_prop_ro_static(
          "Type",
          [op_sharding_type](const nb::object&) { return op_sharding_type; })
      .def_prop_ro_static("ShardGroupType",
                          [op_sharding_shard_group_type](const nb::object&) {
                            return op_sharding_shard_group_type;
                          })
      .def(nb::init<>())
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
  DefRepeatedProperty(op_sharding, "last_tile_dims",
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
      .def_static("unknown", [] { return HloSharding::Unknown(); })
      .def("__eq__", [](const xla::HloSharding& a,
                        const xla::HloSharding& b) { return a == b; })
      .def("__hash__",
           [](const xla::HloSharding& self) { return absl::HashOf(self); })
      .def("is_replicated", &xla::HloSharding::IsReplicated)
      .def("is_manual", &xla::HloSharding::IsManual)
      .def("is_unknown", &xla::HloSharding::IsUnknown)
      .def("is_tiled", &xla::HloSharding::IsTiled)
      .def("tile", [](const xla::HloSharding& self,
                      xla::Shape shape) { return self.TileShape(shape); })
      .def("tuple_elements",
           [](const xla::HloSharding& self) { return self.tuple_elements(); })
      .def("num_devices",
           [](const xla::HloSharding& self) {
             return self.tile_assignment().num_elements();
           })
      .def("num_dimensions",
           [](const xla::HloSharding& self) {
             return self.tile_assignment().num_dimensions();
           })
      .def("tile_assignment_dimensions",
           [](const xla::HloSharding& self) {
             absl::Span<int64_t const> span =
                 self.tile_assignment().dimensions();
             CHECK(span.data());
             return span;
           })
      .def("tile_assignment_devices",
           [](const xla::HloSharding& self) {
             auto span =
                 absl::MakeConstSpan(self.tile_assignment().array().data(),
                                     self.tile_assignment().num_elements());
             CHECK(span.data());
             return span;
           })
      .def("replicate_on_last_tile_dim",
           &xla::HloSharding::ReplicateOnLastTileDim)
      .def("subgroup_types", &xla::HloSharding::subgroup_types)
      .def("__repr__",
           [](const xla::HloSharding& self) { return self.ToString(); })
      .def("to_proto", &xla::HloSharding::ToProto);

  nb::class_<FrontendAttributes> frontend_attributes(m, "FrontendAttributes");
  frontend_attributes.def(nb::init<>())
      .def("__setitem__",
           [](FrontendAttributes* attr, std::string key, std::string value) {
             (*attr->mutable_map())[key] = value;
           });

  nb::enum_<PrecisionConfig::Precision>(m, "PrecisionConfig_Precision")
      .value("DEFAULT", PrecisionConfig::DEFAULT)
      .value("HIGH", PrecisionConfig::HIGH)
      .value("HIGHEST", PrecisionConfig::HIGHEST);

  nb::enum_<FftType>(m, "FftType")
      .value("FFT", FftType::FFT)
      .value("IFFT", FftType::IFFT)
      .value("RFFT", FftType::RFFT)
      .value("IRFFT", FftType::IRFFT);

  // Hlo Module Passes
  nb::class_<HloPassInterface> hlo_pass_interface(m, "HloPassInterface");
  hlo_pass_interface.def_prop_ro("name", &HloPassInterface::name)
      .def("is_pass_pipeline", &HloPassInterface::IsPassPipeline)
      .def("run",
           [](HloPassInterface& pass, HloModule* module) -> bool {
             return xla::ValueOrThrow(pass.Run(module));
           })
      .def("run_on_module_group",
           [](HloPassInterface& pass, HloModuleGroup* module_group) -> bool {
             return xla::ValueOrThrow(pass.RunOnModuleGroup(module_group));
           });

  nb::class_<HloDCE, HloPassInterface>(m, "HloDCE").def(nb::init<>());
  nb::class_<CallInliner, HloPassInterface>(m, "CallInliner").def(nb::init<>());
  nb::class_<FlattenCallGraph, HloPassInterface>(m, "FlattenCallGraph")
      .def(nb::init<>());
  nb::class_<TupleSimplifier, HloPassInterface>(m, "TupleSimplifier")
      .def(nb::init<>());
}  // NOLINT(readability/fn_size)
}  // namespace xla
