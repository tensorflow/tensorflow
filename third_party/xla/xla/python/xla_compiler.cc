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
#include "pybind11/attr.h"  // from @pybind11
#include "pybind11/cast.h"  // from @pybind11
#include "pybind11/numpy.h"  // from @pybind11
#include "pybind11/pybind11.h"  // from @pybind11
#include "pybind11/pytypes.h"  // from @pybind11
#include "pybind11/stl_bind.h"  // from @pybind11
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
#include "xla/pjrt/exceptions.h"
#include "xla/pjrt/status_casters.h"
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

namespace pybind11 {
namespace detail {

template <>
struct type_caster<xla::OpMetadata> {
 public:
  PYBIND11_TYPE_CASTER(xla::OpMetadata, _("xla::OpMetadata"));

  // PyObject -> C++ conversion.
  bool load(handle handle, bool) {
    pybind11::handle op_type = getattr(handle, "op_type");
    if (!op_type.is_none()) {
      value.set_op_type(op_type.cast<std::string>());
    }
    pybind11::handle op_name = getattr(handle, "op_name");
    if (!op_name.is_none()) {
      value.set_op_name(op_name.cast<std::string>());
    }
    pybind11::handle source_file = getattr(handle, "source_file");
    if (!source_file.is_none()) {
      value.set_source_file(source_file.cast<std::string>());
    }
    pybind11::handle source_line = getattr(handle, "source_line");
    if (!source_line.is_none()) {
      value.set_source_line(source_line.cast<int32_t>());
    }
    return true;
  }
};

}  // namespace detail
}  // namespace pybind11

namespace xla {
namespace {

namespace py = pybind11;

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
StatusOr<py::bytes> GetComputationSerializedProto(
    const XlaComputation& computation) {
  std::string result;
  if (!tsl::SerializeToStringDeterministic(computation.proto(), &result)) {
    return Unknown("Failed to serialize the HloModuleProto.");
  }
  return py::bytes(result);
}

// Converts a hlo module to a serialized HloModuleProto.
StatusOr<py::bytes> GetHloModuleSerializedProto(const HloModule& module) {
  std::string result;
  if (!tsl::SerializeToStringDeterministic(module.ToProto(), &result)) {
    return Unknown("Failed to serialize the HloModuleProto.");
  }
  return py::bytes(result);
}

// Converts a serialized HloModuleProto into a HloModule.
StatusOr<std::shared_ptr<HloModule>> HloModuleFromSerializedProto(
    const py::bytes& bytes) {
  HloModuleProto proto;
  proto.ParseFromString(bytes);
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

// Registers a 'fn_capsule' as a CPU custom call target.
// 'fn_capsule' must be a void* pointer encapsulated in a PyCapsule object,
// with name "xla._CUSTOM_CALL_TARGET".
// 'platform' is an XLA platform name, e.g., "Host" or "CUDA".
absl::Status PyRegisterCustomCallTarget(const std::string& fn_name,
                                        py::capsule capsule,
                                        const std::string& platform,
                                        int api_version) {
  static const char* const kName = "xla._CUSTOM_CALL_TARGET";
  if (absl::string_view(capsule.name()) != kName) {
    return InvalidArgument(
        "Argument to RegisterCustomCallTarget was not a "
        "xla._CUSTOM_CALL_TARGET capsule.");
  }
  switch (api_version) {
    case 0:
      CustomCallTargetRegistry::Global()->Register(
          fn_name, static_cast<void*>(capsule), platform);
      return absl::OkStatus();
    case 1:
      ffi::Ffi::RegisterStaticHandler(
          xla::ffi::GetXlaFfiApi(), fn_name, platform,
          reinterpret_cast<XLA_FFI_Handler*>(static_cast<void*>(capsule)));
      return absl::OkStatus();
    default:
      return absl::UnimplementedError(absl::StrFormat(
          "API version %d is not supported by RegisterCustomCallTarget. "
          "Supported versions are 0 and 1.",
          api_version));
  }
}

template <typename T, typename Container>
void DefRepeatedProperty(py::class_<T>& cls, const char* name,
                         Container* (T::*getter)()) {
  cls.def_property(
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

void BuildXlaCompilerSubmodule(py::module& m) {
  // Shapes
  py::class_<Layout> layout_class(m, "Layout");
  layout_class
      .def(py::init([](absl::Span<const int64_t> minor_to_major) {
        return std::make_unique<Layout>(minor_to_major);
      }))
      .def("minor_to_major",
           [](Layout layout) { return SpanToTuple(layout.minor_to_major()); })
      .def("__eq__", [](const Layout& layout,
                        const Layout& other) { return layout == other; })
      .def("__ne__", [](const Layout& layout,
                        const Layout& other) { return layout != other; })
      .def("__hash__",
           [](const Layout& layout) { return absl::HashOf(layout); })
      .def("to_string", &Layout::ToString)
      .def(py::pickle(
          [](const Layout& self) -> py::tuple {
            auto proto = self.ToProto();
            std::string result;
            if (!tsl::SerializeToStringDeterministic(proto, &result)) {
              // throw converted by PyBind to a Python RuntimeError.
              throw XlaRuntimeError(
                  absl::StrCat("Layout.py_pickle: ",
                               "SerializeToStringDeterministic failed"));
            }
            return py::make_tuple(py::bytes(result));
          },
          [](py::tuple t) {
            LayoutProto result;
            result.ParseFromString(t[0].cast<std::string>());
            return Layout::CreateFromProto(result);
          }));

  py::class_<Shape> shape_class(m, "Shape");
  shape_class
      .def(py::init([](const std::string& s) {
        return std::make_unique<Shape>(ValueOrThrow(ParseShape(s)));
      }))
      .def_static(
          "tuple_shape",
          [](std::vector<Shape> shapes) -> Shape {
            return ShapeUtil::MakeTupleShape(shapes);
          },
          "Constructs a tuple shape.")
      .def_static("array_shape",
                  xla::ValueOrThrowWrapper(
                      [](PrimitiveType type, py::object dims_seq,
                         std::optional<py::object> layout_seq,
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
                  "Constructs an array shape.", py::arg("type"),
                  py::arg("dims"), py::arg("layout") = std::nullopt,
                  py::arg("dynamic_dimensions") = std::nullopt)
      .def_static(
          "array_shape",
          xla::ValueOrThrowWrapper(
              [](py::dtype dtype, py::object dims_seq,
                 std::optional<py::object> layout_seq,
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
          "Constructs an array shape.", py::arg("type"), py::arg("dims"),
          py::arg("layout") = std::nullopt,
          py::arg("dynamic_dimensions") = std::nullopt)
      .def_static("token_shape", []() { return ShapeUtil::MakeTokenShape(); })
      .def_static(
          "scalar_shape",
          [](PrimitiveType type) -> Shape {
            return ShapeUtil::MakeScalarShape(type);
          },
          "Constructs a scalar shape.", py::arg("type"))
      .def_static(
          "scalar_shape",
          [](py::dtype dtype) -> Shape {
            PrimitiveType type = xla::ValueOrThrow(DtypeToPrimitiveType(dtype));
            return ShapeUtil::MakeScalarShape(type);
          },
          "Constructs a scalar shape.", py::arg("type"))
      .def("dimensions",
           [](const Shape& shape) -> py::tuple {
             return SpanToTuple(shape.dimensions());
           })
      .def("layout",
           [](const Shape& shape) -> Layout { return shape.layout(); })
      .def("xla_element_type", &Shape::element_type)
      .def("element_type",
           [](const Shape& shape) {
             return xla::ValueOrThrow(
                 PrimitiveTypeToDtype(shape.element_type()));
           })
      .def("numpy_dtype",
           [](const Shape& shape) {
             if (shape.IsTuple()) {
               return py::dtype("O");
             }
             return xla::ValueOrThrow(
                 PrimitiveTypeToDtype(shape.element_type()));
           })
      .def("is_tuple", &Shape::IsTuple)
      .def("is_array", &Shape::IsArray)
      .def("is_token", &Shape::IsToken)
      .def("is_static", &Shape::is_static)
      .def("is_dynamic", &Shape::is_dynamic)
      .def("is_dynamic_dimension", &Shape::is_dynamic_dimension,
           py::arg("dimension"))
      .def("set_dynamic_dimension", &Shape::set_dynamic_dimension,
           py::arg("dimension"), py::arg("is_dynamic"))
      .def("rank", &Shape::rank)
      .def("to_serialized_proto",
           [](const Shape& shape) {
             ShapeProto proto = shape.ToProto();
             return py::bytes(proto.SerializeAsString());
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

  py::class_<ProgramShape>(m, "ProgramShape")
      .def(py::init(
          [](absl::Span<const Shape> params, Shape result) -> ProgramShape {
            ProgramShape program_shape;
            for (const Shape& param : params) {
              *program_shape.add_parameters() = param;
            }
            *program_shape.mutable_result() = result;
            return program_shape;
          }))
      .def("parameter_shapes",
           static_cast<const std::vector<Shape>& (ProgramShape::*)() const>(
               &ProgramShape::parameters))
      .def("result_shape", &ProgramShape::result)
      .def("__repr__", &ProgramShape::ToString);

  py::class_<ShapeIndex>(m, "ShapeIndex")
      .def(py::init([](const std::vector<int64_t>& v) {
        return std::make_unique<ShapeIndex>(v.begin(), v.end());
      }))
      .def("__repr__", &ShapeIndex::ToString)
      .def("__eq__", [](const ShapeIndex& shape_ind,
                        const ShapeIndex& other) { return shape_ind == other; })
      .def("__ne__", [](const ShapeIndex& shape_ind,
                        const ShapeIndex& other) { return shape_ind != other; })
      .def("__hash__",
           [](const ShapeIndex& shape_ind) { return absl::HashOf(shape_ind); });

  // Literals
  py::class_<Literal, std::shared_ptr<Literal>>(m, "Literal")
      .def("__repr__", &Literal::ToString);

  py::class_<XlaComputation>(m, "XlaComputation")
      .def(py::init([](const py::bytes& serialized_hlo_module_proto)
                        -> std::unique_ptr<XlaComputation> {
        HloModuleProto proto;
        proto.ParseFromString(std::string(serialized_hlo_module_proto));
        return std::make_unique<XlaComputation>(proto);
      }))
      .def("get_hlo_module", xla::ValueOrThrowWrapper(GetHloModule))
      .def("program_shape",
           xla::ValueOrThrowWrapper(&XlaComputation::GetProgramShape))
      .def("name", &XlaComputation::name)
      .def("as_serialized_hlo_module_proto",
           xla::ValueOrThrowWrapper(GetComputationSerializedProto))
      .def("as_hlo_text", xla::ValueOrThrowWrapper(GetComputationHloText),
           py::arg("print_large_constants") = false)
      .def("as_hlo_dot_graph",
           xla::ValueOrThrowWrapper(GetComputationHloDotGraph))
      .def("hash", xla::ValueOrThrowWrapper(HashComputation))
      .def("as_hlo_module", xla::ValueOrThrowWrapper(GetHloModule));

  py::class_<HloPrintOptions> hlo_print_options_class(m, "HloPrintOptions");
  hlo_print_options_class.def(py::init<>())
      .def_static("short_parsable", &HloPrintOptions::ShortParsable)
      .def_static("canonical", &HloPrintOptions::Canonical)
      .def_static("fingerprint", &HloPrintOptions::Fingerprint)
      .def_property("print_large_constants",
                    &HloPrintOptions::print_large_constants,
                    &HloPrintOptions::set_print_large_constants)
      .def_property("print_metadata", &HloPrintOptions::print_metadata,
                    &HloPrintOptions::set_print_metadata)
      .def_property("print_backend_config",
                    &HloPrintOptions::print_backend_config,
                    &HloPrintOptions::set_print_backend_config)
      .def_property("print_result_shape", &HloPrintOptions::print_result_shape,
                    &HloPrintOptions::set_print_result_shape)
      .def_property("print_operand_shape",
                    &HloPrintOptions::print_operand_shape,
                    &HloPrintOptions::set_print_operand_shape)
      .def_property("print_operand_names",
                    &HloPrintOptions::print_operand_names,
                    &HloPrintOptions::set_print_operand_names)
      .def_property("print_ids", &HloPrintOptions::print_ids,
                    &HloPrintOptions::set_print_ids)
      .def_property("print_extra_attributes",
                    &HloPrintOptions::print_extra_attributes,
                    &HloPrintOptions::set_print_extra_attributes)
      .def_property("print_program_shape",
                    &HloPrintOptions::print_program_shape,
                    &HloPrintOptions::set_print_program_shape)
      .def_property("print_percent", &HloPrintOptions::print_percent,
                    &HloPrintOptions::set_print_percent)
      .def_property("print_control_dependencies",
                    &HloPrintOptions::print_control_dependencies,
                    &HloPrintOptions::set_print_control_dependencies)
      .def_property("compact_operands", &HloPrintOptions::compact_operands,
                    &HloPrintOptions::set_compact_operands)
      .def_property("include_layout_in_shapes",
                    &HloPrintOptions::include_layout_in_shapes,
                    &HloPrintOptions::set_include_layout_in_shapes)
      .def_property("canonicalize_instruction_names",
                    &HloPrintOptions::canonicalize_instruction_names,
                    &HloPrintOptions::set_canonicalize_instruction_names)
      .def_property("canonicalize_computations",
                    &HloPrintOptions::canonicalize_computations,
                    &HloPrintOptions::set_canonicalize_computations)
      .def_property("indent_amount", &HloPrintOptions::indent_amount,
                    &HloPrintOptions::set_indent_amount)
      .def_property("is_in_nested_computation",
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

  py::class_<ComputationWrapper, std::shared_ptr<ComputationWrapper>>
      hlo_computation_class(m, "HloComputation");

  hlo_computation_class.def_property_readonly("name", &ComputationWrapper::name)
      .def("render_html", &ComputationWrapper::render_html);

  py::class_<HloModule, std::shared_ptr<HloModule>> hlo_module_class(
      m, "HloModule");
  hlo_module_class.def_property_readonly("name", &HloModule::name)
      .def(
          "to_string",
          static_cast<std::string (HloModule::*)(const HloPrintOptions&) const>(
              &HloModule::ToString),
          py::arg("options") = HloPrintOptions())
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
      .def_property_readonly(
          "spmd_output_sharding",
          [](const HloModule& m) -> std::optional<xla::OpSharding> {
            if (!m.has_spmd_output_sharding()) return std::nullopt;
            return m.spmd_output_sharding().ToProto();
          })
      .def_property_readonly(
          "spmd_parameters_shardings",
          [](const HloModule& m)
              -> std::optional<std::vector<xla::OpSharding>> {
            if (!m.has_spmd_parameters_shardings()) return std::nullopt;
            std::vector<xla::OpSharding> param_shardings;
            for (const auto& parameter_sharding :
                 m.spmd_parameters_shardings()) {
              param_shardings.push_back(parameter_sharding.ToProto());
            }
            return param_shardings;
          });

  py::class_<HloModuleGroup, std::shared_ptr<HloModuleGroup>>
      hlo_module_group_class(m, "HloModuleGroup");
  hlo_module_group_class
      .def(py::init(
          [](const std::string& name,
             const std::vector<std::shared_ptr<HloModule>>& hlo_modules)
              -> std::shared_ptr<HloModuleGroup> {
            std::vector<std::unique_ptr<HloModule>> modules;
            modules.reserve(hlo_modules.size());
            for (const auto& m : hlo_modules) {
              modules.push_back(m->Clone(/*suffix=*/""));
            }
            return std::make_shared<HloModuleGroup>(name, std::move(modules));
          }))
      .def_property_readonly("name", &HloModuleGroup::name)
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
  m.def("hlo_module_cost_analysis",
        xla::ValueOrThrowWrapper(
            [](PyClient* client, const HloModule& module)
                -> StatusOr<absl::flat_hash_map<std::string, float>> {
              TF_ASSIGN_OR_RETURN(auto analysis,
                                  client->pjrt_client()->GetHloCostAnalysis());
              TF_RETURN_IF_ERROR(
                  module.entry_computation()->Accept(analysis.get()));

              // Convert from HloCostAnalysis::Properties to a standard map.
              absl::flat_hash_map<std::string, float> ret;
              analysis->properties().ForEach(
                  [&](absl::string_view key, float val) { ret[key] = val; });
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

  py::class_<XlaOp> xla_op_class(m, "XlaOp");

  py::class_<XlaBuilder>(m, "XlaBuilder")
      .def(py::init([](const std::string& name) -> std::unique_ptr<XlaBuilder> {
        return std::make_unique<XlaBuilder>(UniquifyName(name));
      }))
      // TODO(phawkins): delete capitalized names after updating callers.
      .def("Build",
           xla::ValueOrThrowWrapper(
               [](XlaBuilder& builder, std::optional<XlaOp> root) {
                 return root ? builder.Build(*root) : builder.Build();
               }),
           "Builds a computation from the contents of the builder.",
           py::arg("root") = std::nullopt)
      .def("GetShape", xla::ValueOrThrowWrapper(&XlaBuilder::GetShape))
      .def("build",
           xla::ValueOrThrowWrapper(
               [](XlaBuilder& builder, std::optional<XlaOp> root) {
                 return root ? builder.Build(*root) : builder.Build();
               }),
           "Builds a computation from the contents of the builder.",
           py::arg("root") = std::nullopt)
      .def("clear_op_metadata", &XlaBuilder::ClearOpMetadata)
      .def("get_shape", xla::ValueOrThrowWrapper(&XlaBuilder::GetShape))
      .def(
          "get_program_shape",
          [](const XlaBuilder& builder,
             std::optional<XlaOp> root) -> StatusOr<ProgramShape> {
            return root ? builder.GetProgramShape(*root)
                        : builder.GetProgramShape();
          },
          py::arg("root") = std::nullopt)
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
  py::class_<DeviceAssignment>(m, "DeviceAssignment")
      .def_static(
          "create",
          xla::ValueOrThrowWrapper(
              [](py::array_t<int> array) -> StatusOr<DeviceAssignment> {
                if (array.ndim() != 2) {
                  return InvalidArgument(
                      "Argument to DeviceAssignment constructor must be a "
                      "2D array, received an %dD array.",
                      array.ndim());
                }
                DeviceAssignment result(array.shape(0), array.shape(1));
                for (int i = 0; i < array.shape(0); ++i) {
                  for (int j = 0; j < array.shape(1); ++j) {
                    result(i, j) = array.at(i, j);
                  }
                }
                return result;
              }))
      .def("replica_count", &DeviceAssignment::replica_count)
      .def("computation_count", &DeviceAssignment::computation_count)
      .def("__repr__", &DeviceAssignment::ToString)
      .def("serialize", xla::ValueOrThrowWrapper([](const DeviceAssignment& da)
                                                     -> StatusOr<py::bytes> {
             DeviceAssignmentProto proto;
             TF_RETURN_IF_ERROR(da.Serialize(&proto));
             std::string result;
             if (!tsl::SerializeToStringDeterministic(proto, &result)) {
               return Unknown("Failed to serialize the DeviceAssignmentProto.");
             }
             return py::bytes(result);
           }));

  py::class_<CompileOptions> compile_options(m, "CompileOptions");
  compile_options
      .def(py::init([]() -> CompileOptions {
        CompileOptions options;
        DebugOptions* debug_options =
            options.executable_build_options.mutable_debug_options();
        // Sets fast-math-disabling default options expected by JAX.
        debug_options->set_xla_cpu_enable_fast_min_max(false);
        debug_options->set_xla_gpu_enable_fast_min_max(false);
        return options;
      }))
      .def(py::pickle(
          [](const CompileOptions& self) -> py::tuple {
            auto proto = ValueOrThrow(self.ToProto());
            std::string result;
            if (!tsl::SerializeToStringDeterministic(proto, &result)) {
              // throw converted by PyBind to a Python RuntimeError.
              throw XlaRuntimeError(
                  absl::StrCat("CompileOptions.py_pickle: ",
                               "SerializeToStringDeterministic failed"));
            }
            return py::make_tuple(py::bytes(result));
          },
          [](py::tuple t) {
            CompileOptionsProto result;
            result.ParseFromString(t[0].cast<std::string>());
            return ValueOrThrow(CompileOptions::FromProto(result));
          }))
      .def("SerializeAsString",
           [](const CompileOptions& self) -> py::bytes {
             auto proto = ValueOrThrow(self.ToProto());
             std::string result;
             if (!tsl::SerializeToStringDeterministic(proto, &result)) {
               // throw converted by PyBind to a Python RuntimeError.
               throw XlaRuntimeError(
                   absl::StrCat("CompileOptions.SerializeAsString: ",
                                "SerializeToStringDeterministic failed"));
             }
             return py::bytes(result);
           })
      .def_static("ParseFromString",
                  [](py::bytes s) {
                    CompileOptionsProto result;
                    result.ParseFromString(s);
                    return ValueOrThrow(CompileOptions::FromProto(result));
                  })
      .def_readwrite("argument_layouts", &CompileOptions::argument_layouts)
      .def_readwrite("parameter_is_tupled_arguments",
                     &CompileOptions::parameter_is_tupled_arguments)
      .def_readwrite("compile_portable_executable",
                     &CompileOptions::compile_portable_executable)
      .def_readonly("executable_build_options",
                    &CompileOptions::executable_build_options)
      .def_readwrite("env_option_overrides",
                     &CompileOptions::env_option_overrides)
      // TODO(phawkins): the following fields exist for backward compatibility.
      // Remove them after JAX has been updated not to use them.
      .def_readwrite("tuple_arguments",
                     &CompileOptions::parameter_is_tupled_arguments)
      .def_property(
          "num_replicas",
          [](const CompileOptions& options) {
            return options.executable_build_options.num_replicas();
          },
          [](CompileOptions& options, int num_replicas) {
            options.executable_build_options.set_num_replicas(num_replicas);
          })
      .def_property(
          "num_partitions",
          [](const CompileOptions& options) {
            return options.executable_build_options.num_partitions();
          },
          [](CompileOptions& options, int num_partitions) {
            options.executable_build_options.set_num_partitions(num_partitions);
          })
      .def_property(
          "profile_version",
          [](const CompileOptions& options) { return options.profile_version; },
          [](CompileOptions& options, int64_t profile_version) {
            options.profile_version = profile_version;
          })
      .def_property(
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
      [](const std::string& fn_name, py::capsule capsule,
         const std::string& platform, const int api_version) {
        xla::ThrowIfError(PyRegisterCustomCallTarget(
            fn_name, std::move(capsule), platform, api_version));
      },
      py::arg("fn_name"), py::arg("capsule"), py::arg("platform"),
      py::arg("api_version") = 0);

  py::class_<DebugOptions>(m, "DebugOptions")
      .def("__repr__", &DebugOptions::DebugString)
      .def_property("xla_backend_optimization_level",
                    &DebugOptions::xla_backend_optimization_level,
                    &DebugOptions::set_xla_backend_optimization_level)
      .def_property("xla_cpu_enable_fast_math",
                    &DebugOptions::xla_cpu_enable_fast_math,
                    &DebugOptions::set_xla_cpu_enable_fast_math)
      .def_property("xla_cpu_enable_xprof_traceme",
                    &DebugOptions::xla_cpu_enable_xprof_traceme,
                    &DebugOptions::set_xla_cpu_enable_xprof_traceme)
      .def_property("xla_cpu_fast_math_honor_infs",
                    &DebugOptions::xla_cpu_fast_math_honor_infs,
                    &DebugOptions::set_xla_cpu_fast_math_honor_infs)
      .def_property("xla_cpu_fast_math_honor_nans",
                    &DebugOptions::xla_cpu_fast_math_honor_nans,
                    &DebugOptions::set_xla_cpu_fast_math_honor_nans)
      .def_property("xla_cpu_fast_math_honor_division",
                    &DebugOptions::xla_cpu_fast_math_honor_division,
                    &DebugOptions::set_xla_cpu_fast_math_honor_division)
      .def_property("xla_cpu_fast_math_honor_functions",
                    &DebugOptions::xla_cpu_fast_math_honor_functions,
                    &DebugOptions::set_xla_cpu_fast_math_honor_functions)
      .def_property("xla_detailed_logging", &DebugOptions::xla_detailed_logging,
                    &DebugOptions::set_xla_detailed_logging)
      .def_property("xla_enable_dumping", &DebugOptions::xla_enable_dumping,
                    &DebugOptions::set_xla_enable_dumping)
      .def_property("xla_gpu_enable_fast_min_max",
                    &DebugOptions::xla_gpu_enable_fast_min_max,
                    &DebugOptions::set_xla_gpu_enable_fast_min_max)
      .def_property("xla_gpu_dump_autotune_results_to",
                    &DebugOptions::xla_gpu_dump_autotune_results_to,
                    [](DebugOptions* self, std::string value) {
                      self->set_xla_gpu_dump_autotune_results_to(value);
                    })
      .def_property("xla_gpu_load_autotune_results_from",
                    &DebugOptions::xla_gpu_load_autotune_results_from,
                    [](DebugOptions* self, std::string value) {
                      self->set_xla_gpu_load_autotune_results_from(value);
                    })
      .def_property("xla_gpu_cuda_data_dir",
                    &DebugOptions::xla_gpu_cuda_data_dir,
                    [](DebugOptions* self, std::string value) {
                      self->set_xla_gpu_cuda_data_dir(value);
                    })
      .def_property("xla_llvm_disable_expensive_passes",
                    &DebugOptions::xla_llvm_disable_expensive_passes,
                    &DebugOptions::set_xla_llvm_disable_expensive_passes)
      .def_property(
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
      .def_property(
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
      .def_property("xla_test_all_input_layouts",
                    &DebugOptions::xla_test_all_input_layouts,
                    &DebugOptions::set_xla_test_all_input_layouts)
      .def_property("xla_force_host_platform_device_count",
                    &DebugOptions::xla_force_host_platform_device_count,
                    &DebugOptions::set_xla_force_host_platform_device_count)
      .def_property("xla_dump_to", &DebugOptions::xla_dump_to,
                    [](DebugOptions* self, std::string value) {
                      self->set_xla_dump_to(value);
                    })
      .def_property("xla_dump_hlo_module_re",
                    &DebugOptions::xla_dump_hlo_module_re,
                    [](DebugOptions* self, std::string value) {
                      self->set_xla_dump_hlo_module_re(value);
                    })
      .def_property("xla_dump_hlo_pass_re", &DebugOptions::xla_dump_hlo_pass_re,
                    [](DebugOptions* self, std::string value) {
                      self->set_xla_dump_hlo_pass_re(value);
                    })
      .def_property("xla_dump_hlo_as_text", &DebugOptions::xla_dump_hlo_as_text,
                    &DebugOptions::set_xla_dump_hlo_as_text)
      .def_property("xla_dump_hlo_as_proto",
                    &DebugOptions::xla_dump_hlo_as_proto,
                    &DebugOptions::set_xla_dump_hlo_as_proto)
      .def_property("xla_dump_hlo_as_dot", &DebugOptions::xla_dump_hlo_as_dot,
                    &DebugOptions::set_xla_dump_hlo_as_dot)
      .def_property("xla_dump_hlo_as_url", &DebugOptions::xla_dump_hlo_as_url,
                    &DebugOptions::set_xla_dump_hlo_as_url)
      .def_property("xla_dump_hlo_as_html", &DebugOptions::xla_dump_hlo_as_html,
                    &DebugOptions::set_xla_dump_hlo_as_html)
      .def_property("xla_dump_fusion_visualization",
                    &DebugOptions::xla_dump_fusion_visualization,
                    &DebugOptions::set_xla_dump_fusion_visualization)
      .def_property("xla_dump_hlo_snapshots",
                    &DebugOptions::xla_dump_hlo_snapshots,
                    &DebugOptions::set_xla_dump_hlo_snapshots)
      .def_property("xla_dump_max_hlo_modules",
                    &DebugOptions::xla_dump_max_hlo_modules,
                    &DebugOptions::set_xla_dump_max_hlo_modules)
      .def_property("xla_dump_module_metadata",
                    &DebugOptions::xla_dump_module_metadata,
                    &DebugOptions::set_xla_dump_module_metadata)
      .def_property("xla_dump_compress_protos",
                    &DebugOptions::xla_dump_compress_protos,
                    &DebugOptions::set_xla_dump_compress_protos)
      .def_property("xla_dump_hlo_as_long_text",
                    &DebugOptions::xla_dump_hlo_as_long_text,
                    &DebugOptions::set_xla_dump_hlo_as_long_text)
      .def_property("xla_dump_disable_metadata",
                    &DebugOptions::xla_dump_disable_metadata,
                    &DebugOptions::set_xla_dump_disable_metadata)
      .def_property("xla_dump_hlo_pipeline_re",
                    &DebugOptions::xla_dump_hlo_pipeline_re,
                    [](DebugOptions* self, std::string value) {
                      self->set_xla_dump_hlo_pipeline_re(value);
                    })
      .def_property("xla_gpu_enable_async_all_reduce",
                    &DebugOptions::xla_gpu_enable_async_all_reduce,
                    &DebugOptions::set_xla_gpu_enable_async_all_reduce)
      .def_property("xla_gpu_enable_async_all_gather",
                    &DebugOptions::xla_gpu_enable_async_all_gather,
                    &DebugOptions::set_xla_gpu_enable_async_all_gather)
      .def_property("xla_gpu_enable_async_collective_permute",
                    &DebugOptions::xla_gpu_enable_async_collective_permute,
                    &DebugOptions::set_xla_gpu_enable_async_collective_permute)
      .def_property("xla_gpu_enable_async_all_to_all",
                    &DebugOptions::xla_gpu_enable_async_all_to_all,
                    &DebugOptions::set_xla_gpu_enable_async_all_to_all)
      .def_property("xla_gpu_enable_async_reduce_scatter",
                    &DebugOptions::xla_gpu_enable_async_reduce_scatter,
                    &DebugOptions::set_xla_gpu_enable_async_reduce_scatter);

  py::class_<ExecutableBuildOptions>(m, "ExecutableBuildOptions")
      .def(py::init<>())
      .def("__repr__", &ExecutableBuildOptions::ToString)
      .def_property("fdo_profile", &ExecutableBuildOptions::fdo_profile,
                    &ExecutableBuildOptions::set_fdo_profile)
      .def_property(
          "result_layout",
          [](const ExecutableBuildOptions& options) -> std::optional<Shape> {
            return options.result_layout()
                       ? std::optional<Shape>(*options.result_layout())
                       : std::nullopt;
          },
          &ExecutableBuildOptions::set_result_layout)
      .def_property("num_replicas", &ExecutableBuildOptions::num_replicas,
                    &ExecutableBuildOptions::set_num_replicas)
      .def_property("num_partitions", &ExecutableBuildOptions::num_partitions,
                    &ExecutableBuildOptions::set_num_partitions)
      .def_property_readonly(
          "debug_options", &ExecutableBuildOptions::mutable_debug_options,
          py::return_value_policy::reference, py::keep_alive<1, 0>())
      .def_property(
          "device_assignment",
          [](const ExecutableBuildOptions& options)
              -> std::optional<DeviceAssignment> {
            return options.has_device_assignment()
                       ? std::optional<DeviceAssignment>(
                             options.device_assignment())
                       : std::nullopt;
          },
          &ExecutableBuildOptions::set_device_assignment)
      .def_property("use_spmd_partitioning",
                    &ExecutableBuildOptions::use_spmd_partitioning,
                    &ExecutableBuildOptions::set_use_spmd_partitioning)
      .def_property("use_auto_spmd_partitioning",
                    &ExecutableBuildOptions::use_auto_spmd_partitioning,
                    &ExecutableBuildOptions::set_use_auto_spmd_partitioning)
      .def_property(
          "auto_spmd_partitioning_mesh_shape",
          &ExecutableBuildOptions::auto_spmd_partitioning_mesh_shape,
          &ExecutableBuildOptions::set_auto_spmd_partitioning_mesh_shape)
      .def_property(
          "auto_spmd_partitioning_mesh_ids",
          &ExecutableBuildOptions::auto_spmd_partitioning_mesh_ids,
          &ExecutableBuildOptions::set_auto_spmd_partitioning_mesh_ids)
      .def_property(
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
      .def_property(
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

  py::enum_<OpSharding::Type> op_sharding_type(m, "OpSharding_Type");
  op_sharding_type.value("REPLICATED", OpSharding::REPLICATED)
      .value("MAXIMAL", OpSharding::MAXIMAL)
      .value("MANUAL", OpSharding::MANUAL)
      .value("TUPLE", OpSharding::TUPLE)
      .value("OTHER", OpSharding::OTHER)
      .value("UNKNOWN", OpSharding::UNKNOWN);

  py::enum_<OpSharding::ShardGroupType> op_sharding_shard_group_type(
      m, "OpSharding_ShardGroupType");
  op_sharding_shard_group_type.value("AS", OpSharding::AS)
      .value("LIKE", OpSharding::LIKE);

  py::class_<OpSharding> op_sharding(m, "OpSharding");
  op_sharding
      .def_property_readonly_static(
          "Type",
          [op_sharding_type](const py::object&) { return op_sharding_type; })
      .def_property_readonly_static(
          "ShardGroupType",
          [op_sharding_shard_group_type](const py::object&) {
            return op_sharding_shard_group_type;
          })
      .def(py::init<>())
      .def(py::pickle(
          [](const OpSharding& self) {
            return py::make_tuple(py::bytes(self.SerializeAsString()));
          },
          [](py::tuple t) {
            OpSharding result;
            result.ParseFromString(t[0].cast<std::string>());
            return result;
          }))
      .def_property("type", &xla::OpSharding::type, &xla::OpSharding::set_type)
      .def_property("replicate_on_last_tile_dim",
                    &xla::OpSharding::replicate_on_last_tile_dim,
                    &xla::OpSharding::set_replicate_on_last_tile_dim)
      .def_property("is_shard_group", &xla::OpSharding::is_shard_group,
                    &xla::OpSharding::set_is_shard_group)
      .def_property("shard_group_id", &xla::OpSharding::shard_group_id,
                    &xla::OpSharding::set_shard_group_id)
      .def_property("shard_group_type", &xla::OpSharding::shard_group_type,
                    &xla::OpSharding::set_shard_group_type)
      .def("__repr__", &xla::OpSharding::DebugString)
      .def("ParseFromString",
           [](OpSharding& sharding, const std::string& s) {
             sharding.ParseFromString(s);
           })
      .def("SerializeToString",
           [](const OpSharding& sharding) {
             return py::bytes(sharding.SerializeAsString());
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

  py::class_<HloSharding> hlo_sharding(m, "HloSharding");
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
          py::arg("dims"),
          py::arg("reshape_dims") = absl::Span<const int64_t>(),
          py::arg("transpose_perm") = absl::Span<const int>(),
          py::arg("subgroup_types") = absl::Span<const xla::OpSharding::Type>())
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
             return self.tile_assignment().dimensions();
           })
      .def("tile_assignment_devices",
           [](const xla::HloSharding& self) {
             return absl::MakeConstSpan(self.tile_assignment().array().data(),
                                        self.tile_assignment().num_elements());
           })
      .def("replicate_on_last_tile_dim",
           &xla::HloSharding::ReplicateOnLastTileDim)
      .def("subgroup_types", &xla::HloSharding::subgroup_types)
      .def("__repr__",
           [](const xla::HloSharding& self) { return self.ToString(); })
      .def("to_proto", &xla::HloSharding::ToProto);

  py::class_<FrontendAttributes> frontend_attributes(m, "FrontendAttributes");
  frontend_attributes.def(py::init<>())
      .def("__setitem__",
           [](FrontendAttributes* attr, std::string key, std::string value) {
             (*attr->mutable_map())[key] = value;
           });

  py::enum_<PrecisionConfig::Precision>(m, "PrecisionConfig_Precision")
      .value("DEFAULT", PrecisionConfig::DEFAULT)
      .value("HIGH", PrecisionConfig::HIGH)
      .value("HIGHEST", PrecisionConfig::HIGHEST);

  py::enum_<FftType>(m, "FftType")
      .value("FFT", FftType::FFT)
      .value("IFFT", FftType::IFFT)
      .value("RFFT", FftType::RFFT)
      .value("IRFFT", FftType::IRFFT);

  // Hlo Module Passes
  py::class_<HloPassInterface> hlo_pass_interface(m, "HloPassInterface");
  hlo_pass_interface.def_property_readonly("name", &HloPassInterface::name)
      .def("is_pass_pipeline", &HloPassInterface::IsPassPipeline)
      .def("run",
           [](HloPassInterface& pass, HloModule* module) -> bool {
             return xla::ValueOrThrow(pass.Run(module));
           })
      .def("run_on_module_group",
           [](HloPassInterface& pass, HloModuleGroup* module_group) -> bool {
             return xla::ValueOrThrow(pass.RunOnModuleGroup(module_group));
           });

  py::class_<HloDCE, HloPassInterface>(m, "HloDCE").def(py::init<>());
  py::class_<CallInliner, HloPassInterface>(m, "CallInliner").def(py::init<>());
  py::class_<FlattenCallGraph, HloPassInterface>(m, "FlattenCallGraph")
      .def(py::init<>());
  py::class_<TupleSimplifier, HloPassInterface>(m, "TupleSimplifier")
      .def(py::init<>());
}  // NOLINT(readability/fn_size)
}  // namespace xla
