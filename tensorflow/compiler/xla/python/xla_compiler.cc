/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/python/xla_compiler.h"

#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/hash/hash.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "pybind11/attr.h"
#include "pybind11/cast.h"
#include "pybind11/numpy.h"
#include "pybind11/pybind11.h"
#include "pybind11/pytypes.h"
#include "pybind11/stl_bind.h"
#include "tensorflow/compiler/xla/client/executable_build_options.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/client/xla_computation.h"
#include "tensorflow/compiler/xla/debug_options_flags.h"
#include "tensorflow/compiler/xla/layout_util.h"
#include "tensorflow/compiler/xla/python/py_client.h"
#include "tensorflow/compiler/xla/python/types.h"
#include "tensorflow/compiler/xla/service/computation_placer.h"
#include "tensorflow/compiler/xla/service/custom_call_target_registry.h"
#include "tensorflow/compiler/xla/service/hlo.pb.h"
#include "tensorflow/compiler/xla/service/hlo_graph_dumper.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_module_config.h"
#include "tensorflow/compiler/xla/service/hlo_parser.h"
#include "tensorflow/compiler/xla/service/hlo_sharding.h"
#include "tensorflow/compiler/xla/service/name_uniquer.h"
#include "tensorflow/compiler/xla/service/platform_util.h"
#include "tensorflow/compiler/xla/shape.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/compiler/xla/xla.pb.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/lib/strings/proto_serialization.h"

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
  if (!tensorflow::SerializeToStringDeterministic(computation.proto(),
                                                  &result)) {
    return Unknown("Failed to serialize the HloModuleProto.");
  }
  return py::bytes(result);
}

// Converts a hlo module to a serialized HloModuleProto.
StatusOr<py::bytes> GetHloModuleSerializedProto(const HloModule& module) {
  std::string result;
  if (!tensorflow::SerializeToStringDeterministic(module.ToProto(), &result)) {
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
// Safe version of ShapeUtil::MakeShapeWithLayout that fails gracefully on
// invalid input.
StatusOr<Shape> MakeShapeWithLayout(
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

// Registers a 'fn_capsule' as a CPU custom call target.
// 'fn_capsule' must be a void* pointer encapsulated in a PyCapsule object,
// with name "xla._CUSTOM_CALL_TARGET".
// 'platform' is an XLA platform name, e.g., "Host" or "CUDA".
Status PyRegisterCustomCallTarget(const std::string& fn_name,
                                  py::capsule capsule,
                                  const std::string& platform) {
  static const char* const kName = "xla._CUSTOM_CALL_TARGET";
  // TODO(phawkins): remove old name after fixing users.
  static const char* const kOldCpuName = "xla._CPU_CUSTOM_CALL_TARGET";
  if (absl::string_view(capsule.name()) != kName &&
      absl::string_view(capsule.name()) != kOldCpuName) {
    return InvalidArgument(
        "Argument to RegisterCustomCallTargetRegistry was not a "
        "xla._CUSTOM_CALL_TARGET capsule.");
  }
  CustomCallTargetRegistry::Global()->Register(
      fn_name, static_cast<void*>(capsule), platform);
  return OkStatus();
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
      .def("minor_to_major",
           [](Layout layout) { return SpanToTuple(layout.minor_to_major()); })
      .def("__eq__", [](const Layout& layout,
                        const Layout& other) { return layout == other; })
      .def("__ne__", [](const Layout& layout,
                        const Layout& other) { return layout != other; })
      .def("__hash__",
           [](const Layout& layout) { return absl::HashOf(layout); })
      .def("to_string", &Layout::ToString);

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
      .def_static(
          "array_shape",
          [](PrimitiveType type, py::object dims_seq,
             std::optional<py::object> layout_seq,
             std::optional<std::vector<bool>> dynamic_dimensions)
              -> StatusOr<Shape> {
            std::vector<int64_t> dims = SequenceToVector<int64_t>(dims_seq);
            if (layout_seq) {
              std::vector<int64_t> layout =
                  SequenceToVector<int64_t>(*layout_seq);
              return MakeShapeWithLayout(type, dims, layout,
                                         dynamic_dimensions);
            } else {
              return MakeShapeWithLayout(type, dims, std::nullopt,
                                         dynamic_dimensions);
            }
          },
          "Constructs an array shape.", py::arg("type"), py::arg("dims"),
          py::arg("layout") = std::nullopt,
          py::arg("dynamic_dimensions") = std::nullopt)
      .def_static(
          "array_shape",
          [](py::dtype dtype, py::object dims_seq,
             std::optional<py::object> layout_seq,
             std::optional<std::vector<bool>> dynamic_dimensions)
              -> StatusOr<Shape> {
            PrimitiveType type = ValueOrThrow(DtypeToPrimitiveType(dtype));
            std::vector<int64_t> dims = SequenceToVector<int64_t>(dims_seq);
            if (layout_seq) {
              std::vector<int64_t> layout =
                  SequenceToVector<int64_t>(*layout_seq);
              return MakeShapeWithLayout(type, dims, layout,
                                         dynamic_dimensions);
            } else {
              return MakeShapeWithLayout(type, dims, std::nullopt,
                                         dynamic_dimensions);
            }
          },
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
          [](py::dtype dtype) -> StatusOr<Shape> {
            PrimitiveType type = ValueOrThrow(DtypeToPrimitiveType(dtype));
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
             return ValueOrThrow(PrimitiveTypeToDtype(shape.element_type()));
           })
      .def("numpy_dtype",
           [](const Shape& shape) {
             if (shape.IsTuple()) {
               return py::dtype("O");
             }
             return ValueOrThrow(PrimitiveTypeToDtype(shape.element_type()));
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
      .def("get_hlo_module", &GetHloModule)
      .def("program_shape", &XlaComputation::GetProgramShape)
      .def("name", &XlaComputation::name)
      .def("as_serialized_hlo_module_proto", &GetComputationSerializedProto)
      .def("as_hlo_text", &GetComputationHloText,
           py::arg("print_large_constants") = false)
      .def("as_hlo_dot_graph", &GetComputationHloDotGraph)
      .def("hash", &HashComputation)
      .def("as_hlo_module", &GetHloModule);

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

  py::class_<HloModule, std::shared_ptr<HloModule>> hlo_module_class(
      m, "HloModule");
  hlo_module_class.def_property_readonly("name", &HloModule::name)
      .def(
          "to_string",
          static_cast<std::string (HloModule::*)(const HloPrintOptions&) const>(
              &HloModule::ToString),
          py::arg("options") = HloPrintOptions())
      .def("as_serialized_hlo_module_proto", &GetHloModuleSerializedProto)
      .def("from_serialized_hlo_module_proto", &HloModuleFromSerializedProto)
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

  m.def("hlo_module_to_dot_graph",
        [](const HloModule& hlo_module) -> StatusOr<std::string> {
          return RenderGraph(*hlo_module.entry_computation(), /*label=*/"",
                             hlo_module.config().debug_options(),
                             RenderedGraphFormat::kDot);
        });
  m.def(
      "hlo_module_cost_analysis",
      [](PyClient* client,
         const HloModule& module) -> StatusOr<HloCostAnalysis::Properties> {
        TF_ASSIGN_OR_RETURN(auto analysis,
                            client->pjrt_client()->GetHloCostAnalysis());
        TF_RETURN_IF_ERROR(module.entry_computation()->Accept(analysis.get()));
        return analysis->properties();
      });

  py::class_<XlaOp> xla_op_class(m, "XlaOp");

  py::class_<XlaBuilder>(m, "XlaBuilder")
      .def(py::init([](const std::string& name) -> std::unique_ptr<XlaBuilder> {
        return std::make_unique<XlaBuilder>(UniquifyName(name));
      }))
      // TODO(phawkins): delete capitalized names after updating callers.
      .def(
          "Build",
          [](XlaBuilder& builder, std::optional<XlaOp> root) {
            return root ? builder.Build(*root) : builder.Build();
          },
          "Builds a computation from the contents of the builder.",
          py::arg("root") = std::nullopt)
      .def("GetShape", &XlaBuilder::GetShape)
      .def(
          "build",
          [](XlaBuilder& builder, std::optional<XlaOp> root) {
            return root ? builder.Build(*root) : builder.Build();
          },
          "Builds a computation from the contents of the builder.",
          py::arg("root") = std::nullopt)
      .def("clear_op_metadata", &XlaBuilder::ClearOpMetadata)
      .def("get_shape", &XlaBuilder::GetShape)
      .def(
          "get_program_shape",
          [](const XlaBuilder& builder,
             std::optional<XlaOp> root) -> StatusOr<ProgramShape> {
            return root ? builder.GetProgramShape(*root)
                        : builder.GetProgramShape();
          },
          py::arg("root") = std::nullopt)
      .def("is_constant", &XlaBuilder::IsConstant)
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
      .def_static("create",
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
                  })
      .def("replica_count", &DeviceAssignment::replica_count)
      .def("computation_count", &DeviceAssignment::computation_count)
      .def("__repr__", &DeviceAssignment::ToString)
      .def("serialize", [](const DeviceAssignment& da) -> StatusOr<py::bytes> {
        DeviceAssignmentProto proto;
        TF_RETURN_IF_ERROR(da.Serialize(&proto));
        std::string result;
        if (!tensorflow::SerializeToStringDeterministic(proto, &result)) {
          return Unknown("Failed to serialize the DeviceAssignmentProto.");
        }
        return py::bytes(result);
      });

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
      .def_readwrite("argument_layouts", &CompileOptions::argument_layouts)
      .def_readwrite("parameter_is_tupled_arguments",
                     &CompileOptions::parameter_is_tupled_arguments)
      .def_readonly("executable_build_options",
                    &CompileOptions::executable_build_options)
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
  m.def("register_custom_call_target", &PyRegisterCustomCallTarget);

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
      .def_property("xla_detailed_logging_and_dumping",
                    &DebugOptions::xla_detailed_logging_and_dumping,
                    &DebugOptions::set_xla_detailed_logging_and_dumping)
      .def_property("xla_gpu_enable_fast_min_max",
                    &DebugOptions::xla_gpu_enable_fast_min_max,
                    &DebugOptions::set_xla_gpu_enable_fast_min_max)
      .def_property("xla_gpu_cuda_data_dir",
                    &DebugOptions::xla_gpu_cuda_data_dir,
                    [](DebugOptions* self, std::string value) {
                      self->set_xla_gpu_cuda_data_dir(value);
                    })
      .def_property("xla_llvm_disable_expensive_passes",
                    &DebugOptions::xla_llvm_disable_expensive_passes,
                    &DebugOptions::set_xla_llvm_disable_expensive_passes)
      .def_property("xla_test_all_input_layouts",
                    &DebugOptions::xla_test_all_input_layouts,
                    &DebugOptions::set_xla_test_all_input_layouts);

  py::class_<ExecutableBuildOptions>(m, "ExecutableBuildOptions")
      .def(py::init<>())
      .def("__repr__", &ExecutableBuildOptions::ToString)
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
          "allow_spmd_sharding_propagation_to_output",
          &ExecutableBuildOptions::allow_spmd_sharding_propagation_to_output,
          &ExecutableBuildOptions::
              set_allow_spmd_sharding_propagation_to_output);

  py::enum_<OpSharding::Type> op_sharding_type(m, "OpSharding_Type");
  op_sharding_type.value("REPLICATED", OpSharding::REPLICATED)
      .value("MAXIMAL", OpSharding::MAXIMAL)
      .value("MANUAL", OpSharding::MANUAL)
      .value("TUPLE", OpSharding::TUPLE)
      .value("OTHER", OpSharding::OTHER);

  py::class_<OpSharding> op_sharding(m, "OpSharding");
  op_sharding
      .def_property_readonly_static(
          "Type",
          [op_sharding_type](const py::object&) { return op_sharding_type; })
      .def(py::init<>())
      .def_property("type", &xla::OpSharding::type, &xla::OpSharding::set_type)
      .def_property("replicate_on_last_tile_dim",
                    &xla::OpSharding::replicate_on_last_tile_dim,
                    &xla::OpSharding::set_replicate_on_last_tile_dim)
      .def("__repr__", &xla::OpSharding::DebugString)
      .def("SerializeToString", [](const OpSharding& sharding) {
        return py::bytes(sharding.SerializeAsString());
      });
  DefRepeatedProperty(op_sharding, "tile_assignment_dimensions",
                      &xla::OpSharding::mutable_tile_assignment_dimensions);
  DefRepeatedProperty(op_sharding, "tile_assignment_devices",
                      &xla::OpSharding::mutable_tile_assignment_devices);
  DefRepeatedProperty(op_sharding, "tuple_shardings",
                      &xla::OpSharding::mutable_tuple_shardings);
  DefRepeatedProperty(op_sharding, "last_tile_dims",
                      &xla::OpSharding::mutable_last_tile_dims);

  py::class_<HloSharding> hlo_sharding(m, "HloSharding");
  hlo_sharding.def_static("from_proto", &xla::HloSharding::FromProto)
      .def("__eq__", [](const xla::HloSharding& a,
                        const xla::HloSharding& b) { return a == b; })
      .def("__hash__",
           [](const xla::HloSharding& self) { return absl::HashOf(self); });

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

  py::enum_<ChannelHandle::ChannelType>(m, "ChannelHandle_ChannelType")
      .value("CHANNEL_TYPE_INVALID", ChannelHandle::CHANNEL_TYPE_INVALID)
      .value("DEVICE_TO_DEVICE", ChannelHandle::DEVICE_TO_DEVICE)
      .value("DEVICE_TO_HOST", ChannelHandle::DEVICE_TO_HOST)
      .value("HOST_TO_DEVICE", ChannelHandle::HOST_TO_DEVICE);

  py::class_<ChannelHandle>(m, "ChannelHandle")
      .def_property("type", &ChannelHandle::type,
                    [](ChannelHandle* h, ChannelHandle::ChannelType type) {
                      h->set_type(type);
                    })
      .def_property(
          "handle", &ChannelHandle::handle,
          [](ChannelHandle* h, int64_t handle) { h->set_handle(handle); })
      .def("__repr__", [](ChannelHandle* h) { return h->DebugString(); });

  py::enum_<FftType>(m, "FftType")
      .value("FFT", FftType::FFT)
      .value("IFFT", FftType::IFFT)
      .value("RFFT", FftType::RFFT)
      .value("IRFFT", FftType::IRFFT);
}  // NOLINT(readability/fn_size)
}  // namespace xla
