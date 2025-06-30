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

#include "xla/pjrt/utils.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <memory>
#include <numeric>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/ascii.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Support/LLVM.h"
#include "xla/client/executable_build_options.h"
#include "xla/hlo/builder/xla_computation.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_input_output_alias_config.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/ir/hlo_sharding.h"
#include "xla/layout.h"
#include "xla/layout_util.h"
#include "xla/pjrt/layout_mode.h"
#include "xla/primitive_util.h"
#include "xla/service/computation_placer.h"
#include "xla/service/hlo.pb.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/status_macros.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/cpu_info.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/logging.h"  // IWYU pragma: keep
#include "tsl/platform/statusor.h"

namespace xla {

namespace {
absl::StatusOr<Shape> GetShardedShape(const Shape& shape,
                                      const OpSharding& sharding) {
  TF_ASSIGN_OR_RETURN(HloSharding hlo_sharding,
                      HloSharding::FromProto(sharding));
  if (shape.IsTuple()) {
    Shape sharded_shape = shape;
    ShapeUtil::ForEachMutableSubshape(
        &sharded_shape, [&](Shape* subshape, const ShapeIndex& index) {
          if (!subshape->IsTuple()) {
            HloSharding subsharding = hlo_sharding.GetSubSharding(shape, index);
            *subshape = subsharding.TileShape(*subshape);
          }
        });
    return sharded_shape;
  } else {
    return hlo_sharding.TileShape(shape);
  }
}

absl::StatusOr<Shape> GetShardedShape(const HloInstructionProto& instr) {
  TF_ASSIGN_OR_RETURN(Shape unsharded_shape, Shape::FromProto(instr.shape()));
  Shape sharded_shape;
  if (instr.has_sharding()) {
    TF_ASSIGN_OR_RETURN(sharded_shape,
                        GetShardedShape(unsharded_shape, instr.sharding()));
  } else {
    sharded_shape = unsharded_shape;
  }
  LayoutUtil::ClearLayout(&sharded_shape);
  return sharded_shape;
}

// Returns sharded (argument shapes, result shape) without layouts.
absl::StatusOr<std::pair<std::vector<Shape>, Shape>> GetShardedProgramShapes(
    const XlaComputation& computation, const ProgramShape& program_shape) {
  std::vector<Shape> arg_shapes;
  arg_shapes.resize(program_shape.parameters_size());
  Shape result_shape;
  for (const HloComputationProto& comp : computation.proto().computations()) {
    if (comp.id() != computation.proto().entry_computation_id()) {
      continue;
    }
    for (const HloInstructionProto& instr : comp.instructions()) {
      if (instr.opcode() == HloOpcodeString(HloOpcode::kParameter)) {
        if (instr.parameter_number() >= program_shape.parameters_size()) {
          return InvalidArgument(
              "Got invalid parameter number %d, expected %d parameters",
              instr.parameter_number(), program_shape.parameters_size());
        }
        TF_ASSIGN_OR_RETURN(arg_shapes[instr.parameter_number()],
                            GetShardedShape(instr));
      }
      if (instr.id() == comp.root_id()) {
        if (result_shape.element_type() != PRIMITIVE_TYPE_INVALID) {
          return InvalidArgument("Found multiple root instructions");
        }
        TF_ASSIGN_OR_RETURN(result_shape, GetShardedShape(instr));
      }
    }
  }
  for (int i = 0; i < arg_shapes.size(); ++i) {
    if (arg_shapes[i].element_type() == PRIMITIVE_TYPE_INVALID) {
      return InvalidArgument("Couldn't find parameter %d", i);
    }
  }
  if (result_shape.element_type() == PRIMITIVE_TYPE_INVALID) {
    return InvalidArgument("Couldn't find root instruction");
  }
  return std::make_pair(arg_shapes, result_shape);
}
}  // namespace

absl::Status ParseDeviceAssignmentCompileOptions(
    bool compile_portable_executable, ExecutableBuildOptions* build_options,
    std::function<absl::StatusOr<DeviceAssignment>(int, int)>
        GetDefaultDeviceAssignmentFunction,
    int* num_replicas, int* num_partitions,
    std::shared_ptr<DeviceAssignment>* device_assignment) {
  if (compile_portable_executable) {
    if (build_options->has_device_assignment()) {
      return InvalidArgument(
          "CompileOptions requests portable executable but "
          "ExecutableBuildOptions includes a device assignment");
    }
    if (build_options->num_replicas() != 1 ||
        build_options->num_partitions() != 1) {
      return InvalidArgument(
          "CompileOptions requests portable executable but "
          "ExecutableBuildOptions includes num_replicas %d  and num_partitions "
          "%d.",
          build_options->num_replicas(), build_options->num_partitions());
    }
    *num_replicas = 1;
    *num_partitions = 1;
  } else {
    if (!build_options->has_device_assignment()) {
      VLOG(2) << "Compile using default device_assignment.";
      TF_ASSIGN_OR_RETURN(
          DeviceAssignment device_assignment,
          GetDefaultDeviceAssignmentFunction(build_options->num_replicas(),
                                             build_options->num_partitions()));
      build_options->set_device_assignment(device_assignment);
    }
    VLOG(2) << "Compile device_assignment:\n"
            << build_options->device_assignment().ToString();
    *num_replicas = build_options->device_assignment().replica_count();
    *num_partitions = build_options->device_assignment().computation_count();
    *device_assignment =
        std::make_shared<DeviceAssignment>(build_options->device_assignment());
  }
  return absl::OkStatus();
}

// Helper method that takes an ArrayAttr of DictionaryAttrs for each arg or
// result of a function, and looks for "mhlo.layout_mode". `all_attrs` can be
// nullptr. `num_values` is the number of arguments or results.
static absl::StatusOr<std::vector<LayoutMode>> MlirAttrsToLayoutModes(
    mlir::ArrayAttr all_attrs, size_t num_values) {
  if (all_attrs == nullptr) {
    return std::vector<LayoutMode>(num_values);
  }
  if (all_attrs.size() != num_values) {
    return InvalidArgument(
        "MlirAttrsToLayoutModes got unexpected number of attributes: %d, "
        "expected: %d",
        all_attrs.size(), num_values);
  }

  std::vector<LayoutMode> result;
  result.reserve(all_attrs.size());
  for (const mlir::Attribute& dict_attr : all_attrs) {
    mlir::StringAttr attr =
        mlir::cast<mlir::DictionaryAttr>(dict_attr).getAs<mlir::StringAttr>(
            "mhlo.layout_mode");
    if (attr != nullptr) {
      TF_ASSIGN_OR_RETURN(LayoutMode mode,
                          LayoutMode::FromString(attr.getValue().str()));
      result.emplace_back(std::move(mode));
    } else {
      result.emplace_back();
    }
  }
  return result;
}

// TODO(b/329428415): Make this generic enough to be used by the GPU and TPU
// compilers.
absl::StatusOr<MemorySpaceColor> GetMemorySpaceColor(
    const std::string& memory_kind) {
  // TODO(yashkatariya,zce): Unpinned_host is not valid for compiler. Only
  // pinned_host matters. So should there be a different lowering for
  // unpinned_host?
  if (memory_kind == "unpinned_host" || memory_kind == "pinned_host") {
    return xla::Layout::kHostMemorySpace;
  } else if (memory_kind == "device") {
    return xla::Layout::kDefaultMemorySpace;
  } else {
    return InvalidArgument("Unknown memory kind %s", memory_kind);
  }
}

// Helper method that takes an ArrayAttr of DictionaryAttrs for each arg or
// result of a function, and looks for "mhlo.layout_mode". `all_attrs` can be
// nullptr. `num_values` is the number of arguments or results.
static absl::StatusOr<std::vector<MemorySpaceColor>> MlirAttrsToMemoryKinds(
    mlir::ArrayAttr all_attrs, size_t num_values) {
  if (all_attrs == nullptr) {
    return std::vector<MemorySpaceColor>(num_values,
                                         xla::Layout::kDefaultMemorySpace);
  }
  if (all_attrs.size() != num_values) {
    return InvalidArgument(
        "MlirAttrsToMemoryKinds got unexpected number of attributes: %d, "
        "expected: %d",
        all_attrs.size(), num_values);
  }

  std::vector<MemorySpaceColor> result;
  result.reserve(all_attrs.size());
  for (const mlir::Attribute& dict_attr : all_attrs) {
    mlir::StringAttr attr =
        mlir::cast<mlir::DictionaryAttr>(dict_attr).getAs<mlir::StringAttr>(
            "mhlo.memory_kind");
    if (attr != nullptr) {
      TF_ASSIGN_OR_RETURN(MemorySpaceColor memory_space,
                          GetMemorySpaceColor(attr.getValue().str()));
      result.push_back(memory_space);
    } else {
      result.emplace_back(xla::Layout::kDefaultMemorySpace);
    }
  }
  return result;
}

// Helper function for getting default LayoutModes for tupled arguments or
// outputs. Returns nullopt if the arguments/outputs are not tupled. Raises an
// error if layout modes are requested on tupled values.
static absl::StatusOr<std::optional<std::vector<LayoutMode>>>
GetTupleLayoutModes(mlir::ArrayRef<mlir::Type> types,
                    mlir::ArrayAttr all_attrs) {
  if (types.size() != 1 || !llvm::isa<mlir::TupleType>(types[0])) {
    return std::nullopt;
  }
  if (all_attrs != nullptr) {
    if (all_attrs.size() != 1) {
      return InvalidArgument(
          "GetTupleLayoutModes expected single tuple attr, got %d attrs",
          all_attrs.size());
    }
    mlir::StringAttr attr = mlir::cast<mlir::DictionaryAttr>(*all_attrs.begin())
                                .getAs<mlir::StringAttr>("mhlo.layout_mode");
    if (attr != nullptr) {
      return Unimplemented("mhlo.layout_mode not supported with tupled values");
    }
  }
  // Use default layout for all outputs.
  return std::vector<LayoutMode>(mlir::cast<mlir::TupleType>(types[0]).size());
}

// Helper function for getting default LayoutModes for tupled arguments or
// outputs. Returns nullopt if the arguments/outputs are not tupled. Raises an
// error if layout modes are requested on tupled values.
static absl::StatusOr<std::optional<std::vector<MemorySpaceColor>>>
GetTupleMemoryKinds(mlir::ArrayRef<mlir::Type> types,
                    mlir::ArrayAttr all_attrs) {
  if (types.size() != 1 || !llvm::isa<mlir::TupleType>(types[0])) {
    return std::nullopt;
  }
  if (all_attrs != nullptr) {
    if (all_attrs.size() != 1) {
      return InvalidArgument(
          "GetTupleMemoryKinds expected single tuple attr, got %d attrs",
          all_attrs.size());
    }
    mlir::StringAttr attr = mlir::cast<mlir::DictionaryAttr>(*all_attrs.begin())
                                .getAs<mlir::StringAttr>("mhlo.memory_kind");
    if (attr != nullptr) {
      return Unimplemented("mhlo.memory_kind not supported with tupled values");
    }
  }
  // Use default layout for all outputs.
  return std::vector<MemorySpaceColor>(
      mlir::cast<mlir::TupleType>(types[0]).size(),
      xla::Layout::kDefaultMemorySpace);
}

absl::StatusOr<std::vector<LayoutMode>> GetArgLayoutModes(
    mlir::ModuleOp module) {
  mlir::func::FuncOp main = module.lookupSymbol<mlir::func::FuncOp>("main");
  if (main == nullptr) {
    return InvalidArgument(
        "GetArgLayoutModes passed module without main function");
  }

  // Special case: tupled arguments
  TF_ASSIGN_OR_RETURN(std::optional<std::vector<LayoutMode>> maybe_result,
                      GetTupleLayoutModes(main.getFunctionType().getInputs(),
                                          main.getAllArgAttrs()));
  if (maybe_result) return *maybe_result;

  return MlirAttrsToLayoutModes(main.getAllArgAttrs(), main.getNumArguments());
}

absl::StatusOr<std::vector<LayoutMode>> GetOutputLayoutModes(
    mlir::ModuleOp module) {
  mlir::func::FuncOp main = module.lookupSymbol<mlir::func::FuncOp>("main");
  if (main == nullptr) {
    return InvalidArgument(
        "GetOutputLayoutModes passed module without main function");
  }

  // Special case: tupled outputs
  TF_ASSIGN_OR_RETURN(std::optional<std::vector<LayoutMode>> maybe_tuple_result,
                      GetTupleLayoutModes(main.getFunctionType().getResults(),
                                          main.getAllResultAttrs()));
  if (maybe_tuple_result) return *maybe_tuple_result;

  return MlirAttrsToLayoutModes(main.getAllResultAttrs(), main.getNumResults());
}

absl::StatusOr<std::vector<MemorySpaceColor>> GetArgMemoryKinds(
    mlir::ModuleOp module) {
  mlir::func::FuncOp main = module.lookupSymbol<mlir::func::FuncOp>("main");
  if (main == nullptr) {
    return InvalidArgument(
        "GetArgMemoryKinds passed module without main function");
  }

  // Special case: tupled arguments
  TF_ASSIGN_OR_RETURN(
      std::optional<std::vector<MemorySpaceColor>> maybe_tuple_result,
      GetTupleMemoryKinds(main.getFunctionType().getInputs(),
                          main.getAllArgAttrs()));
  if (maybe_tuple_result) return *maybe_tuple_result;

  return MlirAttrsToMemoryKinds(main.getAllArgAttrs(), main.getNumArguments());
}

absl::StatusOr<std::vector<MemorySpaceColor>> GetOutputMemoryKinds(
    mlir::ModuleOp module) {
  mlir::func::FuncOp main = module.lookupSymbol<mlir::func::FuncOp>("main");
  if (main == nullptr) {
    return InvalidArgument(
        "GetOutputMemoryKinds passed module without main function");
  }

  // Special case: tupled outputs
  TF_ASSIGN_OR_RETURN(
      std::optional<std::vector<MemorySpaceColor>> maybe_tuple_result,
      GetTupleMemoryKinds(main.getFunctionType().getResults(),
                          main.getAllResultAttrs()));
  if (maybe_tuple_result) return *maybe_tuple_result;

  return MlirAttrsToMemoryKinds(main.getAllResultAttrs(), main.getNumResults());
}

// Make sure to choose delimiter that will never show up in Layout strings.
static const char* kDelimiter = ";";

static absl::StatusOr<std::vector<LayoutMode>> GetLayoutModesFromFrontendAttr(
    absl::string_view attr) {
  // SkipEmpty() needed to avoid returning the empty string when attr is empty.
  std::vector<std::string> str_modes =
      absl::StrSplit(attr, kDelimiter, absl::SkipEmpty());
  std::vector<LayoutMode> result;
  for (const std::string& str_mode : str_modes) {
    TF_ASSIGN_OR_RETURN(LayoutMode mode, LayoutMode::FromString(str_mode));
    result.emplace_back(std::move(mode));
  }
  return result;
}

static absl::StatusOr<std::vector<LayoutMode>> GetLayoutModes(
    const XlaComputation& computation, absl::string_view frontend_attr_name,
    size_t num_values) {
  const auto& frontend_attrs = computation.proto().frontend_attributes().map();
  auto iter = frontend_attrs.find(frontend_attr_name);
  if (iter == frontend_attrs.end()) {
    // Return all default layouts if frontend attr isn't present.
    return std::vector<LayoutMode>(num_values);
  }
  return GetLayoutModesFromFrontendAttr(iter->second);
}

static absl::StatusOr<std::vector<MemorySpaceColor>>
GetMemoryKindsFromFrontendAttr(absl::string_view attr) {
  // SkipEmpty() needed to avoid returning the empty string when attr is empty.
  std::vector<std::string> str_memory_spaces =
      absl::StrSplit(attr, kDelimiter, absl::SkipEmpty());

  std::vector<MemorySpaceColor> result;
  result.reserve(str_memory_spaces.size());
  for (const std::string& str_mem_space : str_memory_spaces) {
    MemorySpaceColor memory_space;
    CHECK(absl::SimpleAtoi(str_mem_space, &memory_space));
    result.push_back(memory_space);
  }
  return result;
}

static absl::StatusOr<std::vector<MemorySpaceColor>> GetMemoryKinds(
    const XlaComputation& computation, absl::string_view frontend_attr_name,
    size_t num_values) {
  const auto& frontend_attrs = computation.proto().frontend_attributes().map();
  auto iter = frontend_attrs.find(frontend_attr_name);
  if (iter == frontend_attrs.end()) {
    // Return all default memory space i.e. 0 if frontend attr isn't present.
    return std::vector<MemorySpaceColor>(num_values,
                                         xla::Layout::kDefaultMemorySpace);
  }
  return GetMemoryKindsFromFrontendAttr(iter->second);
}

absl::StatusOr<std::vector<LayoutMode>> GetArgLayoutModes(
    const XlaComputation& computation) {
  TF_ASSIGN_OR_RETURN(ProgramShape program_shape,
                      computation.GetProgramShape());
  size_t num_args = program_shape.parameters_size() == 1 &&
                            program_shape.parameters(0).IsTuple()
                        ? program_shape.parameters(0).tuple_shapes().size()
                        : program_shape.parameters_size();
  return GetLayoutModes(computation, "arg_layout_modes", num_args);
}

absl::StatusOr<std::vector<MemorySpaceColor>> GetArgMemoryKinds(
    const XlaComputation& computation) {
  TF_ASSIGN_OR_RETURN(ProgramShape program_shape,
                      computation.GetProgramShape());
  size_t num_args = program_shape.parameters_size() == 1 &&
                            program_shape.parameters(0).IsTuple()
                        ? program_shape.parameters(0).tuple_shapes().size()
                        : program_shape.parameters_size();
  return GetMemoryKinds(computation, "arg_memory_spaces", num_args);
}

absl::StatusOr<std::vector<LayoutMode>> GetOutputLayoutModes(
    const XlaComputation& computation) {
  TF_ASSIGN_OR_RETURN(ProgramShape program_shape,
                      computation.GetProgramShape());
  size_t num_outputs = program_shape.result().IsTuple()
                           ? program_shape.result().tuple_shapes().size()
                           : 1;
  return GetLayoutModes(computation, "out_layout_modes", num_outputs);
}

absl::StatusOr<std::vector<MemorySpaceColor>> GetOutputMemoryKinds(
    const XlaComputation& computation) {
  TF_ASSIGN_OR_RETURN(ProgramShape program_shape,
                      computation.GetProgramShape());
  size_t num_outputs = program_shape.result().IsTuple()
                           ? program_shape.result().tuple_shapes().size()
                           : 1;
  return GetMemoryKinds(computation, "out_memory_spaces", num_outputs);
}

absl::StatusOr<Shape> LayoutModeToXlaShape(
    const LayoutMode& layout_mode, const Shape& unsharded_shape,
    const Shape& sharded_shape, MemorySpaceColor memory_space,
    std::function<absl::StatusOr<Shape>(Shape)>
        choose_compact_layout_for_shape_function) {
  if (unsharded_shape.IsToken() || unsharded_shape.IsOpaque()) {
    return unsharded_shape;
  }
  if (!unsharded_shape.IsArray() || !sharded_shape.IsArray()) {
    return InvalidArgument(
        "LayoutModeToXlaShape must be passed array shapes, got "
        "unsharded_shape: %s, sharded_shape: %s",
        unsharded_shape.ToString(), sharded_shape.ToString());
  }
  // For sharded computations, XLA expects the layout to specified as the global
  // shape with the sharded layout.
  Shape result = unsharded_shape;
  LayoutUtil::ClearLayout(&result);
  switch (layout_mode.mode) {
    case LayoutMode::Mode::kDefault: {
      TF_ASSIGN_OR_RETURN(
          Shape layout,
          choose_compact_layout_for_shape_function(sharded_shape));
      *result.mutable_layout() = layout.layout();
      break;
    }
    case LayoutMode::Mode::kUserSpecified: {
      CHECK(layout_mode.user_layout);
      *result.mutable_layout() = *layout_mode.user_layout;
      break;
    }
    case LayoutMode::Mode::kAuto: {
      // Don't set any layout on `result`.
      break;
    }
  }
  // When layout is AUTO, memory space can't be set since it will be partial.
  if (result.has_layout()) {
    result.mutable_layout()->set_memory_space(memory_space);
  }
  return result;
}

absl::StatusOr<std::pair<std::vector<Shape>, Shape>> LayoutModesToXlaShapes(
    const XlaComputation& computation, std::vector<LayoutMode> arg_layout_modes,
    std::vector<LayoutMode> out_layout_modes,
    const std::vector<MemorySpaceColor>& arg_memory_spaces,
    const std::vector<MemorySpaceColor>& out_memory_spaces,
    std::function<absl::StatusOr<Shape>(Shape)>
        choose_compact_layout_for_shape_function) {
  // Compute sharded argument and output shapes.
  TF_ASSIGN_OR_RETURN(ProgramShape program_shape,
                      computation.GetProgramShape());
  TF_ASSIGN_OR_RETURN(auto sharded_shapes,
                      GetShardedProgramShapes(computation, program_shape));

  // Untuple if necessary.
  bool args_tupled = program_shape.parameters_size() == 1 &&
                     program_shape.parameters(0).IsTuple();
  const std::vector<Shape>& unsharded_arg_shapes =
      args_tupled ? program_shape.parameters(0).tuple_shapes()
                  : program_shape.parameters();
  const std::vector<Shape>& sharded_arg_shapes =
      args_tupled ? sharded_shapes.first[0].tuple_shapes()
                  : sharded_shapes.first;

  bool out_tupled = program_shape.result().IsTuple();
  const std::vector<Shape>& unsharded_out_shapes =
      out_tupled ? program_shape.result().tuple_shapes()
                 : std::vector<Shape>{program_shape.result()};
  const std::vector<Shape>& sharded_out_shapes =
      out_tupled ? sharded_shapes.second.tuple_shapes()
                 : std::vector<Shape>{sharded_shapes.second};

  if (unsharded_arg_shapes.size() != arg_layout_modes.size()) {
    return InvalidArgument(
        "LayoutModesToXlaShapes got mismatched number of arguments and layout "
        "modes (%d vs %d)",
        unsharded_arg_shapes.size(), arg_layout_modes.size());
  }
  if (sharded_arg_shapes.size() != arg_layout_modes.size()) {
    return InvalidArgument(
        "LayoutModesToXlaShapes got mismatched number of sharded arguments and "
        "layout modes (%d vs %d)",
        sharded_arg_shapes.size(), arg_layout_modes.size());
  }
  if (unsharded_out_shapes.size() != out_layout_modes.size()) {
    return InvalidArgument(
        "LayoutModesToXlaShapes got mismatched number of outputs and layout "
        "modes (%d vs %d)",
        unsharded_out_shapes.size(), out_layout_modes.size());
  }
  if (sharded_out_shapes.size() != out_layout_modes.size()) {
    return InvalidArgument(
        "LayoutModesToXlaShapes got mismatched number of sharded outputs and "
        "layout modes (%d vs %d)",
        sharded_out_shapes.size(), out_layout_modes.size());
  }

  // Convert each LayoutMode to an xla::Shape with the appropriate Layout set or
  // unset.
  if (arg_memory_spaces.size() != arg_layout_modes.size()) {
    return InvalidArgument(
        "The sizes of arg_memory_spaces and arg_layout_modes don't match");
  }
  std::vector<Shape> flat_arg_layouts;
  flat_arg_layouts.reserve(arg_layout_modes.size());
  for (int i = 0; i < arg_layout_modes.size(); ++i) {
    TF_ASSIGN_OR_RETURN(
        Shape layout,
        LayoutModeToXlaShape(arg_layout_modes[i], unsharded_arg_shapes[i],
                             sharded_arg_shapes[i], arg_memory_spaces[i],
                             choose_compact_layout_for_shape_function));
    flat_arg_layouts.emplace_back(std::move(layout));
  }

  if (out_memory_spaces.size() != out_layout_modes.size()) {
    return InvalidArgument(
        "The sizes of out_memory_spaces and out_layout_modes don't match");
  }
  std::vector<Shape> flat_out_layouts;
  flat_out_layouts.reserve(out_layout_modes.size());
  for (int i = 0; i < out_layout_modes.size(); ++i) {
    TF_ASSIGN_OR_RETURN(
        Shape layout,
        LayoutModeToXlaShape(out_layout_modes[i], unsharded_out_shapes[i],
                             sharded_out_shapes[i], out_memory_spaces[i],
                             choose_compact_layout_for_shape_function));
    flat_out_layouts.emplace_back(std::move(layout));
  }

  // Tuple final shapes if necessary.
  std::vector<Shape> arg_layouts =
      args_tupled
          ? std::vector<Shape>{ShapeUtil::MakeTupleShape(flat_arg_layouts)}
          : std::move(flat_arg_layouts);
  Shape out_layout = out_tupled ? ShapeUtil::MakeTupleShape(flat_out_layouts)
                                : flat_out_layouts[0];

  return std::pair<std::vector<Shape>, Shape>{std::move(arg_layouts),
                                              std::move(out_layout)};
}

absl::StatusOr<std::pair<std::vector<Shape>, std::vector<const Shape*>>>
LayoutModesToXla(const XlaComputation& computation,
                 std::vector<LayoutMode> arg_layout_modes,
                 std::vector<LayoutMode> out_layout_modes,
                 const std::vector<MemorySpaceColor>& arg_memory_spaces,
                 const std::vector<MemorySpaceColor>& out_memory_spaces,
                 std::function<absl::StatusOr<Shape>(Shape)>
                     choose_compact_layout_for_shape_function,
                 ExecutableBuildOptions& build_options) {
  TF_ASSIGN_OR_RETURN(
      auto pair,
      LayoutModesToXlaShapes(computation, arg_layout_modes, out_layout_modes,
                             arg_memory_spaces, out_memory_spaces,
                             choose_compact_layout_for_shape_function));
  std::vector<Shape>& arg_layouts = pair.first;
  Shape& out_layout = pair.second;

  // Generate result vector of pointers
  std::vector<const Shape*> arg_layout_pointers;
  arg_layout_pointers.reserve(arg_layouts.size());
  for (int i = 0; i < arg_layouts.size(); ++i) {
    arg_layout_pointers.push_back(&arg_layouts[i]);
  }

  // Update build_options
  build_options.set_result_layout(out_layout);

  return std::pair<std::vector<Shape>, std::vector<const Shape*>>{
      std::move(arg_layouts), std::move(arg_layout_pointers)};
}

absl::Status DetermineArgumentLayoutsFromCompileOptions(
    const XlaComputation& computation,
    std::function<absl::StatusOr<Shape>(Shape)>
        choose_compact_layout_for_shape_function,
    std::optional<std::vector<Shape>>& argument_layouts,
    ExecutableBuildOptions* build_options,
    std::vector<const Shape*>* argument_layout_pointers) {
  TF_ASSIGN_OR_RETURN(ProgramShape program_shape,
                      computation.GetProgramShape());
  const bool given_argument_layouts = argument_layouts.has_value();
  if (!argument_layouts) {
    argument_layouts.emplace(program_shape.parameters());
    for (Shape& shape : *argument_layouts) {
      LayoutUtil::ClearLayout(&shape);
    }
  } else if (argument_layouts->size() != program_shape.parameters_size()) {
    return InvalidArgument(
        "CompileOptions specify %d argument layouts, but computation has %d "
        "arguments",
        argument_layouts->size(), program_shape.parameters_size());
  }
  argument_layout_pointers->reserve(argument_layouts->size());

  // Assign a default layout based on `sharded_shape` to any array subshapes in
  // `dst_shape` that are missing layouts.
  auto assign_layouts = [&](const Shape& sharded_shape, Shape* dst_shape) {
    return ShapeUtil::ForEachMutableSubshapeWithStatus(
        dst_shape, [&](Shape* subshape, const ShapeIndex& idx) {
          if (subshape->IsArray() && !subshape->has_layout()) {
            CHECK(ShapeUtil::IndexIsValid(sharded_shape, idx));
            const Shape& sharded_subshape =
                ShapeUtil::GetSubshape(sharded_shape, idx);
            LayoutUtil::SetToDefaultLayout(subshape);
            TF_ASSIGN_OR_RETURN(
                Shape layout,
                choose_compact_layout_for_shape_function(sharded_subshape));
            if (layout.has_layout()) {
              *subshape->mutable_layout() = layout.layout();
            } else {
              subshape->clear_layout();
            }
          }
          return absl::OkStatus();
        });
  };
  TF_ASSIGN_OR_RETURN(auto sharded_shapes,
                      GetShardedProgramShapes(computation, program_shape));

  CHECK_EQ(sharded_shapes.first.size(), argument_layouts->size());
  for (int i = 0; i < argument_layouts->size(); ++i) {
    Shape* layout = &(*argument_layouts)[i];
    argument_layout_pointers->push_back(layout);
    TF_RETURN_IF_ERROR(assign_layouts(sharded_shapes.first[i], layout));
    if (!given_argument_layouts) {
      // Carry memory space forward from program shape.
      ShapeUtil::ForEachSubshape(
          program_shape.parameters(i),
          [&](const Shape& subshape, const ShapeIndex& index) {
            if (subshape.IsArray()) {
              Shape* program_subshape =
                  ShapeUtil::GetMutableSubshape(layout, index);
              if (program_subshape->has_layout() && subshape.has_layout()) {
                program_subshape->mutable_layout()->set_memory_space(
                    subshape.layout().memory_space());
              }
            }
          });
    }
  }

  bool used_program_shape;
  Shape result_layout;
  if (build_options->result_layout()) {
    result_layout = *build_options->result_layout();
    used_program_shape = false;
  } else {
    result_layout = program_shape.result();
    LayoutUtil::ClearLayout(&result_layout);
    used_program_shape = true;
  }
  TF_RETURN_IF_ERROR(assign_layouts(sharded_shapes.second, &result_layout));
  if (used_program_shape) {
    // Carry memory spaces forward from program shape.
    ShapeUtil::ForEachSubshape(
        program_shape.result(),
        [&](const Shape& subshape, const ShapeIndex& index) {
          if (subshape.IsArray()) {
            Shape* result_subshape =
                ShapeUtil::GetMutableSubshape(&result_layout, index);
            if (result_subshape->has_layout() && subshape.has_layout()) {
              result_subshape->mutable_layout()->set_memory_space(
                  subshape.layout().memory_space());
            }
          }
        });
  }
  build_options->set_result_layout(result_layout);
  return absl::OkStatus();
}

absl::StatusOr<std::vector<int>> ComputeParametersThatMustBeDonated(
    const HloModule& module, bool tuple_inputs) {
  HloComputation* computation = module.entry_computation();
  int number_of_parameters = [&]() -> int {
    if (tuple_inputs) {
      CHECK_EQ(computation->num_parameters(), 1);
      const Shape& input_tuple_shape =
          computation->parameter_instruction(0)->shape();
      CHECK(input_tuple_shape.IsTuple());
      return input_tuple_shape.tuple_shapes().size();
    } else {
      return computation->num_parameters();
    }
  }();
  // If any buffer in a parameter is aliased we will donate the entire input
  // parameter.
  std::vector<int> parameters_to_donate;
  parameters_to_donate.reserve(computation->num_parameters());
  const HloInputOutputAliasConfig& config = module.input_output_alias_config();
  TF_RETURN_IF_ERROR(config.ForEachAliasWithStatus(
      [&](const ShapeIndex& output_index,
          const HloInputOutputAliasConfig::Alias& alias) -> absl::Status {
        if (tuple_inputs) {
          if (alias.parameter_number != 0) {
            return InvalidArgument(
                "Unexpected parameter number %d in alias config with tupled "
                "inputs",
                alias.parameter_number);
          }
          const ShapeIndex& index = alias.parameter_index;
          if (!index.empty()) {
            int this_parameter = index.data()[0];
            if (this_parameter >= number_of_parameters) {
              return InvalidArgument(
                  "Unexpected parameter index %s in alias config with tupled "
                  "inputs and %d parameters",
                  index.ToString(), number_of_parameters);
            }
            parameters_to_donate.push_back(this_parameter);
          }
        } else {
          int this_parameter = alias.parameter_number;
          if (this_parameter >= number_of_parameters) {
            return InvalidArgument(
                "Unexpected parameter number %d in alias config without tupled "
                "inputs and %d parameters",
                this_parameter, number_of_parameters);
          }
          parameters_to_donate.push_back(this_parameter);
        }
        return absl::OkStatus();
      }));
  absl::c_sort(parameters_to_donate);
  return parameters_to_donate;
}

int DefaultThreadPoolSize() {
  // Google's CI system exposes an environment variable NPROC that describes
  // a CPU reservation for tests.
  // TODO(phawkins): expose a better thought-out set of knobs to control
  // parallelism.
  for (const char* nproc_env : {"PJRT_NPROC", "NPROC"}) {
    const char* nproc_str = std::getenv(nproc_env);
    int nproc = 0;
    if (nproc_str && absl::SimpleAtoi(nproc_str, &nproc)) {
      return std::max(0, nproc);
    }
  }
  return tsl::port::MaxParallelism();
}

bool HasMajorToMinorLayout(PrimitiveType type, absl::Span<int64_t const> dims,
                           absl::Span<int64_t const> byte_strides) {
  CHECK_EQ(dims.size(), byte_strides.size());
  // If the array is size 0, the strides are irrelevant.
  if (absl::c_find(dims, 0) != dims.end()) {
    return true;
  }
  int64_t stride = primitive_util::ByteWidth(type);
  for (int i = static_cast<int>(dims.size()) - 1; i >= 0; --i) {
    // If a dimension is of size 1, its stride is irrelevant.
    if (dims[i] != 1) {
      if (byte_strides[i] != stride) {
        return false;
      }
      stride *= dims[i];
    }
  }
  return true;
}

absl::StatusOr<Shape> MakeShapeWithTrivialByteStrides(
    PrimitiveType element_type, absl::Span<const int64_t> dimensions,
    absl::Span<const int64_t> byte_strides) {
  TF_RET_CHECK(dimensions.size() == byte_strides.size());
  std::vector<int64_t> minor_to_major(dimensions.size());
  // Begin with a major-to-minor layout that is likey the most common.
  std::iota(minor_to_major.rbegin(), minor_to_major.rend(), 0);
  // Find minor-to-major only if there is no zero dimension size because
  // minor-to-major is irrelevant with any zero dimension size.
  if (absl::c_find(dimensions, 0) == dimensions.end()) {
    absl::c_sort(minor_to_major, [&](int a, int b) {
      if (byte_strides[a] < byte_strides[b]) {
        return true;
      }
      if (byte_strides[a] > byte_strides[b]) {
        return false;
      }
      return dimensions[a] == 1 && dimensions[b] != 1;
    });
    int64_t byte_stride = ShapeUtil::ByteSizeOfPrimitiveType(element_type);
    for (int64_t d : minor_to_major) {
      if (dimensions[d] != 1 && byte_strides[d] != byte_stride) {
        return Unimplemented(
            "Only trivial (compact) byte strides are supported; i.e., byte "
            "striding represents a transposition of the underlying dense "
            "buffer but not broadcasting. Dimensions were: [%s], byte strides "
            "were [%s].",
            absl::StrJoin(dimensions, ","), absl::StrJoin(byte_strides, ","));
      }
      byte_stride *= dimensions[d];
    }
  }
  return ShapeUtil::MakeShapeWithDenseLayout(element_type, dimensions,
                                             minor_to_major);
}

absl::Status TestBufferDonationClashes(
    void* opaque_key,
    absl::flat_hash_map<const void*, std::pair<bool, int>>& donation_clashes,
    bool is_donated, int arg_idx, int replica, int partition) {
  auto [donation_clash_it, first_use] =
      donation_clashes.emplace(opaque_key, std::make_pair(is_donated, arg_idx));
  if (!first_use && (is_donated || donation_clash_it->second.first)) {
    auto [prev_is_donated, prev_arg_idx] = donation_clash_it->second;
    if (is_donated && prev_is_donated) {
      return InvalidArgument(
          "Attempt to donate the same buffer twice in Execute() ("
          "flattened argument %d, replica %d, partition %d, first use: %d). "
          "Toy "
          "example for this bug: `f(donate(a), donate(a))`.",
          arg_idx, replica, partition, prev_arg_idx);
    } else if (is_donated) {
      return InvalidArgument(
          "Attempt to donate a buffer which is also used by the same call "
          "to Execute() (flattened argument %d, replica %d, partition %d, "
          "first use: %d). Toy example for this bug: `f(a, donate(a))`.",
          arg_idx, replica, partition, prev_arg_idx);
    } else {
      return InvalidArgument(
          "Attempt to use a buffer that was previously donated in the same "
          "call to Execute() (flattened argument %d, replica %d, partition "
          "%d, first use: %d). Toy example for this bug: `f(donate(a), "
          "a)`.",
          arg_idx, replica, partition, prev_arg_idx);
    }
  }
  return absl::OkStatus();
}

void MakeAsciiTitlecase(std::string* s) {
  if (!s->empty()) {
    s->at(0) = absl::ascii_toupper(s->at(0));
  }
}

std::string MakeAsciiTitlecase(absl::string_view s) {
  std::string result(s);
  MakeAsciiTitlecase(&result);
  return result;
}

}  // namespace xla
