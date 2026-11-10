/* Copyright 2026 The OpenXLA Authors.

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

#include "xla/pjrt/infer_dispatch_info.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/DebugStringHelper.h"
#include "mlir/Support/LLVM.h"
#include "shardy/dialect/sdy/ir/dialect.h"
#include "shardy/dialect/sdy/ir/utils.h"
#include "xla/hlo/analysis/hlo_dataflow_analysis.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_input_output_alias_config.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instruction_utils.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/ir/hlo_sharding.h"
#include "xla/hlo/translate/mhlo_to_hlo/type_to_shape.h"
#include "xla/mlir/utils/type_util.h"
#include "xla/pjrt/dynamic_shapes.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_compiler.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/pjrt/pjrt_layout.h"
#include "xla/pjrt/utils.h"
#include "xla/service/computation_layout.h"
#include "xla/service/device_assignment.h"
#include "xla/service/hlo_value.h"
#include "xla/service/spmd/shardy/stablehlo_round_trip/export_shardings.h"
#include "xla/shape.h"
#include "xla/shape_layout.h"
#include "xla/shape_util.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"

namespace xla {

std::vector<Shape> GetParameterShapes(const ComputationLayout& layout) {
  // For now, XLA programs compiled with multiple arguments for PJRT cannot use
  // tuples for any of their arguments, so we can assume that a tuple can only
  // arise when there is a single argument.
  std::vector<Shape> shapes;
  if (layout.parameter_count() == 1 && layout.parameter_shape(0).IsTuple()) {
    shapes.reserve(layout.parameter_shape(0).tuple_shapes().size());
    absl::c_copy(layout.parameter_shape(0).tuple_shapes(),
                 std::back_inserter(shapes));
  } else {
    shapes.reserve(layout.parameter_count());
    for (const ShapeLayout& sl : layout.parameter_layouts()) {
      shapes.push_back(sl.shape());
    }
  }
  return shapes;
}

namespace {

bool IsPadRealToStaticOp(mlir::Operation* op) {
  auto attr = op->getAttrOfType<mlir::StringAttr>("call_target_name");
  return attr != nullptr && attr.getValue() == "PadRealToStatic";
}

bool ShapeIndexOverlaps(const ShapeIndex& a, const ShapeIndex& b) {
  const size_t min_size = std::min(a.size(), b.size());
  for (size_t i = 0; i < min_size; ++i) {
    if (a[i] != b[i]) {
      return false;
    }
  }
  return true;
}

bool HasOutputToOperandAlias(const HloInstruction* instruction,
                             int64_t operand_number,
                             const ShapeIndex& operand_index) {
  return absl::c_any_of(
      instruction->output_operand_aliasing(), [&](const auto& alias) {
        return alias.second.first == operand_number &&
               ShapeIndexOverlaps(alias.second.second, operand_index);
      });
}

// Maps a data operand `(async_user, operand_number)` on a `kAsyncStart` or
// `kAsyncUpdate` instruction to the bound parameter index in
// `async_user->async_wrapped_computation()`, or `std::nullopt` if the chain is
// incomplete or `operand_number` is not a bound data operand.
std::optional<int64_t> GetAsyncBoundParameterNumber(
    const HloInstruction* async_user, int64_t operand_number) {
  if (async_user->opcode() != HloOpcode::kAsyncStart &&
      async_user->opcode() != HloOpcode::kAsyncUpdate) {
    return std::nullopt;
  }
  const int64_t user_start_idx =
      async_user->opcode() == HloOpcode::kAsyncStart ? 0 : 1;
  if (operand_number < user_start_idx ||
      operand_number >= async_user->operand_count()) {
    return std::nullopt;
  }
  const HloInstruction* start = async_user->async_chain_start();
  const HloInstruction* done = async_user->async_chain_done();
  const HloComputation* wrapped = async_user->async_wrapped_computation();
  const HloAsyncInstruction* done_async =
      done != nullptr ? DynCast<HloAsyncInstruction>(done) : nullptr;
  const HloAsyncStartInstruction* start_async =
      start != nullptr ? DynCast<HloAsyncStartInstruction>(start) : nullptr;
  if (start_async == nullptr || done_async == nullptr || wrapped == nullptr) {
    return std::nullopt;
  }
  const std::vector<const HloInstruction*> bound_operands =
      hlo_instruction_utils::async::GetAsyncBoundOperands(done_async);
  if (bound_operands.size() != static_cast<size_t>(wrapped->num_parameters())) {
    return std::nullopt;
  }
  int64_t offset = 0;
  for (const HloInstruction* instr : start_async->GetAsyncChain()) {
    const int64_t start_idx = instr->opcode() == HloOpcode::kAsyncStart ? 0 : 1;
    if (instr == async_user) {
      const int64_t param_no = offset + (operand_number - user_start_idx);
      if (param_no < 0 || param_no >= wrapped->num_parameters()) {
        return std::nullopt;
      }
      return param_no;
    }
    if (instr == done) {
      break;
    }
    if (instr->operand_count() > start_idx) {
      offset += instr->operand_count() - start_idx;
    }
  }
  return std::nullopt;
}

// Returns true if `use` of the entry parameter `value` is compatible with the
// zero-prefix compact buffer that `PadRealToStatic` reads: the pad itself
// (operand 0), prefix-agnostic forwarding (`kBitcast`, `kOptimizationBarrier`,
// `kAddDependency`), or a positional subcomputation boundary (`kCall`,
// `kAsyncStart` / `kAsyncUpdate` / `kAsyncDone`) through which
// `HloDataflowAnalysis` forwards `value` unchanged, so that every reader of
// the forwarded value is itself verified by the caller's use walk.
bool IsAllowedPadRealToStaticUse(const HloDataflowAnalysis& dataflow,
                                 const HloValue& value, const HloUse& use) {
  const HloInstruction* user = use.instruction;
  switch (user->opcode()) {
    case HloOpcode::kCustomCall:
      return user->IsCustomCall("PadRealToStatic") && use.operand_number == 0 &&
             use.operand_index.empty();
    case HloOpcode::kBitcast:
    case HloOpcode::kOptimizationBarrier:
      return true;
    case HloOpcode::kAddDependency:
      return use.operand_number == 0;
    case HloOpcode::kCall: {
      const HloComputation* callee = user->to_apply();
      if (callee == nullptr || use.operand_number >= callee->num_parameters() ||
          HasOutputToOperandAlias(user, use.operand_number,
                                  use.operand_index)) {
        return false;
      }
      const HloInstruction* param =
          callee->parameter_instruction(use.operand_number);
      if (!ShapeUtil::IndexIsValid(param->shape(), use.operand_index)) {
        return false;
      }
      return absl::c_linear_search(
          dataflow.GetValueSet(param, use.operand_index).values(), &value);
    }
    case HloOpcode::kAsyncStart:
    case HloOpcode::kAsyncUpdate: {
      if (user->async_chain_start() == nullptr ||
          user->async_chain_done() == nullptr ||
          user->async_wrapped_computation() == nullptr) {
        return false;
      }
      if (user->opcode() == HloOpcode::kAsyncUpdate &&
          use.operand_number == 0) {
        return !use.operand_index.empty() && use.operand_index.front() == 0 &&
               !HasOutputToOperandAlias(user, 0, use.operand_index);
      }
      const std::optional<int64_t> param_no =
          GetAsyncBoundParameterNumber(user, use.operand_number);
      if (!param_no.has_value() ||
          HasOutputToOperandAlias(user, use.operand_number,
                                  use.operand_index) ||
          HasOutputToOperandAlias(user->async_chain_start(), *param_no,
                                  use.operand_index)) {
        return false;
      }
      const HloInstruction* param =
          user->async_wrapped_computation()->parameter_instruction(*param_no);
      if (!ShapeUtil::IndexIsValid(param->shape(), use.operand_index)) {
        return false;
      }
      return absl::c_linear_search(
          dataflow.GetValueSet(param, use.operand_index).values(), &value);
    }
    case HloOpcode::kAsyncDone:
      // Not on the parameter -> pad path: dataflow keeps the bound inputs in
      // element {0} of the async context tuple until `async-done` consumes it.
      return user->async_chain_start() != nullptr && use.operand_number == 0 &&
             !use.operand_index.empty() && use.operand_index.front() == 0;
    default:
      return false;
  }
}

absl::Status VerifyOnlyPadRealToStaticReadsValue(
    const HloDataflowAnalysis& dataflow, const HloValue& value,
    const HloValue& entry_parameter_value) {
  const HloInstruction* parameter =
      entry_parameter_value.defining_instruction();
  const ShapeIndex& leaf_index = entry_parameter_value.defining_index();
  auto format_error = [&](absl::string_view detail,
                          absl::string_view via_name) {
    return InvalidArgument(
        "Entry parameter %d%s of %s feeds PadRealToStatic and is also %s "
        "(through %s). A dynamic parameter consumed by PadRealToStatic is "
        "passed as a compact buffer without dynamic-shape metadata, which no "
        "other instruction can read, so it must have no other users.",
        parameter->parameter_number(),
        leaf_index.empty() ? "" : absl::StrCat(" ", leaf_index.ToString()),
        parameter->GetModule()->name(), detail, via_name);
  };

  for (const HloPosition& pos : value.positions()) {
    if (pos.instruction->IsRoot()) {
      return format_error(absl::StrCat("returned as a result of computation ",
                                       pos.instruction->parent()->name()),
                          pos.instruction->name());
    }
  }
  for (const HloUse& use : value.GetUses()) {
    if (!IsAllowedPadRealToStaticUse(dataflow, value, use)) {
      const HloInstruction* via = use.instruction->operand(use.operand_number);
      return format_error(absl::StrCat("used by ", use.instruction->ToString()),
                          via->name());
    }
  }
  return absl::OkStatus();
}

}  // namespace

absl::Status ClearPadRealToStaticParameterMetadataPrefixes(
    const HloModule& hlo_module, absl::Span<Shape> parameter_shapes) {
  const HloComputation* entry_comp = hlo_module.entry_computation();
  if (entry_comp == nullptr) {
    return absl::OkStatus();
  }

  // Zero-allocation fast path: only modules with a dynamic `PadRealToStatic`
  // operand pay for the dataflow analysis. Offloaded pads live in non-entry
  // computations, so scan every computation.
  std::vector<const HloInstruction*> pad_calls;
  for (const HloComputation* comp : hlo_module.computations()) {
    for (const HloInstruction* instruction : comp->instructions()) {
      if (instruction->IsCustomCall("PadRealToStatic") &&
          instruction->operand_count() > 0 &&
          instruction->operand(0)->shape().is_dynamic()) {
        pad_calls.push_back(instruction);
      }
    }
  }
  if (pad_calls.empty()) {
    return absl::OkStatus();
  }

  ABSL_ASSIGN_OR_RETURN(std::unique_ptr<HloDataflowAnalysis> dataflow,
                   HloDataflowAnalysis::Run(hlo_module));

  const auto& layout = hlo_module.entry_computation_layout();
  const bool flattened_tuple_parameter =
      layout.parameter_count() == 1 && layout.parameter_shape(0).IsTuple();

  // Dataflow forwards entry parameter values unchanged through tuples,
  // `kCall`, and async chains, so every value a pad reads must be defined by an
  // entry parameter; anything else (array copies, reshards, dynamic producers)
  // would carry the 1024-byte metadata prefix the pad does not expect.
  absl::flat_hash_set<HloValue::Id> seen_entry_values;
  std::vector<const HloValue*> entry_parameter_values;
  for (const HloInstruction* pad : pad_calls) {
    const std::vector<const HloValue*>& values =
        dataflow->GetValueSet(pad->operand(0), {}).values();
    if (values.empty()) {
      return InvalidArgument(
          "Dynamic operand of PadRealToStatic (%s) in %s does not resolve to "
          "an entry computation parameter.",
          pad->ToString(), hlo_module.name());
    }
    for (const HloValue* value : values) {
      const HloInstruction* def = value->defining_instruction();
      if (def->opcode() != HloOpcode::kParameter ||
          def->parent() != entry_comp) {
        return InvalidArgument(
            "Dynamic operand of PadRealToStatic (%s) in %s must read an entry "
            "computation parameter directly without intermediate copies or "
            "reshards, but resolves to %s.",
            pad->ToString(), hlo_module.name(), def->ToString());
      }
      if (seen_entry_values.insert(value->id()).second) {
        entry_parameter_values.push_back(value);
      }
    }
  }

  std::vector<const HloValue*> parameter_values;
  parameter_values.reserve(entry_parameter_values.size());
  for (const HloValue* value : entry_parameter_values) {
    const HloInstruction* def = value->defining_instruction();
    const Shape& leaf_shape =
        ShapeUtil::GetSubshape(def->shape(), value->defining_index());
    if (!leaf_shape.IsArray() || !leaf_shape.is_dynamic()) {
      continue;
    }
    ABSL_RETURN_IF_ERROR(
        VerifyOnlyPadRealToStaticReadsValue(*dataflow, *value, *value));
    if (std::optional<ShapeIndex> aliased_output =
            hlo_module.input_output_alias_config().GetAliasedOutput(
                def->parameter_number(), value->defining_index());
        aliased_output.has_value()) {
      return InvalidArgument(
          "Entry parameter %d%s of %s feeds PadRealToStatic and is aliased "
          "to output %s. A dynamic parameter consumed by PadRealToStatic is "
          "passed as a compact runtime-sized buffer without dynamic-shape "
          "metadata, so it cannot back an aliased or donated output.",
          def->parameter_number(),
          value->defining_index().empty()
              ? ""
              : absl::StrCat(" ", value->defining_index().ToString()),
          hlo_module.name(), aliased_output->ToString());
    }
    parameter_values.push_back(value);
  }

  // Mutate `parameter_shapes` only after all validations succeed so an error
  // leaves `parameter_shapes` untouched.
  for (const HloValue* value : parameter_values) {
    int64_t shape_no = value->defining_instruction()->parameter_number();
    ShapeIndex index = value->defining_index();
    if (flattened_tuple_parameter) {
      if (shape_no != 0 || index.empty()) {
        continue;
      }
      shape_no = index.front();
      index = ShapeIndex(index.begin() + 1, index.end());
    }
    if (shape_no < 0 || shape_no >= parameter_shapes.size() ||
        !ShapeUtil::IndexIsValid(parameter_shapes[shape_no], index)) {
      continue;
    }
    Shape* subshape =
        ShapeUtil::GetMutableSubshape(&parameter_shapes[shape_no], index);
    if (subshape->IsArray() && subshape->has_layout()) {
      subshape->mutable_layout()->set_dynamic_shape_metadata_prefix_bytes(0);
    }
  }
  return absl::OkStatus();
}

absl::StatusOr<PjRtLoadedExecutableDispatchInfo> InferDispatchInfo(
    const PjRtTopologyDescription* topology,
    std::vector<Shape> parameter_device_shapes, Shape output_device_shape,
    const HloInputOutputAliasConfig& alias_config,
    std::shared_ptr<DeviceAssignment> device_assignment,
    std::vector<PjRtLoadedExecutable::LogicalDeviceIds>
        addressable_device_logical_ids,
    std::vector<PjRtDevice*> addressable_devices,
    std::unique_ptr<PjRtLoadedExecutableDispatchInfo::Extras> extras,
    bool tuple_inputs,
    std::unique_ptr<PjRtLoadedExecutableDispatchInfo::InputHloSnapshotBits>
        input_hlo_snapshot_bits) {
  PjRtLoadedExecutableDispatchInfo result{
      .parameter_device_shapes = std::move(parameter_device_shapes),
      .output_device_shape =
          std::make_shared<const Shape>(std::move(output_device_shape)),
      .addressable_devices = std::move(addressable_devices),
      .addressable_device_logical_ids =
          std::move(addressable_device_logical_ids),
      .device_assignment = std::move(device_assignment),
      .input_hlo_snapshot_bits = std::move(input_hlo_snapshot_bits),
      .extras = std::move(extras),
  };
  for (const auto& shape : result.parameter_device_shapes) {
    ABSL_ASSIGN_OR_RETURN(int kind, topology->GetMemorySpaceKindForShape(shape));
    result.parameter_memory_space_kind_ids.push_back(kind);
  }
  {
    absl::Span<const Shape> shapes =
        result.output_device_shape->IsTuple()
            ? absl::MakeSpan(result.output_device_shape->tuple_shapes())
            : absl::MakeSpan(&*result.output_device_shape, 1);
    result.output_memory_space_kind_ids.reserve(shapes.size());
    for (const auto& shape : shapes) {
      ABSL_ASSIGN_OR_RETURN(int kind, topology->GetMemorySpaceKindForShape(shape));
      result.output_memory_space_kind_ids.push_back(kind);
    }
  }
  // Initializes information about which arguments to which executables may
  // be donated due to aliases that were specified by the computation.
  ABSL_ASSIGN_OR_RETURN(
      result.parameters_that_may_be_donated,
      ComputeParametersThatMayBeDonated(
          alias_config, result.parameter_device_shapes.size(), tuple_inputs));
  result.input_buffer_sizes_in_bytes.reserve(
      result.parameter_device_shapes.size());
  for (const Shape& shape : result.parameter_device_shapes) {
    DCHECK(!shape.IsTuple());
    ABSL_ASSIGN_OR_RETURN(int64_t size_in_bytes, PjRtGetOnDeviceBytesCount(shape));
    result.input_buffer_sizes_in_bytes.push_back(size_in_bytes);
  }
  return result;
}

namespace {

absl::StatusOr<std::vector<int64_t>> GetShardShape(
    const xla::HloSharding& sharding, llvm::ArrayRef<int64_t> shape,
    size_t num_devices) {
  if (sharding.IsReplicatedOrSingleDevice() || sharding.IsManual() ||
      sharding.IsUnreduced() || sharding.IsUnknown()) {
    return std::vector<int64_t>{shape.begin(), shape.end()};
  }
  if (sharding.TotalNumTiles() != num_devices) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "sharding's tile count and device count does not "
        "match: %d vs. %d; shape=%s, sharding=%s",
        sharding.TotalNumTiles(), num_devices, "??", sharding.ToString()));
  }
  if (shape.size() != sharding.TiledDataRank()) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Numbers of dimensions don't match. From Shape %d vs from "
        "HloSharding %d",
        shape.size(), sharding.TiledDataRank()));
  }
  const absl::Span<const int64_t> sharding_dims = sharding.dimensions();
  std::vector<int64_t> tile_shape;
  tile_shape.reserve(shape.size());
  for (int64_t i = 0; i < shape.size(); ++i) {
    tile_shape.push_back(shape[i] == Shape::kUnboundedSize
                             ? Shape::kUnboundedSize
                             : xla::CeilOfRatio(shape[i], sharding_dims[i]));
  }
  return tile_shape;
}

}  // namespace

absl::StatusOr<PjRtLoadedExecutableDispatchInfo> InferDispatchInfo(
    const PjRtTopologyDescription* topology, mlir::ModuleOp mlir_module,
    const CompileOptions& options,
    std::shared_ptr<DeviceAssignment> device_assignment,
    std::vector<PjRtLoadedExecutable::LogicalDeviceIds>
        addressable_device_logical_ids,
    std::vector<PjRtDevice*> addressable_devices, bool tuple_inputs) {
  if (!device_assignment) {
    return absl::UnimplementedError(
        "Async compilation requires a device_assignment");
  }
  auto extras = std::make_unique<PjRtLoadedExecutableDispatchInfo::Extras>();
  extras->name =
      std::string(mlir_module.getSymName().value_or("?unknown program name?"));
  extras->num_partitions = device_assignment->replica_count();
  extras->num_replicas = device_assignment->replica_count();
  auto mesh = mlir_module.getOps<mlir::sdy::MeshOp>();
  if (mesh.empty() || std::next(mesh.begin()) != mesh.end()) {
    return absl::UnimplementedError("Exactly one shardy mesh is required.");
  }
  auto main = mlir_module.lookupSymbol<mlir::func::FuncOp>("main");
  if (main == nullptr) {
    return absl::UnimplementedError(
        "InferAvalFromModule called on module without main function.");
  }
  auto sharding_fn = [&](mlir::sdy::TensorShardingAttr sharding) {
    return sharding.getMesh(main);
  };
  auto get_xla_shape = [&](xla::HloSharding sharding, mlir::Type type)
      -> absl::StatusOr<
          std::tuple<xla::Shape, std::shared_ptr<const xla::PjRtLayout>>> {
    if (mlir::isa<mlir::UnrankedTensorType>(type)) {
      return absl::InvalidArgumentError(
          absl::StrCat("Unsupported type ", mlir::debugString(type)));
    }
    std::vector<int64_t> shard_shape;
    xla::PrimitiveType primitive_type;
    std::vector<bool> dynamic_dimensions;
    if (auto tensor_type = mlir::dyn_cast<mlir::RankedTensorType>(type)) {
      xla::Shape unsharded_shape = xla::TypeToShape(tensor_type);
      if (unsharded_shape.element_type() == PRIMITIVE_TYPE_INVALID) {
        return absl::InvalidArgumentError(
            absl::StrCat("Unsupported type ", mlir::debugString(type)));
      }
      llvm::ArrayRef<int64_t> dims(unsharded_shape.dimensions().data(),
                                   unsharded_shape.dimensions().size());
      ABSL_ASSIGN_OR_RETURN(shard_shape, GetShardShape(sharding, dims,
                                                  extras->num_replicas *
                                                      extras->num_partitions));
      primitive_type = unsharded_shape.element_type();
      dynamic_dimensions.assign(unsharded_shape.dynamic_dimensions().begin(),
                                unsharded_shape.dynamic_dimensions().end());
    } else {
      primitive_type = xla::ConvertMlirTypeToPrimitiveType(type);
    }
    ABSL_ASSIGN_OR_RETURN(auto* memory_space,
                     addressable_devices[0]->default_memory_space());
    xla::Shape xla_shard_shape = xla::ShapeUtil::MakeShape(
        primitive_type, shard_shape, dynamic_dimensions);
    // TODO(parkers): Fix the nullptr layout.
    ABSL_ASSIGN_OR_RETURN(auto xla_shape,
                     topology->MakeCanonicalShapeForMemorySpace(
                         memory_space->kind_id(), xla_shard_shape, nullptr));
    auto layout = std::make_shared<PjRtLayout>(xla_shape.layout());
    return std::make_tuple(xla_shape, layout);
  };
  std::vector<xla::Shape> parameter_device_shapes;
  extras->parameter_shardings.emplace();
  extras->parameter_layouts = std::vector<std::shared_ptr<const PjRtLayout>>();
  for (size_t i = 0; i < main.getNumArguments(); ++i) {
    auto sharding_attr = mlir::sdy::getSharding(main.getArgument(i));
    if (sharding_attr == nullptr) {
      return absl::InvalidArgumentError(
          absl::StrFormat("Arg[%d] is missing a sharding", i));
    }
    xla::HloSharding hlo_sharding =
        xla::sdy::convertToHloSharding(sharding_attr, sharding_fn,
                                       /*manualAxes=*/{});
    xla::Shape shape;
    std::shared_ptr<const xla::PjRtLayout> layout;
    ABSL_ASSIGN_OR_RETURN((std::tie(shape, layout)),
                     get_xla_shape(hlo_sharding, main.getArgumentTypes()[i]));

    // Inputs feeding PadRealToStatic custom calls are bounded dynamic tensors
    // supplied without the TPU hardware dynamic metadata prefix (0 bytes).
    // On the async-compile path, CommonPjRtClient::CompileAndLoad builds
    // DispatchInfo directly from this MLIR pass without calling LoadInternal,
    // whereas sync/loaded executables validate and clear prefixes post-compile
    // in ClearPadRealToStaticParameterMetadataPrefixes.
    if (shape.is_dynamic()) {
      mlir::BlockArgument arg = main.getArgument(i);
      // Operand 0 of PadRealToStatic is the unpadded data tensor; operands
      // 1..rank are the `s32` runtime dimension sizes.
      const bool is_pad_real_to_static_input =
          llvm::any_of(arg.getUses(), [](mlir::OpOperand& use) {
            return use.getOperandNumber() == 0 &&
                   IsPadRealToStaticOp(use.getOwner());
          });
      if (is_pad_real_to_static_input) {
        if (!arg.hasOneUse()) {
          return absl::InvalidArgumentError(absl::StrFormat(
              "Entry argument %d feeds PadRealToStatic and has other users; "
              "PadRealToStatic dynamic inputs have no metadata prefix and "
              "must have no other users.",
              i));
        }
        if (main.getArgAttr(i, "tf.aliasing_output") != nullptr) {
          return absl::InvalidArgumentError(absl::StrFormat(
              "Entry argument %d feeds PadRealToStatic and is aliased via "
              "tf.aliasing_output; compact PadRealToStatic buffers cannot back "
              "aliased or donated outputs.",
              i));
        }
        shape.mutable_layout()->set_dynamic_shape_metadata_prefix_bytes(0);
        layout = std::make_shared<PjRtLayout>(shape.layout());
      }
    }

    parameter_device_shapes.push_back(shape);
    extras->parameter_shardings->push_back(hlo_sharding.ToProto());
    extras->parameter_layouts->push_back(std::move(layout));
  }
  std::vector<xla::Shape> result_shapes;
  extras->output_shardings.emplace();
  extras->output_layouts = std::vector<std::shared_ptr<const PjRtLayout>>();
  for (size_t i = 0; i < main.getNumResults(); ++i) {
    auto sharding_attr = mlir::sdy::getFuncResultSharding(main, i);
    // sharding_attr.dump();
    if (sharding_attr == nullptr) {
      return absl::InvalidArgumentError(
          absl::StrFormat("Result[%d] is missing a sharding", i));
    }
    xla::HloSharding hlo_sharding =
        xla::sdy::convertToHloSharding(sharding_attr, sharding_fn,
                                       /*manualAxes=*/{});

    xla::Shape shape;
    std::shared_ptr<const xla::PjRtLayout> layout;
    ABSL_ASSIGN_OR_RETURN((std::tie(shape, layout)),
                     get_xla_shape(hlo_sharding, main.getResultTypes()[i]));

    result_shapes.push_back(std::move(shape));
    extras->output_shardings->push_back(hlo_sharding.ToProto());
    extras->output_layouts->push_back(std::move(layout));
  }
  Shape output_device_shape = (result_shapes.size() == 1)
                                  ? std::move(result_shapes[0])
                                  : xla::Shape(std::move(result_shapes));

  // TODO(parkers): implement aliasing.
  extras->input_output_alias_config =
      HloInputOutputAliasConfig(output_device_shape);

  const auto& input_output_alias_config = extras->input_output_alias_config;
  ABSL_ASSIGN_OR_RETURN(
      auto result,
      InferDispatchInfo(topology, std::move(parameter_device_shapes),
                        std::move(output_device_shape),
                        input_output_alias_config, std::move(device_assignment),
                        std::move(addressable_device_logical_ids),
                        std::move(addressable_devices), std::move(extras),
                        tuple_inputs));

  result.extras->fingerprint = "";
  result.extras->parameter_memory_kinds.reserve(
      result.parameter_memory_space_kind_ids.size());
  for (size_t i = 0; i < result.parameter_memory_space_kind_ids.size(); ++i) {
    auto* device = result.addressable_devices[0];
    ABSL_ASSIGN_OR_RETURN(auto* memory_space, device->default_memory_space());
    for (auto* ms : device->memory_spaces()) {
      if (ms->kind_id() == result.parameter_memory_space_kind_ids[i]) {
        memory_space = ms;
      }
    }
    result.extras->parameter_memory_kinds.push_back(memory_space->kind());
  }
  result.extras->output_memory_kinds.reserve(
      result.output_memory_space_kind_ids.size());
  for (size_t i = 0; i < result.output_memory_space_kind_ids.size(); ++i) {
    auto* device = result.addressable_devices[0];
    ABSL_ASSIGN_OR_RETURN(auto* memory_space, device->default_memory_space());
    for (auto* ms : device->memory_spaces()) {
      if (ms->kind_id() == result.output_memory_space_kind_ids[i]) {
        memory_space = ms;
      }
    }
    result.extras->output_memory_kinds.push_back(memory_space->kind());
  }
  return result;
}

}  // namespace xla
