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

#include "xla/python/ifrt/ir/utils.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/container/btree_map.h"
#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Bytecode/BytecodeWriter.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "xla/hlo/ir/hlo_sharding.h"
#include "xla/mlir/utils/type_util.h"
#include "xla/pjrt/pjrt_layout.h"
#include "xla/python/ifrt/array_spec.h"
#include "xla/python/ifrt/client.h"
#include "xla/python/ifrt/device.h"
#include "xla/python/ifrt/device_list.h"
#include "xla/python/ifrt/dtype.h"
#include "xla/python/ifrt/ir/constants.h"
#include "xla/python/ifrt/ir/ifrt_dialect.h"
#include "xla/python/ifrt/ir/ifrt_ops.h"
#include "xla/python/ifrt/ir/support/module_parsing.h"
#include "xla/python/ifrt/ir/support/sharding_conversions.h"
#include "xla/python/ifrt/memory.h"
#include "xla/python/ifrt/remap_plan.h"
#include "xla/python/ifrt/rtti.h"
#include "xla/python/ifrt/shape.h"
#include "xla/python/ifrt/sharding.h"
#include "xla/python/pjrt_ifrt/pjrt_dtype.h"
#include "xla/python/pjrt_ifrt/xla_sharding.h"
#include "xla/status_macros.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace ifrt {

namespace {

// Finds a nested call site location in the given location.
std::optional<mlir::CallSiteLoc> GetCallSiteLoc(mlir::Location loc) {
  if (mlir::dyn_cast<mlir::NameLoc>(loc)) {
    return GetCallSiteLoc(mlir::cast<mlir::NameLoc>(loc).getChildLoc());
  }
  if (auto callLoc = mlir::dyn_cast<mlir::CallSiteLoc>(loc)) {
    return callLoc;
  }
  if (mlir::dyn_cast<mlir::FusedLoc>(loc)) {
    for (auto subLoc : mlir::cast<mlir::FusedLoc>(loc).getLocations()) {
      // If fused return the first call site location.
      if (auto callLoc = GetCallSiteLoc(subLoc)) {
        return callLoc;
      }
    }
    return std::nullopt;
  }
  return std::nullopt;
}

void PrintFileLoc(mlir::FileLineColLoc file_loc,
                  llvm::raw_string_ostream& loc_stream) {
  if (file_loc.getFilename().str() != "-") {
    loc_stream << file_loc.getFilename();
  } else {
    // The location printed is from the MLIR module.
    loc_stream << "mlir";
  }
  loc_stream << ":" << file_loc.getLine() << ":" << file_loc.getStartColumn()
             << " to " << file_loc.getEndColumn() << "\n";
}

// Recurses into the child locations of some of location types to find a nested
// file location and prints info if it is found. Returns true if a file location
// is found.
bool RecursivelyPrintLoc(mlir::Location loc,
                         llvm::raw_string_ostream& loc_stream) {
  return llvm::TypeSwitch<mlir::LocationAttr, bool>(loc)
      .Case([&](mlir::CallSiteLoc call_loc) -> bool {
        // We recurse into the callee of a call site, as the caller will be
        // emitted in a different note on the main diagnostic.
        return RecursivelyPrintLoc(call_loc.getCallee(), loc_stream);
      })
      .Case([&](mlir::FileLineColLoc file_loc) -> bool {
        PrintFileLoc(file_loc, loc_stream);
        return true;
      })
      .Case([&](mlir::FusedLoc fused_loc) -> bool {
        // Fused location is unique in that we try to find a sub-location to
        // show, rather than the top-level location itself.
        for (mlir::Location childLoc : fused_loc.getLocations()) {
          if (RecursivelyPrintLoc(childLoc, loc_stream)) {
            return true;
          }
        }
        return false;
      })
      .Case([&](mlir::NameLoc name_loc) -> bool {
        if (RecursivelyPrintLoc(name_loc.getChildLoc(), loc_stream)) {
          if (!name_loc.getName().empty()) {
            loc_stream << "\t ^ " << name_loc.getName() << "\n";
          }
          return true;
        }
        if (!name_loc.getName().empty()) {
          loc_stream << name_loc.getName() << "\n";
          return true;
        }
        return false;
      })
      .Case([&](mlir::OpaqueLoc opaque_loc) -> bool {
        // OpaqueLoc always falls back to a different source location.
        return RecursivelyPrintLoc(opaque_loc.getFallbackLocation(),
                                   loc_stream);
      })
      .Case([](mlir::UnknownLoc) -> bool {
        // Prefer not to show unknown locations.
        return false;
      });
}

void GetPrettyLocation(mlir::Location loc,
                       llvm::raw_string_ostream& loc_stream) {
  loc_stream << "\t";
  if (mlir::isa<mlir::UnknownLoc>(loc)) {
    loc_stream << "<unknown location>\n";
    return;
  }
  if (std::optional<mlir::CallSiteLoc> call_loc = GetCallSiteLoc(loc)) {
    // Print the file location from the current loc.
    RecursivelyPrintLoc(loc, loc_stream);
    // Print the file locations of the callers.
    GetPrettyLocation(call_loc->getCaller(), loc_stream);
  } else if (!RecursivelyPrintLoc(loc, loc_stream)) {
    loc_stream << "<unknown location>\n";
  }
}

}  // namespace

std::string GetPrettyLocation(mlir::Location loc) {
  std::string loc_str;
  llvm::raw_string_ostream loc_stream(loc_str);
  GetPrettyLocation(loc, loc_stream);
  if (loc_str.empty()) {
    return "\t<unknown location>";
  }
  return loc_str;
}

std::string GetArgPrettyLocation(int index, mlir::ModuleOp module) {
  mlir::func::FuncOp func = GetMainFunction(module);
  CHECK_GE(index, 0);
  CHECK_LT(index, func.getNumArguments());
  return GetPrettyLocation(func.getArgument(index).getLoc());
}

unsigned IfrtCallOpInfo::getHashValue(CallOp call_op) {
  llvm::hash_code hash = {};
  // Use `getInputs()/getOutputs()` instead of `getOperands()/getResults()` to
  // ensure that the control dependencies are not included in the hash.
  for (auto input_type : call_op.getInputs().getTypes()) {
    hash = llvm::hash_combine(hash, input_type);
  }
  for (auto output_type : call_op.getOutputs().getTypes()) {
    hash = llvm::hash_combine(hash, output_type);
  }
  for (mlir::NamedAttribute attr : call_op->getAttrs()) {
    // Exclude `operandSegmentSizes` because its value changes depending on
    // how many control dependencies a CallOp has.
    if (attr.getName() == "operandSegmentSizes") {
      continue;
    }
    hash = llvm::hash_combine(hash, attr);
  }
  return hash;
}

bool IfrtCallOpInfo::isEqual(CallOp lhs, CallOp rhs) {
  if (lhs == rhs) {
    return true;
  }
  // Verify that the input and output types are the same.
  if (lhs.getInputs().getTypes() != rhs.getInputs().getTypes()) {
    return false;
  }
  if (lhs.getOutputs().getTypes() != rhs.getOutputs().getTypes()) {
    return false;
  }
  mlir::NamedAttrList lattrs = lhs->getAttrDictionary();
  mlir::NamedAttrList rattrs = rhs->getAttrDictionary();
  lattrs.erase("operandSegmentSizes");
  rattrs.erase("operandSegmentSizes");
  // Verify that the attributes are the same.
  return lattrs == rattrs;
}

mlir::func::FuncOp GetMainFunction(mlir::ModuleOp module) {
  mlir::func::FuncOp func =
      mlir::dyn_cast_or_null<mlir::func::FuncOp>(module.lookupSymbol("main"));
  CHECK(func);
  return func;
}

llvm::SmallVector<mlir::func::FuncOp> GetIfrtFunctions(mlir::ModuleOp module) {
  llvm::SmallVector<mlir::func::FuncOp> ifrt_funcs;
  for (mlir::func::FuncOp func_op : module.getOps<mlir::func::FuncOp>()) {
    if (IsIfrtFunction(func_op)) {
      ifrt_funcs.push_back(func_op);
    }
  }
  return ifrt_funcs;
}

mlir::SymbolRefAttr MakeOutlinedCalleeSymbolRef(mlir::ModuleOp callee_module,
                                                mlir::func::FuncOp callee) {
  return mlir::SymbolRefAttr::get(
      callee_module.getSymNameAttr(),
      {mlir::FlatSymbolRefAttr::get(callee.getSymNameAttr())});
}

mlir::Operation* GetCalleeRootSymbolOp(mlir::func::FuncOp callee,
                                       mlir::ModuleOp root_module) {
  if (callee == nullptr) {
    return nullptr;
  }
  mlir::Operation* parent = callee->getParentOp();
  if (parent == root_module.getOperation()) {
    return callee.getOperation();
  }
  if (mlir::ModuleOp callee_module =
          llvm::dyn_cast_or_null<mlir::ModuleOp>(parent);
      callee_module != nullptr &&
      callee_module->getParentOp() == root_module.getOperation()) {
    return callee_module.getOperation();
  }
  return nullptr;
}

mlir::FailureOr<mlir::ModuleOp> GetOutlinedAtomProgramModule(
    CallOp call_op, mlir::SymbolTableCollection& symbol_table) {
  mlir::func::FuncOp callee = call_op.getCalleeOp(symbol_table);
  mlir::ModuleOp callee_module =
      llvm::dyn_cast<mlir::ModuleOp>(callee->getParentOp());
  mlir::ModuleOp call_parent_module =
      call_op->getParentOfType<mlir::ModuleOp>();
  if (callee.getSymName() != kCalleeMainFuncName || callee_module == nullptr ||
      call_parent_module == nullptr ||
      callee_module->getParentOp() != call_parent_module.getOperation()) {
    call_op.emitOpError() << "requires callee outlined as `"
                          << kCalleeMainFuncName
                          << "` function in a child ModuleOp. Actual callee "
                             "name: "
                          << callee.getSymName() << ". Actual callee parent: "
                          << callee->getParentOp()->getName();
    return mlir::failure();
  }
  return callee_module;
}

ClonedAtomProgramModule CloneOutlinedAtomProgramModule(
    mlir::ModuleOp callee_module, mlir::SymbolTableCollection& symbol_table) {
  mlir::ModuleOp cloned_module = callee_module.clone();
  mlir::SymbolTable::setSymbolVisibility(
      cloned_module, mlir::SymbolTable::Visibility::Private);
  mlir::Operation* parent_op =
      callee_module->getParentWithTrait<mlir::OpTrait::SymbolTable>();
  CHECK(parent_op != nullptr);
  symbol_table.getSymbolTable(parent_op).insert(
      cloned_module, ++mlir::Block::iterator(callee_module));
  mlir::func::FuncOp cloned_main = GetMainFunction(cloned_module);
  return ClonedAtomProgramModule{
      cloned_module, cloned_main,
      MakeOutlinedCalleeSymbolRef(cloned_module, cloned_main)};
}

bool IsTrueOrUnitAttr(mlir::Attribute attr) {
  if (attr == nullptr) {
    return false;
  }
  if (auto bool_attr = llvm::dyn_cast<mlir::BoolAttr>(attr);
      bool_attr != nullptr) {
    return bool_attr.getValue();
  }
  return llvm::isa<mlir::UnitAttr>(attr);
}

bool HasDonatedOrAliasedDictAttr(mlir::DictionaryAttr dict) {
  if (dict == nullptr) {
    return false;
  }
  return dict.get(kAliasingOutputAttrName) != nullptr ||
         IsTrueOrUnitAttr(dict.get(kBufferDonationAttrName)) ||
         IsTrueOrUnitAttr(dict.get(kIfrtDonatedArgAttrName));
}

bool IsTokenArrayType(mlir::Type type) {
  if (type == nullptr) {
    return false;
  }
  auto array_type = llvm::dyn_cast<IfrtArrayType>(type);
  return array_type != nullptr &&
         llvm::isa<IfrtTokenType>(array_type.getShape().getElementType());
}

std::optional<unsigned> GetArrayInputIndex(mlir::OperandRange inputs,
                                           unsigned operand_num) {
  if (inputs.empty() || operand_num < inputs.getBeginOperandIndex() ||
      operand_num >= inputs.getBeginOperandIndex() + inputs.size()) {
    return std::nullopt;
  }
  return operand_num - inputs.getBeginOperandIndex();
}

bool IsReshard(IfrtArrayType from, IfrtArrayType to) {
  if (from.getShape() == to.getShape() &&
      from.getShardingAttr() == to.getShardingAttr() &&
      from.getDevices().size() == to.getDevices().size()) {
    return false;
  }
  return true;
}

absl::StatusOr<DType> ToIfrtDType(mlir::Type type) {
  if (llvm::isa<IfrtTokenType>(type)) {
    return ToDType(xla::PrimitiveType::TOKEN);
  }
  if (llvm::isa<IfrtStringType>(type)) {
    return DType(DType::kString);
  }
  xla::PrimitiveType primitive_type = xla::ConvertMlirTypeToPrimitiveType(type);
  return ToDType(primitive_type);
}

std::string OperationToString(mlir::Operation* op,
                              const mlir::OpPrintingFlags& flags) {
  std::string out;
  {
    llvm::raw_string_ostream os(out);
    op->print(os, flags);
  }
  return out;
}

absl::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>> CloneModuleIntoContext(
    mlir::ModuleOp module, mlir::MLIRContext& context) {
  std::string bytecode;
  llvm::raw_string_ostream os(bytecode);
  TF_RET_CHECK(mlir::succeeded(mlir::writeBytecodeToFile(module, os)));
  return support::ParseMlirModuleString(bytecode, context);
}

absl::StatusOr<ShardingRef> ShardingFromIfrtArrayType(
    const IfrtArrayType array_type, Client* client,
    const DeviceListRef& device_list) {
  DeviceListRef array_device_list;
  {
    std::vector<Device*> array_devices;
    array_devices.reserve(array_type.getDevices().size());
    absl::Span<Device* const> devices = device_list->devices();
    for (int logical_id : array_type.getDevices()) {
      if (logical_id < 0 || logical_id >= devices.size()) {
        return absl::InvalidArgumentError(absl::StrFormat(
            "Logical device id %d is out of range for device list of size %d",
            logical_id, devices.size()));
      }
      TF_RET_CHECK(devices[logical_id] != nullptr);
      array_devices.push_back(devices[logical_id]);
    }
    ABSL_ASSIGN_OR_RETURN(array_device_list,
                     client->MakeDeviceList(std::move(array_devices)));
  }

  IfrtShardingParamAttr sharding_attr =
      mlir::dyn_cast_or_null<IfrtShardingParamAttr>(
          array_type.getShardingAttr());
  if (sharding_attr == nullptr) {
    return absl::InvalidArgumentError(
        "Array type sharding attribute must be `IfrtShardingParamAttr`");
  }

  ABSL_ASSIGN_OR_RETURN(
      xla::HloSharding hlo_sharding,
      xla::ifrt::support::ToHloSharding(sharding_attr.getSharding()));
  return xla::ifrt::HloSharding::Create(std::move(array_device_list),
                                        array_type.MemoryKind(),
                                        std::move(hlo_sharding));
}

absl::StatusOr<ArraySpec> ArraySpecFromMlirType(
    mlir::Type array_type, Client* client, const DeviceListRef& device_list) {
  IfrtArrayType ifrt_array_type = GetArrayType(array_type);

  ABSL_ASSIGN_OR_RETURN(DType dtype,
                   ToIfrtDType(ifrt_array_type.getShape().getElementType()));

  ABSL_ASSIGN_OR_RETURN(
      ShardingRef sharding,
      ShardingFromIfrtArrayType(ifrt_array_type, client, device_list));
  return ArraySpec{
      /*dtype=*/dtype,
      /*shape=*/Shape(ifrt_array_type.getShape().getShape()),
      /*sharding=*/std::move(sharding),
  };
}

// Returns a DeviceList for the given device ids.
absl::StatusOr<DeviceListRef> LookUpDevices(Client* client,
                                            absl::Span<const DeviceId> ids) {
  std::vector<Device*> devices;
  devices.reserve(ids.size());
  for (DeviceId id : ids) {
    ABSL_ASSIGN_OR_RETURN(devices.emplace_back(), client->LookupDevice(id));
  }
  return client->MakeDeviceList(devices);
}

namespace {

// Resolves a subset `DeviceListRef` of size `count` from `device_list` by
// stepping through `intervals`.
absl::StatusOr<DeviceListRef> ComputeDeviceListFromIntervals(
    Client* absl_nonnull client, const DeviceListRef& device_list,
    int64_t count, absl::Span<const IfrtIntervalAttr> intervals) {
  TF_RET_CHECK(count >= 0);
  std::vector<Device*> devices;
  devices.reserve(count);
  for (const IfrtIntervalAttr& interval : intervals) {
    TF_RET_CHECK(interval.getStep() > 0);
    int64_t index = interval.getStart();
    while (index < interval.getEnd()) {
      TF_RET_CHECK(index >= 0 && index < device_list->size());
      devices.push_back(device_list->devices()[index]);
      index += interval.getStep();
    }
  }
  TF_RET_CHECK(devices.size() == count);
  return client->MakeDeviceList(devices);
}

using InputDevicesForOutputMap =
    absl::flat_hash_map<int, std::vector<RemapPlan::InputDeviceRange>>;

// Computes the per-output `InputDeviceRange` map for `remap_op`.
absl::StatusOr<InputDevicesForOutputMap> ComputeInputDevicesForOutputMap(
    Client* absl_nonnull client, absl::Span<const ArraySpec> input_specs,
    absl::Span<const ArraySpec> output_specs, RemapArraysOp remap_op) {
  struct IntervalsAndCount {
    std::vector<IfrtIntervalAttr> intervals;
    int64_t count = 0;
  };

  // Using `btree_map` ensures deterministic iteration order across runs.
  absl::btree_map<int, absl::btree_map<int, IntervalsAndCount>>
      output_to_inputs_and_intervals;
  for (const mlir::Attribute array_mapping : remap_op.getMappings()) {
    const auto array_mapping_attr =
        llvm::cast<IfrtArrayMappingAttr>(array_mapping);
    int in_array = array_mapping_attr.getInArrayIndex();
    int out_array = array_mapping_attr.getOutArrayIndex();
    TF_RET_CHECK(in_array >= 0 && in_array < input_specs.size());
    TF_RET_CHECK(out_array >= 0 && out_array < output_specs.size());

    IntervalsAndCount& intervals =
        output_to_inputs_and_intervals[out_array][in_array];
    for (const mlir::Attribute m : array_mapping_attr.getMappings()) {
      const auto mapping_attr = llvm::cast<IfrtMappingAttr>(m);
      IfrtIntervalAttr from_shards = mapping_attr.getFromShards();
      intervals.intervals.push_back(from_shards);
      intervals.count += from_shards.size();
    }
  }

  absl::flat_hash_map<int, std::vector<RemapPlan::InputDeviceRange>>
      input_devices_for_output_map;
  input_devices_for_output_map.reserve(output_specs.size());
  for (int out_array = 0; out_array < output_specs.size(); ++out_array) {
    input_devices_for_output_map.insert({out_array, {}});
  }

  for (const auto& [out_array, input_intervals] :
       output_to_inputs_and_intervals) {
    TF_RET_CHECK(out_array >= 0 && out_array < output_specs.size());
    const DeviceListRef& out_devices =
        output_specs[out_array].sharding->devices();
    std::vector<RemapPlan::InputDeviceRange>& out_input_devices =
        input_devices_for_output_map[out_array];
    for (const auto& [in_array, intervals] : input_intervals) {
      TF_RET_CHECK(in_array >= 0 && in_array < input_specs.size());
      const DeviceListRef& in_devices =
          input_specs[in_array].sharding->devices();
      TF_RET_CHECK(intervals.count >= 0 &&
                   intervals.count <= out_devices->size());
      TF_RET_CHECK(intervals.count >= 0 &&
                   intervals.count <= in_devices->size());
      if (intervals.count == in_devices->size() &&
          intervals.intervals.size() == 1 &&
          intervals.intervals[0].getStart() == 0 &&
          intervals.intervals[0].getStep() == 1) {
        out_input_devices.push_back(
            RemapPlan::InputDeviceRange{/*in_array=*/in_array,
                                        /*input_devices=*/in_devices});
      } else {
        ABSL_ASSIGN_OR_RETURN(
            DeviceListRef interval_device_list,
            ComputeDeviceListFromIntervals(client, in_devices, intervals.count,
                                           intervals.intervals));
        out_input_devices.push_back(RemapPlan::InputDeviceRange{
            /*in_array=*/in_array,
            /*input_devices=*/std::move(interval_device_list)});
      }
    }
  }
  return input_devices_for_output_map;
}

}  // namespace

absl::StatusOr<RemapPlan> BuildRemapPlan(Client* absl_nonnull client,
                                         const DeviceListRef& device_list,
                                         RemapArraysOp remap_op) {
  std::vector<ArraySpec> input_specs;
  input_specs.reserve(remap_op.getInputs().size());
  for (const mlir::Value input : remap_op.getInputs()) {
    ABSL_ASSIGN_OR_RETURN(ArraySpec spec, ArraySpecFromMlirType(
                                         input.getType(), client, device_list));
    input_specs.push_back(std::move(spec));
  }

  std::vector<ArraySpec> output_specs;
  output_specs.reserve(remap_op.getOutputs().size());
  for (const mlir::Value output : remap_op.getOutputs()) {
    ABSL_ASSIGN_OR_RETURN(
        ArraySpec spec,
        ArraySpecFromMlirType(output.getType(), client, device_list));
    output_specs.push_back(std::move(spec));
  }

  ABSL_ASSIGN_OR_RETURN(InputDevicesForOutputMap input_devices_for_output_map,
                   ComputeInputDevicesForOutputMap(client, input_specs,
                                                   output_specs, remap_op));
  RemapPlan remap_plan(std::move(input_specs), std::move(output_specs),
                       std::move(input_devices_for_output_map));
  ABSL_RETURN_IF_ERROR(remap_plan.Validate());
  return remap_plan;
}

std::optional<std::vector<xla::OpSharding>> GetOpShardings(
    absl::Span<const ArraySpec> specs, absl::string_view kind) {
  std::vector<xla::OpSharding> shardings;
  shardings.reserve(specs.size());
  for (size_t idx = 0; idx < specs.size(); ++idx) {
    const ArraySpec& spec = specs[idx];
    const auto* hlo_sharding = dyn_cast<HloSharding>(spec.sharding.get());
    if (hlo_sharding == nullptr) {
      LOG(ERROR) << "Failed to get sharding of " << kind << " #" << idx
                 << ". Sharding is not an HloSharding.";
      return std::nullopt;
    }
    shardings.push_back(hlo_sharding->xla_hlo_sharding().ToProto());
  }
  return shardings;
}

absl::StatusOr<std::vector<std::shared_ptr<const xla::PjRtLayout>>> GetLayouts(
    const absl::Status& layout_status, absl::Span<const ArraySpec> specs) {
  ABSL_RETURN_IF_ERROR(layout_status);
  std::vector<std::shared_ptr<const xla::PjRtLayout>> layouts;
  layouts.reserve(specs.size());
  for (const ArraySpec& spec : specs) {
    layouts.push_back(spec.layout);
  }
  return layouts;
}

absl::StatusOr<std::vector<std::vector<absl::string_view>>>
GetOutputMemoryKinds(absl::Span<const ArraySpec> out_specs) {
  std::vector<absl::string_view> output_memory_kinds;
  output_memory_kinds.reserve(out_specs.size());
  for (const ArraySpec& spec : out_specs) {
    output_memory_kinds.push_back(spec.sharding->memory_kind().value());
  }
  // Return a single outer entry representing the top-level IFRT IR program
  // outputs rather than per-atom memory kinds (which have no single program
  // output ordering).
  return std::vector<std::vector<absl::string_view>>{
      std::move(output_memory_kinds)};
}

}  // namespace ifrt
}  // namespace xla
