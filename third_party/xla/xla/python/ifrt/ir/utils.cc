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

#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Bytecode/BytecodeWriter.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/LLVM.h"
#include "xla/hlo/ir/hlo_sharding.h"
#include "xla/mlir/utils/type_util.h"
#include "xla/python/ifrt/array_spec.h"
#include "xla/python/ifrt/client.h"
#include "xla/python/ifrt/device.h"
#include "xla/python/ifrt/device_list.h"
#include "xla/python/ifrt/dtype.h"
#include "xla/python/ifrt/ir/ifrt_dialect.h"
#include "xla/python/ifrt/ir/ifrt_ops.h"
#include "xla/python/ifrt/ir/support/module_parsing.h"
#include "xla/python/ifrt/ir/support/sharding_conversions.h"
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
      TF_RET_CHECK(devices[logical_id] != nullptr);
      array_devices.push_back(devices[logical_id]);
    }
    ABSL_ASSIGN_OR_RETURN(array_device_list,
                     client->MakeDeviceList(std::move(array_devices)));
  }

  IfrtShardingParamAttr sharding_attr = GetShardingParamAttr(array_type);

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

}  // namespace ifrt
}  // namespace xla
