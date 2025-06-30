/* Copyright 2024 The OpenXLA Authors.

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

#include "xla/python/ifrt/ir/transforms/utils.h"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "xla/mlir/utils/type_util.h"
#include "xla/python/ifrt/dtype.h"
#include "xla/python/ifrt/ir/ifrt_dialect.h"
#include "xla/python/ifrt/ir/ifrt_ops.h"
#include "xla/python/pjrt_ifrt/pjrt_dtype.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/fingerprint.h"

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
          loc_stream << "\t ^ " << name_loc.getName() << "\n";
          return true;
        };
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
  if (auto call_loc = GetCallSiteLoc(loc)) {
    // Print the file location from the current loc.
    RecursivelyPrintLoc(*call_loc, loc_stream);
    // Print the file locations of the callers.
    GetPrettyLocation(call_loc->getCaller(), loc_stream);
  } else if (auto file_loc = mlir::dyn_cast<mlir::FileLineColLoc>(loc)) {
    PrintFileLoc(file_loc, loc_stream);
  }
}

}  // namespace

std::string GetPrettyLocation(mlir::Location loc) {
  std::string loc_str;
  llvm::raw_string_ostream loc_stream(loc_str);
  GetPrettyLocation(loc, loc_stream);
  return loc_str;
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
  if (lhs == getEmptyKey() || lhs == getTombstoneKey() ||
      rhs == getEmptyKey() || rhs == getTombstoneKey()) {
    return false;
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

void UpdateFunctionType(mlir::func::FuncOp func_op) {
  func_op.setType(mlir::FunctionType::get(
      func_op.getContext(), func_op.getBody().getArgumentTypes(),
      func_op.getBody().front().getTerminator()->getOperandTypes()));
}

absl::StatusOr<DType> ToIfrtDType(mlir::Type type) {
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

mlir::ModuleOp CloneModuleUsingBuilder(mlir::ModuleOp module,
                                       mlir::OpBuilder& builder) {
  // Create a stub for the new module.
  mlir::ModuleOp cloned_module =
      builder.create<mlir::ModuleOp>(module.getLoc(), module.getName());
  cloned_module->setAttrs(module->getAttrs());
  mlir::IRMapping mapper;
  // Clone each operation in the body of the module into the new module.
  for (mlir::Operation& op : module.getBody()->getOperations()) {
    cloned_module.getBody()->push_back(op.clone(mapper));
  }
  return cloned_module;
}

absl::StatusOr<std::vector<std::string>> ExpandPlatformNames(
    const mlir::Pass::ListOption<std::string>& platform_names) {
  std::vector<std::string> expanded_platform_names;
  for (const auto& platform_entry : platform_names) {
    std::vector<absl::string_view> parts = absl::StrSplit(platform_entry, ':');
    if (parts.size() == 1) {
      expanded_platform_names.push_back(platform_entry);
    } else if (parts.size() == 2) {
      std::string platform_name(parts[0]);
      int num_occurences;
      if (!absl::SimpleAtoi(parts[1], &num_occurences)) {
        return absl::InvalidArgumentError(
            absl::StrCat("Invalid platform name: `", platform_entry,
                         "` in platform_names pass option"));
      }
      for (int i = 0; i < num_occurences; ++i) {
        expanded_platform_names.push_back(platform_name);
      }
    } else {
      return absl::InvalidArgumentError(
          absl::StrCat("Invalid platform name: `", platform_entry,
                       "` in platform_names pass option"));
    }
  }
  return expanded_platform_names;
}

uint64_t MlirModuleFingerprint(mlir::ModuleOp module) {
  std::string s;
  llvm::raw_string_ostream os(s);
  mlir::OpPrintingFlags flags;
  flags.enableDebugInfo(false);
  module.print(os, flags);
  return tsl::Fingerprint64(os.str());
}

}  // namespace ifrt
}  // namespace xla
