/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/compiler/mlir/quantization/tensorflow/cc/quantization_unit_loc.h"

#include <cstddef>
#include <optional>
#include <string>
#include <string_view>

#include "absl/strings/str_cat.h"
#include "llvm/Support/Casting.h"
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/Location.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project

namespace mlir {
namespace quant {
namespace {

// Prefix and suffix to the QuantizationUnit string representation.
constexpr std::string_view kQuantizationUnitPrefix = "QuantizationUnit(";
constexpr std::string_view kQuantizationUnitSuffix = ")";

// Concatenates node name and func name with a "@" separator.
std::string ConcatNodeAndFuncName(std::string_view node_name,
                                  std::string_view func_name) {
  return absl::StrCat(node_name, "@", func_name);
}

// Generate a string to represent the QuantizationUnit.
std::string GenerateQuantizationUnitString(
    const QuantizationUnitLoc::QuantizationUnit& unit) {
  return absl::StrCat(kQuantizationUnitPrefix, unit.SerializeAsString(),
                      kQuantizationUnitSuffix);
}

}  // namespace

QuantizationUnitLoc::QuantizationUnitLoc(MLIRContext* context,
                                         const QuantizationUnit& unit)
    : CallSiteLoc(CallSiteLoc::get(
          /*callee=*/NameLoc::get(
              StringAttr::get(context, ConcatNodeAndFuncName(unit.node_name(),
                                                             unit.func_name())),
              /*childLoc=*/NameLoc::get(
                  StringAttr::get(context, unit.op_type()))),
          /*caller=*/NameLoc::get(StringAttr::get(
              context, GenerateQuantizationUnitString(unit))))) {}

bool QuantizationUnitLoc::classof(Attribute attr) {
  if (!llvm::isa<CallSiteLoc>(attr)) return false;
  auto callsite_loc = llvm::dyn_cast<CallSiteLoc>(attr);

  if (!mlir::isa<NameLoc>(callsite_loc.getCaller())) return false;
  StringRef caller_name =
      mlir::cast<NameLoc>(callsite_loc.getCaller()).getName().strref();
  return caller_name.starts_with(kQuantizationUnitPrefix) &&
         caller_name.ends_with(kQuantizationUnitSuffix);
}

std::optional<QuantizationUnitLoc::QuantizationUnit>
FindQuantizationUnitFromLoc(Location loc) {
  if (isa<QuantizationUnitLoc>(loc)) {
    Location caller = mlir::cast<CallSiteLoc>(loc).getCaller();
    StringRef caller_name = mlir::cast<NameLoc>(caller).getName().strref();
    const size_t start_index = kQuantizationUnitPrefix.size();
    const size_t end_index = caller_name.rfind(kQuantizationUnitSuffix);
    std::string serialized_proto =
        caller_name.substr(start_index, end_index - start_index).str();
    QuantizationUnitLoc::QuantizationUnit quant_unit;
    if (quant_unit.ParseFromString(serialized_proto)) {
      return quant_unit;
    }
  } else if (isa<FusedLoc>(loc)) {
    // If the op is rewritten, FusedLoc can be created.
    for (Location child_loc : mlir::cast<FusedLoc>(loc).getLocations()) {
      std::optional<QuantizationUnitLoc::QuantizationUnit> found_unit =
          FindQuantizationUnitFromLoc(child_loc);
      if (found_unit.has_value()) return found_unit;
    }
  } else if (isa<CallSiteLoc>(loc)) {
    // If the graph is inlined, CallSiteLoc can be created.
    return FindQuantizationUnitFromLoc(
        mlir::cast<CallSiteLoc>(loc).getCallee());
  }

  return std::nullopt;
}

}  // namespace quant
}  // namespace mlir
