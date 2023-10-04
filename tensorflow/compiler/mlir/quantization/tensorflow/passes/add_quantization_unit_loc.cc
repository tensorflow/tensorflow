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
#include <memory>
#include <optional>
#include <string>
#include <utility>

#include "absl/algorithm/container.h"
#include "absl/strings/match.h"
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/Location.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Pass/PassRegistry.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Support/TypeID.h"  // from @llvm-project
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/transforms/passes.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/cc/quantization_unit_loc.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/ops/tf_op_quant_spec.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/passes/passes.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/quantization_options.pb.h"

namespace mlir {
namespace quant {
namespace {

using QuantizationUnit =
    tensorflow::quantization::UnitWiseQuantizationSpec::QuantizationUnit;

// Adds QuantizationUnitLoc to quantizable layers.
class AddQuantizationUnitLocPass
    : public PassWrapper<AddQuantizationUnitLocPass,
                         OperationPass<func::FuncOp>> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(AddQuantizationUnitLocPass)
  explicit AddQuantizationUnitLocPass() = default;

  StringRef getArgument() const final {
    // This is the argument used to refer to the pass in
    // the textual format (on the commandline for example).
    return "quant-add-quantization-unit-loc";
  }
  StringRef getDescription() const final {
    return "Add QuantizationUnitLoc to quantizable layers.";
  }

 private:
  void runOnOperation() override;
};

// TF graph nodes are imported with one of following location patterns:
//   FusedLoc[NameLoc(op_type:), ..., NameLoc(node_name@func_name)] or
//   FusedLoc[NameLoc(op_type:), ..., CallSiteLoc(node_name@func_name)]. See
// tensorflow/compiler/mlir/tensorflow/translate/import_model.cc for more
// details.
bool IsImportLocPattern(FusedLoc loc) {
  ArrayRef<Location> locations = loc.cast<FusedLoc>().getLocations();
  if (locations.size() < 2 || !isa<NameLoc>(locations.front())) return false;

  StringRef op_type_with_suffix =
      locations.front().cast<NameLoc>().getName().strref();
  if (!op_type_with_suffix.ends_with(":")) return false;

  return absl::c_all_of(locations, [](Location loc) {
    return isa<NameLoc>(loc) ||
           (isa<CallSiteLoc>(loc) &&
            isa<NameLoc>(loc.cast<CallSiteLoc>().getCallee()));
  });
}

// Finds the pattern of the location created by `ImporterBase::GetLocation`
// in `tensorflow/compiler/mlir/tensorflow/translate/import_model.cc`.
void FindQuantizationUnitsRecursively(Location loc,
                                      SmallVector<QuantizationUnit>& units) {
  if (!isa<FusedLoc>(loc)) return;

  auto set_node_and_func_name = [](QuantizationUnit& new_unit,
                                   StringRef name_loc_id) {
    if (name_loc_id.contains("@")) {
      new_unit.set_node_name(name_loc_id.split('@').first.str());
      new_unit.set_func_name(name_loc_id.split('@').second.str());
    } else {
      new_unit.set_node_name(name_loc_id.str());
    }
  };

  ArrayRef<Location> locations = loc.cast<FusedLoc>().getLocations();
  if (IsImportLocPattern(loc.cast<FusedLoc>())) {
    QuantizationUnit new_unit;
    // Op type is a NameLoc with the ":" suffix.
    StringRef op_type_with_suffix =
        locations.front().cast<NameLoc>().getName().strref();
    StringRef op_type =
        op_type_with_suffix.substr(0, op_type_with_suffix.size() - 1);
    new_unit.set_op_type(op_type.str());

    if (isa<NameLoc>(locations.back())) {
      StringRef name_loc_id =
          locations.back().cast<NameLoc>().getName().strref();
      set_node_and_func_name(new_unit, name_loc_id);
    } else {
      Location callee = locations.back().cast<CallSiteLoc>().getCallee();
      StringRef name_loc_id = callee.cast<NameLoc>().getName().strref();
      set_node_and_func_name(new_unit, name_loc_id);
    }
    units.push_back(new_unit);
  } else {
    for (Location child_loc : locations) {
      FindQuantizationUnitsRecursively(child_loc, units);
    }
  }
}

// Finds the QuantizationUnit from location.
std::optional<QuantizationUnit> FindQuantizationUnit(Operation* op) {
  SmallVector<QuantizationUnit> quant_units;
  FindQuantizationUnitsRecursively(op->getLoc(), quant_units);

  if (quant_units.size() == 1) {
    return *quant_units.begin();
  }
  // Among units, return the one with the same type as given op.
  StringRef given_op_type = op->getName().getStringRef();
  for (const QuantizationUnit& quant_unit : quant_units) {
    if (absl::StrContains(given_op_type.lower(),
                          StringRef(quant_unit.op_type()).lower())) {
      return quant_unit;
    }
  }

  return std::nullopt;
}

class AddQuantizationUnitLoc : public RewritePattern {
 public:
  explicit AddQuantizationUnitLoc(MLIRContext* context)
      : RewritePattern(MatchAnyOpTypeTag(), /*benefit=*/1, context) {}

 private:
  LogicalResult matchAndRewrite(Operation* op,
                                PatternRewriter& rewriter) const override {
    if (!IsOpWithQuantizableTrait(op) ||
        FindQuantizationUnitFromLoc(op->getLoc()).has_value()) {
      return failure();
    }

    std::optional<QuantizationUnit> quantization_unit =
        FindQuantizationUnit(op);
    if (!quantization_unit.has_value()) return failure();

    if (quantization_unit->func_name().empty()) {
      std::string func_name =
          op->getParentOfType<func::FuncOp>().getSymNameAttr().str();
      quantization_unit->set_func_name(func_name);
    }
    QuantizationUnitLoc unit_loc(getContext(), quantization_unit.value());
    op->setLoc(unit_loc);

    return success();
  }
};

void AddQuantizationUnitLocPass::runOnOperation() {
  MLIRContext* ctx = &getContext();
  RewritePatternSet patterns(ctx);
  func::FuncOp func = getOperation();

  patterns.add<AddQuantizationUnitLoc>(ctx);
  if (failed(applyPatternsAndFoldGreedily(func, std::move(patterns)))) {
    func.emitError() << "quant-add-quantization-unit-loc pattern "
                        "conversion did not converge.";
    signalPassFailure();
  }
}

}  // namespace

// Creates an instance of `AddQuantizationUnitLocPass`.
std::unique_ptr<OperationPass<func::FuncOp>>
CreateAddQuantizationUnitLocPass() {
  return std::make_unique<AddQuantizationUnitLocPass>();
}

static PassRegistration<AddQuantizationUnitLocPass> pass;

}  // namespace quant
}  // namespace mlir
