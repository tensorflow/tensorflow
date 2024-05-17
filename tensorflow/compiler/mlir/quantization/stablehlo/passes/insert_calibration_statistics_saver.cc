/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_set>
#include <vector>

#include "absl/strings/string_view.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/SymbolTable.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project  // IWYU pragma: keep
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/quantization/tensorflow/passes/tf_quant_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"  // IWYU pragma: keep
#include "tsl/platform/path.h"

namespace mlir::quant::stablehlo {
namespace {

std::string GetOutputFilePath(absl::string_view calibration_data_dir,
                              absl::string_view func_name,
                              int32_t output_file_idx) {
  return tsl::io::JoinPath(calibration_data_dir,
                           llvm::Twine(func_name)
                               .concat("_")
                               .concat(std::to_string(output_file_idx))
                               .concat(".pb")
                               .str());
}

// Finds `CustomAggregator` ops and collects their outputs and attributes.
void FindCustomAggregatorOps(
    Region& region,
    const std::unordered_set<std::string>& aggregator_ops_to_ignore,
    SmallVector<Value>& statistics_outputs, SmallVector<StringRef>& ids,
    SmallVector<int32_t>& calibration_methods) {
  for (auto op : region.getOps<TF::CustomAggregatorOp>()) {
    if (aggregator_ops_to_ignore.count(op.getId().str())) continue;

    ids.push_back(op.getId());
    calibration_methods.push_back(op.getCalibrationMethod());
    statistics_outputs.push_back(op.getMin());
    statistics_outputs.push_back(op.getMax());
    statistics_outputs.push_back(op.getHistogram());
  }
}

// Inserts a `CalibrationStatisticsSaverOp` to the end of the region.
LogicalResult InsertCalibrationStatisticsSaverOp(
    Region& region, MLIRContext& ctx, absl::string_view output_file_path,
    const std::unordered_set<std::string>& aggregator_ops_to_ignore) {
  SmallVector<Value> statistics_outputs;
  SmallVector<StringRef> ids;
  SmallVector<int32_t> calibration_methods;
  FindCustomAggregatorOps(region, aggregator_ops_to_ignore, statistics_outputs,
                          ids, calibration_methods);
  if (statistics_outputs.empty()) return failure();

  OpBuilder builder(&ctx);
  // Set the insertion point right before the return op.
  builder.setInsertionPoint(&region.back().back());

  StringAttr output_file_path_attr = builder.getStringAttr(output_file_path);
  ArrayAttr ids_attr = builder.getStrArrayAttr(ids);
  ArrayAttr calibration_methods_attr =
      builder.getI32ArrayAttr(calibration_methods);
  builder.create<TF::CalibrationStatisticsSaverOp>(
      region.getLoc(), statistics_outputs, output_file_path_attr, ids_attr,
      calibration_methods_attr);
  return success();
}

// Returns true if the op contains a `CalibrationStatisticsSaverOp`.
bool ContainCalibrationStatisticsSaverOp(Operation* op) {
  // Check the region for CaseRegionOp, IfRegionOp and WhileRegionOp.
  for (Region& region : op->getRegions()) {
    if (!region.getOps<TF::CalibrationStatisticsSaverOp>().empty()) {
      return true;
    }
  }

  SymbolTable symbol_table(op->getParentOfType<ModuleOp>());
  // Check the functions associated to CaseOp, IfOp and WhileOp.
  for (const NamedAttribute& attr : op->getAttrs()) {
    FlatSymbolRefAttr symbol_attr =
        dyn_cast_or_null<FlatSymbolRefAttr>(attr.getValue());
    if (!symbol_attr) continue;

    func::FuncOp target_func = dyn_cast_or_null<func::FuncOp>(
        symbol_table.lookup(symbol_attr.getValue()));
    if (!target_func) continue;

    if (!target_func.getBody()
             .getOps<TF::CalibrationStatisticsSaverOp>()
             .empty()) {
      return true;
    }
  }
  return false;
}

}  // namespace

#define GEN_PASS_DECL_INSERTCALIBRATIONSTATISTICSSAVERPASS
#define GEN_PASS_DEF_INSERTCALIBRATIONSTATISTICSSAVERPASS
#include "tensorflow/compiler/mlir/quantization/stablehlo/passes/passes.h.inc"

class InsertCalibrationStatisticsSaverPass
    : public impl::InsertCalibrationStatisticsSaverPassBase<
          InsertCalibrationStatisticsSaverPass> {
 public:
  using impl::InsertCalibrationStatisticsSaverPassBase<
      InsertCalibrationStatisticsSaverPass>::
      InsertCalibrationStatisticsSaverPassBase;

 private:
  void runOnOperation() override;
};

void InsertCalibrationStatisticsSaverPass::runOnOperation() {
  ModuleOp module_op = getOperation();
  MLIRContext& ctx = getContext();

  std::unordered_set<std::string> aggregator_ops_to_ignore(
      aggregator_ops_to_ignore_.begin(), aggregator_ops_to_ignore_.end());

  // Insert CalibrationStatisticsSaverOp to the end of each region.
  for (auto func_op : module_op.getOps<func::FuncOp>()) {
    int32_t output_file_idx = 0;
    StringRef func_name = func_op.getSymName();

    func_op.walk([&output_file_idx, &ctx, &func_name, &aggregator_ops_to_ignore,
                  this](Operation* op) {
      for (Region& region : op->getRegions()) {
        if (succeeded(InsertCalibrationStatisticsSaverOp(
                region, ctx,
                GetOutputFilePath(calibration_data_dir_, func_name,
                                  output_file_idx),
                aggregator_ops_to_ignore))) {
          ++output_file_idx;
        };
      }
    });
  }

  // Control flow ops that contains CalibrationStatisticsSaver ops must be set
  // to stateful, otherwise the op will not be executed.
  OpBuilder builder(&ctx);
  module_op.walk([&builder](Operation* op) {
    if (op->hasAttrOfType<BoolAttr>("is_stateless") &&
        ContainCalibrationStatisticsSaverOp(op)) {
      op->setAttr("is_stateless", builder.getBoolAttr(false));
    }
  });
}

std::unique_ptr<OperationPass<ModuleOp>>
CreateInsertCalibrationStatisticsSaverPass(
    StringRef calibration_data_dir,
    const std::vector<std::string>& aggregator_ops_to_ignore) {
  InsertCalibrationStatisticsSaverPassOptions options = {
      .aggregator_ops_to_ignore_ = aggregator_ops_to_ignore,
      .calibration_data_dir_ = calibration_data_dir.str(),
  };
  return std::make_unique<InsertCalibrationStatisticsSaverPass>(options);
}

}  // namespace mlir::quant::stablehlo
