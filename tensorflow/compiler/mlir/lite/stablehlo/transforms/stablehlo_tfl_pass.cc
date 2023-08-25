/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/compiler/mlir/lite/stablehlo/transforms/stablehlo_tfl_pass.h"

#include <memory>
#include <string>
#include <utility>

#include "flatbuffers/flatbuffers.h"  // from @flatbuffers
#include "flatbuffers/flexbuffers.h"  // from @flatbuffers
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Block.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/Diagnostics.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/OperationSupport.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Pass/PassRegistry.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Transforms/DialectConversion.h"  // from @llvm-project
#include "stablehlo/dialect/StablehloOps.h"  // from @stablehlo
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"

namespace mlir {
namespace odml {

class StablehloToTflPass
    : public mlir::PassWrapper<StablehloToTflPass,
                               mlir::OperationPass<mlir::func::FuncOp>> {
 public:
  explicit StablehloToTflPass() : PassWrapper() {}
  StringRef getArgument() const final { return "stablehlo-tfl"; }
  StringRef getDescription() const final {
    return "This pass will legalize StableHLO Ops to TFLite custom Ops.";
  }

 private:
  void runOnOperation() override;

  void getDependentDialects(DialectRegistry& registry) const override {
    registry.insert<TFL::TensorFlowLiteDialect>();
  }
  inline TFL::ConstBytesAttr CustomOption(OpBuilder* builder,
                                          const std::string& content) {
    return TFL::ConstBytesAttr::get(builder->getContext(),
                                    StringRef(content.data(), content.size()));
  }

  void AddIntegerArray(flexbuffers::Builder* fbb,
                       ::llvm::ArrayRef<int64_t> vec) {
    auto start_input_dim = fbb->StartVector();
    for (auto int_value : vec) {
      fbb->Add(int_value);
    }
    fbb->EndVector(start_input_dim, /*typed=*/false, /*fixed=*/false);
  }
};

void StablehloToTflPass::runOnOperation() {
  func::FuncOp fn = getOperation();
  OpBuilder builder(fn.getContext());
  fn.walk([&](Operation* op) {
    // Process only StableHLO ops.
    if (op->getDialect()->getNamespace() != "stablehlo") return;

    // Build options.
    std::string custom_option_buffer;
    auto fbb = std::make_unique<flexbuffers::Builder>();
    size_t map_start = fbb->StartMap();
    for (auto pair : op->getAttrDictionary().getValue()) {
      const char* key = pair.getName().data();
      const auto attr = pair.getValue();

      if (attr.isa<::mlir::IntegerAttr>()) {
        fbb->Int(key, attr.dyn_cast<mlir::IntegerAttr>().getInt());
        continue;
      }

      if (attr.isa<::mlir::FloatAttr>()) {
        fbb->Double(key, attr.dyn_cast<mlir::FloatAttr>().getValueAsDouble());
        continue;
      }

      if (attr.isa<::mlir::ElementsAttr>()) {
        auto start = fbb->StartVector(key);
        auto array_attr = attr.dyn_cast<mlir::ElementsAttr>();
        const auto ftype = array_attr.getElementType();
        if (ftype.isInteger(16) || ftype.isInteger(32) || ftype.isInteger(64) ||
            ftype.isInteger(128) || ftype.isInteger(1)) {
          for (auto value : array_attr.getValues<IntegerAttr>()) {
            auto int_value =
                value.dyn_cast_or_null<mlir::IntegerAttr>().getInt();
            fbb->Add(int_value);
          }
        } else if (ftype.isF32() || ftype.isF64() || ftype.isF128()) {
          for (auto value : array_attr.getValues<FloatAttr>()) {
            auto double_value =
                value.dyn_cast_or_null<mlir::FloatAttr>().getValueAsDouble();
            fbb->Add(double_value);
          }
        } else {
          emitWarning(op->getLoc(), "serialization of ElementsAttr for ")
              << key << " only supports Integer and Float.";
        }
        fbb->EndVector(start, /*typed=*/true, /*fixed=*/false);
        continue;
      }

      if (attr.isa<::mlir::StringAttr>()) {
        fbb->String(key, attr.dyn_cast<mlir::StringAttr>().data());
        continue;
      }

      if (attr.isa<::mlir::ArrayAttr>()) {
        auto start = fbb->StartVector(key);
        auto array_attr = attr.dyn_cast<mlir::ArrayAttr>();
        if (array_attr.size() > 1 && !array_attr[0].isa<mlir::StringAttr>() &&
            !array_attr[0].isa<mlir::stablehlo::PrecisionAttr>()) {
          emitWarning(op->getLoc(), "serialization of ArrayAttr for ")
              << key << " only supports Strings.";
          continue;
        }
        for (auto value : array_attr) {
          if (value.isa<mlir::stablehlo::PrecisionAttr>()) {
            auto string_value =
                mlir::stablehlo::stringifyPrecision(
                    value.cast<mlir::stablehlo::PrecisionAttr>().getValue())
                    .data();
            fbb->Add(string_value);
          } else {
            auto string_value =
                value.dyn_cast_or_null<mlir::StringAttr>().data();
            fbb->Add(string_value);
          }
        }
        fbb->EndVector(start, /*typed=*/true, /*fixed=*/false);
        continue;
      }

      if (attr.isa<::mlir::stablehlo::ConvDimensionNumbersAttr>()) {
        auto dimension_attr =
            attr.dyn_cast<::mlir::stablehlo::ConvDimensionNumbersAttr>();
        auto start = fbb->StartVector(key);
        fbb->Add(dimension_attr.getInputBatchDimension());
        fbb->Add(dimension_attr.getInputFeatureDimension());
        AddIntegerArray(fbb.get(), dimension_attr.getInputSpatialDimensions());
        fbb->Add(dimension_attr.getKernelInputFeatureDimension());
        fbb->Add(dimension_attr.getKernelOutputFeatureDimension());
        AddIntegerArray(fbb.get(), dimension_attr.getKernelSpatialDimensions());
        fbb->Add(dimension_attr.getOutputBatchDimension());
        fbb->Add(dimension_attr.getOutputFeatureDimension());
        AddIntegerArray(fbb.get(), dimension_attr.getOutputSpatialDimensions());
        fbb->EndVector(start, /*typed=*/false, /*fixed=*/false);
        continue;
      }

      if (attr.isa<::mlir::stablehlo::GatherDimensionNumbersAttr>()) {
        auto dimension_attr =
            attr.dyn_cast<::mlir::stablehlo::GatherDimensionNumbersAttr>();
        auto start = fbb->StartVector(key);
        AddIntegerArray(fbb.get(), dimension_attr.getOffsetDims());
        AddIntegerArray(fbb.get(), dimension_attr.getCollapsedSliceDims());
        AddIntegerArray(fbb.get(), dimension_attr.getStartIndexMap());
        fbb->Add(dimension_attr.getIndexVectorDim());
        fbb->EndVector(start, /*typed=*/false, /*fixed=*/false);
        continue;
      }

      if (attr.isa<::mlir::stablehlo::ScatterDimensionNumbersAttr>()) {
        auto dimension_attr =
            attr.dyn_cast<::mlir::stablehlo::ScatterDimensionNumbersAttr>();
        auto start = fbb->StartVector(key);
        AddIntegerArray(fbb.get(), dimension_attr.getUpdateWindowDims());
        AddIntegerArray(fbb.get(), dimension_attr.getInsertedWindowDims());
        AddIntegerArray(fbb.get(),
                        dimension_attr.getScatterDimsToOperandDims());
        fbb->Add(dimension_attr.getIndexVectorDim());
        fbb->EndVector(start, /*typed=*/false, /*fixed=*/false);
        continue;
      }

      if (attr.isa<::mlir::stablehlo::DotDimensionNumbersAttr>()) {
        auto dimension_attr =
            attr.dyn_cast<::mlir::stablehlo::DotDimensionNumbersAttr>();
        auto start = fbb->StartVector(key);
        AddIntegerArray(fbb.get(), dimension_attr.getLhsBatchingDimensions());
        AddIntegerArray(fbb.get(), dimension_attr.getRhsBatchingDimensions());
        AddIntegerArray(fbb.get(),
                        dimension_attr.getLhsContractingDimensions());
        AddIntegerArray(fbb.get(),
                        dimension_attr.getRhsContractingDimensions());
        fbb->EndVector(start, /*typed=*/false, /*fixed=*/false);
        continue;
      }

      if (attr.isa<::mlir::stablehlo::ComparisonDirectionAttr>()) {
        auto string_value =
            mlir::stablehlo::stringifyComparisonDirection(
                attr.cast<mlir::stablehlo::ComparisonDirectionAttr>()
                    .getValue())
                .str();
        fbb->String(key, string_value);
        continue;
      }

      if (attr.isa<::mlir::stablehlo::ComparisonTypeAttr>()) {
        auto string_value =
            mlir::stablehlo::stringifyComparisonType(
                attr.cast<mlir::stablehlo::ComparisonTypeAttr>().getValue())
                .str();
        fbb->String(key, string_value);
        continue;
      }

      // default
      emitWarning(op->getLoc(), "serialization not supported for : ") << key;
    }
    fbb->EndMap(map_start);
    fbb->Finish();
    custom_option_buffer.assign(fbb->GetBuffer().begin(),
                                fbb->GetBuffer().end());

    // Build custom op.
    builder.setInsertionPoint(op);
    auto tfl_custom_op = builder.create<TFL::CustomOp>(
        op->getLoc(), op->getResultTypes(), op->getOperands(),
        op->getName().getStringRef(),
        CustomOption(&builder, custom_option_buffer));
    op->replaceAllUsesWith(tfl_custom_op);
    op->erase();
  });
}
std::unique_ptr<OperationPass<func::FuncOp>> CreateStablehloToTflPass() {
  return std::make_unique<StablehloToTflPass>();
}

static PassRegistration<StablehloToTflPass> pass;

}  // namespace odml
}  // namespace mlir
