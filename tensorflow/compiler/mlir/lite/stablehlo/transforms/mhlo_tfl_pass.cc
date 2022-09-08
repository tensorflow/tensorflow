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
#include "tensorflow/compiler/mlir/lite/stablehlo/transforms/mhlo_tfl_pass.h"

#include <memory>
#include <string>
#include <utility>

#include "flatbuffers/flatbuffers.h"  // from @flatbuffers
#include "flatbuffers/flexbuffers.h"  // from @flatbuffers
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Block.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
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
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"
#include "tensorflow/compiler/xla/mlir_hlo/include/mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "tensorflow/compiler/xla/mlir_hlo/include/mlir-hlo/Dialect/mhlo/IR/register.h"
#include "tensorflow/compiler/xla/mlir_hlo/stablehlo/stablehlo/dialect/ChloOps.h"
#include "tensorflow/compiler/xla/mlir_hlo/stablehlo/stablehlo/dialect/Register.h"

namespace mlir {
namespace TFL {
namespace mhlo {

class MhloToTflPass
    : public mlir::PassWrapper<MhloToTflPass,
                               mlir::OperationPass<mlir::func::FuncOp>> {
 public:
  explicit MhloToTflPass() : PassWrapper() {}
  StringRef getArgument() const final { return "mhlo-tfl"; }
  StringRef getDescription() const final {
    return "This pass will legalize MHLO Ops to TFLite custom Ops.";
  }

 private:
  void runOnOperation() override;

  void getDependentDialects(DialectRegistry &registry) const override {
    mlir::mhlo::registerAllMhloDialects(registry);
    mlir::stablehlo::registerAllDialects(registry);
    registry.insert<mlir::func::FuncDialect, mlir::arith::ArithmeticDialect>();
    registry.insert<::mlir::mhlo::MhloDialect>();
    registry.insert<shape::ShapeDialect>();
    registry.insert<TensorFlowLiteDialect>();
  }
  inline ConstBytesAttr CustomOption(OpBuilder* builder,
                                     const std::string& content) {
    return ConstBytesAttr::get(builder->getContext(),
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

void MhloToTflPass::runOnOperation() {
  func::FuncOp fn = getOperation();
  OpBuilder builder(fn.getContext());
  fn.walk([&](Operation *op) {
    // Process only MHLO ops.
    if (op->getDialect()->getNamespace() != "mhlo") return;

    // Build options.
    std::string custom_option_buffer;
    auto fbb = std::make_unique<flexbuffers::Builder>();
    size_t map_start = fbb->StartMap();
    for (auto pair : op->getAttrDictionary().getValue()) {
      const char* key = pair.getName().data();
      const auto attr = pair.getValue();
      if (attr.isa<::mlir::IntegerAttr>()) {
        fbb->Int(key, attr.dyn_cast<mlir::IntegerAttr>().getInt());
      } else if (attr.isa<::mlir::ElementsAttr>()) {
        auto start = fbb->StartVector(key);
        auto array_attr = attr.dyn_cast<mlir::ElementsAttr>();
        for (auto value : array_attr.getValues<IntegerAttr>()) {
          auto int_value = value.dyn_cast_or_null<mlir::IntegerAttr>().getInt();
          fbb->Add(int_value);
        }
        fbb->EndVector(start, /*typed=*/true, /*fixed=*/false);
      } else if (attr.isa<::mlir::StringAttr>()) {
        fbb->String(key, attr.dyn_cast<mlir::StringAttr>().data());
      } else if (attr.isa<::mlir::ArrayAttr>()) {
        auto start = fbb->StartVector(key);
        auto array_attr = attr.dyn_cast<mlir::ArrayAttr>();
        if (array_attr.size() > 1 && !array_attr[0].isa<mlir::StringAttr>() &&
            !array_attr[0].isa<mlir::mhlo::PrecisionAttr>()) {
          emitWarning(op->getLoc(), "seralization of ArrayAttr for ")
              << key << " only supports Strings.";
          continue;
        }
        for (auto value : array_attr) {
          if (value.isa<mlir::mhlo::PrecisionAttr>()) {
            auto string_value =
                mlir::mhlo::stringifyPrecision(
                    value.cast<mlir::mhlo::PrecisionAttr>().getValue())
                    .data();
            fbb->Add(string_value);
          } else {
            auto string_value =
                value.dyn_cast_or_null<mlir::StringAttr>().data();
            fbb->Add(string_value);
          }
        }
        fbb->EndVector(start, /*typed=*/true, /*fixed=*/false);
      } else if (attr.isa<::mlir::mhlo::ConvDimensionNumbersAttr>()) {
        auto dimension_attr =
            attr.dyn_cast<::mlir::mhlo::ConvDimensionNumbersAttr>();
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
      } else {
        emitWarning(op->getLoc(), "seralization not supported for : ") << key;
      }
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
std::unique_ptr<OperationPass<func::FuncOp>> CreateMhloToTflPass() {
  return std::make_unique<MhloToTflPass>();
}

static PassRegistration<MhloToTflPass> pass;

}  // namespace mhlo
}  // namespace TFL
}  // namespace mlir
