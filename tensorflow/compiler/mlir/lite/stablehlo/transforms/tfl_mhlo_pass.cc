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
#include "tensorflow/compiler/mlir/lite/stablehlo/transforms/tfl_mhlo_pass.h"

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
#include "stablehlo/dialect/ChloOps.h"  // from @stablehlo
#include "stablehlo/dialect/Register.h"  // from @stablehlo
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/dynamic_shape_utils.h"
#include "tensorflow/compiler/xla/mlir_hlo/mhlo/IR/hlo_ops.h"
#include "tensorflow/compiler/xla/mlir_hlo/mhlo/IR/register.h"

namespace mlir {
namespace TFL {
namespace mhlo {

class TflToMhloPass
    : public mlir::PassWrapper<TflToMhloPass,
                               mlir::OperationPass<mlir::func::FuncOp>> {
 public:
  explicit TflToMhloPass() : PassWrapper() {}
  StringRef getArgument() const final { return "tfl-parse-mhlo-ops"; }
  StringRef getDescription() const final {
    return "This pass will legalize TFLite custom Ops to MHLO ops.";
  }

 private:
  void runOnOperation() override;

  void getDependentDialects(DialectRegistry& registry) const override {
    mlir::mhlo::registerAllMhloDialects(registry);
    mlir::stablehlo::registerAllDialects(registry);
    registry.insert<::mlir::mhlo::MhloDialect>();
    registry.insert<shape::ShapeDialect>();
  }
  inline ConstBytesAttr CustomOption(OpBuilder* builder,
                                     const std::string& content) {
    return ConstBytesAttr::get(builder->getContext(),
                               StringRef(content.data(), content.size()));
  }

  std::vector<int64_t> FlatbufferVecToMlirVec(const flexbuffers::Vector& vec) {
    std::vector<int64_t> temp(vec.size(), 0);
    for (int i = 0; i < vec.size(); i++) {
      temp[i] = vec[i].AsInt64();
    }
    return temp;
  }

  llvm::SmallVector<mlir::NamedAttribute, 4> ReadAttr(const flexbuffers::Map& m,
                                                      Builder* builder) {
    llvm::SmallVector<mlir::NamedAttribute, 4> attrs;
    const auto& keys = m.Keys();
    for (size_t i = 0; i < keys.size(); ++i) {
      const auto key = keys[i].AsKey();
      const auto& value = m[key];
      switch (value.GetType()) {
        case flexbuffers::FBT_INT: {
          auto attr = value.AsInt64();
          auto named_attr = builder->getNamedAttr(
              key, builder->getIntegerAttr(builder->getIntegerType(64), attr));
          attrs.push_back(named_attr);
          break;
        }
        case flexbuffers::FBT_VECTOR_INT: {
          const auto& vector = value.AsTypedVector();
          std::vector<int64_t> vec;
          for (size_t i = 0; i < vector.size(); i++) {
            vec.push_back(vector[i].AsInt64());
          }
          RankedTensorType ty = tensorflow::GetTypeFromTFTensorShape(
              {static_cast<int64_t>(vec.size())}, builder->getIntegerType(64));
          auto named_attr =
              builder->getNamedAttr(key, DenseIntElementsAttr::get(ty, vec));
          attrs.push_back(named_attr);
          break;
        }
        case flexbuffers::FBT_VECTOR_STRING_DEPRECATED: {
          const auto& vector = value.AsTypedVector();

          if (std::string{key} == "precision_config") {
            llvm::SmallVector<mlir::Attribute> precision_attrs;
            for (size_t i = 0; i < vector.size(); i++) {
              auto conf_attr = mlir::mhlo::PrecisionAttr::get(
                  builder->getContext(),
                  mlir::mhlo::symbolizePrecision(vector[i].AsString().str())
                      .value());
              precision_attrs.push_back(conf_attr);
            }
            auto named_attr = builder->getNamedAttr(
                key, builder->getArrayAttr(precision_attrs));
            attrs.push_back(named_attr);
          } else {
            std::vector<StringRef> temp;
            for (size_t i = 0; i < vector.size(); i++) {
              auto conf_str =
                  builder->getStringAttr(vector[i].AsString().str());
              temp.push_back(conf_str);
            }
            ArrayRef<StringRef> values(temp);
            auto named_attr =
                builder->getNamedAttr(key, builder->getStrArrayAttr(values));
            attrs.push_back(named_attr);
          }

          break;
        }
        case flexbuffers::FBT_VECTOR: {
          if (std::string{key} == "dimension_numbers") {
            auto value_vec = value.AsVector();
            auto vec1 = FlatbufferVecToMlirVec(value_vec[2].AsVector());
            auto vec2 = FlatbufferVecToMlirVec(value_vec[5].AsVector());
            auto vec3 = FlatbufferVecToMlirVec(value_vec[8].AsVector());
            auto conv_dimension_numbers_attr =
                mlir::mhlo::ConvDimensionNumbersAttr::get(
                    builder->getContext(), value_vec[0].AsInt64(),
                    value_vec[1].AsInt64(), llvm::ArrayRef<int64_t>(vec1),
                    value_vec[3].AsInt64(), value_vec[4].AsInt64(),
                    llvm::ArrayRef<int64_t>(vec2), value_vec[6].AsInt64(),
                    value_vec[7].AsInt64(), llvm::ArrayRef<int64_t>(vec3));
            auto named_attr =
                builder->getNamedAttr(key, conv_dimension_numbers_attr);
            attrs.push_back(named_attr);
          }
          break;
        }
        default: {
          emitWarning(builder->getUnknownLoc(),
                      "seralization not supported for : ")
              << key;
          break;
        }
      }
    }
    return attrs;
  }
};

void TflToMhloPass::runOnOperation() {
  func::FuncOp fn = getOperation();
  OpBuilder builder(fn.getContext());
  fn.walk([&](TFL::CustomOp custom_op) {
    builder.setInsertionPoint(custom_op);
    const uint8_t* option_buf = reinterpret_cast<const uint8_t*>(
        custom_op.getCustomOption().getValue().data());
    auto flex_buffer_map =
        flexbuffers::GetRoot(option_buf,
                             custom_op.getCustomOption().getValue().size())
            .AsMap();
    auto attr = ReadAttr(flex_buffer_map, &builder);
    OperationState op_state(custom_op.getLoc(),
                            custom_op.getCustomCode().str());
    op_state.addOperands(custom_op.getOperands());
    llvm::SmallVector<mlir::Type, 4> output_tys;
    for (int i = 0; i < custom_op.getNumResults(); i++) {
      output_tys.push_back(custom_op.getType(i));
    }
    op_state.addTypes(output_tys);
    op_state.addAttributes(attr);
    auto mhlo_op = builder.create(op_state);
    custom_op.replaceAllUsesWith(mhlo_op);
    custom_op.erase();
  });
}

std::unique_ptr<OperationPass<func::FuncOp>> CreateTflToMhloPass() {
  return std::make_unique<TflToMhloPass>();
}

static PassRegistration<TflToMhloPass> pass;

}  // namespace mhlo
}  // namespace TFL
}  // namespace mlir
