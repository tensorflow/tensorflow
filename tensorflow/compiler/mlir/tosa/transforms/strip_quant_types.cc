// Copyright 2021 The TensorFlow Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// ==============================================================================*/

// This pass removes the quantized type from output Tensors attached to TOSA Ops
//
// We only strip away the quantization element type from types in between two
// tosa ops and if all successors of that value is tosa.
//
// Everything else is left as is since we do not know who
// else will consume that value


#include <climits>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <numeric>

#include "mlir/Dialect/Quant/QuantOps.h"  // from @llvm-project
#include "mlir/Dialect/Tosa/IR/TosaOps.h"  // from @llvm-project
#include "mlir/Dialect/Tosa/Utils/QuantUtils.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/Pass/PassRegistry.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Transforms/DialectConversion.h"  // from @llvm-project
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/Support/LLVM.h"

#include "tensorflow/compiler/mlir/tosa/transforms/legalize_common.h"
#include "tensorflow/compiler/mlir/tosa/transforms/legalize_utils.h"
#include "tensorflow/compiler/mlir/tosa/transforms/passes.h"

#define PASS_NAME "tosa-strip-quantization"
#define DEBUG_TYPE PASS_NAME

namespace mlir {
namespace tosa {
namespace {
#define GEN_PASS_CLASSES
#include "tensorflow/compiler/mlir/tosa/transforms/passes.h.inc"

bool isOpTosa(mlir::Operation *op)
{
  return (isa<tosa::TosaDialect>(op->getDialect()));
}

Type convertType(Type type)
{
  if (auto qType = type.dyn_cast<quant::QuantizedType>()) {
    return IntegerType::get(type.getContext(),
                            qType.getStorageTypeIntegralWidth(),
                            qType.isSigned() ? IntegerType::SignednessSemantics::Signed:
                                               IntegerType::SignednessSemantics::Unsigned);
  }
  return type;
}

Type convertTensor(RankedTensorType type)
{
  auto newType = RankedTensorType::get(type.getShape(),
                                       convertType(type.getElementType()));
  return newType;
}

static bool isIllegalType(Type type) {
  if (type.isa<quant::QuantizedType>()) return true;
  if (auto shapedType = type.dyn_cast<ShapedType>()) {
    return isIllegalType(shapedType.getElementType());
  }
  return false;
}

struct StripQuantizationTypes : public RewritePattern {
  explicit StripQuantizationTypes(MLIRContext *context)
      : RewritePattern(MatchAnyOpTypeTag(), 1, context) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override
  {
    if (not(isOpTosa(op)))
        return failure();

    bool matched = false;

    for (auto result_value: op->getResults())
    {
      /*
         Checks that the Type of the Value being stripped is actually illegal
      */
      bool type_needs_stripping = isIllegalType(result_value.getType());

      /*
         Checking that all consumers of the value are TOSA Ops
      */
      bool all_successors_tosa = true;
      for (auto consumer: result_value.getUsers())
        all_successors_tosa &= (isOpTosa(consumer));

      if (all_successors_tosa and type_needs_stripping)
      {
        auto new_type = convertTensor(result_value.getType().dyn_cast<mlir::RankedTensorType>());
        result_value.setType(new_type);
        matched = true;
      }
    }

    if (matched)
       return success();
    return failure();
  }
};

class StripQuantTypes : public TosaStripQuantTypesPassBase<StripQuantTypes> {
 public:
  explicit StripQuantTypes() {}
  void runOnFunction() override;
};

void StripQuantTypes::runOnFunction() {
  auto* ctx = &getContext();
  auto func = getFunction();

  RewritePatternSet patterns(ctx);

  patterns.insert<StripQuantizationTypes>(ctx);

  GreedyRewriteConfig config;
  config.useTopDownTraversal = true;

  (void)applyPatternsAndFoldGreedily(func, std::move(patterns), config);
}

}  // anonymous namespace

std::unique_ptr<OperationPass<FuncOp>> createStripQuantTypesPass() {
  return std::make_unique<StripQuantTypes>();
}
}  // namespace tosa
}  // namespace mlir
