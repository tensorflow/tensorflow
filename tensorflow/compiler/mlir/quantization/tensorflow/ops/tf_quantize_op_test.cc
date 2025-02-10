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

#include "tensorflow/compiler/mlir/quantization/tensorflow/ops/tf_quantize_op.h"

#include <optional>

#include <gtest/gtest.h>
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Dialect/Quant/IR/Quant.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/OwningOpRef.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/quantization/common/attrs_and_constraints.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/quantization_options.pb.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_dialect.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"

namespace mlir::quant {
namespace {

using QuantizationComponentSpec =
    tensorflow::quantization::QuantizationComponentSpec;

class EmptyPatternRewriter : public mlir::PatternRewriter {
 public:
  explicit EmptyPatternRewriter(const OpBuilder& other_builder)
      : mlir::PatternRewriter(other_builder) {}
  ~EmptyPatternRewriter() override = default;
};

TEST(TfQuantOpTest, applyUniformQuantization) {
  MLIRContext context;
  OwningOpRef<ModuleOp> module(ModuleOp::create(UnknownLoc::get(&context)));
  OpBuilder builder(&module->getBodyRegion());
  context.loadDialect<TF::TensorFlowDialect, quant::QuantDialect,
                      func::FuncDialect>();
  EmptyPatternRewriter pattern_rewriter(builder);
  Value value = CreateConstValue<float>(builder, module->getLoc(), {1024, 2},
                                        SmallVector<float>(2048, 0));

  QuantizationComponentSpec quant_spec;
  quant_spec.set_quantization_component(
      QuantizationComponentSpec::COMPONENT_WEIGHT);
  quant_spec.set_tensor_type(QuantizationComponentSpec::TENSORTYPE_INT_8);

  std::optional<TF::PartitionedCallOp> dequantize_op = ApplyUniformQuantization(
      pattern_rewriter, cast<TF::ConstOp>(value.getDefiningOp()), quant_spec);
  EXPECT_TRUE(dequantize_op.has_value());
  EXPECT_EQ(dequantize_op.value().func().getName().str(),
            "composite_dequantize_uniform");
}

}  // namespace
}  // namespace mlir::quant
