/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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
#include <initializer_list>
#include <iterator>
#include <numeric>
#include <string>

#include "absl/memory/memory.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Quant/QuantOps.h"  // from @llvm-project
#include "mlir/Dialect/StandardOps/IR/Ops.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/BlockAndValueMapping.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/Matchers.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/IR/StandardTypes.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Transforms/DialectConversion.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/quantization/quantization_config.h"
#include "tensorflow/compiler/mlir/lite/quantization/quantization_traits.h"
#include "tensorflow/compiler/mlir/lite/quantization/quantization_utils.h"
#include "tensorflow/compiler/mlir/xla/ir/hlo_ops.h"
#include "tensorflow/compiler/xla/client/lib/quantize.h"

#define DEBUG_TYPE "quant-kernel-fusion"

constexpr int kFakeQuantOperandsNum = 5;
constexpr int kFakeQuantPerChannelOperandsNum = 6;

namespace mlir {
namespace xla_hlo {

namespace {

TypeAttr GetQuantSpec(Operation* op) {
  auto fake_quant = llvm::dyn_cast_or_null<CustomCallOp>(op);
  if (!fake_quant || fake_quant.getNumOperands() < kFakeQuantOperandsNum ||
      fake_quant.getNumOperands() > kFakeQuantPerChannelOperandsNum ||
      fake_quant.call_target_name() != "fake_quant_with_min_max_vars")
    return {};

  DenseFPElementsAttr min, max;
  DenseIntElementsAttr bit_width, narrow_range, quant_dim;
  if (!matchPattern(fake_quant.getOperand(1), m_Constant(&min)) ||
      !matchPattern(fake_quant.getOperand(2), m_Constant(&max)) ||
      !matchPattern(fake_quant.getOperand(3), m_Constant(&bit_width)) ||
      !matchPattern(fake_quant.getOperand(4), m_Constant(&narrow_range)))
    return {};

  auto bit_width_val = (*bit_width.attr_value_begin()).cast<IntegerAttr>();
  auto narrow_range_val = (*narrow_range.int_value_begin()).getSExtValue();
  int quant_dim_val = -1;
  if (fake_quant.getNumOperands() == kFakeQuantPerChannelOperandsNum &&
      matchPattern(fake_quant.getOperand(kFakeQuantPerChannelOperandsNum - 1),
                   m_Constant(&quant_dim))) {
    quant_dim_val = (*quant_dim.int_value_begin()).getSExtValue();
  }

  OpBuilder builder(op);
  Type input_type =
      fake_quant.getOperand(0).getType().cast<ShapedType>().getElementType();
  return quant::GetQuantizedTypeAttr(
      builder, input_type, min, max, quant_dim_val, bit_width_val,
      builder.getBoolAttr(narrow_range_val), /*is_signed=*/true);
}

// Collects input values from outside for 'ops'.
void CollectInputs(llvm::ArrayRef<Operation*> ops,
                   llvm::SmallVectorImpl<Value>* inputs,
                   llvm::SmallVectorImpl<Attribute>* input_specs) {
  for (Operation* op : ops) {
    for (Value operand : op->getOperands()) {
      if (std::find(inputs->begin(), inputs->end(), operand) != inputs->end()) {
        continue;
      }
      if (Operation* def_op = operand.getDefiningOp()) {
        if (std::find(ops.begin(), ops.end(), def_op) == ops.end()) {
          inputs->push_back(operand);
        }
      } else {  // argument value
        inputs->push_back(operand);
      }
    }
  }

  for (Value input : *inputs) {
    ShapedType input_type = input.getType().cast<ShapedType>();
    if (TypeAttr spec = GetQuantSpec(input.getDefiningOp())) {
      input_specs->push_back(spec);
    } else {
      input_specs->push_back(TypeAttr::get(input_type.getElementType()));
    }
  }
}

// Collects values that are produced by 'ops' and have use outside of 'ops'.
// TODO(fengliuai): if it is a single user and QDQ, write that to the specs.
void CollectRets(llvm::ArrayRef<Operation*> ops,
                 llvm::SmallVectorImpl<Value>* rets,
                 llvm::SmallVectorImpl<Type>* ret_types,
                 llvm::SmallVectorImpl<Attribute>* ret_specs) {
  for (Operation* op : ops) {
    for (Value result : op->getResults()) {
      for (Operation* user : result.getUsers()) {
        // If there are any user outside of 'ops'
        if (std::find(ops.begin(), ops.end(), user) == ops.end()) {
          ShapedType ret_type = result.getType().cast<ShapedType>();
          rets->push_back(result);
          ret_types->push_back(ret_type);
          if (TypeAttr spec = GetQuantSpec(user)) {
            ret_specs->push_back(spec);
          } else {
            ret_specs->push_back(TypeAttr::get(ret_type.getElementType()));
          }
          break;
        }
      }
    }
  }
}

llvm::SmallVector<Value, 0> fuseOps(PatternRewriter* rewriter,
                                    const std::initializer_list<Value>& results,
                                    StringRef kernel) {
  // Collect all the operations to be fused.
  llvm::SmallVector<Operation*, 4> fused;
  llvm::SmallVector<Location, 4> locs;
  fused.reserve(results.size());
  locs.reserve(results.size());
  for (auto value : results) {
    Operation* op = value.getDefiningOp();
    fused.push_back(op);
    locs.push_back(op->getLoc());
  }

  // Collect inputs from outside to 'ops'.
  llvm::SmallVector<Value, 4> inputs;
  llvm::SmallVector<Attribute, 4> input_specs;
  CollectInputs(fused, &inputs, &input_specs);

  // Collect outputs from 'ops' to outside.
  llvm::SmallVector<Value, 4> rets;
  llvm::SmallVector<Type, 4> ret_types;
  llvm::SmallVector<Attribute, 4> ret_specs;
  CollectRets(fused, &rets, &ret_types, &ret_specs);

  // Create the region op with the return.
  auto region = rewriter->create<quant::QuantizeRegionOp>(
      rewriter->getFusedLoc(locs), ret_types, inputs,
      rewriter->getArrayAttr(input_specs), rewriter->getArrayAttr(ret_specs),
      kernel);
  auto* body = new Block();
  region.body().push_back(body);

  OpBuilder builder(body);
  BlockAndValueMapping mapping;

  // Make block arguments and add it to the block value mapping.
  for (Value input : inputs) {
    mapping.map(input, body->addArgument(input.getType()));
  }

  // Clone the operations 'ops' to the region.
  for (Operation* op : fused) {
    builder.clone(*op, mapping);
  }

  llvm::SmallVector<Value, 4> new_rets;
  new_rets.reserve(rets.size());
  for (auto ret : llvm::enumerate(rets)) {
    Value new_ret = mapping.lookupOrNull(ret.value());
    assert(new_ret && "couldn't find return value.");
    new_rets.push_back(new_ret);
    ret.value().replaceAllUsesWith(region.getResult(ret.index()));
  }
  builder.create<quant::ReturnOp>(builder.getUnknownLoc(), new_rets);

  LLVM_DEBUG({
    assert(region.verify().Success && "failed to create quant region.");
    llvm::dbgs() << "\ncreated region: ";
    region.print(llvm::dbgs());
    llvm::dbgs() << "\n\n\n";
  });

  SmallVector<Value, 0> new_values(fused.back()->getNumResults());
  return new_values;
}

struct CpuKernelFusionPass : public FunctionPass<CpuKernelFusionPass> {
  explicit CpuKernelFusionPass() = default;
  CpuKernelFusionPass(const CpuKernelFusionPass&) {}

  void runOnFunction() override;

 private:
  LogicalResult fuseCpuKernels(Operation* op);
};

#include "tensorflow/compiler/mlir/lite/quantization/xla/generated_cpu_kernel_fusion.inc"

LogicalResult CpuKernelFusionPass::fuseCpuKernels(Operation* op) {
  MLIRContext* ctx = op->getContext();
  OwningRewritePatternList patterns;
  populateWithGenerated(ctx, &patterns);

  ConversionTarget target(*ctx);
  target.addLegalDialect<quant::QuantizationDialect>();
  target.addLegalOp<CallOp, ModuleOp, FuncOp, ModuleTerminatorOp,
                    ::mlir::ReturnOp>();
  return applyPartialConversion(op, target, patterns);
}

void CpuKernelFusionPass::runOnFunction() {
  if (failed(fuseCpuKernels(getFunction()))) signalPassFailure();
}

}  // namespace

// Creates an instance of the xla_hlo cpu kernel fusion pass.
std::unique_ptr<OpPassBase<FuncOp>> CreateCpuKernelFusionPass() {
  return std::make_unique<CpuKernelFusionPass>();
}

static PassRegistration<CpuKernelFusionPass> pass(
    "xla-hlo-cpu-fusion", "Fuse xla hlo ops into cpu kernels");

}  // namespace xla_hlo
}  // namespace mlir
