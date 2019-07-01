/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

// This transformation pass applies soem clean up steps after quantization.

#include "mlir/IR/MLIRContext.h"  // TF:local_config_mlir
#include "mlir/Pass/Pass.h"  // TF:local_config_mlir
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"
#include "tensorflow/compiler/mlir/lite/transforms/passes.h"
#include "tensorflow/compiler/mlir/lite/utils/quantization_utils.h"

//===----------------------------------------------------------------------===//
// The post-quantize Pass.
//
namespace mlir {
namespace TFL {
namespace {

// Applies all the clean up steps after quantization.
class PostQuantizePass : public FunctionPass<PostQuantizePass> {
 public:
  // Constructor used by the PassRegistration. This will remove the adaptor ops.
  explicit PostQuantizePass() : emit_quant_adaptor_ops_(false) {}

  // Constructor used by manually creating the pass.
  explicit PostQuantizePass(bool emit_quant_adaptor_ops)
      : emit_quant_adaptor_ops_(emit_quant_adaptor_ops) {}

  void runOnFunction() override;

 private:
  // Set this flag to true if the inputs and outputs are in floating point. The
  // quant adaptor ops convert them to fixed point values (i.e. quantize) before
  // feeding them to the model and convert them back to floating point
  // (i.e. dequantize) as the output.
  bool emit_quant_adaptor_ops_;
};

void RemoveQuantizationAdaptorOps(Function* func) {
  mlir::OpBuilder builder(func->getBody());
  auto& bb = func->getBlocks().front();
  auto* terminator = bb.getTerminator();

  int num_args = bb.getNumArguments();
  llvm::SmallVector<Type, 4> input_types;
  input_types.reserve(num_args);
  // Edit the block arguments and create the new input ops in place to replace
  // the old input ops and quantize ops.
  for (int i = 0; i != num_args; ++i) {
    // Previous loop iteration may invalidate the insertion point so we have to
    // reset insertion point each iteration.
    builder.setInsertionPointToStart(&bb);

    // In each iteration, a new argument is appended to the end of the list
    // and the current argument is erased, so here we always process the first
    // argument in the list.
    auto* arg = bb.getArgument(0);
    auto* input_op = *arg->user_begin();
    auto input_result = input_op->getResult(0);
    // We can drop the quantization adaptor only when the pseudo input op has
    // one user and it is the quantize op. Otherwise, we have to keep the
    // adaptor and allow the floating point inputs.
    if (input_result->hasOneUse() &&
        isa<QuantizeOp>(*input_result->user_begin())) {
      auto* second_op = *input_result->user_begin();
      auto quantize_output = second_op->getResult(0);
      auto quantize_type = quantize_output->getType();
      input_types.push_back(quantize_type);
      auto* new_arg = bb.addArgument(quantize_type);
      // Make a copy of input op with quantized input and output type.
      auto new_input =
          builder.create<InputOp>(input_op->getLoc(), quantize_type, new_arg);
      quantize_output->replaceAllUsesWith(new_input);
      second_op->erase();
      input_op->erase();
    } else {
      // Make a copy of current argument and append it to the end of the list.
      Type arg_type = arg->getType();
      input_types.push_back(arg_type);
      auto* new_arg = bb.addArgument(arg_type);
      arg->replaceAllUsesWith(new_arg);
    }
    arg->dropAllUses();
    bb.eraseArgument(0);
  }

  // Edit the return ops and remove the dequantize ops in place.
  int num_return_operands = terminator->getNumOperands();
  llvm::SmallVector<Type, 4> output_types;
  output_types.reserve(num_return_operands);
  for (int i = 0; i != num_return_operands; ++i) {
    auto* returned_value = terminator->getOperand(i);
    Operation* returned_op = returned_value->getDefiningOp();
    if (isa<DequantizeOp>(returned_op)) {
      auto* dequantized_result = returned_op->getOperand(0);
      output_types.push_back(dequantized_result->getType());
      terminator->setOperand(i, dequantized_result);
      returned_op->erase();
    } else {
      output_types.push_back(returned_value->getType());
    }
  }
  auto new_func_type = builder.getFunctionType(input_types, output_types);
  func->setType(new_func_type);
}

void PostQuantizePass::runOnFunction() {
  auto& func = getFunction();
  if (!emit_quant_adaptor_ops_) {
    RemoveQuantizationAdaptorOps(&func);
  }
}

}  // namespace

// Creates an instance of the TensorFlow Lite dialect PostQuantize pass.
FunctionPassBase* CreatePostQuantizePass(bool emit_quant_adaptor_ops) {
  return new PostQuantizePass(emit_quant_adaptor_ops);
}

static PassRegistration<PostQuantizePass> pass(
    "tfl-post-quantize", "Apply post quantization clean up after quantization");

}  // namespace TFL
}  // namespace mlir
