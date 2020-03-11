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

#include <string>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/None.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"  // TF:llvm-project
#include "mlir/IR/Attributes.h"  // TF:llvm-project
#include "mlir/IR/Builders.h"  // TF:llvm-project
#include "mlir/IR/Function.h"  // TF:llvm-project
#include "mlir/IR/Identifier.h"  // TF:llvm-project
#include "mlir/IR/Location.h"  // TF:llvm-project
#include "mlir/IR/MLIRContext.h"  // TF:llvm-project
#include "mlir/IR/Module.h"  // TF:llvm-project
#include "mlir/IR/Operation.h"  // TF:llvm-project
#include "mlir/IR/StandardTypes.h"  // TF:llvm-project
#include "mlir/IR/SymbolTable.h"  // TF:llvm-project
#include "mlir/Interfaces/CallInterfaces.h"  // TF:llvm-project
#include "mlir/Pass/Pass.h"  // TF:llvm-project
#include "mlir/Support/LLVM.h"  // TF:llvm-project
#include "mlir/Support/LogicalResult.h"  // TF:llvm-project
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"
#include "tensorflow/compiler/mlir/lite/transforms/passes.h"
#include "tensorflow/compiler/mlir/lite/utils/lstm_utils.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"

// NOLINTNEXTLINE

namespace mlir {
namespace TFL {
namespace {

constexpr char kTFAPIImplements[] = "tf.api_implements";

// Abstracts the conversion of the embedded lookup composite function.
class ConvertEmbeddedLookupFunc {
 public:
  explicit ConvertEmbeddedLookupFunc(FuncOp func) : func_(func) {}

  void RewriteFunc() {
    func_.setAttr(kTFImplements,
                  StringAttr::get("embedding_lookup", func_.getContext()));
    Value lookup = func_.getArgument(1);
    Value value = func_.getArgument(0);
    auto output_type = func_.getType().getResult(0);

    OpBuilder builder(func_.getBody());
    auto op = builder.create<mlir::TFL::EmbeddingLookupOp>(
        func_.getLoc(), output_type, lookup, value);

    builder.create<mlir::ReturnOp>(func_.getLoc(), op.getResult());
  }

  LogicalResult VerifySignature() {
    if (func_.getNumArguments() != 2) {
      return func_.emitError()
             << "Invalid number of arguments in the embedding "
                "matmul composite function";
    }
    if (func_.getType().getNumResults() != 1) {
      return func_.emitError() << "Invalid number of results in the embedding "
                                  "matmul composite function";
    }
    return success();
  }

 private:
  FuncOp func_;
};

// This pass uses mechanisms listed in RFC:
// https://github.com/tensorflow/community/pull/113
// It prepares composite functions that are attributed to indicate
// a specific interface (LSTM, SVDF, Embedding lookup etc.) by replacing the
// body with the corresponding fused TFLite op. The replacement need not always
// be a fused op, though that is the primary use case.
class PrepareCompositeFunctionsPass
    : public ModulePass<PrepareCompositeFunctionsPass> {
 public:
  explicit PrepareCompositeFunctionsPass() {}

 private:
  void ConvertTFImplements(FuncOp func, StringAttr attr);
  void ConvertTFAPIImplements(FuncOp func, StringAttr attr, ModuleOp module);
  void runOnModule() override;
};

void PrepareCompositeFunctionsPass::ConvertTFImplements(FuncOp func,
                                                        StringAttr attr) {
  if (attr.getValue() == "embedding_matmul") {
    func.eraseBody();
    func.addEntryBlock();
    // Convert the composite embedding_matmul function body to a
    // TFLite fused embedding_lookup op.
    ConvertEmbeddedLookupFunc convert_embedded_lookup(func);
    if (failed(convert_embedded_lookup.VerifySignature())) {
      return signalPassFailure();
    }
    convert_embedded_lookup.RewriteFunc();
  } else if (attr.getValue() == mlir::TFL::kLstmCellSimple) {
    func.eraseBody();
    func.addEntryBlock();
    ConvertLSTMCellSimpleToFusedLSTM convert_lstm_cell_simple(func);
    if (failed(convert_lstm_cell_simple.RewriteFunc())) {
      return signalPassFailure();
    }
  } else if (attr.getValue() == mlir::TFL::kLayerNormalizedLstmCellSimple) {
    func.eraseBody();
    func.addEntryBlock();
    ConvertLayerNormalizedLSTMCellSimpleToFusedLSTM
        convert_layer_norm_lstm_cell_simple(func);
    if (failed(convert_layer_norm_lstm_cell_simple.RewriteFunc())) {
      return signalPassFailure();
    }
  }
}

LogicalResult CheckOutputConsumer(
    Operation* call_op, int expected_num_outputs,
    llvm::DenseSet<int> expected_consumer_indices) {
  if (call_op->getNumResults() != expected_num_outputs) return failure();

  for (int i = 0; i < expected_num_outputs; ++i) {
    auto it = expected_consumer_indices.find(i);
    if (it == expected_consumer_indices.end()) {
      // Unexpected consumer.
      if (!call_op->getResult(i).use_empty()) return failure();
    }
  }
  return success();
}

LogicalResult CheckFusableKerasLstm(FuncOp lstm_func, ModuleOp module) {
  bool check_failed = false;
  for (auto func : module.getOps<FuncOp>()) {
    func.walk([&](Operation* op) {
      auto call_op = dyn_cast_or_null<CallOpInterface>(op);
      if (call_op && op->getAttrOfType<SymbolRefAttr>("f").getRootReference() ==
                         lstm_func.getName()) {
        // Keras LSTM have 5 outputs.
        // We should make sure only the first or the second output are consumed.
        if (failed(CheckOutputConsumer(call_op, 5, {0, 1})))
          check_failed = true;
      }
    });
  }

  if (check_failed) return failure();
  return success();
}

void PrepareCompositeFunctionsPass::ConvertTFAPIImplements(FuncOp func,
                                                           StringAttr attr,
                                                           ModuleOp module) {
  // Keras lstm tf.api_implements usually has attribute like "lstm_abcde91...".
  // TODO(b/147436982): we need to make sure that only the
  // outputs(full sequence) is used, not the last_output, not the new_states.
  // We will discard everything except the outputs.
  // And the outputs is in the shape of [batch, time, units].
  if (attr.getValue().startswith("lstm_")) {
    // Check if the keras lstm can be fused, if not, we just don't do anything.
    if (failed(CheckFusableKerasLstm(func, module))) return;

    func.eraseBody();
    func.addEntryBlock();

    OpBuilder builder(func.getBody());
    if (failed(ConvertKerasLSTMLayer(func, &builder)))
      return signalPassFailure();
  }
}

void PrepareCompositeFunctionsPass::runOnModule() {
  auto module = getModule();
  for (auto func : module.getOps<FuncOp>()) {
    // We have two kinds of implements:
    // 1) tf._implements.
    // 2) tf.api_implements.
    // We need to handle them separately.
    auto tf_implements_attr = func.getAttrOfType<StringAttr>(kTFImplements);
    if (tf_implements_attr) {
      ConvertTFImplements(func, tf_implements_attr);
    }

    auto tf_api_implements_attr =
        func.getAttrOfType<StringAttr>(kTFAPIImplements);
    if (tf_api_implements_attr) {
      // TODO(b/147536816): Keras lstm should set up the correct attributes.
      ConvertTFAPIImplements(func, tf_api_implements_attr, module);
    }
  }
}
}  // namespace

std::unique_ptr<OpPassBase<ModuleOp>> CreatePrepareCompositeFunctionsPass() {
  return std::make_unique<PrepareCompositeFunctionsPass>();
}

static PassRegistration<PrepareCompositeFunctionsPass> pass(
    "tfl-prepare-composite-funcs-tf",
    "Prepares composite functions in Tensorflow dialect of MLIR ");

}  // namespace TFL
}  // namespace mlir
