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
#include "mlir/Dialect/StandardOps/IR/Ops.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/Identifier.h"  // from @llvm-project
#include "mlir/IR/Location.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/SymbolTable.h"  // from @llvm-project
#include "mlir/IR/Visitors.h"  // from @llvm-project
#include "mlir/Interfaces/CallInterfaces.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"
#include "tensorflow/compiler/mlir/lite/transforms/passes.h"
#include "tensorflow/compiler/mlir/lite/utils/lstm_utils.h"
#include "tensorflow/compiler/mlir/lite/utils/nms_utils.h"
#include "tensorflow/compiler/mlir/lite/utils/perception_ops_utils.h"
#include "tensorflow/compiler/mlir/lite/utils/tftext_utils.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_attributes.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"

// The cmd line flag to turn on/off Tf.Text API fusion.
// NOLINTNEXTLINE
static llvm::cl::opt<bool> fuse_tftext_flag(
    "tfl-fuse-tftext", llvm::cl::value_desc("bool"),
    llvm::cl::desc("Fuse TF.Text API ops when it's true"),
    llvm::cl::init(false));

namespace mlir {
namespace TFL {
namespace {

constexpr char kTFAPIImplements[] = "tf.api_implements";
constexpr char kTFTextAPIPrefix[] = "tftext:";
constexpr char kCustomSSDPostprocessing[] = "TFLite_Detection_PostProcess";
constexpr char kTfNMSPadded[] = "non_max_suppression_padded_v2";
constexpr char kCustomMaxUnpooling[] = "addons:MaxUnpooling2D";
constexpr char kCustomDenseImageWarp[] = "addons:DenseImageWarp";

using mlir::TF::FuncAttr;

// Abstracts the conversion of the embedded lookup composite function.
class ConvertEmbeddedLookupFunc {
 public:
  explicit ConvertEmbeddedLookupFunc(FuncOp func) : func_(func) {}

  void RewriteFunc() {
    func_->setAttr(kTFImplements,
                   StringAttr::get(func_.getContext(), "embedding_lookup"));
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
    : public PassWrapper<PrepareCompositeFunctionsPass,
                         OperationPass<ModuleOp>> {
  void getDependentDialects(DialectRegistry& registry) const override {
    registry.insert<TFL::TensorFlowLiteDialect>();
  }

 public:
  explicit PrepareCompositeFunctionsPass() {}

 private:
  // TODO(b/160915525): Consolidate FuncAttr and StringAttr into one.
  void ConvertTFImplements(FuncOp func, StringAttr attr);
  void ConvertTFImplementsWithAttributes(FuncOp func, FuncAttr attr);
  void ConvertTFAPIImplements(FuncOp func, StringAttr attr, ModuleOp module);
  void runOnOperation() override;
};

LogicalResult CheckFusableLayerNormalizedLstmCellSimple(FuncOp lstm_func) {
  for (int i = 0; i < 5; ++i) {
    auto input = lstm_func.getArgument(i);
    auto input_type = input.getType().dyn_cast_or_null<RankedTensorType>();
    if (!input_type) {
      lstm_func.emitWarning(
          "we cannot fuse this lstm func because all the inputs have not "
          "ranked tensor type.");
      return failure();
    }
  }

  return success();
}

LogicalResult CheckFusableLstmCellSimple(FuncOp lstm_func) {
  for (int i = 0; i < 4; ++i) {
    auto input = lstm_func.getArgument(i);
    auto input_type = input.getType().dyn_cast_or_null<RankedTensorType>();
    if (!input_type) {
      lstm_func.emitWarning(
          "we cannot fuse this lstm func because all the inputs have not "
          "ranked tensor type.");
      return failure();
    }
  }

  return success();
}

LogicalResult CheckOutputConsumer(
    Operation* call_op, int expected_num_outputs,
    llvm::DenseSet<int> expected_consumer_indices) {
  const int num_results = call_op->getNumResults();
  if (num_results != expected_num_outputs) return failure();

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
  for (auto func : module.getOps<FuncOp>()) {
    if (func == lstm_func) continue;
    auto result = func.walk([&](CallOpInterface op) {
      if (dyn_cast<FuncOp>(op.resolveCallable()) == lstm_func) {
        // Keras LSTM have 5 outputs.
        // We should make sure only the first or the second output are
        // consumed.
        if (failed(CheckOutputConsumer(op.getOperation(), 5, {0, 1})))
          return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });

    if (result.wasInterrupted()) return failure();
  }

  // We should know the batch size in advance for the lstm fusion.
  // A good indicator of batch size is both cell state and input state (indices
  // 1 & 2) have fixed shape and other input tenors should have ranked tensor
  // types.
  for (int i = 0; i < 6; ++i) {
    auto input = lstm_func.getArgument(i);
    auto input_type = input.getType().dyn_cast_or_null<RankedTensorType>();
    if (!input_type) {
      lstm_func.emitWarning(
          "we cannot fuse this lstm func because all the inputs have not "
          "ranked tensor type.");
      return failure();
    }
    switch (i) {
      case 1:  // output_init_state
      case 2:  // hidden_init_state
        if (!input_type.hasStaticShape()) {
          lstm_func.emitWarning(
              "we cannot fuse this lstm func because the batch size is not "
              "fixed, please consider setting fixed batch size like "
              "https://github.com/tensorflow/tensorflow/blob/master/tensorflow/"
              "lite/examples/experimental_new_converter/"
              "Keras_LSTM_fusion_Codelab.ipynb");
          return failure();
        }
        break;
      case 3:  // wiehgt
      case 4:  // recurrent_kernel
      case 5:  // bias
        if (!input_type.hasStaticShape()) {
          lstm_func.emitWarning(
              "we cannot fuse this lstm func because the weight & bias are not "
              "fixed, please consider setting fixed batch size like "
              "https://github.com/tensorflow/tensorflow/blob/master/tensorflow/"
              "lite/examples/experimental_new_converter/"
              "Keras_LSTM_fusion_Codelab.ipynb");
          return failure();
        }
        break;
      default:
        // No op.
        break;
    }
  }

  return success();
}

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
    // Check if the lstm cell simple can be fused, if not, we just don't do
    // anything.
    if (failed(CheckFusableLstmCellSimple(func))) return;
    func.eraseBody();
    func.addEntryBlock();
    ConvertLSTMCellSimpleToFusedLSTM convert_lstm_cell_simple(func);
    if (failed(convert_lstm_cell_simple.RewriteFunc())) {
      return signalPassFailure();
    }
  } else if (attr.getValue() == mlir::TFL::kLayerNormalizedLstmCellSimple) {
    // Check if the layer normalized lstm cell simple can be fused, if not, we
    // just don't do anything.
    if (failed(CheckFusableLayerNormalizedLstmCellSimple(func))) return;
    func.eraseBody();
    func.addEntryBlock();
    ConvertLayerNormalizedLSTMCellSimpleToFusedLSTM
        convert_layer_norm_lstm_cell_simple(func);
    if (failed(convert_layer_norm_lstm_cell_simple.RewriteFunc())) {
      return signalPassFailure();
    }
  } else if (attr.getValue() == kTfNMSPadded) {
    func.eraseBody();
    func.addEntryBlock();
    ConvertNMSPaddedFunc convert_nms_padded(func);
    if (failed(convert_nms_padded.VerifySignature())) {
      return signalPassFailure();
    }
    convert_nms_padded.RewriteFunc();
  } else if (attr.getValue() == kCustomDenseImageWarp) {
    ConvertDenseImageWarpFunc image_warping(func);
    if (failed(image_warping.VerifySignature()) ||
        failed(image_warping.RewriteFunc())) {
      return signalPassFailure();
    }
  }
}

void PrepareCompositeFunctionsPass::ConvertTFImplementsWithAttributes(
    FuncOp func, FuncAttr attr) {
  auto api_name = attr.GetName().getLeafReference();
  bool enable_fuse_tftext =
      fuse_tftext_flag || IsTFTextRegistered(tensorflow::OpRegistry::Global());
  if (api_name.startswith(kTFTextAPIPrefix) && enable_fuse_tftext) {
    if (failed(ConvertTFTextAPI(func, api_name, attr))) {
      return signalPassFailure();
    }
  } else if (api_name == kCustomSSDPostprocessing) {
    ConvertSSDPostProcessFunc convert_ssd_postprocess(func, attr);
    if (failed(convert_ssd_postprocess.VerifySignature()) ||
        failed(convert_ssd_postprocess.RewriteFunc())) {
      return signalPassFailure();
    }
  } else if (api_name == kCustomMaxUnpooling) {
    ConvertMaxUnpoolingFunc max_unpooling(func, attr);
    if (failed(max_unpooling.VerifySignature()) ||
        failed(max_unpooling.RewriteFunc())) {
      return signalPassFailure();
    }
  }
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

void PrepareCompositeFunctionsPass::runOnOperation() {
  auto module = getOperation();
  for (auto func : module.getOps<FuncOp>()) {
    // We have three kinds of implements:
    // 1) tf._implements, with string attributes.
    // 2) tf._implements, with proto attributes.
    // 3) tf.api_implements.
    // We need to handle them separately.
    auto tf_implements_attr_str =
        func->getAttrOfType<StringAttr>(kTFImplements);
    if (tf_implements_attr_str) {
      ConvertTFImplements(func, tf_implements_attr_str);
      continue;
    }

    auto tf_implements_attr = func->getAttrOfType<FuncAttr>(kTFImplements);
    if (tf_implements_attr) {
      ConvertTFImplementsWithAttributes(func, tf_implements_attr);
      continue;
    }

    auto tf_api_implements_attr =
        func->getAttrOfType<StringAttr>(kTFAPIImplements);
    if (tf_api_implements_attr) {
      // TODO(b/147536816): Keras lstm should set up the correct attributes.
      ConvertTFAPIImplements(func, tf_api_implements_attr, module);
    }
  }
}
}  // namespace

std::unique_ptr<OperationPass<ModuleOp>> CreatePrepareCompositeFunctionsPass() {
  return std::make_unique<PrepareCompositeFunctionsPass>();
}

static PassRegistration<PrepareCompositeFunctionsPass> pass(
    "tfl-prepare-composite-funcs-tf",
    "Prepares composite functions in Tensorflow dialect of MLIR ");

}  // namespace TFL
}  // namespace mlir
