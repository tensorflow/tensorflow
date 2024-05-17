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

#include <optional>
#include <string>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
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

namespace mlir {
namespace TFL {
namespace {
#define GEN_PASS_DEF_PREPARECOMPOSITEFUNCTIONSPASS
#include "tensorflow/compiler/mlir/lite/transforms/passes.h.inc"

constexpr char kTFAPIImplements[] = "tf.api_implements";
constexpr char kTFTextAPIPrefix[] = "tftext:";
constexpr char kCustomSSDPostprocessing[] = "TFLite_Detection_PostProcess";
constexpr char kTfNMSPadded[] = "non_max_suppression_padded_v2";
constexpr char kCustomMaxUnpooling[] = "addons:MaxUnpooling2D";
constexpr char kCustomDenseImageWarp[] = "addons:DenseImageWarp";
constexpr char kTFLFusableOp[] = "tfl_fusable_op";

using mlir::TF::FuncAttr;

inline ConstBytesAttr CustomOption(OpBuilder* builder,
                                   const std::string& content) {
  return ConstBytesAttr::get(builder->getContext(),
                             StringRef(content.data(), content.size()));
}

LogicalResult CreateTflFusableOpCustomOptions(
    ArrayRef<std::pair<StringRef, Attribute>> attrs, OpBuilder* builder,
    std::string& custom_option_buffer) {
  // There is something worth noting in the ordering of the custom op option:
  // At the MLIR level, all the option is ordered alphabetcially, so there is
  // no way for us to retrieve the original order, so please make sure you are
  // reading custom option from dictionary rather than depending on the order.
  flexbuffers::Builder fbb;
  size_t start_map = fbb.StartMap();

  for (auto attr : attrs) {
    if (auto float_attr = mlir::dyn_cast_or_null<FloatAttr>(attr.second)) {
      fbb.Float(attr.first.data(), float_attr.getValue().convertToFloat());
    } else if (auto int_attr =
                   mlir::dyn_cast_or_null<IntegerAttr>(attr.second)) {
      fbb.Int(attr.first.data(), int_attr.getInt());
    } else if (auto bool_attr = mlir::dyn_cast_or_null<BoolAttr>(attr.second)) {
      fbb.Bool(attr.first.data(), bool_attr.getValue());
    } else if (auto string_attr =
                   mlir::dyn_cast_or_null<StringAttr>(attr.second)) {
      fbb.String(attr.first.data(), string_attr.getValue().str());
    } else {
      // TODO(b/201482289): support other data types.
      return failure();
    }
  }

  fbb.EndMap(start_map);
  fbb.Finish();
  custom_option_buffer.assign(fbb.GetBuffer().begin(), fbb.GetBuffer().end());
  return success();
}

// Convert func annotated with `tfl_fusable_op` attribute to tfl custom op.
LogicalResult ConvertTflFusableOp(
    func::FuncOp func, StringRef custom_op_name,
    ArrayRef<std::pair<StringRef, Attribute>> attrs) {
  func.eraseBody();
  func.addEntryBlock();

  OpBuilder builder(func.getBody());
  std::string custom_option_buffer;
  if (failed(CreateTflFusableOpCustomOptions(attrs, &builder,
                                             custom_option_buffer))) {
    return failure();
  }

  auto tfl_fusable_op = builder.create<TFL::CustomOp>(
      func->getLoc(), func.getFunctionType().getResults(), func.getArguments(),
      custom_op_name, CustomOption(&builder, custom_option_buffer));
  builder.create<func::ReturnOp>(func->getLoc(), tfl_fusable_op.getResults());
  return success();
}

// Abstracts the conversion of the embedded lookup composite function.
class ConvertEmbeddedLookupFunc {
 public:
  explicit ConvertEmbeddedLookupFunc(func::FuncOp func) : func_(func) {}

  void RewriteFunc() {
    func_->setAttr(kTFImplements,
                   StringAttr::get(func_.getContext(), "embedding_lookup"));
    Value lookup = func_.getArgument(1);
    Value value = func_.getArgument(0);
    auto output_type = func_.getFunctionType().getResult(0);

    OpBuilder builder(func_.getBody());
    auto op = builder.create<mlir::TFL::EmbeddingLookupOp>(
        func_.getLoc(), output_type, lookup, value);

    builder.create<mlir::func::ReturnOp>(func_.getLoc(), op.getResult());
  }

  LogicalResult VerifySignature() {
    if (func_.getNumArguments() != 2) {
      return func_.emitWarning()
             << "Invalid number of arguments in the embedding "
                "matmul composite function";
    }
    if (func_.getFunctionType().getNumResults() != 1) {
      return func_.emitWarning() << "Invalid number of results in the "
                                    "embedding matmul composite function";
    }
    return success();
  }

 private:
  func::FuncOp func_;
};

class PrepareCompositeFunctionsPass
    : public impl::PrepareCompositeFunctionsPassBase<
          PrepareCompositeFunctionsPass> {
  void getDependentDialects(DialectRegistry& registry) const override {
    registry.insert<TFL::TensorFlowLiteDialect>();
  }

 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PrepareCompositeFunctionsPass)

  explicit PrepareCompositeFunctionsPass() {}

 private:
  // TODO(b/160915525): Consolidate FuncAttr and StringAttr into one.
  void ConvertTFImplements(func::FuncOp func, StringAttr attr);
  void ConvertTFImplementsWithAttributes(func::FuncOp func, FuncAttr attr);
  void ConvertTFAPIImplements(func::FuncOp func, StringAttr attr,
                              ModuleOp module);
  void runOnOperation() override;
};

LogicalResult CheckFusableLayerNormalizedLstmCellSimple(
    func::FuncOp lstm_func) {
  for (int i = 0; i < 5; ++i) {
    auto input = lstm_func.getArgument(i);
    auto input_type = mlir::dyn_cast_or_null<RankedTensorType>(input.getType());
    if (!input_type) {
      lstm_func.emitWarning(
          "we cannot fuse this lstm func because all the inputs have not "
          "ranked tensor type.");
      return failure();
    }
  }

  return success();
}

LogicalResult CheckFusableLstmCellSimple(func::FuncOp lstm_func) {
  for (int i = 0; i < 4; ++i) {
    auto input = lstm_func.getArgument(i);
    auto input_type = mlir::dyn_cast_or_null<RankedTensorType>(input.getType());
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

LogicalResult CheckFusableKerasLstm(func::FuncOp lstm_func, ModuleOp module) {
  for (auto func : module.getOps<func::FuncOp>()) {
    if (func == lstm_func) continue;
    auto result = func.walk([&](CallOpInterface op) {
      if (dyn_cast<func::FuncOp>(op.resolveCallable()) == lstm_func) {
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
  // Current UnidirectionalSequenceLSTMOp doesn't support mask input.
  if (lstm_func.getNumArguments() == 7) return failure();

  // We should know the batch size in advance for the lstm fusion.
  // A good indicator of batch size is both cell state and input state (indices
  // 1 & 2) have fixed shape and other input tenors should have ranked tensor
  // types.
  for (int i = 0; i < 6; ++i) {
    auto input = lstm_func.getArgument(i);
    auto input_type = mlir::dyn_cast_or_null<RankedTensorType>(input.getType());
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

void PrepareCompositeFunctionsPass::ConvertTFImplements(func::FuncOp func,
                                                        StringAttr attr) {
  if (attr.getValue() == "embedding_matmul") {
    // Convert the composite embedding_matmul function body to a
    // TFLite fused embedding_lookup op.
    ConvertEmbeddedLookupFunc convert_embedded_lookup(func);
    if (failed(convert_embedded_lookup.VerifySignature())) return;
    func.eraseBody();
    func.addEntryBlock();
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
    ConvertNMSPaddedFunc convert_nms_padded(func);
    if (failed(convert_nms_padded.VerifySignature())) return;
    func.eraseBody();
    func.addEntryBlock();
    convert_nms_padded.RewriteFunc();
  } else if (attr.getValue() == kCustomDenseImageWarp) {
    ConvertDenseImageWarpFunc image_warping(func);
    if (failed(image_warping.VerifySignature())) return;
    if (failed(image_warping.RewriteFunc())) {
      return signalPassFailure();
    }
  }
}

void PrepareCompositeFunctionsPass::ConvertTFImplementsWithAttributes(
    func::FuncOp func, FuncAttr attr) {
  StringRef api_name = attr.getName().getLeafReference().getValue();
  bool enable_fuse_tftext =
      tfl_fuse_tftext_ || IsTFTextRegistered(tensorflow::OpRegistry::Global());
  if (api_name.starts_with(kTFTextAPIPrefix) && enable_fuse_tftext) {
    if (failed(ConvertTFTextAPI(func, api_name, attr))) {
      return signalPassFailure();
    }
  } else if (api_name == kCustomSSDPostprocessing) {
    ConvertSSDPostProcessFunc convert_ssd_postprocess(func, attr);
    if (failed(convert_ssd_postprocess.VerifySignature())) return;
    if (failed(convert_ssd_postprocess.RewriteFunc())) {
      return signalPassFailure();
    }
  } else if (api_name == kCustomMaxUnpooling) {
    ConvertMaxUnpoolingFunc max_unpooling(func, attr);
    if (failed(max_unpooling.VerifySignature())) return;
    if (failed(max_unpooling.RewriteFunc())) {
      return signalPassFailure();
    }
  } else {
    // We will look for the `tfl_fusable_op` attribute and fuse as a custom op.
    DictionaryAttr dict_attr = attr.getAttrs();

    SmallVector<std::pair<StringRef, Attribute>, 4> attributes;
    bool tfl_fusable_op = false;
    for (auto attr_item : dict_attr) {
      // Push other attributes except the TFLFusableOp.
      if (attr_item.getName() == kTFLFusableOp &&
          mlir::dyn_cast<BoolAttr>(attr_item.getValue()).getValue()) {
        tfl_fusable_op = true;
      } else {
        attributes.push_back({attr_item.getName(), attr_item.getValue()});
      }
    }

    if (!tfl_fusable_op) return;

    if (failed(ConvertTflFusableOp(func, api_name, attributes))) {
      func->emitError(absl::StrCat("failed to fuse for op: ", api_name.str()));
      return signalPassFailure();
    }
  }
}

void PrepareCompositeFunctionsPass::ConvertTFAPIImplements(func::FuncOp func,
                                                           StringAttr attr,
                                                           ModuleOp module) {
  // Keras lstm tf.api_implements usually has attribute like "lstm_abcde91...".
  // TODO(b/147436982): we need to make sure that only the
  // outputs(full sequence) is used, not the last_output, not the new_states.
  // We will discard everything except the outputs.
  // And the outputs is in the shape of [batch, time, units].
  if (attr.getValue().starts_with("lstm_")) {
    // Check if the keras lstm can be fused, if not, we just don't do anything.
    if (failed(CheckFusableKerasLstm(func, module))) return;
    func.eraseBody();
    func.addEntryBlock();
    OpBuilder builder(func.getBody());
    if (failed(ConvertKerasLSTMLayer(func, &builder)))
      return signalPassFailure();
  }

  // LSTM `func::FuncOps` with indy behavior always have the `tf.api_implements`
  // function attribute prefixed with `"indy_lstm_"`.
  // IndyLSTMs have diagonal recurrent weight matrices and can benefit from
  // more efficent operations in TFLite with the correct conversion (i.e. when
  // the diagonal recurrent weight matrices are provided as vectors).
  if (attr.getValue().starts_with("indy_lstm_")) {
    // Check if the keras lstm can be fused, if not, we just don't do anything.
    if (failed(CheckFusableKerasLstm(func, module))) return;
    func.eraseBody();
    func.addEntryBlock();
    OpBuilder builder(func.getBody());
    if (failed(ConvertKerasLSTMLayer(func, &builder, true)))
      return signalPassFailure();
  }
}

void PrepareCompositeFunctionsPass::runOnOperation() {
  auto module = getOperation();
  for (auto func : module.getOps<func::FuncOp>()) {
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

}  // namespace TFL
}  // namespace mlir
