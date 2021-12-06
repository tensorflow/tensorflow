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

#include "absl/strings/match.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/FormatVariadic.h"
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/IR/SymbolTable.h"  // from @llvm-project
#include "mlir/IR/Visitors.h"  // from @llvm-project
#include "mlir/Transforms/DialectConversion.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/dialect_registration.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_executor.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes_detail.h"
#include "tensorflow/core/transforms/toposort/toposort_pass.h"
#include "tensorflow/core/util/device_name_utils.h"

namespace mlir {
namespace TF {
namespace {

// FIXME: This should be consistent with
// tensorflow::kImportModelDefaultGraphFuncName
static const char kImportModelDefaultGraphFuncName[] = "main";

// Please refer to the TFG dialect description for the list of used attributes.
// Belows are the attributes in TFE.
// TFE Arguments and Results (Got from "_Arg",
// "_Retval", .etc)
//  NodeDef.device <-> "tf.device"
//  NodeDef.attr <-> "tf."
//
// TFE general operations
//  NodeDef.device <-> "device"
//
// The following two functions are only used for mapping/excluding attributes
// which are inconsistent between TFG and TFE.
//
static mlir::LogicalResult FilterTfgSpecificArgResultAttributes(
    mlir::MLIRContext *context, mlir::ArrayAttr array_attr,
    llvm::SmallVector<mlir::DictionaryAttr> &output) {
  for (mlir::DictionaryAttr dict_attr :
       array_attr.template getAsRange<mlir::DictionaryAttr>()) {
    mlir::NamedAttrList list;
    for (mlir::NamedAttribute attr : dict_attr.getValue()) {
      // Skip if the attribute has "tfg" prefix.
      if (attr.getName().getValue().startswith("tfg")) continue;
      list.append(attr);
    }
    output.push_back(list.getDictionary(context));
  }
  return mlir::success();
}

static mlir::LogicalResult ReformatOpAttributes(
    mlir::MLIRContext *context, llvm::ArrayRef<mlir::NamedAttribute> attrs,
    llvm::SmallVectorImpl<mlir::NamedAttribute> &output) {
  for (mlir::NamedAttribute attr : attrs) {
    if (attr.getName().strref().contains(
            mlir::tfg::TFGraphDialect::getDeviceAttrKey())) {
      tensorflow::DeviceNameUtils::ParsedName parsed_name;
      if (!tensorflow::DeviceNameUtils::ParseFullName(
              attr.getValue().cast<mlir::StringAttr>().getValue().str(),
              &parsed_name))
        return mlir::failure();
      if (!parsed_name.has_type) {
        parsed_name.type = "CPU";
        parsed_name.has_type = true;
      }
      if (!parsed_name.has_id) {
        parsed_name.id = 0;
        parsed_name.has_id = true;
      }
      output.push_back(mlir::NamedAttribute(
          mlir::Identifier::get("device", context),
          mlir::StringAttr::get(
              context,
              tensorflow::DeviceNameUtils::ParsedNameToString(parsed_name))));
    } else {
      output.push_back(attr);
    }
  }
  return mlir::success();
}

// Split the tfg.NextIteration into tf_executor::NextIterationSourceOp and
// tf_executor::NextIterationSinkOp to break the cycle introduced by itself.
static void SplitNextIteration(Block &block) {
  // TODO(b/207144333): Supports callback for unregistered ops
  block.walk([&](Operation *op) {
    if (!op->getName().getStringRef().equals("tfg.NextIteration")) return;
    mlir::OpBuilder builder(op);
    auto source_op = builder.create<tf_executor::NextIterationSourceOp>(
        op->getLoc(), op->getOperand(0).getType());
    builder.create<tf_executor::NextIterationSinkOp>(
        op->getLoc(), source_op.token(), /*input=*/op->getOperand(0),
        /*controlInputs=*/op->getOperands().drop_front());
    op->replaceAllUsesWith(
        ValueRange({source_op.output(), source_op.control()}));
    op->erase();
  });
}

class ConvertGraphOp : public OpConversionPattern<tfg::GraphOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      tfg::GraphOp graph, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const final {
    Location loc = graph.getLoc();
    // To keep the import-as-graph logic taken by TFG, we create `void func()`
    // to contain the ops in the tfg::GraphOp. That means the arguments/results
    // will be the operations inside the function body rather than representing
    // them in the function signature.
    FunctionType func_type = rewriter.getFunctionType({}, {});
    FuncOp func = rewriter.create<FuncOp>(loc, kImportModelDefaultGraphFuncName,
                                          func_type);
    rewriter.setInsertionPointToStart(func.addEntryBlock());
    auto executor_graph =
        rewriter.create<tf_executor::GraphOp>(loc, func_type.getResults());
    executor_graph.body().takeBody(graph.nodes());

    // Add terminator of tf_executor::graph
    rewriter.setInsertionPointToEnd(&executor_graph.body().front());
    rewriter.create<tf_executor::FetchOp>(loc);

    // Add terminator of func
    rewriter.setInsertionPointToEnd(&func.body().front());
    rewriter.create<ReturnOp>(loc);

    rewriter.replaceOp(graph.getOperation(), func.getOperation()->getResults());

    return success();
  }
};

class ConvertGraphFuncOp : public OpConversionPattern<tfg::GraphFuncOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      tfg::GraphFuncOp graph_func, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const final {
    assert(!graph_func.generic());

    FuncOp func = rewriter.create<FuncOp>(
        graph_func.getLoc(),
        graph_func->getAttrOfType<StringAttr>(SymbolTable::getSymbolAttrName())
            .getValue(),
        graph_func.getType());

    func->setAttrs(graph_func->getAttrs());

    llvm::SmallVector<DictionaryAttr> arg_attrs;
    llvm::SmallVector<DictionaryAttr> res_attrs;
    if (failed(FilterTfgSpecificArgResultAttributes(
            getContext(), graph_func.getAllArgAttrs(), arg_attrs)) ||
        failed(FilterTfgSpecificArgResultAttributes(
            getContext(), graph_func.getAllResultAttrs(), res_attrs)))
      return failure();

    // Update arg/result attributes.
    func.setAllArgAttrs(arg_attrs);
    func.setAllResultAttrs(res_attrs);

    rewriter.setInsertionPointToStart(func.addEntryBlock());
    func.body().takeBody(graph_func.body());

    rewriter.replaceOp(graph_func.getOperation(),
                       func.getOperation()->getResults());

    return success();
  }
};

class ConvertReturnOp : public OpConversionPattern<tfg::ReturnOp> {
 public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      tfg::ReturnOp ret, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const final {
    rewriter.replaceOpWithNewOp<ReturnOp>(ret.getOperation(),
                                          adaptor.getOperands());
    return success();
  }
};

class ConvertControlTriggerOp : public ConversionPattern {
 public:
  explicit ConvertControlTriggerOp(MLIRContext *context)
      : ConversionPattern("tfg.ControlTrigger", PatternBenefit(1), context) {}

  LogicalResult matchAndRewrite(
      Operation *op, llvm::ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    llvm::SmallVector<Type, 2> new_types(op->getResultTypes());
    new_types.back() = rewriter.getType<tf_executor::ControlType>();

    rewriter.replaceOpWithNewOp<tf_executor::ControlTriggerOp>(
        op, new_types, operands, op->getAttrs());
    return success();
  }
};

class ConvertEnterOp : public ConversionPattern {
 public:
  explicit ConvertEnterOp(MLIRContext *context)
      : ConversionPattern("tfg.Enter", PatternBenefit(1), context) {}

  LogicalResult matchAndRewrite(
      Operation *op, llvm::ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    llvm::SmallVector<Type, 2> new_types(op->getResultTypes());
    new_types.back() = rewriter.getType<tf_executor::ControlType>();

    rewriter.replaceOpWithNewOp<tf_executor::EnterOp>(op, new_types, operands,
                                                      op->getAttrs());
    return success();
  }
};

class ConvertExitOp : public ConversionPattern {
 public:
  explicit ConvertExitOp(MLIRContext *context)
      : ConversionPattern("tfg.Exit", PatternBenefit(1), context) {}

  LogicalResult matchAndRewrite(
      Operation *op, llvm::ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    llvm::SmallVector<Type, 2> new_types(op->getResultTypes());
    new_types.back() = rewriter.getType<tf_executor::ControlType>();

    rewriter.replaceOpWithNewOp<tf_executor::ExitOp>(op, new_types, operands,
                                                     op->getAttrs());
    return success();
  }
};

class ConvertLoopCondOp : public ConversionPattern {
 public:
  explicit ConvertLoopCondOp(MLIRContext *context)
      : ConversionPattern("tfg.LoopCond", PatternBenefit(1), context) {}

  LogicalResult matchAndRewrite(
      Operation *op, llvm::ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    llvm::SmallVector<Type, 2> new_types(op->getResultTypes());
    new_types.back() = rewriter.getType<tf_executor::ControlType>();

    rewriter.replaceOpWithNewOp<tf_executor::LoopCondOp>(
        op, new_types, operands, op->getAttrs());
    return success();
  }
};

class ConvertMergeOp : public ConversionPattern {
 public:
  explicit ConvertMergeOp(MLIRContext *context)
      : ConversionPattern("tfg.Merge", PatternBenefit(1), context) {}

  LogicalResult matchAndRewrite(
      Operation *op, llvm::ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    llvm::SmallVector<Type, 2> new_types(op->getResultTypes());
    new_types.back() = rewriter.getType<tf_executor::ControlType>();

    rewriter.replaceOpWithNewOp<tf_executor::MergeOp>(op, new_types, operands,
                                                      op->getAttrs());
    return success();
  }
};

class ConvertSwitchOp : public ConversionPattern {
 public:
  explicit ConvertSwitchOp(MLIRContext *context)
      : ConversionPattern("tfg.Switch", PatternBenefit(1), context) {}

  LogicalResult matchAndRewrite(
      Operation *op, llvm::ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    llvm::SmallVector<Type, 2> new_types(op->getResultTypes());
    new_types.back() = rewriter.getType<tf_executor::ControlType>();

    rewriter.replaceOpWithNewOp<tf_executor::SwitchOp>(op, new_types, operands,
                                                       op->getAttrs());
    return success();
  }
};

class ConvertSwitchNOp : public ConversionPattern {
 public:
  explicit ConvertSwitchNOp(MLIRContext *context)
      : ConversionPattern("tfg.SwitchN", PatternBenefit(1), context) {}

  LogicalResult matchAndRewrite(
      Operation *op, llvm::ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    llvm::SmallVector<Type, 2> new_types(op->getResultTypes());
    new_types.back() = rewriter.getType<tf_executor::ControlType>();

    rewriter.replaceOpWithNewOp<tf_executor::SwitchNOp>(op, new_types, operands,
                                                        op->getAttrs());
    return success();
  }
};

class ConvertGeneralOp : public ConversionPattern {
 public:
  ConvertGeneralOp(MLIRContext *context,
                   const DenseSet<StringRef> &func_symbols)
      : ConversionPattern(MatchAnyOpTypeTag(), PatternBenefit(1), context),
        func_symbols_(func_symbols) {}

  LogicalResult matchAndRewrite(
      Operation *op, llvm::ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    if (!llvm::isa<tfg::TFGraphDialect>(op->getDialect())) return failure();

    Location loc = op->getLoc();
    llvm::SmallVector<mlir::Type, 2> new_types(op->getResultTypes());
    // Update the control type from tf_type.control to tf_executor.control.
    new_types.back() = rewriter.getType<tf_executor::ControlType>();

    // Control operand is attached on tf_executor::IslandOp.
    llvm::SmallVector<Value> island_control_operands;
    llvm::SmallVector<Value> inner_op_operands;

    for (Value value : operands) {
      // Because of the property of graph region, the control operands may
      // not have been converted to tf_executor::ControlType.
      if (value.getType().isa<tfg::ControlType>() ||
          value.getType().isa<tf_executor::ControlType>()) {
        island_control_operands.push_back(value);
      } else {
        inner_op_operands.push_back(value);
      }
    }

    auto island = rewriter.create<tf_executor::IslandOp>(
        loc, new_types, island_control_operands);
    island.body().push_back(new mlir::Block);

    rewriter.setInsertionPointToEnd(&island.body().front());

    // Control dependency has been applied on tf_executor.island. Remove it
    // while creating the tf operations.
    new_types.pop_back();

    llvm::SmallVector<std::unique_ptr<Region>, 1> new_regions;
    for (auto &region : op->getRegions()) {
      new_regions.push_back(std::make_unique<Region>());
      new_regions.back()->takeBody(region);
    }

    llvm::SmallVector<NamedAttribute, 4> attrs;
    if (failed(ReformatOpAttributes(getContext(), op->getAttrs(), attrs)))
      return failure();

    Operation *inner_op;

    StringRef op_name = op->getName().stripDialect();
    if (!func_symbols_.contains(op_name)) {
      std::string tf_op_name = llvm::formatv(
          "{0}.{1}", TF::TensorFlowDialect::getDialectNamespace(), op_name);
      OperationState state =
          OperationState(loc, tf_op_name, inner_op_operands, new_types, attrs,
                         op->getSuccessors(), new_regions);
      inner_op = rewriter.createOperation(state);
    } else {
      bool disable_call_shape_inference = false;
      if (op->hasAttr("_disable_call_shape_inference")) {
        disable_call_shape_inference =
            op->getAttrOfType<BoolAttr>("_disable_call_shape_inference")
                .getValue();
      }
      inner_op =
          rewriter.create<LegacyCallOp>(loc, new_types, inner_op_operands,
                                        op_name, disable_call_shape_inference);
    }

    rewriter.create<tf_executor::YieldOp>(loc, inner_op->getResults());

    rewriter.replaceOp(op, island.getOperation()->getResults());

    return success();
  }

 private:
  const DenseSet<StringRef> &func_symbols_;
};

class LegalizeTFGToTFE : public TF::LegalizeTFGToTFPassBase<LegalizeTFGToTFE> {
  void getDependentDialects(DialectRegistry &registry) const override {
    RegisterAllTensorFlowDialects(registry);
  }

  void runOnOperation() override;
};

}  // namespace

void LegalizeTFGToTFE::runOnOperation() {
  MLIRContext &context = getContext();
  ModuleOp module = getOperation();

  DenseSet<StringRef> func_symbols;
  for (auto &op : module.body().getOps()) {
    if (auto func = llvm::dyn_cast<tfg::GraphFuncOp>(op)) {
      func_symbols.insert(
          func->getAttrOfType<StringAttr>(SymbolTable::getSymbolAttrName())
              .getValue());
    }
  }

  ConversionTarget target(context);
  target.addLegalDialect<TF::TensorFlowDialect>();
  target.addLegalDialect<tf_executor::TensorFlowExecutorDialect>();
  target.addLegalOp<ModuleOp>();
  target.addLegalOp<FuncOp>();
  target.addLegalOp<ReturnOp>();

  OwningRewritePatternList patterns(&context);
  patterns.insert<ConvertGraphOp>(&context);
  patterns.insert<ConvertGraphFuncOp>(&context);
  patterns.insert<ConvertReturnOp>(&context);
  patterns.insert<ConvertGeneralOp>(&context, func_symbols);
  // Control flow V1 operation conversion patterns.
  patterns.insert<ConvertControlTriggerOp>(&context);
  patterns.insert<ConvertEnterOp>(&context);
  patterns.insert<ConvertExitOp>(&context);
  patterns.insert<ConvertLoopCondOp>(&context);
  patterns.insert<ConvertMergeOp>(&context);
  patterns.insert<ConvertSwitchOp>(&context);
  patterns.insert<ConvertSwitchNOp>(&context);
  FrozenRewritePatternSet finalPatterns(std::move(patterns));

  // Turn the graph region into SSACFG region by applying an order to the
  // operations.
  for (auto &op : module.body().getOps()) {
    for (auto &region : op.getRegions()) {
      for (auto &block : region) {
        // Split tfg.NextIteration to break the cycle.
        SplitNextIteration(block);
        tfg::SortTopologically(&block);
      }
    }
  }

  // Version information is embedded in graph operation in TFG. In TFE, it's
  // embedded in the module operation.
  for (auto &op : module.body().getOps()) {
    auto graph = dyn_cast<tfg::GraphOp>(op);
    if (!graph) continue;
    Builder b(&context);
    auto producer = b.getNamedAttr(
        "producer", b.getI32IntegerAttr(graph.version().getProducer()));
    auto min_consumer = b.getNamedAttr(
        "min_consumer", b.getI32IntegerAttr(graph.version().getMinConsumer()));
    auto bad_consumers = b.getNamedAttr(
        "bad_consumers", b.getI32ArrayAttr(graph.version().getBadConsumers()));
    module->setAttr("tf.versions",
                    b.getDictionaryAttr(llvm::ArrayRef<NamedAttribute>(
                        {producer, min_consumer, bad_consumers})));
    break;
  }

  if (failed(applyFullConversion(module.getOperation(), target, finalPatterns)))
    signalPassFailure();
}

std::unique_ptr<Pass> CreateLegalizeTFGToTFEPass() {
  return std::make_unique<LegalizeTFGToTFE>();
}

}  // end namespace TF
}  // end namespace mlir
