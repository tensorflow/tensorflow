/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/transforms/func_to_graph/pass.h"

#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "tensorflow/core/ir/ops.h"
#include "tensorflow/core/transforms/pass_detail.h"

namespace mlir {
namespace tfg {
namespace {
class FuncToGraphPass : public FuncToGraphBase<FuncToGraphPass> {
 public:
  FuncToGraphPass() = default;

  LogicalResult initialize(MLIRContext *context) override {
    graph_version_attr_name_ =
        StringAttr::get(context, "tfg.lifted_graph_version");
    lifted_value_attr_name_ = StringAttr::get(context, "tfg.lifted_value_attr");
    return success();
  }

  // This will try to lower the function that has attribute
  // `tfg.lifted_graph_version` to a graph. It replaces all the uses of
  // arguments with related op results. The relation between args and ops is
  // identified by the tfg.name attr. The arg's tfg.name attr will be prefixed
  // with the related op's tfg.name. Besides, The ReturnOp will be dropped
  // directly.
  void runOnOperation() override;

 private:
  StringAttr graph_version_attr_name_;
  StringAttr lifted_value_attr_name_;
};
}  // namespace

void FuncToGraphPass::runOnOperation() {
  ModuleOp module = getOperation();

  GraphFuncOp lifted_graph_func;
  for (auto func : module.getOps<GraphFuncOp>()) {
    // Only lifted graph function will have this attribute and there will be at
    // most one lifted graph function in a module.
    if (!func->hasAttr(graph_version_attr_name_)) continue;
    if (!lifted_graph_func) {
      lifted_graph_func = func;
    } else {
      module.emitError(
          "Only one lifted graph function is allowed in a module, but see ")
          << lifted_graph_func.sym_name() << " and " << func.sym_name();
      return signalPassFailure();
    }
  }

  if (!lifted_graph_func) return;

  DenseMap<StringRef, Operation *> referred_ops;

  if (ArrayAttr all_arg_attrs = lifted_graph_func.getAllArgAttrs()) {
    for (auto arg_attr : all_arg_attrs.getAsRange<DictionaryAttr>()) {
      auto lifted_value_attr =
          arg_attr.getAs<ArrayAttr>(lifted_value_attr_name_);
      // Control arg doesn't have lifted_value_attr, just skip it here. For
      // non-control arg, this attribute is required. This invariant will be
      // checked below.
      if (!lifted_value_attr) continue;

      // Init the entry with nullptr and it'll be updated with associated op
      // later.
      referred_ops.insert({lifted_value_attr[0].cast<StringAttr>().getValue(),
                           /*Operation=*/nullptr});
    }
  }

  for (Operation &op : lifted_graph_func.getBody()->without_terminator()) {
    StringRef op_name = TFOp(op).name();
    auto it = referred_ops.find(op_name);
    if (it != referred_ops.end()) it->second = &op;
  }

  for (auto &it : llvm::enumerate(lifted_graph_func.getArguments())) {
    if (it.value().getType().isa<ControlType>()) continue;

    auto lifted_value_attr = lifted_graph_func.getArgAttrOfType<ArrayAttr>(
        it.index(), lifted_value_attr_name_);
    if (!lifted_value_attr) {
      lifted_graph_func.emitError("arg #")
          << it.index()
          << " is missing tfg.lifted_value_attr, can't be lowered";
      return signalPassFailure();
    }

    StringRef value_defining_op_name =
        lifted_value_attr[0].cast<StringAttr>().getValue();
    Operation *op = referred_ops[value_defining_op_name];
    if (!op) {
      lifted_graph_func.emitError(
          "lifted arg can't find the associated operation: ")
          << value_defining_op_name;
      return signalPassFailure();
    }

    unsigned result_index =
        lifted_value_attr[1].cast<IntegerAttr>().getValue().getZExtValue();
    if (result_index >= op->getNumResults()) {
      lifted_graph_func.emitError("result index out of bound: seeing index ")
          << result_index << " from lifted_value_attr of arg #" << it.index()
          << ", but op only has " << op->getNumResults() << " results";
      return signalPassFailure();
    }

    it.value().replaceAllUsesWith(op->getResult(result_index));
  }

  OpBuilder builder(lifted_graph_func);
  auto graph = builder.create<GraphOp>(
      lifted_graph_func.getLoc(),
      lifted_graph_func->getAttr(graph_version_attr_name_).cast<VersionAttr>());

  // Remove the terminator.
  lifted_graph_func.getBody()->getTerminator()->erase();
  graph.getRegion().takeBody(lifted_graph_func.getRegion());
  lifted_graph_func.erase();
}

std::unique_ptr<Pass> CreateFuncToGraphPass() {
  return std::make_unique<FuncToGraphPass>();
}

}  // namespace tfg
}  // namespace mlir
