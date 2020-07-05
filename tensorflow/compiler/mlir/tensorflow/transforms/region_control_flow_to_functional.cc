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

// This transformation pass transforms region bases control flow operations in
// the TensorFlow dialect to their functional counterparts, i.e.,
// tf.IfRegion ->  tf.If

#include "llvm/ADT/STLExtras.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/Function.h"  // from @llvm-project
#include "mlir/IR/Module.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/TypeUtilities.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/IR/Verifier.h"  // from @llvm-project
#include "mlir/IR/Visitors.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Pass/PassRegistry.h"  // from @llvm-project
#include "mlir/Transforms/RegionUtils.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/op_or_arg_name_mapper.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_types.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes.h"

namespace mlir {
namespace TF {

namespace {

struct RegionControlFlowToFunctional
    : public PassWrapper<RegionControlFlowToFunctional,
                         OperationPass<ModuleOp>> {
  void runOnOperation() override;

 private:
  LogicalResult ConvertIfOp(IfRegionOp if_region);

  // Get unique name by using the loc to name mapping.
  std::string GetName(Operation* op, StringRef suffix);

  tensorflow::OpOrArgLocNameMapper mapper;
  llvm::SmallVector<FuncOp, 4> worklist;
};

std::string RegionControlFlowToFunctional::GetName(Operation* op,
                                                   StringRef suffix) {
  return (mapper.GetUniqueName(op) + suffix).str();
}

// Returns all the external values referenced from the given set of regions. If
// the external value is a constant, sink it into the region instead (and do not
// add it to the returned vector).
llvm::SmallVector<Value, 4> CollectExternValues(ArrayRef<Region*> regions) {
  llvm::SetVector<Value> extern_values_set;

  for (auto region : regions) {
    llvm::SetVector<Value> region_extern_values;
    getUsedValuesDefinedAbove(*region, region_extern_values);

    // Sink down constants into the functions.
    for (auto extern_value : region_extern_values) {
      if (!matchPattern(extern_value, m_Constant())) {
        extern_values_set.insert(extern_value);
        continue;
      }
      // Add constant at start of region.
      auto const_builder = OpBuilder::atBlockBegin(&region->front());
      auto const_value = const_builder.clone(*extern_value.getDefiningOp());
      replaceAllUsesInRegionWith(extern_value, const_value->getResult(0),
                                 *region);
    }
  }

  return {extern_values_set.begin(), extern_values_set.end()};
}

// Extracts the contents of a region with a single block into a new function.
// `extern_values` is the set of external values that the region refers to.
//
// Any inputs to the terminator of the region are converted to return values of
// the function. If any of these values is not exact type as the function's
// return type, appropriate cast operations will be inserted
void ExtractSingleBlockRegion(Region& region, FunctionType type, StringRef name,
                              llvm::SmallVectorImpl<Value>& extern_values,
                              llvm::SmallVectorImpl<FuncOp>& worklist) {
  ModuleOp module = region.getParentOfType<ModuleOp>();
  auto builder = OpBuilder::atBlockBegin(module.getBody());
  auto loc = region.getParentOp()->getLoc();

  // Create new function and extract region body into the function.
  auto outlined_func =
      builder.create<FuncOp>(loc, name, type, ArrayRef<NamedAttribute>{});

  outlined_func.getBody().takeBody(region);
  Region& func_region = outlined_func.getBody();
  Block& first_block = func_region.front();

  // Replace all external uses with function arguments.
  for (auto it : llvm::enumerate(extern_values)) {
    Value arg = first_block.addArgument(it.value().getType());
    replaceAllUsesInRegionWith(it.value(), arg, func_region);
  }

  // Replace the existing terminator with a return.
  Operation* terminator = outlined_func.getBody().front().getTerminator();
  builder.setInsertionPoint(terminator);

  SmallVector<Value, 4> return_values;
  return_values.reserve(terminator->getNumOperands());
  for (auto it : llvm::enumerate(type.getResults())) {
    Value ret_val = terminator->getOperand(it.index());
    // Add a cast operation if types do not match.
    if (ret_val.getType() != it.value()) {
      ret_val =
          builder.create<CastOp>(terminator->getLoc(), it.value(), ret_val);
    }
    return_values.push_back(ret_val);
  }
  builder.create<ReturnOp>(terminator->getLoc(), return_values);
  terminator->erase();
  outlined_func.setVisibility(FuncOp::Visibility::Private);

  // Add the outlined function to the worklist in case its body has
  // IfRegion ops that need to converted.
  worklist.push_back(outlined_func);
}

// Returns call for region with single call whose result feeds into the
// terminator of the region. Returns none if the region doesn't contain just
// call and non-truncting casts ops.
llvm::Optional<CallOp> IsSingleCallRegion(Region& region) {
  if (!llvm::hasSingleElement(region)) return llvm::None;

  Block& block = region.front();
  auto it = block.rbegin();
  YieldOp yield = dyn_cast<YieldOp>(*it++);

  if (it == block.rend()) return llvm::None;

  // Check if there is a Call before the Yield.
  CallOp call = dyn_cast<CallOp>(*it++);
  if (!call) return llvm::None;

  // There can only be non-truncating cast op's prior to the call.
  for (; it != block.rend(); ++it) {
    CastOp cast = dyn_cast<CastOp>(*it);
    if (!cast || cast.Truncate()) return llvm::None;
  }

  // All results of the call should feed into the yield.
  if (call.getNumResults() != yield.getNumOperands()) return llvm::None;

  for (auto res_it : llvm::zip(call.getResults(), yield.getOperands()))
    if (std::get<0>(res_it) != std::get<1>(res_it)) return llvm::None;

  return call;
}

// Returns whether the arguments of the given call are same as the given list of
// arguments (after looking through cast ops).
bool MatchCallArgs(CallOp call, llvm::SmallVectorImpl<Value>& args) {
  if (call.getNumOperands() != args.size()) return false;

  for (auto it : llvm::enumerate(args)) {
    Value arg = call.getOperand(it.index());
    if (auto cast = dyn_cast_or_null<CastOp>(arg.getDefiningOp()))
      arg = cast.getOperand();

    if (arg != it.value()) return false;
  }
  return true;
}

// Summary information for trivially transforming region based op's to
// functional ops. A trivial transformation can be done when the regions are
// just calls to functions, in which case no outlining is needed.
struct TrivialTransformInfo {
  // Can the op be transformed trivially?
  bool can_transform = false;

  // List of callee names (one for each region).
  llvm::SmallVector<StringRef, 4> callee_names;

  // List of arguments used in these call (each call uses the same arguments
  // potentially through casts).
  llvm::SmallVector<Value, 4> call_args;
};

// Analyzes the given set of regions (attached to the same parent op) to check
// if the parent op be transformed to functional form trivially (i.e., reusing
// existing functions and without outlining). This is possible when all the
// regions are single call regions and the all the calls have the same
// arguments.
//
// If this trivial transformation is possible, return the relevant information
// needed for the transformation (in `TrivialTransformInfo`), else indicate that
// a trivial transformation is not possible by setting `can_transform` false.
TrivialTransformInfo AnalyzeForTrivialTransform(ArrayRef<Region*> regions) {
  const TrivialTransformInfo cannot_transform;

  if (regions.empty()) return cannot_transform;

  llvm::SmallVector<CallOp, 2> calls;
  calls.reserve(regions.size());

  // Verify each region is a single call and collect these calls.
  for (Region* region : regions) {
    auto call = IsSingleCallRegion(*region);
    if (!call.hasValue()) return cannot_transform;
    calls.push_back(call.getValue());
  }

  llvm::SmallVector<StringRef, 4> callees;
  callees.reserve(regions.size());

  CallOp call0 = calls[0];
  int num_args = call0.getNumOperands();

  // Collect arguments of the first call.
  llvm::SmallVector<Value, 4> call0_args;
  call0_args.reserve(num_args);
  for (Value arg : call0.getArgOperands()) {
    if (auto cast = dyn_cast_or_null<CastOp>(arg.getDefiningOp()))
      arg = cast.getOperand();
    call0_args.push_back(arg);
  }

  // Match arguments of rest of the calls with those of the first call.
  for (auto call : calls) {
    if (call != call0 && !MatchCallArgs(call, call0_args))
      return cannot_transform;
    callees.push_back(call.getCallee());
  }

  return {true, callees, call0_args};
}

// Transform IfRegionOp to IfOp.
LogicalResult RegionControlFlowToFunctional::ConvertIfOp(IfRegionOp if_region) {
  const TrivialTransformInfo tti = AnalyzeForTrivialTransform(
      {&if_region.then_branch(), &if_region.else_branch()});

  std::string then_name, else_name;
  llvm::SmallVector<Value, 4> extern_values;

  if (tti.can_transform) {
    // We can transform to functional form trivially without outlining.
    then_name = tti.callee_names[0].str();
    else_name = tti.callee_names[1].str();
    extern_values = tti.call_args;
  } else {
    // Collect external values that are used within the else and then bodies.
    extern_values = CollectExternValues(
        {&if_region.then_branch(), &if_region.else_branch()});

    // These external values need to be added as inputs to the generated If. The
    // order is determined by the order of these values the `extern_vales`.

    // Build the type for the outlined function.
    llvm::SmallVector<Type, 4> input_types;
    input_types.reserve(extern_values.size());
    for (auto input : extern_values) input_types.push_back(input.getType());

    FunctionType func_type = FunctionType::get(
        input_types, if_region.getResultTypes(), if_region.getContext());

    // Create 2 new functions with the input signature matching this order,
    // and outline the `then` and `else` regions by moving the bodies of these
    // regions into these functions. Replace tf.yield with a regular return.
    then_name = GetName(if_region, "_then");
    ExtractSingleBlockRegion(if_region.then_branch(), func_type, then_name,
                             extern_values, worklist);

    else_name = GetName(if_region, "_else");
    ExtractSingleBlockRegion(if_region.else_branch(), func_type, else_name,
                             extern_values, worklist);
  }

  // Once we have the `then` and `else` functions ready (either outlined or
  // existing ones), replace the region based op with a functional control flow
  // op.
  OpBuilder builder(if_region);
  auto if_op = builder.create<IfOp>(
      if_region.getLoc(), if_region.getResultTypes(), if_region.cond(),
      extern_values, then_name, else_name, if_region.is_stateless());
  if_region.replaceAllUsesWith(if_op.getResults());
  if_region.erase();
  return success();
}

void RegionControlFlowToFunctional::runOnOperation() {
  ModuleOp module = getOperation();

  // Seed worklist with all functions in the module.
  worklist = llvm::to_vector<4>(module.getOps<FuncOp>());

  while (!worklist.empty()) {
    FuncOp function = worklist.pop_back_val();

    auto result = function.walk([&](Operation* op) {
      if (IfRegionOp if_region = llvm::dyn_cast<IfRegionOp>(op)) {
        if (failed(ConvertIfOp(if_region))) {
          if_region.emitOpError() << " failed to convert to functional form";
          return WalkResult::interrupt();
        }
      }
      return WalkResult::advance();
    });

    if (result.wasInterrupted()) return signalPassFailure();
  }
}

}  // namespace

std::unique_ptr<OperationPass<ModuleOp>>
CreateTFRegionControlFlowToFunctional() {
  return std::make_unique<RegionControlFlowToFunctional>();
}

static PassRegistration<RegionControlFlowToFunctional> pass(
    "tf-region-control-flow-to-functional",
    "Transform region bases control flow Ops to functional counterparts");

}  // namespace TF
}  // namespace mlir
