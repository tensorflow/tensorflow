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
// tf.IfRegion ->  tf.If and tf.WhileRegion -> tf.While

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
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
#include "tensorflow/compiler/mlir/tensorflow/utils/attribute_utils.h"

#define DEBUG_TYPE "tf-region-cf-to-functional"

namespace mlir {
namespace TF {

namespace {

constexpr char kElseFuncNameAttr[] = "_else_func_name";
constexpr char kThenFuncNameAttr[] = "_then_func_name";
constexpr char kXlaPropagateCompileTimeConsts[] =
    "_xla_propagate_compile_time_consts";

#define GEN_PASS_DEF_REGIONCONTROLFLOWTOFUNCTIONALPASS
#include "tensorflow/compiler/mlir/tensorflow/transforms/tf_passes.h.inc"

struct RegionControlFlowToFunctional
    : public impl::RegionControlFlowToFunctionalPassBase<
          RegionControlFlowToFunctional> {
  void runOnOperation() override;

 private:
  LogicalResult ConvertIfOp(IfRegionOp if_region);
  LogicalResult ConvertWhileOp(WhileRegionOp while_region);

  // Get unique name by using the loc to name mapping.
  std::string GetName(Operation* op, StringRef suffix);

  tensorflow::OpOrArgLocNameMapper mapper;
  llvm::SmallVector<func::FuncOp, 4> worklist;
};

std::string RegionControlFlowToFunctional::GetName(Operation* op,
                                                   StringRef suffix) {
  return (mapper.GetUniqueName(op) + suffix).str();
}

// Returns all the external values referenced from the given regions. If the
// external value is a constant, sink it into the region instead (and do not
// add it to the returned vector).
llvm::SmallVector<Value, 4> CollectExternValues(Region& first, Region& second) {
  llvm::SetVector<Value> extern_values;

  for (Region* region : {&first, &second}) {
    llvm::SetVector<Value> region_extern_values;
    getUsedValuesDefinedAbove(*region, region_extern_values);

    // Sink down constants into the functions.
    for (auto extern_value : region_extern_values) {
      if (!matchPattern(extern_value, m_Constant())) {
        extern_values.insert(extern_value);
        continue;
      }
      // Add constant at start of region.
      auto const_builder = OpBuilder::atBlockBegin(&region->front());
      auto const_value = const_builder.clone(*extern_value.getDefiningOp());
      replaceAllUsesInRegionWith(extern_value, const_value->getResult(0),
                                 *region);
    }
  }

  return llvm::to_vector<4>(extern_values);
}

// Copies over optional attributes from source region op `src` to the given
// functional op `dst` and appropriately overrides any necessary attributes.
void CopyAndOverrideAttributes(Operation* src, Operation* dst,
                               OpBuilder* builder) {
  CopyDeviceAndUnderscoredAttributes(src, dst);

  // Explicitly override attribute to propagate constants to the functions
  // before compiling to XLA. This is necessary along with conversion to
  // functional format because inlined regions may have moved loop invariant ops
  // outside of the region which may cause some new legalization failures.
  // TODO(b/126739593): Enable this attribute in TensorFlow by default. Also,
  // see b/185542519 for the context.
  dst->setAttr(kXlaPropagateCompileTimeConsts, builder->getBoolAttr(true));
}

// Extracts the contents of a region with a single block into a new function.
// `extern_values` is the set of external values that the region refers to.
//
// Inputs to the terminator of the region are converted to return values of
// the function. If `extern_values_passthrough` is true, all the extern values
// are also added as return values from the function
void ExtractSingleBlockRegion(Region& region, StringRef name,
                              llvm::SmallVectorImpl<Value>& extern_values,
                              llvm::SmallVectorImpl<func::FuncOp>& worklist,
                              bool extern_values_passthrough) {
  ModuleOp module = region.getParentOfType<ModuleOp>();
  auto builder = OpBuilder::atBlockBegin(module.getBody());
  auto loc = region.getParentOp()->getLoc();
  Block& entry = region.front();
  int num_region_arguments = entry.getNumArguments();
  Operation* terminator = entry.getTerminator();

  // Build the function type. Region arguments and extern values together
  // become the function arguments, with region arguments going first.
  auto input_types = llvm::to_vector<4>(entry.getArgumentTypes());
  for (auto input : extern_values) input_types.push_back(input.getType());

  // Terminator operands and pass through extern values (if enabled) together
  // become the function return values.
  auto return_types = llvm::to_vector<4>(terminator->getOperandTypes());
  if (extern_values_passthrough)
    for (auto input : extern_values) return_types.push_back(input.getType());

  auto type = FunctionType::get(region.getContext(), input_types, return_types);

  // Create new function and extract region body into the function.
  auto outlined_func = builder.create<func::FuncOp>(loc, name, type);
  Region& func_region = outlined_func.getBody();
  func_region.takeBody(region);
  Block& first_block = func_region.front();

  // Replace all external uses with function arguments.
  for (auto it : llvm::enumerate(extern_values)) {
    Value arg = first_block.addArgument(it.value().getType(), loc);
    replaceAllUsesInRegionWith(it.value(), arg, func_region);
  }

  // Function return values are all the terminator operands + pass through
  // extern values (if enabled).
  auto return_values = llvm::to_vector<4>(terminator->getOperands());
  if (extern_values_passthrough)
    return_values.insert(return_values.end(),
                         first_block.args_begin() + num_region_arguments,
                         first_block.args_end());

  // Replace the existing terminator with a return.
  terminator = first_block.getTerminator();
  builder.setInsertionPoint(terminator);
  builder.create<func::ReturnOp>(terminator->getLoc(), return_values);
  terminator->erase();

  outlined_func.setPrivate();

  // Add the outlined function to the worklist in case its body has
  // IfRegion or WhileRegion ops that need to converted.
  worklist.push_back(outlined_func);
}

// Returns call for region with single call whose result feeds into the
// terminator of the region. if `allow_to_bool` is true, also allows a single
// ToBoolOp between the region yield and the call. Returns none if the region
// does not conform to this pattern.
llvm::Optional<func::CallOp> IsSingleCallRegion(Region& region,
                                                bool allow_to_bool = false) {
  if (!llvm::hasSingleElement(region)) return llvm::None;

  Block& block = region.front();
  auto it = block.rbegin();
  YieldOp yield = dyn_cast<YieldOp>(*it++);

  if (it == block.rend()) return llvm::None;

  // Operation which is expected to consume all the call results.
  Operation* call_consumer = yield;

  // Allow a single ToBoolOp between the call and the yield (valid only
  // when the yield has a single operand)
  if (allow_to_bool && yield.getNumOperands() == 1 && isa<ToBoolOp>(*it)) {
    if (it->getResult(0) != yield.getOperand(0)) return llvm::None;
    call_consumer = cast<ToBoolOp>(*it);
    it++;
    if (it == block.rend()) return llvm::None;
  }

  // Check if there is a Call before the Yield.
  func::CallOp call = dyn_cast<func::CallOp>(*it++);
  if (!call) return llvm::None;

  // All call results should feed into expected consumer
  // All results of the call should feed into the yield.
  if (call.getNumResults() != call_consumer->getNumOperands())
    return llvm::None;

  for (auto res_it : llvm::zip(call.getResults(), call_consumer->getOperands()))
    if (std::get<0>(res_it) != std::get<1>(res_it)) return llvm::None;

  // There can only be non-truncating cast op's prior to the call.
  for (; it != block.rend(); ++it) {
    CastOp cast = dyn_cast<CastOp>(*it);
    if (!cast || cast.getTruncate()) return llvm::None;
  }

  return call;
}

using ArgMatcherFn = function_ref<bool(Value, Region&, Value, Region&)>;

// Returns whether the arguments of the given 2 calls are match (after looking
// through cast ops). `matcher` is the predicate used to check if two arguments
// match.
bool MatchCallArgs(func::CallOp first, func::CallOp second,
                   ArgMatcherFn matcher) {
  if (first.getNumOperands() != second.getNumOperands()) return false;

  Region& first_region = *first->getParentRegion();
  Region& second_region = *second->getParentRegion();

  for (auto it : llvm::zip(first.getArgOperands(), second.getArgOperands())) {
    // Get the defining Op, skipping over casts.
    auto get_defining_op = [](Value value) {
      while (auto cast_op =
                 llvm::dyn_cast_or_null<CastOp>(value.getDefiningOp())) {
        // Consider cast compatibility in case
        //    %cast = "tf.Cast"(%0) : (tensor<2xi64>) -> tensor<2xf32>
        // is skipped.
        if (cast_op.getSrcT() != cast_op.getDstT()) {
          break;
        }
        value = cast_op.getOperand();
      }
      return value;
    };
    Value first_arg = get_defining_op(std::get<0>(it));
    Value second_arg = get_defining_op(std::get<1>(it));

    if (!matcher(first_arg, first_region, second_arg, second_region))
      return false;
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
  llvm::SmallVector<StringRef, 2> callee_names;

  // Analyzes the given calls (from regions attached to the same parent op) to
  // check if the parent op be transformed to functional form trivially (i.e.,
  // reusing existing functions and without outlining). This is possible when
  // all the regions are single call regions (checked using matchers outside
  // this class) and the all the calls match using the given argument matcher.
  //
  // If such a trivial transformation is possible, stash the relevant
  // information needed for the transformation, else indicate that a trivial
  // transformation is not possible by setting `can_transform` to false.
  TrivialTransformInfo(llvm::Optional<func::CallOp> first_call,
                       llvm::Optional<func::CallOp> second_call,
                       ArgMatcherFn arg_matcher) {
    if (!first_call || !second_call) return;

    if (!MatchCallArgs(first_call.value(), second_call.value(), arg_matcher))
      return;

    can_transform = true;
    callee_names = {first_call.value().getCallee(),
                    second_call.value().getCallee()};
  }
};

// Transform IfRegionOp to IfOp.
LogicalResult RegionControlFlowToFunctional::ConvertIfOp(IfRegionOp if_region) {
  llvm::SmallVector<Value, 4> extern_values;

  // For IfOp, arguments of calls in the then and else regions match if they
  // are the same value.
  auto if_arg_matcher = [&](Value first, Region&, Value second, Region&) {
    if (first != second) return false;

    // collect the call arguments post lookup through cast Op's
    extern_values.push_back(first);
    return true;
  };

  const TrivialTransformInfo tti(IsSingleCallRegion(if_region.getThenBranch()),
                                 IsSingleCallRegion(if_region.getElseBranch()),
                                 if_arg_matcher);

  std::string then_name, else_name;

  if (tti.can_transform) {
    // We can transform to functional form trivially without outlining.
    then_name = tti.callee_names[0].str();
    else_name = tti.callee_names[1].str();
  } else {
    // Collect external values that are used within the else and then bodies.
    extern_values = CollectExternValues(if_region.getThenBranch(),
                                        if_region.getElseBranch());

    // These external values need to be added as inputs to the generated If. The
    // order is determined by the order of these values the `extern_vales`.

    // Create 2 new functions with the input signature matching this order,
    // and outline the `then` and `else` regions by moving the bodies of these
    // regions into these functions. Replace tf.yield with a regular return.
    if (if_region->hasAttrOfType<StringAttr>(kThenFuncNameAttr) &&
        !if_region.get_thenFuncNameAttr().getValue().empty()) {
      then_name =
          mapper.GetUniqueName(if_region.get_thenFuncNameAttr().getValue())
              .str();
    } else {
      then_name = GetName(if_region, "_then");
    }
    ExtractSingleBlockRegion(if_region.getThenBranch(), then_name,
                             extern_values, worklist,
                             /*extern_values_passthrough=*/false);

    if (if_region->hasAttrOfType<StringAttr>(kElseFuncNameAttr) &&
        !if_region.get_elseFuncNameAttr().getValue().empty()) {
      else_name =
          mapper.GetUniqueName(if_region.get_elseFuncNameAttr().getValue())
              .str();
    } else {
      else_name = GetName(if_region, "_else");
    }
    ExtractSingleBlockRegion(if_region.getElseBranch(), else_name,
                             extern_values, worklist,
                             /*extern_values_passthrough=*/false);
  }

  // Look through ToBool operations for the condition.
  Value cond = if_region.getCond();
  auto to_bool = dyn_cast_or_null<ToBoolOp>(cond.getDefiningOp());
  if (to_bool) cond = to_bool.getOperand();

  // Once we have the `then` and `else` functions ready (either outlined or
  // existing ones), replace the region based op with a functional control flow
  // op.
  OpBuilder builder(if_region);
  auto if_op = builder.create<IfOp>(
      if_region.getLoc(), if_region.getResultTypes(), cond, extern_values,
      then_name, else_name, if_region.getIsStateless());
  CopyAndOverrideAttributes(if_region, if_op, &builder);

  if_region.replaceAllUsesWith(if_op.getResults());
  if_region.erase();

  if (to_bool && to_bool.use_empty()) to_bool.erase();
  return success();
}

// Transform WhileRegion to WhileOp.
LogicalResult RegionControlFlowToFunctional::ConvertWhileOp(
    WhileRegionOp while_region) {
  // For While, the arguments of the calls in the body and cond regions match
  // if they are region arguments with the same region argument numbers. If the
  // 2 calls have the same value (an extern value) used as an argument, we
  // cannot do a trivial transformation because post transform, we will need to
  // pass this extern value as an argument to the function, so we cannot use the
  // existing function as is.
  auto while_arg_matcher = [](Value first, Region& first_region, Value second,
                              Region& second_region) {
    if (!first.isa<BlockArgument>() || !second.isa<BlockArgument>())
      return false;
    BlockArgument first_block_arg = first.cast<BlockArgument>();
    BlockArgument second_block_arg = second.cast<BlockArgument>();

    // 2 block arguments will match if they are the same argument number, and
    // are block arguments of the corresponding containing regions.
    return first_block_arg.getArgNumber() == second_block_arg.getArgNumber() &&
           first_block_arg.getParentBlock() == &first_region.front() &&
           second_block_arg.getParentBlock() == &second_region.front();
  };

  const TrivialTransformInfo tti(
      IsSingleCallRegion(while_region.getCond(), /*allow_to_bool=*/true),
      IsSingleCallRegion(while_region.getBody()), while_arg_matcher);

  // All existing inputs to while region are inputs to the functional while.
  auto new_inputs = llvm::to_vector<4>(while_region.getOperands());

  // All existing results will also be generated by the functional while.
  auto new_result_types = llvm::to_vector<4>(while_region.getResultTypes());

  std::string cond_name, body_name;
  if (tti.can_transform) {
    // We can transform to functional form trivially without outlining.
    cond_name = tti.callee_names[0].str();
    body_name = tti.callee_names[1].str();
  } else {
    // The WhileRegion regions can refer to either arguments of the region, or
    // external values implicitly captured by the region. When converting to
    // functional form, all such external values need to become function
    // arguments of the outlined functions, and become pass through values in
    // the outlined body function. So when outlining the while body, in addition
    // to the region arguments, all these external references need to be added
    // as function arguments.
    llvm::SmallVector<Value, 4> extern_values =
        CollectExternValues(while_region.getCond(), while_region.getBody());

    // Outline the `cond` and `body` regions by moving the bodies of these
    // regions into new functions. Replace tf.yield with a regular return.
    cond_name = GetName(while_region, "_cond");
    ExtractSingleBlockRegion(while_region.getCond(), cond_name, extern_values,
                             worklist, /*extern_values_passthrough=*/false);

    body_name = GetName(while_region, "_body");
    ExtractSingleBlockRegion(while_region.getBody(), body_name, extern_values,
                             worklist, /*extern_values_passthrough=*/true);

    // All extern values become additional inputs and additional output types
    // for the functional while.
    new_inputs.append(extern_values.begin(), extern_values.end());
    for (auto ext : extern_values) new_result_types.push_back(ext.getType());
  }

  // Once we have the `cond` and `body` functions ready (either outlined or
  // existing ones), replace the region based op with a functional op.
  OpBuilder builder(while_region);
  auto while_op = builder.create<WhileOp>(
      while_region.getLoc(), new_result_types, new_inputs, cond_name, body_name,
      while_region.getParallelIterations(), while_region.getIsStateless(),
      while_region.getShapeInvariant());
  CopyAndOverrideAttributes(while_region, while_op, &builder);

  // Redirect old results to new results.
  for (auto it : llvm::zip(
           while_region.getResults(),
           while_op.getResults().take_front(while_region.getNumResults())))
    std::get<0>(it).replaceAllUsesWith(std::get<1>(it));

  while_region.erase();
  return success();
}

void RegionControlFlowToFunctional::runOnOperation() {
  ModuleOp module = getOperation();

  // Seed worklist with all functions in the module.
  worklist = llvm::to_vector<4>(module.getOps<func::FuncOp>());
  while (!worklist.empty()) {
    func::FuncOp function = worklist.pop_back_val();

    auto result = function.walk([&](Operation* op) {
      if (auto if_region = llvm::dyn_cast<IfRegionOp>(op)) {
        if (failed(ConvertIfOp(if_region))) {
          op->emitOpError() << "failed to convert to functional form";
          return WalkResult::interrupt();
        }
      } else if (auto while_region = llvm::dyn_cast<WhileRegionOp>(op)) {
        if (failed(ConvertWhileOp(while_region))) {
          op->emitOpError() << "failed to convert to functional form";
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

}  // namespace TF
}  // namespace mlir
