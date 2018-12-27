//===- ConvertToCFG.cpp - ML function to CFG function conversion ----------===//
//
// Copyright 2019 The MLIR Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================
//
// This file implements APIs to convert ML functions into CFG functions.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/StmtVisitor.h"
#include "mlir/Pass.h"
#include "mlir/StandardOps/StandardOps.h"
#include "mlir/Support/Functional.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Transforms/Utils.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/CommandLine.h"
using namespace mlir;

//===----------------------------------------------------------------------===//
// ML function converter
//===----------------------------------------------------------------------===//

namespace {
// Generates CFG function equivalent to the given ML function.
class FunctionConverter : public StmtVisitor<FunctionConverter> {
public:
  FunctionConverter(CFGFunction *cfgFunc)
      : cfgFunc(cfgFunc), builder(cfgFunc) {}
  CFGFunction *convert(MLFunction *mlFunc);

  void visitForStmt(ForStmt *forStmt);
  void visitIfStmt(IfStmt *ifStmt);
  void visitOperationStmt(OperationStmt *opStmt);

private:
  CFGValue *getConstantIndexValue(int64_t value);
  void visitStmtBlock(StmtBlock *stmtBlock);
  CFGValue *buildMinMaxReductionSeq(
      Location loc, CmpIPredicate predicate,
      llvm::iterator_range<Operation::result_iterator> values);

  CFGFunction *cfgFunc;
  CFGFuncBuilder builder;

  // Mapping between original MLValues and lowered CFGValues.
  llvm::DenseMap<const MLValue *, CFGValue *> valueRemapping;
};
} // end anonymous namespace

// Return a vector of OperationStmt's arguments as SSAValues.  For each
// statement operands, represented as MLValue, lookup its CFGValue conterpart in
// the valueRemapping table.
static llvm::SmallVector<SSAValue *, 4>
operandsAs(Statement *opStmt,
           const llvm::DenseMap<const MLValue *, CFGValue *> &valueRemapping) {
  llvm::SmallVector<SSAValue *, 4> operands;
  for (const MLValue *operand : opStmt->getOperands()) {
    assert(valueRemapping.count(operand) != 0 && "operand is not defined");
    operands.push_back(valueRemapping.lookup(operand));
  }
  return operands;
}

// Convert an operation statement into an operation instruction.
//
// The operation description (name, number and types of operands or results)
// remains the same but the values must be updated to be CFGValues.  Update the
// mapping MLValue->CFGValue as the conversion is performed.  The operation
// instruction is appended to current block (end of SESE region).
void FunctionConverter::visitOperationStmt(OperationStmt *opStmt) {
  // Set up basic operation state (context, name, operands).
  OperationState state(cfgFunc->getContext(), opStmt->getLoc(),
                       opStmt->getName());
  state.addOperands(operandsAs(opStmt, valueRemapping));

  // Set up operation return types.  The corresponding SSAValues will become
  // available after the operation is created.
  state.addTypes(
      functional::map([](SSAValue *result) { return result->getType(); },
                      opStmt->getResults()));

  // Copy attributes.
  for (auto attr : opStmt->getAttrs()) {
    state.addAttribute(attr.first.strref(), attr.second);
  }

  auto opInst = builder.createOperation(state);

  // Make results of the operation accessible to the following operations
  // through remapping.
  assert(opInst->getNumResults() == opStmt->getNumResults());
  for (unsigned i = 0, n = opInst->getNumResults(); i < n; ++i) {
    valueRemapping.insert(
        std::make_pair(opStmt->getResult(i), opInst->getResult(i)));
  }
}

// Create a CFGValue for the given integer constant of index type.
CFGValue *FunctionConverter::getConstantIndexValue(int64_t value) {
  auto op = builder.create<ConstantIndexOp>(builder.getUnknownLoc(), value);
  return cast<CFGValue>(op->getResult());
}

// Visit all statements in the given statement block.
void FunctionConverter::visitStmtBlock(StmtBlock *stmtBlock) {
  for (auto &stmt : *stmtBlock)
    this->visit(&stmt);
}

// Given a range of values, emit the code that reduces them with "min" or "max"
// depending on the provided comparison predicate.  The predicate defines which
// comparison to perform, "lt" for "min", "gt" for "max" and is used for the
// `cmpi` operation followed by the `select` operation:
//
//   %cond   = cmpi "predicate" %v0, %v1
//   %result = select %cond, %v0, %v1
//
// Multiple values are scanned in a linear sequence.  This creates a data
// dependences that wouldn't exist in a tree reduction, but is easier to
// recognize as a reduction by the subsequent passes.
CFGValue *FunctionConverter::buildMinMaxReductionSeq(
    Location loc, CmpIPredicate predicate,
    llvm::iterator_range<Operation::result_iterator> values) {
  assert(!llvm::empty(values) && "empty min/max chain");

  auto valueIt = values.begin();
  CFGValue *value = cast<CFGValue>(*valueIt++);
  for (; valueIt != values.end(); ++valueIt) {
    auto cmpOp = builder.create<CmpIOp>(loc, predicate, value, *valueIt);
    auto selectOp =
        builder.create<SelectOp>(loc, cmpOp->getResult(), value, *valueIt);
    value = cast<CFGValue>(selectOp->getResult());
  }

  return value;
}

// Convert a "for" loop to a flow of basic blocks.
//
// Create an SESE region for the loop (including its body) and append it to the
// end of the current region.  The loop region consists of the initialization
// block that sets up the initial value of the loop induction variable (%iv) and
// computes the loop bounds that are loop-invariant in MLFunctions; the
// condition block that checks the exit condition of the loop; the body SESE
// region; and the end block that post-dominates the loop.  The end block of the
// loop becomes the new end of the current SESE region.  The body of the loop is
// constructed recursively after starting a new region (it may be, for example,
// a nested loop).  Induction variable modification is appended to the body SESE
// region that always loops back to the condition block.
//
//      +--------------------------------+
//      | <end of current SESE region>   |
//      | <current insertion point>      |
//      | br init                        |
//      +--------------------------------+
//             |
//             v
//      +--------------------------------+
//      | init:                          |
//      |   <start of loop SESE region>  |
//      |   <compute initial %iv value>  |
//      |   br cond(%iv)                 |
//      +--------------------------------+
//             |
//  -------|   |
//  |      v   v
//  |   +--------------------------------+
//  |   | cond(%iv):                     |
//  |   |   <compare %iv to upper bound> |
//  |   |   cond_br %r, body, end        |
//  |   +--------------------------------+
//  |          |               |
//  |          |               -------------|
//  |          v                            |
//  |   +--------------------------------+  |
//  |   | body:                          |  |
//  |   |   <body SESE region start>     |  |
//  |   |   <...>                        |  |
//  |   +--------------------------------+  |
//  |          |                            |
//  |         ... <SESE region of the body> |
//  |          |                            |
//  |          v                            |
//  |   +--------------------------------+  |
//  |   | body-end:                      |  |
//  |   |   <body SESE region end>       |  |
//  |   |   %new_iv =<add step to %iv>   |  |
//  |   |   br cond(%new_iv)             |  |
//  |   +--------------------------------+  |
//  |          |                            |
//  |-----------        |--------------------
//                      v
//      +--------------------------------+
//      | end:                           |
//      |   <end of loop SESE region>    |
//      |   <new insertion point>        |
//      +--------------------------------+
//
void FunctionConverter::visitForStmt(ForStmt *forStmt) {
  // First, store the loop insertion location so that we can go back to it after
  // creating the new blocks (block creation updates the insertion point).
  BasicBlock *loopInsertionPoint = builder.getInsertionBlock();

  // Create blocks so that they appear in more human-readable order in the
  // output.
  BasicBlock *loopInitBlock = builder.createBlock();
  BasicBlock *loopConditionBlock = builder.createBlock();
  BasicBlock *loopBodyFirstBlock = builder.createBlock();

  // At the loop insertion location, branch immediately to the loop init block.
  builder.setInsertionPoint(loopInsertionPoint);
  builder.create<BranchOp>(builder.getUnknownLoc(), loopInitBlock);

  // The loop condition block has an argument for loop induction variable.
  // Create it upfront and make the loop induction variable -> basic block
  // argument remapping available to the following instructions.  ForStatement
  // is-a MLValue corresponding to the loop induction variable.
  builder.setInsertionPoint(loopConditionBlock);
  CFGValue *iv = loopConditionBlock->addArgument(builder.getIndexType());
  valueRemapping.insert(std::make_pair(forStmt, iv));

  // Recursively construct loop body region.
  // Walking manually because we need custom logic before and after traversing
  // the list of children.
  builder.setInsertionPoint(loopBodyFirstBlock);
  visitStmtBlock(forStmt->getBody());

  // Builder point is currently at the last block of the loop body.  Append the
  // induction variable stepping to this block and branch back to the exit
  // condition block.  Construct an affine map f : (x -> x+step) and apply this
  // map to the induction variable.
  auto affStep = builder.getAffineConstantExpr(forStmt->getStep());
  auto affDim = builder.getAffineDimExpr(0);
  auto affStepMap = builder.getAffineMap(1, 0, {affDim + affStep}, {});
  auto stepOp =
      builder.create<AffineApplyOp>(forStmt->getLoc(), affStepMap, iv);
  CFGValue *nextIvValue = cast<CFGValue>(stepOp->getResult(0));
  builder.create<BranchOp>(builder.getUnknownLoc(), loopConditionBlock,
                           nextIvValue);

  // Create post-loop block here so that it appears after all loop body blocks.
  BasicBlock *postLoopBlock = builder.createBlock();

  builder.setInsertionPoint(loopInitBlock);
  // Compute loop bounds using affine_apply after remapping its operands.
  auto remapOperands = [this](const SSAValue *value) -> SSAValue * {
    const MLValue *mlValue = dyn_cast<MLValue>(value);
    return valueRemapping.lookup(mlValue);
  };
  auto operands =
      functional::map(remapOperands, forStmt->getLowerBoundOperands());
  auto lbAffineApply = builder.create<AffineApplyOp>(
      forStmt->getLoc(), forStmt->getLowerBoundMap(), operands);
  CFGValue *lowerBound = buildMinMaxReductionSeq(
      forStmt->getLoc(), CmpIPredicate::SGT, lbAffineApply->getResults());
  operands = functional::map(remapOperands, forStmt->getUpperBoundOperands());
  auto ubAffineApply = builder.create<AffineApplyOp>(
      forStmt->getLoc(), forStmt->getUpperBoundMap(), operands);
  CFGValue *upperBound = buildMinMaxReductionSeq(
      forStmt->getLoc(), CmpIPredicate::SLT, ubAffineApply->getResults());
  builder.create<BranchOp>(builder.getUnknownLoc(), loopConditionBlock,
                           lowerBound);

  builder.setInsertionPoint(loopConditionBlock);
  auto comparisonOp = builder.create<CmpIOp>(
      forStmt->getLoc(), CmpIPredicate::SLT, iv, upperBound);
  auto comparisonResult = cast<CFGValue>(comparisonOp->getResult());
  builder.create<CondBranchOp>(builder.getUnknownLoc(), comparisonResult,
                               loopBodyFirstBlock, ArrayRef<SSAValue *>(),
                               postLoopBlock, ArrayRef<SSAValue *>());

  // Finally, make sure building can continue by setting the post-loop block
  // (end of loop SESE region) as the insertion point.
  builder.setInsertionPoint(postLoopBlock);
}

// Convert an "if" statement into a flow of basic blocks.
//
// Create an SESE region for the if statement (including its "then" and optional
// "else" statement blocks) and append it to the end of the current region.  The
// conditional region consists of a sequence of condition-checking blocks that
// implement the short-circuit scheme, followed by a "then" SESE region and an
// "else" SESE region, and the continuation block that post-dominates all blocks
// of the "if" statement.  The flow of blocks that correspond to the "then" and
// "else" clauses are constructed recursively, enabling easy nesting of "if"
// statements and if-then-else-if chains.
//
//      +--------------------------------+
//      | <end of current SESE region>   |
//      | <current insertion point>      |
//      | %zero = constant 0 : index     |
//      | %v = affine_apply #expr1(%ops) |
//      | %c = cmpi "sge" %v, %zero      |
//      | cond_br %c, %next, %else       |
//      +--------------------------------+
//             |              |
//             |              --------------|
//             v                            |
//      +--------------------------------+  |
//      | next:                          |  |
//      |   <repeat the check for expr2> |  |
//      |   cond_br %c, %next2, %else    |  |
//      +--------------------------------+  |
//             |              |             |
//            ...             --------------|
//             |   <Per-expression checks>  |
//             v                            |
//      +--------------------------------+  |
//      | last:                          |  |
//      |   <repeat the check for exprN> |  |
//      |   cond_br %c, %then, %else     |  |
//      +--------------------------------+  |
//             |              |             |
//             |              --------------|
//             v                            |
//      +--------------------------------+  |
//      | then:                          |  |
//      |   <then SESE region>           |  |
//      +--------------------------------+  |
//             |                            |
//            ... <SESE region of "then">   |
//             |                            |
//             v                            |
//      +--------------------------------+  |
//      | then_end:                      |  |
//      |   <then SESE region end>       |  |
//      |   br continue                  |  |
//      +--------------------------------+  |
//             |                            |
//   |----------               |-------------
//   |                         V
//   |  +--------------------------------+
//   |  | else:                          |
//   |  |   <else SESE region>           |
//   |  +--------------------------------+
//   |         |
//   |        ... <SESE region of "else">
//   |         |
//   |         v
//   |  +--------------------------------+
//   |  | else_end:                      |
//   |  |   <else SESE region>           |
//   |  |   br continue                  |
//   |  +--------------------------------+
//   |         |
//   ------|   |
//         v   v
//      +--------------------------------+
//      | continue:                      |
//      |   <end of "if" SESE region>    |
//      |   <new insertion point>        |
//      +--------------------------------+
//
void FunctionConverter::visitIfStmt(IfStmt *ifStmt) {
  assert(ifStmt != nullptr);

  auto integerSet = ifStmt->getCondition().getIntegerSet();

  // Create basic blocks for the 'then' block and for the 'else' block.
  // Although 'else' block may be empty in absence of an 'else' clause, create
  // it anyway for the sake of consistency and output IR readability.  Also
  // create extra blocks for condition checking to prepare for short-circuit
  // logic: conditions in the 'if' statement are conjunctive, so we can jump to
  // the false branch as soon as one condition fails.  `cond_br` requires
  // another block as a target when the condition is true, and that block will
  // contain the next condition.
  BasicBlock *ifInsertionBlock = builder.getInsertionBlock();
  SmallVector<BasicBlock *, 4> ifConditionExtraBlocks;
  unsigned numConstraints = integerSet.getNumConstraints();
  ifConditionExtraBlocks.reserve(numConstraints - 1);
  for (unsigned i = 0, e = numConstraints - 1; i < e; ++i) {
    ifConditionExtraBlocks.push_back(builder.createBlock());
  }
  BasicBlock *thenBlock = builder.createBlock();
  BasicBlock *elseBlock = builder.createBlock();
  builder.setInsertionPoint(ifInsertionBlock);

  // Implement short-circuit logic.  For each affine expression in the 'if'
  // condition, convert it into an affine map and call `affine_apply` to obtain
  // the resulting value.  Perform the equality or the greater-than-or-equality
  // test between this value and zero depending on the equality flag of the
  // condition.  If the test fails, jump immediately to the false branch, which
  // may be the else block if it is present or the continuation block otherwise.
  // If the test succeeds, jump to the next block testing testing the next
  // conjunct of the condition in the similar way.  When all conjuncts have been
  // handled, jump to the 'then' block instead.
  SSAValue *zeroConstant = getConstantIndexValue(0);
  ifConditionExtraBlocks.push_back(thenBlock);
  for (auto tuple :
       llvm::zip(integerSet.getConstraints(), integerSet.getEqFlags(),
                 ifConditionExtraBlocks)) {
    AffineExpr constraintExpr = std::get<0>(tuple);
    bool isEquality = std::get<1>(tuple);
    BasicBlock *nextBlock = std::get<2>(tuple);

    // Build and apply an affine map.
    auto affineMap =
        builder.getAffineMap(integerSet.getNumDims(),
                             integerSet.getNumSymbols(), constraintExpr, {});
    auto affineApplyOp = builder.create<AffineApplyOp>(
        ifStmt->getLoc(), affineMap, operandsAs(ifStmt, valueRemapping));
    SSAValue *affResult = affineApplyOp->getResult(0);

    // Compare the result of the apply and branch.
    auto comparisonOp = builder.create<CmpIOp>(
        ifStmt->getLoc(), isEquality ? CmpIPredicate::EQ : CmpIPredicate::SGE,
        affResult, zeroConstant);
    builder.create<CondBranchOp>(ifStmt->getLoc(), comparisonOp->getResult(),
                                 nextBlock, /*trueArgs*/ ArrayRef<SSAValue *>(),
                                 elseBlock,
                                 /*falseArgs*/ ArrayRef<SSAValue *>());
    builder.setInsertionPoint(nextBlock);
  }
  ifConditionExtraBlocks.pop_back();

  // Recursively traverse the 'then' block.
  builder.setInsertionPoint(thenBlock);
  visitStmtBlock(ifStmt->getThen());
  BasicBlock *lastThenBlock = builder.getInsertionBlock();

  // Recursively traverse the 'else' block if present.
  builder.setInsertionPoint(elseBlock);
  if (ifStmt->hasElse())
    visitStmtBlock(ifStmt->getElse());
  BasicBlock *lastElseBlock = builder.getInsertionBlock();

  // Create the continuation block here so that it appears lexically after the
  // 'then' and 'else' blocks, branch from end of 'then' and 'else' SESE regions
  // to the continuation block.
  BasicBlock *continuationBlock = builder.createBlock();
  builder.setInsertionPoint(lastThenBlock);
  builder.create<BranchOp>(ifStmt->getLoc(), continuationBlock);
  builder.setInsertionPoint(lastElseBlock);
  builder.create<BranchOp>(ifStmt->getLoc(), continuationBlock);

  // Make sure building can continue by setting up the continuation block as the
  // insertion point.
  builder.setInsertionPoint(continuationBlock);
}

// Entry point of the function convertor.
//
// Conversion is performed by recursively visiting statements of an MLFunction.
// It reasons in terms of single-entry single-exit (SESE) regions that are not
// materialized in the code.  Instead, the pointer to the last block of the
// region is maintained throughout the conversion as the insertion point of the
// IR builder since we never change the first block after its creation.  "Block"
// statements such as loops and branches create new SESE regions for their
// bodies, and surround them with additional basic blocks for the control flow.
// Individual operations are simply appended to the end of the last basic block
// of the current region.  The SESE invariant allows us to easily handle nested
// structures of arbitrary complexity.
//
// During the conversion, we maintain a mapping between the MLValues present in
// the original function and their CFGValue images in the function under
// construction.  When an MLValue is used, it gets replaced with the
// corresponding CFGValue that has been defined previously.  The value flow
// starts with function arguments converted to basic block arguments.
CFGFunction *FunctionConverter::convert(MLFunction *mlFunc) {
  auto outerBlock = builder.createBlock();

  // CFGFunctions do not have explicit arguments but use the arguments to the
  // first basic block instead.  Create those from the MLFunction arguments and
  // set up the value remapping.
  outerBlock->addArguments(mlFunc->getType().getInputs());
  assert(mlFunc->getNumArguments() == outerBlock->getNumArguments());
  for (unsigned i = 0, n = mlFunc->getNumArguments(); i < n; ++i) {
    const MLValue *mlArgument = mlFunc->getArgument(i);
    CFGValue *cfgArgument = outerBlock->getArgument(i);
    valueRemapping.insert(std::make_pair(mlArgument, cfgArgument));
  }

  // Convert statements in order.
  for (auto &stmt : *mlFunc->getBody()) {
    visit(&stmt);
  }

  return cfgFunc;
}

//===----------------------------------------------------------------------===//
// Module converter
//===----------------------------------------------------------------------===//

namespace {
// ModuleConverter class does CFG conversion for the whole module.
class ModuleConverter : public ModulePass {
public:
  explicit ModuleConverter() : ModulePass(&ModuleConverter::passID) {}

  PassResult runOnModule(Module *m) override;

  static char passID;

private:
  // Generates CFG functions for all ML functions in the module.
  void convertMLFunctions();
  // Generates CFG function for the given ML function.
  CFGFunction *convert(MLFunction *mlFunc);
  // Replaces all ML function references in the module
  // with references to the generated CFG functions.
  void replaceReferences();
  // Replaces function references in the given function.
  void replaceReferences(CFGFunction *cfgFunc);
  // Replaces MLFunctions with their CFG counterparts in the module.
  void replaceFunctions();

  // Map from ML functions to generated CFG functions.
  llvm::DenseMap<MLFunction *, CFGFunction *> generatedFuncs;
  Module *module = nullptr;
};
} // end anonymous namespace

char ModuleConverter::passID = 0;

// Iterates over all functions in the module generating CFG functions
// equivalent to ML functions and replacing references to ML functions
// with references to the generated ML functions.  The names of the converted
// functions match those of the original functions to avoid breaking any
// external references to the current module.  Therefore, converted functions
// are added to the module at the end of the pass, after removing the original
// functions to avoid name clashes.  Conversion procedure has access to the
// module as member of ModuleConverter and must not rely on the converted
// function to belong to the module.
PassResult ModuleConverter::runOnModule(Module *m) {
  module = m;
  convertMLFunctions();
  replaceReferences();
  replaceFunctions();

  return success();
}

void ModuleConverter::convertMLFunctions() {
  for (Function &fn : *module) {
    if (fn.isML())
      generatedFuncs[&fn] = convert(&fn);
  }
}

// Creates CFG function equivalent to the given ML function.
CFGFunction *ModuleConverter::convert(MLFunction *mlFunc) {
  // Use the same name as for ML function; do not add the converted function to
  // the module yet to avoid collision.
  auto name = mlFunc->getName().str();
  auto *cfgFunc = new Function(Function::Kind::CFGFunc, mlFunc->getLoc(), name,
                               mlFunc->getType(), mlFunc->getAttrs());

  // Generates the body of the CFG function.
  return FunctionConverter(cfgFunc).convert(mlFunc);
}

// Replace references to MLFunctions with the references to the converted
// CFGFunctions.  Since this all MLFunctions are converted at this point, it is
// unnecessary to replace references in the MLFunctions that are going to be
// removed anyway.  However, it is necessary to replace the references in the
// converted CFGFunctions that have not been added to the module yet.
void ModuleConverter::replaceReferences() {
  // Build the remapping between function attributes pointing to ML functions
  // and the newly created function attributes pointing to the converted CFG
  // functions.
  llvm::DenseMap<Attribute, FunctionAttr> remappingTable;
  for (const Function &fn : *module) {
    if (!fn.isML())
      continue;
    CFGFunction *convertedFunc = generatedFuncs.lookup(&fn);
    assert(convertedFunc && "ML function was not converted");

    MLIRContext *context = module->getContext();
    auto mlFuncAttr = FunctionAttr::get(&fn, context);
    auto cfgFuncAttr = FunctionAttr::get(convertedFunc, module->getContext());
    remappingTable.insert({mlFuncAttr, cfgFuncAttr});
  }

  // Remap in existing functions.
  remapFunctionAttrs(*module, remappingTable);

  // Remap in generated functions.
  for (auto pair : generatedFuncs) {
    remapFunctionAttrs(*pair.second, remappingTable);
  }
}

// Replace the value of a function attribute named "name" attached to the
// operation "op" and containing an MLFunction-typed value with the result of
// converting "func" to a CFGFunction.
static inline void replaceMLFunctionAttr(
    Operation &op, Identifier name, const Function *func,
    const llvm::DenseMap<MLFunction *, CFGFunction *> &generatedFuncs) {
  if (!func->isML())
    return;

  Builder b(op.getContext());
  auto *cfgFunc = generatedFuncs.lookup(func);
  op.setAttr(name, b.getFunctionAttr(cfgFunc));
}

// The CFG and ML functions have the same name.  First, erase the MLFunction.
// Then insert the CFGFunction at the same place.
void ModuleConverter::replaceFunctions() {
  for (auto pair : generatedFuncs) {
    auto &functions = module->getFunctions();
    auto it = functions.erase(pair.first);
    functions.insert(it, pair.second);
  }
}

//===----------------------------------------------------------------------===//
// Entry point method
//===----------------------------------------------------------------------===//

/// Replaces all ML functions in the module with equivalent CFG functions.
/// Function references are appropriately patched to refer to the newly
/// generated CFG functions.  Converted functions have the same names as the
/// original functions to preserve module linking.
ModulePass *mlir::createConvertToCFGPass() { return new ModuleConverter(); }

static PassRegistration<ModuleConverter>
    pass("convert-to-cfg",
         "Convert all ML functions in the module to CFG ones");
