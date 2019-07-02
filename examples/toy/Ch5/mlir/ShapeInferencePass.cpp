//===- ShapeInferencePass.cpp - Toy Shape Inference / Func Specialization -===//
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
// This file implements a Module level pass performing interprocedural
// propagation of array shapes through function specialization.
//
//===----------------------------------------------------------------------===//

#include "toy/Dialect.h"

#include "mlir/Analysis/Verifier.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/StandardOps/Ops.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>

#define DEBUG_TYPE "toy-shape-inference"

using namespace toy;
using llvm::MutableArrayRef;
using llvm::SmallVector;
using llvm::SmallVectorImpl;
using llvm::StringRef;
using llvm::Twine;

/// Create mangled name for function specialization. We will simply append the
/// shape of the arguments to the function name. For example calling
///
///   "toy.generic_call"(%1, %3) {callee: "foo"}
///       : (!toy<"array<2, 3>">, !toy<"array<2, 3>">) -> !toy<"array">
///
/// would be mangled foo_2x3_2x3. This mangling isn't robust as the user could
/// have provide a function with a similar name. But we will claim this as a
/// feature: this allow the user to provide custom specialization!
static std::string mangle(StringRef funcName,
                          MutableArrayRef<mlir::OpOperand> operands) {
  std::string mangledName;
  mangledName.reserve(funcName.size() + operands.size() * 6);
  mangledName = funcName;
  for (auto &operand : operands) {
    auto arrayTy = operand.get()->getType().cast<ToyArrayType>();
    mangledName += "_";
    const char *sep = "";
    for (auto dim : arrayTy.getShape()) {
      mangledName += (sep + Twine(dim)).str();
      sep = "x";
    }
  }
  return mangledName;
}

namespace {

/// The ShapeInferencePass is a ModulePass: it will run on the Module as a
/// whole. MLIR also supports FunctionPass which are restricted to modify a
/// single function at a time. This pass couldn't be a function pass due the
/// nature of its interprocedural transformations.
///
/// The algorithm has two levels, first intra-procedurally:
///
///   1) Build a worklist containing all the operations that are returning
///      a generic Toy array: these are the operations that need shape
///      inference.
///   2) Iterate on the worklist:
///     a) find an operation to process: the next ready operation in the
///        worklist has all of its arguments non-generic,
///     b) if no operation is found, break out of the loop,
///     c) remove the operation from the worklist,
///     d) infer the shape of its output from the arguments type.
///   3) If the worklist is empty, the algorithm succeeded and we infer the
///      return type for the function from the return operation.
///
/// There is a twist though: when a call to a generic function is encountered,
/// shape inference requires the return type of the callee to be inferred first.
/// At this point we need to run specialize the callee by cloning it. Here is
/// the inter-procedural flow:
///
///   1) Keep a worklist of function to process. Start with function "main".
///   2) While the worklist isn't empty:
///     a) Take the last inserted function in the worklist.
///     b) Run the intra-procedural shape inference on this function.
///     c) If the intra-procedural shape inference can't complete, it returns
///        a Function that needs to be inferred first. In this case, queue this
///        new function and continue. Otherwise the inference succeeded and we
///        can pop from the queue.
///
class ShapeInferencePass : public mlir::ModulePass<ShapeInferencePass> {
public:
  // One entry in the inter-procedural worklist. It keeps track of the
  // function to process, the mangled name for this specialization, and the
  // types of the arguments on which to specialize.
  struct FunctionToSpecialize {
    mlir::Function function;
    std::string mangledName;
    SmallVector<mlir::Type, 4> argumentsType;
  };

  void runOnModule() override {
    auto module = getModule();
    mlir::ModuleManager moduleManager(module);
    auto main = moduleManager.getNamedFunction("main");
    if (!main) {
      emitError(mlir::UnknownLoc::get(module.getContext()),
                "Shape inference failed: can't find a main function\n");
      signalPassFailure();
      return;
    }

    /// Inter-procedural loop, initialize with `main` and iterate till
    /// successfully infer the full reachable call-graph from main.
    SmallVector<FunctionToSpecialize, 8> worklist;
    worklist.push_back({main, "", {}});
    while (!worklist.empty()) {
      if (failed(specialize(worklist, moduleManager)))
        return;
    }

    // Delete any generic function left
    // FIXME: we may want this as a separate pass.
    for (mlir::Function function : llvm::make_early_inc_range(module)) {
      if (auto genericAttr =
              function.getAttrOfType<mlir::BoolAttr>("toy.generic")) {
        if (genericAttr.getValue())
          function.erase();
      }
    }
  }

  /// Run inference on a function. If a mangledName is provided, we need to
  /// specialize the function: to this end clone it first.
  mlir::LogicalResult
  specialize(SmallVectorImpl<FunctionToSpecialize> &funcWorklist,
             mlir::ModuleManager &moduleManager) {
    FunctionToSpecialize &functionToSpecialize = funcWorklist.back();
    mlir::Function f = functionToSpecialize.function;

    // Check if cloning for specialization is needed (usually anything but main)
    // We will create a new function with the concrete types for the parameters
    // and clone the body into it.
    if (!functionToSpecialize.mangledName.empty()) {
      if (moduleManager.getNamedFunction(functionToSpecialize.mangledName)) {
        funcWorklist.pop_back();
        // Function already specialized, move on.
        return mlir::success();
      }
      // Create a new function with a generic array return type, it will be
      // updated when the inference for the function body completes.
      auto type = mlir::FunctionType::get(functionToSpecialize.argumentsType,
                                          {ToyArrayType::get(&getContext())},
                                          &getContext());
      auto newFunction = mlir::Function::create(
          f.getLoc(), functionToSpecialize.mangledName, type, f.getAttrs());
      moduleManager.insert(newFunction);

      // Clone the function body
      mlir::BlockAndValueMapping mapper;
      f.cloneInto(newFunction, mapper);
      LLVM_DEBUG({
        llvm::dbgs() << "====== Cloned : \n";
        f.dump();
        llvm::dbgs() << "====== Into : \n";
        newFunction.dump();
      });
      f = newFunction;
      f.setAttr("toy.generic", mlir::BoolAttr::get(false, &getContext()));
      // Remap the entry-block arguments
      // FIXME: this seems like a bug in `cloneInto()` above?
      auto &entryBlock = f.getBlocks().front();
      int blockArgSize = entryBlock.getArguments().size();
      assert(blockArgSize == static_cast<int>(f.getType().getInputs().size()));
      entryBlock.addArguments(f.getType().getInputs());
      auto argList = entryBlock.getArguments();
      for (int argNum = 0; argNum < blockArgSize; ++argNum) {
        argList[0]->replaceAllUsesWith(argList[blockArgSize]);
        entryBlock.eraseArgument(0);
      }
      assert(succeeded(verify(f)));
    }
    LLVM_DEBUG(llvm::dbgs()
               << "Run shape inference on : '" << f.getName() << "'\n");

    auto *toyDialect = getContext().getRegisteredDialect("toy");
    if (!toyDialect) {
      signalPassFailure();
      return emitError(mlir::UnknownLoc::get(&getContext()),
                       "Toy dialect is not registered");
    }

    // Populate the worklist with the operations that need shape inference:
    // these are the Toy operations that return a generic array.
    llvm::SmallPtrSet<mlir::Operation *, 16> opWorklist;
    f.walk([&](mlir::Operation *op) {
      if (op->getDialect() == toyDialect) {
        if (op->getNumResults() == 1 &&
            op->getResult(0)->getType().cast<ToyArrayType>().isGeneric())
          opWorklist.insert(op);
      }
    });

    // Iterate on the operations in the worklist until all operations have been
    // inferred or no change happened (fix point).
    while (!opWorklist.empty()) {
      // Find the next operation ready for inference, that is an operation
      // with all operands already resolved (non-generic).
      auto nextop = llvm::find_if(opWorklist, [](mlir::Operation *op) {
        return llvm::all_of(op->getOperandTypes(), [](mlir::Type ty) {
          return !ty.cast<ToyArrayType>().isGeneric();
        });
      });
      if (nextop == opWorklist.end())
        break; // failure: no operations can be inferred.

      mlir::Operation *op = *nextop;
      opWorklist.erase(op);
      LLVM_DEBUG(llvm::dbgs() << "Inferring shape for: " << *op << "\n");

      // The add operation is trivial: propagate the input type as is.
      if (auto addOp = llvm::dyn_cast<AddOp>(op)) {
        op->getResult(0)->setType(op->getOperand(0)->getType());
        continue;
      }

      // Transpose is easy: just invert the dimensions.
      if (op->getName().getStringRef() == "toy.transpose") {
        SmallVector<int64_t, 2> dims;
        auto arrayTy = op->getOperand(0)->getType().cast<ToyArrayType>();
        dims.insert(dims.end(), arrayTy.getShape().begin(),
                    arrayTy.getShape().end());
        if (dims.size() == 2)
          std::swap(dims[0], dims[1]);
        op->getResult(0)->setType(ToyArrayType::get(&getContext(), dims));
        continue;
      }

      // Multiplication is a bit trickier, handle rank 1 as dot product and rank
      // 2 as matrix multiplications.
      // We need to be careful about rank mismatch here: the verifier could
      // catch it but shape inference earlier in the pass could generate an
      // invalid IR (from an invalid Toy input of course) and we wouldn't want
      // to crash here.
      if (auto mulOp = llvm::dyn_cast<MulOp>(op)) {
        auto lhs = mulOp.getLHS()->getType().cast<ToyArrayType>();
        auto rhs = mulOp.getRHS()->getType().cast<ToyArrayType>();
        auto lhsRank = lhs.getShape().size();
        auto rhsRank = rhs.getShape().size();
        if (lhsRank != rhsRank) {
          return op->emitError("Shape mismatch: LHS and RHS must have the same "
                               "rank for multiplication, got ")
                 << lhsRank << " vs  " << lhsRank;
        }
        SmallVector<int64_t, 2> dims;
        if (lhsRank == 1) {
          // dot product, result shape is <1>
          dims.push_back(1);
        } else {
          if (lhsRank != 2) {
            return op->emitError("Shape mismatch: expect rank 1 or 2 for mul "
                                 "operands, got ")
                   << lhsRank;
          }
          dims.push_back(lhs.getShape()[0]);
          dims.push_back(rhs.getShape()[1]);
        }
        op->getResult(0)->setType(ToyArrayType::get(&getContext(), dims));
        continue;
      }

      // Process calls: lookup the callee after mangling the name with the
      // argument shapes. If the callee does not exist, we stop the inference
      // for this function, queue the callee in the inter-procedural work list,
      // and return. The current function stays in the work list and will
      // restart after the callee is processed.
      if (auto callOp = llvm::dyn_cast<GenericCallOp>(op)) {
        auto calleeName = callOp.getCalleeName();
        auto callee = moduleManager.getNamedFunction(calleeName);
        if (!callee) {
          signalPassFailure();
          return f.emitError("Shape inference failed, call to unknown '")
                 << calleeName << "'";
        }
        auto mangledName = mangle(calleeName, op->getOpOperands());
        LLVM_DEBUG(llvm::dbgs() << "Found callee to infer: '" << calleeName
                                << "', mangled: '" << mangledName << "'\n");
        auto mangledCallee = moduleManager.getNamedFunction(mangledName);
        if (!mangledCallee) {
          // Can't find the target, this is where we queue the request for the
          // callee and stop the inference for the current function now.
          funcWorklist.push_back({callee, std::move(mangledName),
                                  llvm::to_vector<4>(op->getOperandTypes())});
          return mlir::success();
        }
        // Found a specialized callee! Let's turn this into a normal call
        // operation.
        SmallVector<mlir::Value *, 8> operands(op->getOperands());
        mlir::OpBuilder builder(f.getBody());
        builder.setInsertionPoint(op);
        auto newCall =
            builder.create<mlir::CallOp>(op->getLoc(), mangledCallee, operands);
        if (newCall.getNumResults()) {
          op->getResult(0)->replaceAllUsesWith(newCall.getResult(0));
          op->erase();
          continue;
        }
      }
    }

    // Done with inference on this function, removing it from the worklist.
    funcWorklist.pop_back();
    // Mark the function as non-generic now that inference has succeeded
    f.setAttr("toy.generic", mlir::BoolAttr::get(false, &getContext()));

    // If the operation worklist isn't empty, this indicates a failure.
    if (!opWorklist.empty()) {
      signalPassFailure();
      auto diag = f.emitError("Shape inference failed, ")
                  << opWorklist.size() << " operations couldn't be inferred\n";
      for (auto *ope : opWorklist)
        diag << " - " << *ope << "\n";
      return diag;
    }

    // Finally, update the return type of the function based on the argument to
    // the return operation.
    for (auto &block : f.getBlocks()) {
      auto ret = llvm::cast<ReturnOp>(block.getTerminator());
      if (!ret)
        continue;
      if (ret.getNumOperands() &&
          f.getType().getResult(0) == ret.getOperand()->getType())
        // type match, we're done
        break;
      SmallVector<mlir::Type, 1> retTy;
      if (ret.getNumOperands())
        retTy.push_back(ret.getOperand()->getType());
      std::vector<mlir::Type> argumentsType;
      for (auto arg : f.getArguments())
        argumentsType.push_back(arg->getType());
      auto newType =
          mlir::FunctionType::get(argumentsType, retTy, &getContext());
      f.setType(newType);
      assert(succeeded(verify(f)));
      break;
    }
    return mlir::success();
  }
};
} // end anonymous namespace

namespace toy {
mlir::Pass *createShapeInferencePass() { return new ShapeInferencePass(); }
} // namespace toy
