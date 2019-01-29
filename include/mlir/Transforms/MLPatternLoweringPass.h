//===- MLPatternLoweringPass.h - Generic ML lowering pass -------*- C++ -*-===//
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
// Defines a generic class to implement lowering passes on ML functions as a
// list of pattern rewriters.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TRANSFORMS_MLPATTERNLOWERINGPASS_H
#define MLIR_TRANSFORMS_MLPATTERNLOWERINGPASS_H

#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass.h"
#include <type_traits>

namespace mlir {

/// Specialization of the pattern rewriter to ML functions.
class MLFuncLoweringRewriter : public PatternRewriter {
public:
  explicit MLFuncLoweringRewriter(FuncBuilder *builder)
      : PatternRewriter(builder->getContext()), builder(builder) {}

  FuncBuilder *getBuilder() { return builder; }

  OperationInst *createOperation(const OperationState &state) override {
    auto *result = builder->createOperation(state);
    return result;
  }

private:
  FuncBuilder *builder;
};

/// Base class for the Function-wise lowering state.  A pointer to the same
/// instance of the subclass will be passed to all `rewrite` calls on operations
/// that belong to the same Function.
class MLFuncGlobalLoweringState {
public:
  virtual ~MLFuncGlobalLoweringState() {}

protected:
  // Must be subclassed.
  MLFuncGlobalLoweringState() {}
};

/// Base class for Function lowering patterns.
class MLLoweringPattern : public Pattern {
public:
  /// Subclasses must override this function to implement rewriting.  It will be
  /// called on all operations found by `match` (declared in Pattern, subclasses
  /// must override).  It will be passed the function-wise state, common to all
  /// matches, and the state returned by the `match` call, if any.  The subclass
  /// must use `rewriter` to modify the function.
  virtual void rewriteOpInst(OperationInst *op,
                             MLFuncGlobalLoweringState *funcWiseState,
                             std::unique_ptr<PatternState> opState,
                             MLFuncLoweringRewriter *rewriter) const = 0;

protected:
  // Must be subclassed.
  MLLoweringPattern(StringRef opName, int64_t benefit, MLIRContext *context)
      : Pattern(opName, benefit, context) {}
};

namespace detail {
/// Owning list of ML lowering patterns.
using OwningMLLoweringPatternList =
    std::vector<std::unique_ptr<mlir::MLLoweringPattern>>;
} // namespace detail

/// Generic lowering pass for ML functions.  The lowering details are defined as
/// a sequence of pattern matchers.  The following constraints on matchers
/// apply:
/// - only one (match root) operation can be removed;
/// - the code produced by rewriters is final, it is not pattern-matched;
/// - the matchers are applied in their order of appearance in the list;
/// - if the match is found, the operation is rewritten immediately and the
///   next _original_ operation is considered.
/// In other words, for each operation, the pass applies the first matching
/// rewriter in the list and advances to the (lexically) next operation.
/// Non-operation instructions (ForInst) are ignored.
/// This is similar to greedy worklist-based pattern rewriter, except that this
/// operates on ML functions using an ML builder and does not maintain the work
/// list.  Note that, as of the time of writing, worklist-based rewriter did not
/// support removing multiple operations either.
template <typename... Patterns>
class MLPatternLoweringPass : public FunctionPass {
public:
  explicit MLPatternLoweringPass(void *ID) : FunctionPass(ID) {}

  virtual std::unique_ptr<MLFuncGlobalLoweringState>
  makeFuncWiseState(Function *f) const {
    return nullptr;
  }

  PassResult runOnFunction(Function *f) override;
};

/////////////////////////////////////////////////////////////////////
// MLPatternLoweringPass template implementations
/////////////////////////////////////////////////////////////////////

namespace detail {
template <typename Pattern, typename... Patterns> struct ListAdder {
  static void addPatternsToList(OwningMLLoweringPatternList *list,
                                MLIRContext *context) {
    static_assert(std::is_base_of<MLLoweringPattern, Pattern>::value,
                  "can only add subclasses of MLLoweringPattern");
    list->emplace_back(new Pattern(context));
    ListAdder<Patterns...>::addPatternsToList(list, context);
  }
};

template <typename Pattern> struct ListAdder<Pattern> {
  static void addPatternsToList(OwningMLLoweringPatternList *list,
                                MLIRContext *context) {
    list->emplace_back(new Pattern(context));
  }
};
} // namespace detail

template <typename... Patterns>
PassResult MLPatternLoweringPass<Patterns...>::runOnFunction(Function *f) {
  detail::OwningMLLoweringPatternList patterns;
  detail::ListAdder<Patterns...>::addPatternsToList(&patterns, f->getContext());
  auto funcWiseState = makeFuncWiseState(f);

  FuncBuilder builder(f);
  MLFuncLoweringRewriter rewriter(&builder);

  llvm::SmallVector<OperationInst *, 16> ops;
  f->walkOps([&ops](OperationInst *inst) { ops.push_back(inst); });

  for (OperationInst *inst : ops) {
    for (const auto &pattern : patterns) {
      rewriter.getBuilder()->setInsertionPoint(inst);
      auto matchResult = pattern->match(inst);
      if (matchResult) {
        pattern->rewriteOpInst(inst, funcWiseState.get(),
                               std::move(*matchResult), &rewriter);
        break;
      }
    }
  }

  return PassResult::Success;
}

} // end namespace mlir

#endif // MLIR_TRANSFORMS_MLPATTERNLOWERINGPASS_H
