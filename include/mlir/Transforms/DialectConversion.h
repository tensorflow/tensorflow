//===- DialectConversion.h - MLIR dialect conversion pass -------*- C++ -*-===//
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
// This file declares a generic pass for converting between MLIR dialects.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TRANSFORMS_DIALECTCONVERSION_H_
#define MLIR_TRANSFORMS_DIALECTCONVERSION_H_

#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass.h"
#include "mlir/Support/LLVM.h"

namespace mlir {

// Forward declarations.
class Block;
class FuncBuilder;
class Instruction;
class MLIRContext;
class Type;
class Value;

// Private implementation class.
namespace impl {
class FunctionConversion;
}

/// Base class for the dialect op conversion patterns.  Specific conversions
/// must derive this class and implement `PatternMatch match(Instruction *)`
/// defined in `Pattern` and at least one of `rewrite` and `rewriteTerminator`.
//
// TODO(zinenko): this should eventually converge with RewritePattern.  So far,
// rewritePattern is missing support for operations with successors as well as
// an ability to accept new operands instead of reusing those of the existing
// operation.
class DialectOpConversion : public Pattern {
public:
  /// Construct an DialectOpConversion.  `rootName` must correspond to the
  /// canonical name of the first operation matched by the pattern.
  DialectOpConversion(StringRef rootName, PatternBenefit benefit,
                      MLIRContext *ctx)
      : Pattern(rootName, benefit, ctx) {}

  /// Hook for derived classes to implement rewriting.  `op` is the (first)
  /// operation matched by the pattern, `operands` is a list of rewritten values
  /// that are passed to this operation, `rewriter` can be used to emit the new
  /// operations.  This function returns the values produced by the newly
  /// created operation(s).  These values will be used instead of those produced
  /// by the original operation.  This function must be reimplemented if the
  /// DialectOpConversion ever needs to replace an operation that does not have
  /// successors.  This function should not fail.  If some specific cases of the
  /// operation are not supported, these cases should not be matched.
  virtual SmallVector<Value *, 4> rewrite(Instruction *op,
                                          ArrayRef<Value *> operands,
                                          FuncBuilder &rewriter) const {
    llvm_unreachable("unimplemented rewrite, did you mean rewriteTerminator?");
  };

  /// Hook for derived classes to implement rewriting.  `op` is the (first)
  /// operation matched by the pattern, `properOperands` is a list of rewritten
  /// values that are passed to the operation itself, `destinations` is a list
  /// of (potentially rewritten) successor blocks, `operands` is a list of lists
  /// of rewritten values passed to each of the successors, co-indexed with
  /// `destinations`, `rewriter` can be used to emit the new operations.  Since
  /// terminators never produce results (which could not be used anyway), this
  /// function does not return anything.  It must be reimplemented if the
  /// DialectOpConversion ever needs to replace a terminator operation that has
  /// successors.  This function should not fail the pass.  If some specific
  /// cases of the operation are not supported, these cases should not be
  /// matched.
  virtual void rewriteTerminator(Instruction *op,
                                 ArrayRef<Value *> properOperands,
                                 ArrayRef<Block *> destinations,
                                 ArrayRef<ArrayRef<Value *>> operands,
                                 FuncBuilder &rewriter) const {
    llvm_unreachable("unimplemented rewriteTerminator, did you mean rewrite?");
  }
};

// Helper class to create a list of dialect conversion patterns given a list of
// their types and a list of attributes perfect-forwarded to each of the
// conversion constructors.
template <typename Arg, typename... Args> struct ConversionListBuilder {
  template <typename... ConstructorArgs>
  static llvm::DenseSet<DialectOpConversion *>
  build(llvm::BumpPtrAllocator *allocator,
        ConstructorArgs &&... constructorArgs) {
    auto sub = ConversionListBuilder<Args...>::build(
        allocator, std::forward<ConstructorArgs>(constructorArgs)...);
    auto *ptr = allocator->Allocate<Arg>();
    new (ptr) Arg(std::forward<ConstructorArgs>(constructorArgs)...);
    sub.insert(ptr);
    return sub;
  }
};

// Template specialization to stop recursion.
template <typename Arg> struct ConversionListBuilder<Arg> {
  template <typename... ConstructorArgs>
  static llvm::DenseSet<DialectOpConversion *>
  build(llvm::BumpPtrAllocator *allocator,
        ConstructorArgs &&... constructorArgs) {
    auto *ptr = allocator->Allocate<Arg>();
    new (ptr) Arg(std::forward<ConstructorArgs>(constructorArgs)...);
    return {ptr};
  }
};

/// Base class for dialect conversion passes.  Specific passes must derive this
/// class and implement the pure virtual functions.
///
/// The module pass proceeds as follows.
/// 1. Call `initConverters` to obtain a set of conversions to apply, given the
/// current MLIR context.
/// 2. For each function in the module do the following.
//    a. Create a new function with the same name and convert its signature
//    using `convertType`.
//    b. For each block in the function, create a block in the function with
//    its arguments converted using `convertType`.
//    c. Traverse blocks in DFS-preorder of successors starting from the entry
//    block (if any), and convert individual operations as follows.  Pattern
//    match against the list of conversions.  On the first match, call
//    `rewriteTerminator` for terminator operations with successors and
//    `rewrite` for other operations, and advance to the next iteration.  If no
//    match is found, replicate the operation as is.  Note that if two patterns
//    match the same operation, it is undefined which of them will be applied.
/// 3. Update all attributes of function type to point to the new functions.
/// 4. Replace old functions with new functions in the module.
/// If any error happend during the conversion, the pass fails as soon as
/// possible.
///
/// If the pass fails, the module is not modified.
class DialectConversion : public ModulePass {
  friend class impl::FunctionConversion;

public:
  /// Construct a pass given its unique identifier.
  DialectConversion(const void *passID) : ModulePass(passID) {}

  /// Run the pass on the module.
  PassResult runOnModule(Module *m) override;

protected:
  /// Derived classes must implement this hook to produce a set of conversion
  /// patterns to apply.  They may use `mlirContext` to obtain registered
  /// dialects or operations.  This will be called in the beginning of the pass.
  virtual llvm::DenseSet<DialectOpConversion *>
  initConverters(MLIRContext *mlirContext) = 0;

  /// Derived classes must reimplement this hook if they need to convert
  /// block or function argument types or function result types.
  ///
  /// For functions types, this function will be passed a function type and the
  /// result must be another function type with arguments and results converted.
  /// Note: even if some target dialects have first-class function types, they
  /// cannot be used at the top level of MLIR function signature.
  ///
  /// For block attribute types, this function will be called for each attribute
  /// individually.
  ///
  /// If type conversion can fail, this function should return a
  /// default-constructed Type.  The failure will be then propagated to trigger
  /// the pass failure.
  virtual Type convertType(Type t) { return t; }
};

} // end namespace mlir

#endif // MLIR_TRANSFORMS_DIALECTCONVERSION_H_
