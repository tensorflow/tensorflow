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
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir {

// Forward declarations.
class Block;
class FuncBuilder;
class MLIRContext;
class Operation;
class Type;
class Value;

// Private implementation class.
namespace impl {
class FunctionConverter;
}

/// Base class for the dialect op conversion patterns.  Specific conversions
/// must derive this class and implement least one `rewrite` method. Optionally
/// they can also override `PatternMatch match(Operation *)` to match more
/// specific operations than the `rootName` provided in the constructor.
/// NOTE: These conversion patterns can only be used with the DialectConversion
/// class.
class DialectOpConversion : public RewritePattern {
public:
  /// Construct an DialectOpConversion.  `rootName` must correspond to the
  /// canonical name of the first operation matched by the pattern.
  DialectOpConversion(StringRef rootName, PatternBenefit benefit,
                      MLIRContext *ctx)
      : RewritePattern(rootName, benefit, ctx) {}

  /// Hook for derived classes to implement matching. Dialect conversion
  /// generally unconditionally match the root operation, so default to success
  /// here.
  virtual PatternMatchResult match(Operation *op) const override {
    return matchSuccess();
  }

  /// Hook for derived classes to implement rewriting. `op` is the (first)
  /// operation matched by the pattern, `operands` is a list of rewritten values
  /// that are passed to this operation, `rewriter` can be used to emit the new
  /// operations. This function must be reimplemented if the DialectOpConversion
  /// ever needs to replace an operation that does not have successors. This
  /// function should not fail. If some specific cases of the operation are not
  /// supported, these cases should not be matched.
  virtual void rewrite(Operation *op, ArrayRef<Value *> operands,
                       PatternRewriter &rewriter) const {
    llvm_unreachable("unimplemented rewrite");
  }

  /// Hook for derived classes to implement rewriting. `op` is the (first)
  /// operation matched by the pattern, `properOperands` is a list of rewritten
  /// values that are passed to the operation itself, `destinations` is a list
  /// of (potentially rewritten) successor blocks, `operands` is a list of lists
  /// of rewritten values passed to each of the successors, co-indexed with
  /// `destinations`, `rewriter` can be used to emit the new operations. It must
  /// be reimplemented if the DialectOpConversion ever needs to replace a
  /// terminator operation that has successors. This function should not fail
  /// the pass. If some specific cases of the operation are not supported,
  /// these cases should not be matched.
  virtual void rewrite(Operation *op, ArrayRef<Value *> properOperands,
                       ArrayRef<Block *> destinations,
                       ArrayRef<ArrayRef<Value *>> operands,
                       PatternRewriter &rewriter) const {
    llvm_unreachable("unimplemented rewrite for terminators");
  }

  /// Rewrite the IR rooted at the specified operation with the result of
  /// this pattern, generating any new operations with the specified
  /// builder. If an unexpected error is encountered (an internal
  /// compiler error), it is emitted through the normal MLIR diagnostic
  /// hooks and the IR is left in a valid state.
  void rewrite(Operation *op, PatternRewriter &rewriter) const final;

private:
  using RewritePattern::matchAndRewrite;
  using RewritePattern::rewrite;
};

/// Base class for dialect conversion interface.  Specific converters must
/// derive this class and implement the pure virtual functions.
///
/// The module conversion proceeds as follows.
/// 1. Call `initConverters` to obtain a set of conversions to apply, given the
///    current MLIR context.
/// 2. For each function in the module do the following.
//    a. Create a new function with the same name and convert its signature
//       using `convertType`.
//    b. For each block in the function, create a block in the function with
//       its arguments converted using `convertType`.
//    c. Traverse blocks in DFS-preorder of successors starting from the entry
//       block (if any), and convert individual operations as follows.  Pattern
//       match against the list of conversions.  On the first match, call
//       `rewrite` for the operations, and advance to the next iteration.  If no
//       match is found, replicate the operation as is.
/// 3. Update all attributes of function type to point to the new functions.
/// 4. Replace old functions with new functions in the module.
/// If any error happened during the conversion, the pass fails as soon as
/// possible.
///
/// If the conversion fails, the module is not modified.
class DialectConversion {
  friend class impl::FunctionConverter;

public:
  virtual ~DialectConversion() = default;

  /// Run the converter on the provided module.
  LLVM_NODISCARD
  LogicalResult convert(Module *m);

protected:
  /// Derived classes must implement this hook to produce a set of conversion
  /// patterns to apply.  They may use `mlirContext` to obtain registered
  /// dialects or operations.  This will be called in the beginning of the
  /// conversion.
  virtual void initConverters(OwningRewritePatternList &patterns,
                              MLIRContext *mlirContext) = 0;

  /// Derived classes must reimplement this hook if they need to convert
  /// block or function argument types or function result types.  If the target
  /// dialect has support for custom first-class function types, convertType
  /// should create those types for arguments of MLIR function type.  It can be
  /// used for values (constant, operands, results) of function type but not for
  /// the function signatures.  For the latter, convertFunctionSignatureType is
  /// used instead.
  ///
  /// For block attribute types, this function will be called for each attribute
  /// individually.
  ///
  /// If type conversion can fail, this function should return a
  /// default-constructed Type.  The failure will be then propagated to trigger
  /// the pass failure.
  virtual Type convertType(Type t) { return t; }

  /// Derived classes must reimplement this hook if they need to change the
  /// function signature during conversion.  This function will be called on
  /// a function type corresponding to a function signature and must produce the
  /// converted MLIR function type.
  ///
  /// Note: even if some target dialects have first-class function types, they
  /// cannot be used at the top level of MLIR function signature.
  ///
  /// The default behavior of this function is to call convertType on individual
  /// function operands and results, and then create a new MLIR function type
  /// from those.
  ///
  /// Post-condition: if the returned optional attribute list does not have
  /// a value then no argument attribute conversion happened.
  virtual FunctionType convertFunctionSignatureType(
      FunctionType t, ArrayRef<NamedAttributeList> argAttrs,
      SmallVectorImpl<NamedAttributeList> &convertedArgAttrs);
};

} // end namespace mlir

#endif // MLIR_TRANSFORMS_DIALECTCONVERSION_H_
