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
#include "llvm/ADT/MapVector.h"

namespace mlir {

// Forward declarations.
class Block;
class FuncBuilder;
class MLIRContext;
class Operation;
class Type;
class Value;

/// Base class for the conversion patterns that require type changes. Specific
/// conversions must derive this class and implement least one `rewrite` method.
/// NOTE: These conversion patterns can only be used with the 'apply*' methods
/// below.
class ConversionPattern : public RewritePattern {
public:
  /// Construct an ConversionPattern.  `rootName` must correspond to the
  /// canonical name of the first operation matched by the pattern.
  ConversionPattern(StringRef rootName, PatternBenefit benefit,
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
  /// operations. This function must be reimplemented if the
  /// ConversionPattern ever needs to replace an operation that does not
  /// have successors. This function should not fail. If some specific cases of
  /// the operation are not supported, these cases should not be matched.
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
  /// be reimplemented if the ConversionPattern ever needs to replace a
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
  /// this pattern. If an unexpected error is encountered (an internal compiler
  /// error), it is emitted through the normal MLIR diagnostic hooks and the IR
  /// is left in a valid state.
  void rewrite(Operation *op, PatternRewriter &rewriter) const final;

private:
  using RewritePattern::matchAndRewrite;
  using RewritePattern::rewrite;
};

/// Base class for type conversion interface. Specific converters must
/// derive this class and implement the pure virtual functions.
class TypeConverter {
public:
  virtual ~TypeConverter() = default;

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

/// This class describes a specific conversion target.
class ConversionTarget {
public:
  /// This enumeration corresponds to the specific action to take when
  /// considering an operation legal for this conversion target.
  enum class LegalizationAction {
    /// The target supports this operation.
    Legal,

    /// This operation has dynamic legalization constraints that must be checked
    /// by the target.
    Dynamic
  };

  /// The type used to store operation legality information.
  using LegalityMapTy = llvm::MapVector<OperationName, LegalizationAction>;

  ConversionTarget(MLIRContext &ctx) : ctx(ctx) {}
  virtual ~ConversionTarget() = default;

  /// Runs a custom legalization query for the given operation. This should
  /// return true if the given operation is legal, otherwise false.
  virtual bool isLegal(Operation *op) const {
    llvm_unreachable(
        "targets with custom legalization must override 'isLegal'");
  }

  /// Register a legality action for the given operation.
  void setOpAction(OperationName op, LegalizationAction action) {
    legalOperations[op] = action;
  }
  template <typename OpT> void setOpAction(LegalizationAction action) {
    setOpAction(OperationName(OpT::getOperationName(), &ctx), action);
  }

  /// Register the given operations as legal.
  template <typename OpT> void addLegalOp() {
    setOpAction<OpT>(LegalizationAction::Legal);
  }
  template <typename OpT, typename OpT2, typename... OpTs> void addLegalOp() {
    addLegalOp<OpT>();
    addLegalOp<OpT2, OpTs...>();
  }

  /// Register the operations of the given dialects as legal.
  void addLegalDialects(ArrayRef<StringRef> dialectNames);
  template <typename... Names>
  void addLegalDialects(StringRef name, Names... names) {
    SmallVector<StringRef, 2> dialectNames({name, names...});
    addLegalDialects(dialectNames);
  }
  template <typename... Args> void addLegalDialects() {
    SmallVector<StringRef, 2> dialectNames({Args::getDialectNamespace()...});
    addLegalDialects(dialectNames);
  }

  /// Register the given operation as dynamically legal, i.e. requiring custom
  /// handling by the target via 'isLegal'.
  template <typename OpT> void addDynamicallyLegalOp() {
    setOpAction<OpT>(LegalizationAction::Dynamic);
  }
  template <typename OpT, typename OpT2, typename... OpTs>
  void addDynamicallyLegalOp() {
    addDynamicallyLegalOp<OpT>();
    addDynamicallyLegalOp<OpT2, OpTs...>();
  }

  /// Get the legality action for the given operation.
  llvm::Optional<LegalizationAction> getOpAction(OperationName op) const {
    auto it = legalOperations.find(op);
    if (it != legalOperations.end())
      return it->second;
    return llvm::None;
  }

  /// Returns a range of operations that this target has defined to be legal in
  /// some capacity.
  llvm::iterator_range<LegalityMapTy::const_iterator> getLegalOps() const {
    return llvm::make_range(legalOperations.begin(), legalOperations.end());
  }

private:
  /// A deterministic mapping of operation name to the specific legality action
  /// to take.
  LegalityMapTy legalOperations;

  /// The current context this target applies to.
  MLIRContext &ctx;
};

/// Convert the given module with the provided conversion patterns and type
/// conversion object. If conversion fails for specific functions, those
/// functions remains unmodified.
LLVM_NODISCARD LogicalResult applyConversionPatterns(
    Module &module, ConversionTarget &target, TypeConverter &converter,
    OwningRewritePatternList &&patterns);

/// Convert the given function with the provided conversion patterns. This will
/// convert as many of the operations within 'fn' as possible given the set of
/// patterns.
LLVM_NODISCARD
LogicalResult applyConversionPatterns(Function &fn, ConversionTarget &target,
                                      OwningRewritePatternList &&patterns);

} // end namespace mlir

#endif // MLIR_TRANSFORMS_DIALECTCONVERSION_H_
