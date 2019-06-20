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

  /// Hook for derived classes to implement combined matching and rewriting.
  virtual PatternMatchResult
  matchAndRewrite(Operation *op, ArrayRef<Value *> properOperands,
                  ArrayRef<Block *> destinations,
                  ArrayRef<ArrayRef<Value *>> operands,
                  PatternRewriter &rewriter) const {
    if (!match(op))
      return matchFailure();
    rewrite(op, properOperands, destinations, operands, rewriter);
    return matchSuccess();
  }

  /// Hook for derived classes to implement combined matching and rewriting.
  virtual PatternMatchResult matchAndRewrite(Operation *op,
                                             ArrayRef<Value *> operands,
                                             PatternRewriter &rewriter) const {
    if (!match(op))
      return matchFailure();
    rewrite(op, operands, rewriter);
    return matchSuccess();
  }

  /// Attempt to match and rewrite the IR root at the specified operation.
  PatternMatchResult matchAndRewrite(Operation *op,
                                     PatternRewriter &rewriter) const final;

private:
  using RewritePattern::rewrite;
};

/// Base class for type conversion interface. Specific converters must
/// derive this class and implement the pure virtual functions.
class TypeConverter {
public:
  virtual ~TypeConverter() = default;

  /// This class provides all of the information necessary to convert a
  /// FunctionType signature.
  class SignatureConversion {
  public:
    SignatureConversion(unsigned numOrigInputs)
        : remappedInputs(numOrigInputs) {}

    /// This struct represents a range of new types that remap an existing
    /// signature input.
    struct InputMapping {
      size_t inputNo, size;
    };

    /// Return the converted type signature.
    FunctionType getConvertedType(MLIRContext *ctx) const {
      return FunctionType::get(argTypes, resultTypes, ctx);
    }

    /// Return the argument types for the new signature.
    ArrayRef<Type> getConvertedArgTypes() const { return argTypes; }

    /// Return the result types for the new signature.
    ArrayRef<Type> getConvertedResultTypes() const { return resultTypes; }

    /// Returns the attributes for the arguments of the new signature.
    ArrayRef<NamedAttributeList> getConvertedArgAttrs() const {
      return argAttrs;
    }

    /// Get the input mapping for the given argument.
    llvm::Optional<InputMapping> getInputMapping(unsigned input) const {
      return remappedInputs[input];
    }

    //===------------------------------------------------------------------===//
    // Conversion Hooks
    //===------------------------------------------------------------------===//

    /// Append new result types to the signature conversion.
    void addResults(ArrayRef<Type> results);

    /// Remap an input of the original signature with a new set of types. The
    /// new types are appended to the new signature conversion.
    void addInputs(unsigned origInputNo, ArrayRef<Type> types,
                   ArrayRef<NamedAttributeList> attrs = llvm::None);

    /// Append new input types to the signature conversion, this should only be
    /// used if the new types are not intended to remap an existing input.
    void addInputs(ArrayRef<Type> types,
                   ArrayRef<NamedAttributeList> attrs = llvm::None);

    /// Remap an input of the original signature with a range of types in the
    /// new signature.
    void remapInput(unsigned origInputNo, unsigned newInputNo,
                    unsigned newInputCount = 1);

  private:
    /// The remapping information for each of the original arguments.
    SmallVector<llvm::Optional<InputMapping>, 4> remappedInputs;

    /// The set of argument and results types.
    SmallVector<Type, 4> argTypes, resultTypes;

    /// The set of attributes for each new argument type.
    SmallVector<NamedAttributeList, 4> argAttrs;
  };

  /// This hooks allows for converting a type. This function should return
  /// failure if no valid conversion exists, success otherwise. If the new set
  /// of types is empty, the type is removed and any usages of the existing
  /// value are expected to be removed during conversion.
  virtual LogicalResult convertType(Type t, SmallVectorImpl<Type> &results);

  /// This hook simplifies defining 1-1 type conversions. This function returns
  /// the type convert to on success, and a null type on failure.
  virtual Type convertType(Type t) { return t; }

  /// Convert the given FunctionType signature. This functions returns a valid
  /// SignatureConversion on success, None otherwise.
  llvm::Optional<SignatureConversion>
  convertSignature(FunctionType type, ArrayRef<NamedAttributeList> argAttrs);
  llvm::Optional<SignatureConversion> convertSignature(FunctionType type) {
    SmallVector<NamedAttributeList, 4> argAttrs(type.getNumInputs());
    return convertSignature(type, argAttrs);
  }

  /// This hook allows for changing a FunctionType signature. This function
  /// should populate 'result' with the new arguments and result on success,
  /// otherwise return failure.
  ///
  /// The default behavior of this function is to call 'convertType' on
  /// individual function operands and results. Any argument attributes are
  /// dropped if the resultant conversion is not a 1->1 mapping.
  virtual LogicalResult convertSignature(FunctionType type,
                                         ArrayRef<NamedAttributeList> argAttrs,
                                         SignatureConversion &result);

  /// This hook allows for converting a specific argument of a signature. It
  /// takes as inputs the original argument input number, type, and attributes.
  /// On success, this function should populate 'result' with any new mappings.
  virtual LogicalResult convertSignatureArg(unsigned inputNo, Type type,
                                            NamedAttributeList attrs,
                                            SignatureConversion &result);
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
  virtual bool isDynamicallyLegal(Operation *op) const {
    llvm_unreachable(
        "targets with custom legalization must override 'isDynamicallyLegal'");
  }

  //===--------------------------------------------------------------------===//
  // Legality Registration
  //===--------------------------------------------------------------------===//

  /// Register a legality action for the given operation.
  void setOpAction(OperationName op, LegalizationAction action);
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

  /// Register the given operation as dynamically legal, i.e. requiring custom
  /// handling by the target via 'isDynamicallyLegal'.
  template <typename OpT> void addDynamicallyLegalOp() {
    setOpAction<OpT>(LegalizationAction::Dynamic);
  }
  template <typename OpT, typename OpT2, typename... OpTs>
  void addDynamicallyLegalOp() {
    addDynamicallyLegalOp<OpT>();
    addDynamicallyLegalOp<OpT2, OpTs...>();
  }

  /// Register a legality action for the given dialects.
  void setDialectAction(ArrayRef<StringRef> dialectNames,
                        LegalizationAction action);

  /// Register the operations of the given dialects as legal.
  template <typename... Names>
  void addLegalDialect(StringRef name, Names... names) {
    SmallVector<StringRef, 2> dialectNames({name, names...});
    setDialectAction(dialectNames, LegalizationAction::Legal);
  }
  template <typename... Args> void addLegalDialect() {
    SmallVector<StringRef, 2> dialectNames({Args::getDialectNamespace()...});
    setDialectAction(dialectNames, LegalizationAction::Legal);
  }

  /// Register the operations of the given dialects as dynamically legal, i.e.
  /// requiring custom handling by the target via 'isDynamicallyLegal'.
  template <typename... Names>
  void addDynamicallyLegalDialect(StringRef name, Names... names) {
    SmallVector<StringRef, 2> dialectNames({name, names...});
    setDialectAction(dialectNames, LegalizationAction::Dynamic);
  }
  template <typename... Args> void addDynamicallyLegalDialect() {
    SmallVector<StringRef, 2> dialectNames({Args::getDialectNamespace()...});
    setDialectAction(dialectNames, LegalizationAction::Dynamic);
  }

  //===--------------------------------------------------------------------===//
  // Legality Querying
  //===--------------------------------------------------------------------===//

  /// Get the legality action for the given operation.
  llvm::Optional<LegalizationAction> getOpAction(OperationName op) const;

private:
  /// A deterministic mapping of operation name to the specific legality action
  /// to take.
  LegalityMapTy legalOperations;

  /// A deterministic mapping of dialect name to the specific legality action to
  /// take.
  llvm::StringMap<LegalizationAction> legalDialects;

  /// The current context this target applies to.
  MLIRContext &ctx;
};

/// Convert the given module with the provided conversion patterns and type
/// conversion object. This function returns failure if a type conversion
/// failed, potentially leaving the IR in an invalid state.
LLVM_NODISCARD LogicalResult applyConversionPatterns(
    Module &module, ConversionTarget &target, TypeConverter &converter,
    OwningRewritePatternList &&patterns);

/// Convert the given functions with the provided conversion patterns. This
/// function returns failure if a type conversion failed, potentially leaving
/// the IR in an invalid state.
LLVM_NODISCARD
LogicalResult applyConversionPatterns(ArrayRef<Function *> fns,
                                      ConversionTarget &target,
                                      TypeConverter &converter,
                                      OwningRewritePatternList &&patterns);

/// Convert the given function with the provided conversion patterns. This will
/// convert as many of the operations within 'fn' as possible given the set of
/// patterns.
LLVM_NODISCARD
LogicalResult applyConversionPatterns(Function &fn, ConversionTarget &target,
                                      OwningRewritePatternList &&patterns);

} // end namespace mlir

#endif // MLIR_TRANSFORMS_DIALECTCONVERSION_H_
