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
#include "llvm/ADT/StringMap.h"

namespace mlir {

// Forward declarations.
class Block;
class ConversionPatternRewriter;
class FuncOp;
class MLIRContext;
class Operation;
class Type;
class Value;

//===----------------------------------------------------------------------===//
// Type Conversion
//===----------------------------------------------------------------------===//

/// Base class for type conversion interface. Specific converters must
/// derive this class and implement the pure virtual functions.
class TypeConverter {
public:
  virtual ~TypeConverter() = default;

  /// This class provides all of the information necessary to convert a type
  /// signature.
  class SignatureConversion {
  public:
    SignatureConversion(unsigned numOrigInputs)
        : remappedInputs(numOrigInputs) {}

    /// This struct represents a range of new types or a single value that
    /// remaps an existing signature input.
    struct InputMapping {
      size_t inputNo, size;
      Value *replacementValue;
    };

    /// Return the argument types for the new signature.
    ArrayRef<Type> getConvertedTypes() const { return argTypes; }

    /// Get the input mapping for the given argument.
    Optional<InputMapping> getInputMapping(unsigned input) const {
      return remappedInputs[input];
    }

    //===------------------------------------------------------------------===//
    // Conversion Hooks
    //===------------------------------------------------------------------===//

    /// Remap an input of the original signature with a new set of types. The
    /// new types are appended to the new signature conversion.
    void addInputs(unsigned origInputNo, ArrayRef<Type> types);

    /// Append new input types to the signature conversion, this should only be
    /// used if the new types are not intended to remap an existing input.
    void addInputs(ArrayRef<Type> types);

    /// Remap an input of the original signature with a range of types in the
    /// new signature.
    void remapInput(unsigned origInputNo, unsigned newInputNo,
                    unsigned newInputCount = 1);

    /// Remap an input of the original signature to another `replacement`
    /// value. This drops the original argument.
    void remapInput(unsigned origInputNo, Value *replacement);

  private:
    /// The remapping information for each of the original arguments.
    SmallVector<llvm::Optional<InputMapping>, 4> remappedInputs;

    /// The set of new argument types.
    SmallVector<Type, 4> argTypes;
  };

  /// This hook allows for converting a type. This function should return
  /// failure if no valid conversion exists, success otherwise. If the new set
  /// of types is empty, the type is removed and any usages of the existing
  /// value are expected to be removed during conversion.
  virtual LogicalResult convertType(Type t, SmallVectorImpl<Type> &results);

  /// This hook simplifies defining 1-1 type conversions. This function returns
  /// the type to convert to on success, and a null type on failure.
  virtual Type convertType(Type t) { return t; }

  /// Convert the given set of types, filling 'results' as necessary. This
  /// returns failure if the conversion of any of the types fails, success
  /// otherwise.
  LogicalResult convertTypes(ArrayRef<Type> types,
                             SmallVectorImpl<Type> &results);

  /// Return true if the given type is legal for this type converter, i.e. the
  /// type converts to itself.
  bool isLegal(Type type);

  /// Return true if the inputs and outputs of the given function type are
  /// legal.
  bool isSignatureLegal(FunctionType funcType);

  /// This hook allows for converting a specific argument of a signature. It
  /// takes as inputs the original argument input number, type.
  /// On success, this function should populate 'result' with any new mappings.
  virtual LogicalResult convertSignatureArg(unsigned inputNo, Type type,
                                            SignatureConversion &result);

  /// This function converts the type signature of the given block, by invoking
  /// 'convertSignatureArg' for each argument. This function should return a
  /// valid conversion for the signature on success, None otherwise.
  llvm::Optional<SignatureConversion> convertBlockSignature(Block *block);

  /// This hook allows for materializing a conversion from a set of types into
  /// one result type by generating a cast operation of some kind. The generated
  /// operation should produce one result, of 'resultType', with the provided
  /// 'inputs' as operands. This hook must be overridden when a type conversion
  /// results in more than one type, or if a type conversion may persist after
  /// the conversion has finished.
  virtual Operation *materializeConversion(PatternRewriter &rewriter,
                                           Type resultType,
                                           ArrayRef<Value *> inputs,
                                           Location loc) {
    llvm_unreachable("expected 'materializeConversion' to be overridden");
  }
};

//===----------------------------------------------------------------------===//
// Conversion Patterns
//===----------------------------------------------------------------------===//

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
                       ConversionPatternRewriter &rewriter) const {
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
                       ConversionPatternRewriter &rewriter) const {
    llvm_unreachable("unimplemented rewrite for terminators");
  }

  /// Hook for derived classes to implement combined matching and rewriting.
  virtual PatternMatchResult
  matchAndRewrite(Operation *op, ArrayRef<Value *> properOperands,
                  ArrayRef<Block *> destinations,
                  ArrayRef<ArrayRef<Value *>> operands,
                  ConversionPatternRewriter &rewriter) const {
    if (!match(op))
      return matchFailure();
    rewrite(op, properOperands, destinations, operands, rewriter);
    return matchSuccess();
  }

  /// Hook for derived classes to implement combined matching and rewriting.
  virtual PatternMatchResult
  matchAndRewrite(Operation *op, ArrayRef<Value *> operands,
                  ConversionPatternRewriter &rewriter) const {
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

/// OpConversionPattern is a wrapper around ConversionPattern that allows for
/// matching and rewriting against an instance of a derived operation class as
/// opposed to a raw Operation.
template <typename SourceOp>
struct OpConversionPattern : public ConversionPattern {
  OpConversionPattern(MLIRContext *context, PatternBenefit benefit = 1)
      : ConversionPattern(SourceOp::getOperationName(), benefit, context) {}

  /// Wrappers around the ConversionPattern methods that pass the derived op
  /// type.
  void rewrite(Operation *op, ArrayRef<Value *> operands,
               ConversionPatternRewriter &rewriter) const final {
    rewrite(llvm::cast<SourceOp>(op), operands, rewriter);
  }
  void rewrite(Operation *op, ArrayRef<Value *> properOperands,
               ArrayRef<Block *> destinations,
               ArrayRef<ArrayRef<Value *>> operands,
               ConversionPatternRewriter &rewriter) const final {
    rewrite(llvm::cast<SourceOp>(op), properOperands, destinations, operands,
            rewriter);
  }
  PatternMatchResult
  matchAndRewrite(Operation *op, ArrayRef<Value *> properOperands,
                  ArrayRef<Block *> destinations,
                  ArrayRef<ArrayRef<Value *>> operands,
                  ConversionPatternRewriter &rewriter) const final {
    return matchAndRewrite(llvm::cast<SourceOp>(op), properOperands,
                           destinations, operands, rewriter);
  }
  PatternMatchResult
  matchAndRewrite(Operation *op, ArrayRef<Value *> operands,
                  ConversionPatternRewriter &rewriter) const final {
    return matchAndRewrite(llvm::cast<SourceOp>(op), operands, rewriter);
  }

  // TODO(b/142763075): Use OperandAdaptor when it supports access to unnamed
  // operands.

  /// Rewrite and Match methods that operate on the SourceOp type. These must be
  /// overridden by the derived pattern class.
  virtual void rewrite(SourceOp op, ArrayRef<Value *> operands,
                       ConversionPatternRewriter &rewriter) const {
    llvm_unreachable("must override matchAndRewrite or a rewrite method");
  }

  virtual void rewrite(SourceOp op, ArrayRef<Value *> properOperands,
                       ArrayRef<Block *> destinations,
                       ArrayRef<ArrayRef<Value *>> operands,
                       ConversionPatternRewriter &rewriter) const {
    llvm_unreachable("unimplemented rewrite for terminators");
  }

  virtual PatternMatchResult
  matchAndRewrite(SourceOp op, ArrayRef<Value *> properOperands,
                  ArrayRef<Block *> destinations,
                  ArrayRef<ArrayRef<Value *>> operands,
                  ConversionPatternRewriter &rewriter) const {
    if (!match(op))
      return matchFailure();
    rewrite(op, properOperands, destinations, operands, rewriter);
    return matchSuccess();
  }

  virtual PatternMatchResult
  matchAndRewrite(SourceOp op, ArrayRef<Value *> operands,
                  ConversionPatternRewriter &rewriter) const {
    if (!match(op))
      return matchFailure();
    rewrite(op, operands, rewriter);
    return matchSuccess();
  }

private:
  using ConversionPattern::matchAndRewrite;
};

/// Add a pattern to the given pattern list to convert the signature of a FuncOp
/// with the given type converter.
void populateFuncOpTypeConversionPattern(OwningRewritePatternList &patterns,
                                         MLIRContext *ctx,
                                         TypeConverter &converter);

//===----------------------------------------------------------------------===//
// Conversion PatternRewriter
//===----------------------------------------------------------------------===//

namespace detail {
struct ConversionPatternRewriterImpl;
} // end namespace detail

/// This class implements a pattern rewriter for use with ConversionPatterns. It
/// extends the base PatternRewriter and provides special conversion specific
/// hooks.
class ConversionPatternRewriter final : public PatternRewriter {
public:
  ConversionPatternRewriter(MLIRContext *ctx, TypeConverter *converter);
  ~ConversionPatternRewriter() override;

  /// Apply a signature conversion to the entry block of the given region. This
  /// replaces the entry block with a new block containing the updated
  /// signature. The new entry block to the region is returned for convenience.
  Block *
  applySignatureConversion(Region *region,
                           TypeConverter::SignatureConversion &conversion);

  /// Replace all the uses of the block argument `from` with value `to`.
  void replaceUsesOfBlockArgument(BlockArgument *from, Value *to);

  /// Clone the given operation without cloning its regions.
  Operation *cloneWithoutRegions(Operation *op);
  template <typename OpT> OpT cloneWithoutRegions(OpT op) {
    return cast<OpT>(cloneWithoutRegions(op.getOperation()));
  }

  /// Return the converted value that replaces 'key'. Return 'key' if there is
  /// no such a converted value.
  Value *getRemappedValue(Value *key);

  //===--------------------------------------------------------------------===//
  // PatternRewriter Hooks
  //===--------------------------------------------------------------------===//

  /// PatternRewriter hook for replacing the results of an operation.
  void replaceOp(Operation *op, ValueRange newValues,
                 ValueRange valuesToRemoveIfDead) override;
  using PatternRewriter::replaceOp;

  /// PatternRewriter hook for erasing a dead operation. The uses of this
  /// operation *must* be made dead by the end of the conversion process,
  /// otherwise an assert will be issued.
  void eraseOp(Operation *op) override;

  /// PatternRewriter hook for splitting a block into two parts.
  Block *splitBlock(Block *block, Block::iterator before) override;

  /// PatternRewriter hook for merging a block into another.
  void mergeBlocks(Block *source, Block *dest, ValueRange argValues) override;

  /// PatternRewriter hook for moving blocks out of a region.
  void inlineRegionBefore(Region &region, Region &parent,
                          Region::iterator before) override;
  using PatternRewriter::inlineRegionBefore;

  /// PatternRewriter hook for cloning blocks of one region into another. The
  /// given region to clone *must* not have been modified as part of conversion
  /// yet, i.e. it must be within an operation that is either in the process of
  /// conversion, or has not yet been converted.
  void cloneRegionBefore(Region &region, Region &parent,
                         Region::iterator before,
                         BlockAndValueMapping &mapping) override;
  using PatternRewriter::cloneRegionBefore;

  /// PatternRewriter hook for creating a new operation.
  Operation *createOperation(const OperationState &state) override;

  /// PatternRewriter hook for updating the root operation in-place.
  void notifyRootUpdated(Operation *op) override;

  /// Return a reference to the internal implementation.
  detail::ConversionPatternRewriterImpl &getImpl();

private:
  std::unique_ptr<detail::ConversionPatternRewriterImpl> impl;
};

//===----------------------------------------------------------------------===//
// ConversionTarget
//===----------------------------------------------------------------------===//

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
    Dynamic,

    /// The target explicitly does not support this operation.
    Illegal,
  };

  /// A structure containing additional information describing a specific legal
  /// operation instance.
  struct LegalOpDetails {
    /// A flag that indicates if this operation is 'recursively' legal. This
    /// means that if an operation is legal, either statically or dynamically,
    /// all of the operations nested within are also considered legal.
    bool isRecursivelyLegal = false;
  };

  /// The signature of the callback used to determine if an operation is
  /// dynamically legal on the target.
  using DynamicLegalityCallbackFn = std::function<bool(Operation *)>;

  ConversionTarget(MLIRContext &ctx) : ctx(ctx) {}
  virtual ~ConversionTarget() = default;

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

  /// Register the given operation as dynamically legal and set the dynamic
  /// legalization callback to the one provided.
  template <typename OpT>
  void addDynamicallyLegalOp(const DynamicLegalityCallbackFn &callback) {
    OperationName opName(OpT::getOperationName(), &ctx);
    setOpAction(opName, LegalizationAction::Dynamic);
    setLegalityCallback(opName, callback);
  }
  template <typename OpT, typename OpT2, typename... OpTs>
  void addDynamicallyLegalOp(const DynamicLegalityCallbackFn &callback) {
    addDynamicallyLegalOp<OpT>(callback);
    addDynamicallyLegalOp<OpT2, OpTs...>(callback);
  }
  template <typename OpT, class Callable>
  typename std::enable_if<!is_invocable<Callable, Operation *>::value>::type
  addDynamicallyLegalOp(Callable &&callback) {
    addDynamicallyLegalOp<OpT>(
        [=](Operation *op) { return callback(cast<OpT>(op)); });
  }

  /// Register the given operation as illegal, i.e. this operation is known to
  /// not be supported by this target.
  template <typename OpT> void addIllegalOp() {
    setOpAction<OpT>(LegalizationAction::Illegal);
  }
  template <typename OpT, typename OpT2, typename... OpTs> void addIllegalOp() {
    addIllegalOp<OpT>();
    addIllegalOp<OpT2, OpTs...>();
  }

  /// Mark an operation, that *must* have either been set as `Legal` or
  /// `DynamicallyLegal`, as being recursively legal. This means that in
  /// addition to the operation itself, all of the operations nested within are
  /// also considered legal. An optional dynamic legality callback may be
  /// provided to mark subsets of legal instances as recursively legal.
  template <typename OpT>
  void markOpRecursivelyLegal(const DynamicLegalityCallbackFn &callback = {}) {
    OperationName opName(OpT::getOperationName(), &ctx);
    markOpRecursivelyLegal(opName, callback);
  }
  template <typename OpT, typename OpT2, typename... OpTs>
  void markOpRecursivelyLegal(const DynamicLegalityCallbackFn &callback = {}) {
    markOpRecursivelyLegal<OpT>(callback);
    markOpRecursivelyLegal<OpT2, OpTs...>(callback);
  }
  template <typename OpT, class Callable>
  typename std::enable_if<!is_invocable<Callable, Operation *>::value>::type
  markOpRecursivelyLegal(Callable &&callback) {
    markOpRecursivelyLegal<OpT>(
        [=](Operation *op) { return callback(cast<OpT>(op)); });
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
  template <typename... Args>
  void addDynamicallyLegalDialect(
      Optional<DynamicLegalityCallbackFn> callback = llvm::None) {
    SmallVector<StringRef, 2> dialectNames({Args::getDialectNamespace()...});
    setDialectAction(dialectNames, LegalizationAction::Dynamic);
    if (callback)
      setLegalityCallback(dialectNames, *callback);
  }

  /// Register the operations of the given dialects as illegal, i.e.
  /// operations of this dialect are not supported by the target.
  template <typename... Names>
  void addIllegalDialect(StringRef name, Names... names) {
    SmallVector<StringRef, 2> dialectNames({name, names...});
    setDialectAction(dialectNames, LegalizationAction::Illegal);
  }
  template <typename... Args> void addIllegalDialect() {
    SmallVector<StringRef, 2> dialectNames({Args::getDialectNamespace()...});
    setDialectAction(dialectNames, LegalizationAction::Illegal);
  }

  //===--------------------------------------------------------------------===//
  // Legality Querying
  //===--------------------------------------------------------------------===//

  /// Get the legality action for the given operation.
  Optional<LegalizationAction> getOpAction(OperationName op) const;

  /// If the given operation instance is legal on this target, a structure
  /// containing legality information is returned. If the operation is not
  /// legal, None is returned.
  Optional<LegalOpDetails> isLegal(Operation *op) const;

protected:
  /// Runs a custom legalization query for the given operation. This should
  /// return true if the given operation is legal, otherwise false.
  virtual bool isDynamicallyLegal(Operation *op) const {
    llvm_unreachable(
        "targets with custom legalization must override 'isDynamicallyLegal'");
  }

private:
  /// Set the dynamic legality callback for the given operation.
  void setLegalityCallback(OperationName name,
                           const DynamicLegalityCallbackFn &callback);

  /// Set the dynamic legality callback for the given dialects.
  void setLegalityCallback(ArrayRef<StringRef> dialects,
                           const DynamicLegalityCallbackFn &callback);

  /// Set the recursive legality callback for the given operation and mark the
  /// operation as recursively legal.
  void markOpRecursivelyLegal(OperationName name,
                              const DynamicLegalityCallbackFn &callback);

  /// The set of information that configures the legalization of an operation.
  struct LegalizationInfo {
    /// The legality action this operation was given.
    LegalizationAction action;

    /// If some legal instances of this operation may also be recursively legal.
    bool isRecursivelyLegal;
  };

  /// Get the legalization information for the given operation.
  Optional<LegalizationInfo> getOpInfo(OperationName op) const;

  /// A deterministic mapping of operation name and its respective legality
  /// information.
  llvm::MapVector<OperationName, LegalizationInfo> legalOperations;

  /// A set of dynamic legality callbacks for given operation names.
  DenseMap<OperationName, DynamicLegalityCallbackFn> opLegalityFns;

  /// A set of legality callbacks for given operation names that are used to
  /// check if an operation instance is recursively legal.
  DenseMap<OperationName, DynamicLegalityCallbackFn> opRecursiveLegalityFns;

  /// A deterministic mapping of dialect name to the specific legality action to
  /// take.
  llvm::StringMap<LegalizationAction> legalDialects;

  /// A set of dynamic legality callbacks for given dialect names.
  llvm::StringMap<DynamicLegalityCallbackFn> dialectLegalityFns;

  /// The current context this target applies to.
  MLIRContext &ctx;
};

//===----------------------------------------------------------------------===//
// Op Conversion Entry Points
//===----------------------------------------------------------------------===//

/// Below we define several entry points for operation conversion. It is
/// important to note that the patterns provided to the conversion framework may
/// have additional constraints. See the `PatternRewriter Hooks` section of the
/// ConversionPatternRewriter, to see what additional constraints are imposed on
/// the use of the PatternRewriter.

/// Apply a partial conversion on the given operations, and all nested
/// operations. This method converts as many operations to the target as
/// possible, ignoring operations that failed to legalize. This method only
/// returns failure if there are unreachable blocks in any of the regions nested
/// within 'ops'. If 'converter' is provided, the signatures of blocks and
/// regions are also converted.
LLVM_NODISCARD LogicalResult
applyPartialConversion(ArrayRef<Operation *> ops, ConversionTarget &target,
                       const OwningRewritePatternList &patterns,
                       TypeConverter *converter = nullptr);
LLVM_NODISCARD LogicalResult
applyPartialConversion(Operation *op, ConversionTarget &target,
                       const OwningRewritePatternList &patterns,
                       TypeConverter *converter = nullptr);

/// Apply a complete conversion on the given operations, and all nested
/// operations. This method returns failure if the conversion of any operation
/// fails, or if there are unreachable blocks in any of the regions nested
/// within 'ops'. If 'converter' is provided, the signatures of blocks and
/// regions are also converted.
LLVM_NODISCARD LogicalResult
applyFullConversion(ArrayRef<Operation *> ops, ConversionTarget &target,
                    const OwningRewritePatternList &patterns,
                    TypeConverter *converter = nullptr);
LLVM_NODISCARD LogicalResult
applyFullConversion(Operation *op, ConversionTarget &target,
                    const OwningRewritePatternList &patterns,
                    TypeConverter *converter = nullptr);

/// Apply an analysis conversion on the given operations, and all nested
/// operations. This method analyzes which operations would be successfully
/// converted to the target if a conversion was applied. All operations that
/// were found to be legalizable to the given 'target' are placed within the
/// provided 'convertedOps' set; note that no actual rewrites are applied to the
/// operations on success and only pre-existing operations are added to the set.
/// This method only returns failure if there are unreachable blocks in any of
/// the regions nested within 'ops', or if a type conversion failed. If
/// 'converter' is provided, the signatures of blocks and regions are also
/// considered for conversion.
LLVM_NODISCARD LogicalResult applyAnalysisConversion(
    ArrayRef<Operation *> ops, ConversionTarget &target,
    const OwningRewritePatternList &patterns,
    DenseSet<Operation *> &convertedOps, TypeConverter *converter = nullptr);
LLVM_NODISCARD LogicalResult applyAnalysisConversion(
    Operation *op, ConversionTarget &target,
    const OwningRewritePatternList &patterns,
    DenseSet<Operation *> &convertedOps, TypeConverter *converter = nullptr);
} // end namespace mlir

#endif // MLIR_TRANSFORMS_DIALECTCONVERSION_H_
