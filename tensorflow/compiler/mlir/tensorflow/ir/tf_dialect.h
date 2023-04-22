/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

// This file defines the standard MLIR TensorFlow dialect after control
// dependences are raise to the standard form.

#ifndef TENSORFLOW_COMPILER_MLIR_TENSORFLOW_IR_TF_DIALECT_H_
#define TENSORFLOW_COMPILER_MLIR_TENSORFLOW_IR_TF_DIALECT_H_

#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/Dialect.h"  // from @llvm-project

namespace mlir {
namespace TF {

class ResourceType;
class TensorFlowRegistryEffectInterfaceFallback;
class VariantType;

class TensorFlowDialect final : public Dialect {
 public:
  explicit TensorFlowDialect(MLIRContext *context);
  ~TensorFlowDialect() override;

  static StringRef getDialectNamespace() { return "tf"; }

  // Gradient attribute ("tf.gradient") in the list of NamedAttributes in a
  // function references to its gradient function. This attribute in TensorFlow
  // Dialect is used to model TF GradientDef. GetGradientAttrName() returns the
  // string description of gradient attribute.
  static StringRef GetGradientAttrName() { return "tf.gradient"; }

  // This attribute marks if a function is stateful.
  // Returns the string description of stateful attribute.
  static StringRef GetStatefulAttrName() { return "tf.signature.is_stateful"; }

  // Returns true if the op can be duplicated during transformations.
  static bool CanDuplicate(Operation *op);

  // Returns true if the op can have side effects.
  static bool CanHaveSideEffects(Operation *op);

  Attribute parseAttribute(DialectAsmParser &parser, Type type) const override;

  void printAttribute(Attribute attr, DialectAsmPrinter &os) const override;

  // Parse a type registered to this dialect.
  Type parseType(DialectAsmParser &parser) const override;

  // Prints a type registered to this dialect.
  void printType(Type ty, DialectAsmPrinter &os) const override;

  // Parses resource type with potential subtypes.
  Type ParseResourceType(DialectAsmParser &parser) const;

  // Prints resource type with potential subtypes.
  void PrintResourceType(ResourceType ty, DialectAsmPrinter &os) const;

  // Parse and print variant type. It may have subtypes inferred using shape
  // inference.
  Type ParseVariantType(DialectAsmParser &parser) const;
  void PrintVariantType(VariantType ty, DialectAsmPrinter &os) const;

  // Registered hook to materialize a constant operation from a given attribute
  // value with the desired resultant type.
  Operation *materializeConstant(OpBuilder &builder, Attribute value, Type type,
                                 Location loc) override;

  typedef std::function<void(TensorFlowDialect &dialect)> AdditionalOpFunction;

  // Register an op registration hook which is invoked during construction.
  //
  // A hook may use the public addOperations() method to add additional
  // operations to the dialect. Hooks will only apply to subsequent
  // instantations of the Dialect/MLIRContext.
  static void RegisterAdditionalOperationHook(AdditionalOpFunction fn) {
    GetAdditionalOperationHooks()->push_back(std::move(fn));
  }

  // Re-define publicly the protected addOperations() method from the Dialect
  // class, usually used in a Dialect constructor. This allows hook
  // functions to register operations on the TensorFlow dialect using the
  // same interface.
  template <typename... Args>
  void addOperations() {
    Dialect::addOperations<Args...>();
  }

  using ConstantFoldHook = LogicalResult (*)(Operation *, ArrayRef<Attribute>,
                                             SmallVectorImpl<OpFoldResult> &);
  static void RegisterConstantFoldHook(ConstantFoldHook fn) {
    constant_fold_hook_ = std::move(fn);
  }

  static LogicalResult constantFold(Operation *op, ArrayRef<Attribute> operands,
                                    SmallVectorImpl<OpFoldResult> &results) {
    if (constant_fold_hook_) return constant_fold_hook_(op, operands, results);
    return failure();
  }

  using DecodeConstantHook = LogicalResult (*)(OpaqueElementsAttr input,
                                               ElementsAttr &output);
  static void RegisterDecodeConstantHook(DecodeConstantHook fn) {
    decode_constant_hook_ = std::move(fn);
  }
  static LogicalResult decode(OpaqueElementsAttr input, ElementsAttr &output) {
    if (decode_constant_hook_) return decode_constant_hook_(input, output);
    return failure();
  }

  // Provides a hook for op interface.
  void *getRegisteredInterfaceForOp(mlir::TypeID interface,
                                    mlir::OperationName opName) override;

 private:
  /// Register the attributes of this dialect.
  void registerAttributes();
  /// Register the types of this dialect.
  void registerTypes();

  // Hook functions which may add additional operations to the dialect.
  // These are invoked at construction time.
  static std::vector<AdditionalOpFunction> *GetAdditionalOperationHooks();

  static ConstantFoldHook constant_fold_hook_;
  static DecodeConstantHook decode_constant_hook_;

  // Storage for a custom fallback interface.
  TensorFlowRegistryEffectInterfaceFallback *fallback_effect_op_interface_;
};

}  // namespace TF
}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_TENSORFLOW_IR_TF_DIALECT_H_
