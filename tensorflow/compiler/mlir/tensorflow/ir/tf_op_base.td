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

// This is the base operation definition file for TensorFlow.
//
// This file includes the definition for the TensorFlow dialect, base TensorFlow
// op, and various commonly used TensorFlow traits, types, attributes, and
// builders.

#ifndef TF_OP_BASE
#define TF_OP_BASE

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "tensorflow/compiler/mlir/tensorflow/ir/tf_op_interfaces.td"

//===----------------------------------------------------------------------===//
// TensorFlow dialect definitions
//===----------------------------------------------------------------------===//

def TF_Dialect : Dialect {
  let name = "tf";

  let description = [{
The TensorFlow dialect.

This dialect maps to TensorFlow operations.

Invariants:

* All values are of Tensor type (in particular, scalars are
  represented using zero-dimensional tensors);

TODO: Make invariants more structured so that we can reference them in ops.
  }];

  let cppNamespace = "TF";
}

//===----------------------------------------------------------------------===//
// TensorFlow traits
//===----------------------------------------------------------------------===//

// Specify this trait if the op requires all outputs to have the same type and
// the inputs either have the same type as result or a ref type corresponding to
// the result type.
def TF_OperandsSameAsResultsTypeOrRef : NativeOpTrait<
  "TF::OperandsSameAsResultsTypeOrRef">;

// Layout agnostic operations do not depend on the operands data layout (data
// format), as an example all element wise operations are layout agnostic.
def TF_LayoutAgnostic : NativeOpTrait<"TF::LayoutAgnostic">;

// Variant of broadcastable trait that considers TF's subtype behavior.
class TF_OpIsBroadcastableToRes<int opId, int resId> : And<[
    TCOpResIsShapedTypePred<opId, resId>,
    CPred<"mlir::TF::BroadcastCompatible("
              "$_op.getOperand(" # opId # ").getType(), "
              "$_op.getResult(" # resId # ").getType())">]>;


class TF_AllTypesMatchPred<list<string> values> :
    CPred<"TF::AreCastCompatible(llvm::makeArrayRef({"# StrJoin<values>.result #"}))">;

class TF_AllTypesMatch<list<string> names> :
    PredOpTrait<
        "all of {" # StrJoin<names>.result # "} have dynamically equal types ",
        TF_AllTypesMatchPred<
            !foreach(n, names, !subst("$_self", "$" # n, "$_self.getType()"))>>;

//===----------------------------------------------------------------------===//
// TensorFlow op side effects
//===----------------------------------------------------------------------===//

class TF_ResourceBase<string resourceKind> :
  Resource<!strconcat("::mlir::TF::ResourceEffects::", resourceKind)> {
}

def TF_VariableResource : TF_ResourceBase<"Variable">;
def TF_StackResource : TF_ResourceBase<"Stack">;
def TF_TensorArrayResource : TF_ResourceBase<"TensorArray">;

def TF_VariableRead : MemRead<TF_VariableResource>;
def TF_StackRead : MemRead<TF_StackResource>;
def TF_TensorArrayRead : MemRead<TF_TensorArrayResource>;

def TF_VariableWrite : MemWrite<TF_VariableResource>;
def TF_StackWrite : MemWrite<TF_StackResource>;
def TF_TensorArrayWrite : MemWrite<TF_TensorArrayResource>;

//===----------------------------------------------------------------------===//
// TensorFlow op definitions
//===----------------------------------------------------------------------===//

class TF_Op<string mnemonic, list<OpTrait> traits = []> :
    Op<TF_Dialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// TensorFlow attribute definitions
//===----------------------------------------------------------------------===//

class TF_TensorFlowAttr <string name, string description> :
    Attr<CPred<"$_self.isa<mlir::TF::" # name # "Attr>()">,
         "TensorFlow " # description # " attribute">;

def TF_ShapeAttr : TF_TensorFlowAttr<"Shape", "shape"> {
  let returnType = "llvm::Optional<llvm::ArrayRef<int64_t>>";
  let convertFromStorage = "$_self.cast<mlir::TF::ShapeAttr>().getValue()";

  // Create a ranked shape attr by default.
  let constBuilderCall = "mlir::TF::ShapeAttr::get($_builder.getContext(), $0)";
}

def TF_ShapeAttrArray :
    TypedArrayAttrBase<TF_ShapeAttr, "tensorflow shape attribute array">;

//===----------------------------------------------------------------------===//
// TensorFlow type definitions
//===----------------------------------------------------------------------===//

// Any tensor element type defined in the TensorFlow dialect
def TF_TFDialectType :
    Type<CPred<"$_self.isa<mlir::TF::TensorFlowType>()">, "TensorFlow type">;

// Class for any TensorFlow dialect specific type
class TF_TensorFlowType <string name, string description> :
    Type<CPred<"$_self.isa<mlir::TF::" # name # "Type>()">,
         "TensorFlow " # description # " type">,
    BuildableType<"getType<mlir::TF::" # name # "Type>()">;

// Any tensor element type allowed in TensorFlow ops
def TF_ElementType : Type<Or<[AnyFloat.predicate,
                              AnySignlessInteger.predicate,
                              AnyUnsignedInteger.predicate,
                              AnyComplex.predicate,
                              TF_TFDialectType.predicate]>,
                          "tf.dtype">;

// Any TensorFlow tensor type
def TF_Tensor : TensorOf<[TF_ElementType]>;

//===----------------------------------------------------------------------===//
// Integer types

def TF_I32Or64 : SignlessIntOfWidths<[32, 64]>;

def TF_I32OrI64Tensor : TensorOf<[TF_I32Or64]>;

def TF_Uint8 : UI<8>;
def TF_Uint8Tensor : TensorOf<[TF_Uint8]>;

def TF_Uint16 : UI<16>;
def TF_Uint16Tensor : TensorOf<[TF_Uint16]>;

def TF_Uint32 : UI<32>;
def TF_Uint32Tensor : TensorOf<[TF_Uint32]>;

def TF_Uint64 : UI<64>;
def TF_Uint64Tensor : TensorOf<[TF_Uint64]>;

// Any unsigned integer type
def TF_UInt : UnsignedIntOfWidths<[8, 16, 32, 64]>;

// Any signed integer type
def TF_SInt : SignlessIntOfWidths<[8, 16, 32, 64]>;

// Any integer type
def TF_Int : AnyTypeOf<[TF_SInt, TF_UInt]>;

// Any integer tensor types
def TF_IntTensor : TensorOf<[TF_Int]>;

//===----------------------------------------------------------------------===//
// Quantized types
def TF_Qint8   : TF_TensorFlowType<"Qint8", "qint8">;
def TF_Qint16  : TF_TensorFlowType<"Qint16", "qint16">;
def TF_Qint32  : TF_TensorFlowType<"Qint32", "qint32">;
def TF_Quint8  : TF_TensorFlowType<"Quint8", "quint8">;
def TF_Quint16 : TF_TensorFlowType<"Quint16", "quint16">;

// Any quantized type
def TF_AnyQuantized : AnyTypeOf<[TF_Qint8, TF_Qint16, TF_Qint32, TF_Quint8,
                              TF_Quint16]>;
//===----------------------------------------------------------------------===//
// Floating-point types

def TF_F32Or64 : FloatOfWidths<[32, 64]>;

def TF_F32OrF64Tensor : TensorOf<[TF_F32Or64]>;

// Any floating-point tensor types
def TF_FpTensor : TensorOf<[AnyFloat]>;

//===----------------------------------------------------------------------===//
// Complex types

// TODO(suderman): Remove TF_Complex64 and use a standard ops declaration, along
// with the associated cleanup.
def TF_Complex64 : Complex<F<32>>;
def TF_Complex64Tensor : TensorOf<[TF_Complex64]>;

def TF_Complex128 : Complex<F<64>>;
def TF_Complex128Tensor : TensorOf<[TF_Complex128]>;

def TF_AnyComplex : AnyTypeOf<[TF_Complex64, TF_Complex128],
                              "64/128-bit complex type">;

def TF_ComplexTensor : TensorOf<[TF_AnyComplex]>;

//===----------------------------------------------------------------------===//
// String/variant/resource types

def TF_Str : TF_TensorFlowType<"String", "string">;
def TF_StrTensor : TensorOf<[TF_Str]>;

def TF_Variant : TF_TensorFlowType<"Variant", "variant">;
def TF_VariantTensor : TensorOf<[TF_Variant]>;

def TF_Resource : TF_TensorFlowType<"Resource", "resource">;
def TF_ResourceTensor : TensorOf<[TF_Resource]>;

//===----------------------------------------------------------------------===//
// Multi-category type constraints

def TF_IntOrF32OrF64Tensor: TensorOf<[TF_Int, TF_F32Or64]>;

def TF_FpOrI32OrI64Tensor : TensorOf<[AnyFloat, TF_I32Or64]>;

// Any integer or floating-point tensor types
def TF_IntOrFpTensor : TensorOf<[TF_Int, AnyFloat]>;

def TF_SintOrFpTensor : TensorOf<[TF_SInt, AnyFloat]>;

def TF_FpOrComplexTensor : TensorOf<[AnyFloat, TF_AnyComplex]>;

def TF_AnyNumber : AnyTypeOf<[TF_Int, AnyFloat, TF_AnyQuantized, TF_AnyComplex],
                             "number">;

def TF_NumberTensor : TensorOf<[TF_AnyNumber]>;

def TF_NumberOrStr : AnyTypeOf<[AnyFloat, TF_SInt, TF_AnyComplex, TF_Uint8, TF_Str]>;
def TF_NumberOrStrTensor : TensorOf<[TF_NumberOrStr]>;

//===----------------------------------------------------------------------===//
// TensorFlow attribute definitions
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Tensorflow devices metadata

// Tensorflow GPU device metadata.
def TF_GpuDeviceMetadata : StructAttr<"GpuDeviceMetadata", TF_Dialect, [
    // GPU device compute capability: major:minor.
    StructFieldAttr<"cc_major", I32Attr>,
    StructFieldAttr<"cc_minor", I32Attr>
]>;

//===----------------------------------------------------------------------===//
// String attribute constraints

// A string attribute whose value are one of the values in `cases`.
class TF_AnyStrAttrOf<list<string> cases> : StringBasedAttr<
  CPred<!foldl(
      "$_self.cast<StringAttr>().getValue() == \"" # !head(cases) # "\"",
      !foreach(case, !tail(cases),
               "$_self.cast<StringAttr>().getValue() == \"" # case # "\""),
      prev, cur, prev # " || " # cur)>,
  "string attribute whose value is " #
    !foldl(/*init*/!head(cases), /*list*/!tail(cases),
           prev, cur, prev # ", or " # cur)>;

// TODO: Use EnumAttr to define the common attribute cases

def TF_ConvnetDataFormatAttr : StringBasedAttr<
    CPred<"$_self.cast<StringAttr>().getValue() == \"NHWC\" || " #
          "$_self.cast<StringAttr>().getValue() == \"NCHW\"">,
    "'NHWC' or 'NCHW' convnet data format">;

//===----------------------------------------------------------------------===//
// Type attributes

// A derived attribute that returns the size of `idx`-th ODS-declared variadic
// operand.
class TF_DerivedOperandSizeAttr<int idx> : DerivedAttr<
  "size_t",
  "auto range = getODSOperands(" # idx # ");\n"
  "return std::distance(range.begin(), range.end());",
  [{ $_builder.getI64IntegerAttr($_self) }]>;

// A derived attribute that returns the element type of `idx`-th ODS-declared
// operand. If the `idx`-th operand is a variadic operand, then this attribute
// just returns the element type of its first tensor, which is only meaningful
// when the variadic operand has at least one tensor and the tensors all have
// the same element type.
class TF_DerivedOperandTypeAttr<int idx> : DerivedTypeAttr<
  "return mlir::getElementTypeOrSelf(*getODSOperands(" # idx # ").begin());">;

// A derived attribute that returns the element types of the tensors in the
// actual value pack that corresponds to the `idx`-th ODS-declared variadic
// operand. This returns a list of element types so it is used for variadic
// operands that can have different element types.
class TF_DerivedOperandTypeListAttr<int idx> : DerivedAttr<
  "mlir::OperandElementTypeRange",
  "auto values = getODSOperands(" # idx # ");\n"
  "return {mlir::OperandElementTypeIterator(values.begin()), "
          "mlir::OperandElementTypeIterator(values.end())};",
  [{
    ArrayAttr::get(
    [&]() {
      llvm::SmallVector<Attribute, 4> ret;
      for (auto t : $_self)
        ret.push_back(TypeAttr::get(t));
      return ret;
    }(), $_ctx)
  }]
>;

// A derived attribute that returns the shapes of the tensors in the actual
// value pack that corresponds to the `idx`-th ODS-declared variadic operand.
// This returns a list of shapes so it is used for variadic operands that
// can have different shapes.
class TF_DerivedOperandShapeListAttr<int idx> : DerivedAttr<
  "mlir::TF::OperandShapeRange",
  "auto values = getODSOperands(" # idx # ");\n"
  "return {mlir::TF::OperandShapeIterator(values.begin()), "
          "mlir::TF::OperandShapeIterator(values.end())};",
  [{
    ArrayAttr::get(
      [&](){
        llvm::SmallVector<Attribute, 4> ret;
        for (auto shape : $_self)
          ret.push_back(mlir::TF::ShapeAttr::get($_ctx, shape));
        return ret;
      }(), $_ctx)
  }]
>;

// A derived attribute that returns the size of `idx`-th ODS-declared variadic
// result.
class TF_DerivedResultSizeAttr<int idx> : DerivedAttr<
  "size_t",
  "auto range = getODSResults(" # idx # ");\n"
  "return std::distance(range.begin(), range.end());",
  [{ $_builder.getI64IntegerAttr($_self) }]>;

// A derived attribute that returns the element type of `idx`-th ODS-declared
// result. If the `idx`-th result is a variadic result, then this attribute
// just returns the element type of its first tensor, which is only meaningful
// when the variadic result has at least one tensor and the tensors all have
// the same element type.
class TF_DerivedResultTypeAttr<int idx> : DerivedTypeAttr<
  "return mlir::getElementTypeOrSelf(*getODSResults(" # idx # ").begin());">;

// A derived attribute that returns the element types of the tensors in the
// actual value pack that corresponds to the `idx`-th ODS-declared variadic
// result. This returns a list of element types so it is used for variadic
// results that can have different element types.
class TF_DerivedResultTypeListAttr<int idx> : DerivedAttr<
  "mlir::ResultElementTypeRange",
  "auto values = getODSResults(" # idx # ");\n"
  "return {mlir::ResultElementTypeIterator(values.begin()), "
          "mlir::ResultElementTypeIterator(values.end())};",
  [{
    ArrayAttr::get(
    [&]() {
      llvm::SmallVector<Attribute, 4> ret;
      for (auto t : $_self)
        ret.push_back(TypeAttr::get(t));
      return ret;
    }(), $_ctx)
  }]
>;

// A derived attribute that returns the shapes of the tensors in the actual
// value pack that corresponds to the `idx`-th ODS-declared variadic result.
// This returns a list of shapes so it is used for variadic results that
// can have different shapes.
class TF_DerivedResultShapeListAttr<int idx> : DerivedAttr<
  "mlir::TF::ResultShapeRange",
  "auto values = getODSResults(" # idx # ");\n"
  "return {mlir::TF::ResultShapeIterator(values.begin()), "
          "mlir::TF::ResultShapeIterator(values.end())};",
  [{
    ArrayAttr::get(
      [&](){
        llvm::SmallVector<Attribute, 4> ret;
        for (auto shape : $_self)
          ret.push_back(mlir::TF::ShapeAttr::get($_ctx, shape));
        return ret;
      }(), $_ctx)
  }]
>;

// A derived attribute that returns the shape of the first result type.
def TF_DerivedResultShapeAttr : DerivedAttr<"ShapedType",
  "return (*getOperation()->result_type_begin()).cast<ShapedType>();",
  [{ TypeAttr::get($_self) }]>;

// A derived attribute that returns the element type of the tensor held by a
// named resource-type operand or result.
class TF_DerivedOperandOrResultHandleTypeAttr<string name> : DerivedTypeAttr<
  "auto resource_type =\n"
  "  mlir::getElementTypeOrSelf(this->" # name # "())\n"
  "  .cast<TF::ResourceType>();\n"
  "assert(!resource_type.getSubtypes().empty() && \"unknown type\");\n"
  "return mlir::getElementTypeOrSelf(*resource_type.getSubtypes().begin());">;

// A derived attribute that returns the shape of the tensor held by a named
// resource-type operand or result.
class TF_DerivedOperandOrResultHandleShapeAttr<string name> : DerivedAttr<
  "ShapedType",
  "auto resource_type =\n"
  "  mlir::getElementTypeOrSelf(this->" # name # "())\n"
  "  .cast<TF::ResourceType>();\n"
  "assert(!resource_type.getSubtypes().empty() && \"unknown shape\");\n"
  "return resource_type.getSubtypes().begin()->cast<ShapedType>();",
  [{ TypeAttr::get($_self) }]>;

def TF_IntTypeAttr : TypeAttrBase<"IntegerType", "integer type"> {
  let returnType = "Type";
}

//===----------------------------------------------------------------------===//
// TensorFlow common builders
//===----------------------------------------------------------------------===//

// Mixin class defining a builder for binary ops supporting broadcast
// behavior. The result type has the same element type as both operands.
class WithBroadcastableBinOpBuilder {
  list<OpBuilder> builders = [OpBuilder<
"OpBuilder &builder, OperationState &result, Value  x, Value  y",
[{
  auto resultType =
      OpTrait::util::getBroadcastedType(x.getType(), y.getType());
  if (!resultType)
    mlir::emitError(result.location, "non-broadcastable operands");
  return build(builder, result, resultType, x, y);
}]
  >];
}

// Mixin class defining a builder for comparison ops supporting broadcast
// behavior. The result type has bool element type.
class WithBroadcastableCmpOpBuilder {
  list<OpBuilder> builders = [OpBuilder<
"OpBuilder &builder, OperationState &result, Value  x, Value  y",
[{
  Type resultType;
  if (x.getType().isa<UnrankedTensorType>() ||
      y.getType().isa<UnrankedTensorType>()) {
    resultType = UnrankedTensorType::get(builder.getI1Type());
  } else {
    SmallVector<int64_t, 4> resultShape;
    if (!OpTrait::util::getBroadcastedShape(
            x.getType().cast<ShapedType>().getShape(),
            y.getType().cast<ShapedType>().getShape(), resultShape)) {
      mlir::emitError(result.location,
                      "operands have no broadcastable shapes");
    }

    resultType = RankedTensorType::get(resultShape, builder.getI1Type());
  }
  return build(builder, result, resultType, x, y);
}]
  >];
}

#endif // TF_OP_BASE
