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
// op, and various commonly used TensorFlow types, attributes, and builders.

#ifdef TF_OP_BASE
#else
#define TF_OP_BASE

#ifdef OP_BASE
#else
include "mlir/IR/OpBase.td"
#endif // OP_BASE

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
  represented using zero-dimentional tensors);

TODO: Make invariants more structured so that we can reference them in ops.
  }];

  let cppNamespace = "TF";
}

//===----------------------------------------------------------------------===//
// TensorFlow op definitions
//===----------------------------------------------------------------------===//

class TF_Op<string mnemonic, list<OpTrait> traits = []> :
    Op<TF_Dialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// TensorFlow type definitions
//===----------------------------------------------------------------------===//

// Any tensor element type defined in the TensorFlow dialect
def TF_TFDialectType :
    Type<CPred<"$_self.isa<TensorFlowType>()">, "TensorFlow type">;

// Any tensor element type allowed in TensorFlow ops
def TF_ElementType : Type<Or<[AnyFloat.predicate, AnyInteger.predicate,
                                 TF_TFDialectType.predicate]>,
                          "tf.dtype">;

// Any TensorFlow tensor type
def TF_Tensor : TensorOf<[TF_ElementType]>;

//===----------------------------------------------------------------------===//
// Integer types

def TF_I32Or64 : IntOfWidths<[32, 64]>;

def TF_I32OrI64Tensor : TensorOf<[TF_I32Or64]>;

def TF_Uint8 : Type<CPred<"$_self.isa<mlir::TF::Uint8Type>()">,
                    "TensorFlow uint8 type">,
               BuildableType<"getType<mlir::TF::Uint8Type>()">;

def TF_Uint16 : Type<CPred<"$_self.isa<mlir::TF::Uint16Type>()">,
                     "TensorFlow uint16 type">,
               BuildableType<"getType<mlir::TF::Uint16Type>()">;

def TF_Uint32 : Type<CPred<"$_self.isa<mlir::TF::Uint32Type>()">,
                     "TensorFlow uint32 type">,
               BuildableType<"getType<mlir::TF::Uint32Type>()">;

def TF_Uint64 : Type<CPred<"$_self.isa<mlir::TF::Uint64Type>()">,
                     "TensorFlow uint64 type">,
                BuildableType<"getType<mlir::TF::Uint64Type>()">;

// Any unsigned integer type
def TF_UInt : AnyTypeOf<[TF_Uint8, TF_Uint16, TF_Uint32, TF_Uint64]>;

// Any signed integer type
def TF_SInt : IntOfWidths<[8, 16, 32, 64]>;

// Any integer type
def TF_Int : AnyTypeOf<[TF_SInt, TF_UInt]>;

// Any integer tensor types
def TF_IntTensor : TensorOf<[TF_Int]>;

//===----------------------------------------------------------------------===//
// Floating-point types

def TF_F32Or64 : FloatOfWidths<[32, 64]>;

def TF_F32OrF64Tensor : TensorOf<[TF_F32Or64]>;

// Any floating-point tensor types
def TF_FpTensor : TensorOf<[AnyFloat]>;

//===----------------------------------------------------------------------===//
// Complex types

def TF_Complex64 :
    Type<CPred<"$_self.isa<TF::Complex64Type>()">, "complex64 type">;
def TF_Complex64Tensor : TensorOf<[TF_Complex64]>;

def TF_Complex128 :
    Type<CPred<"$_self.isa<TF::Complex128Type>()">, "complex128 type">;
def TF_Complex128Tensor : TensorOf<[TF_Complex128]>;

def TF_AnyComplex : AnyTypeOf<[TF_Complex64, TF_Complex128],
                              "64/128-bit complex type">;

def TF_ComplexTensor : TensorOf<[TF_AnyComplex]>;

//===----------------------------------------------------------------------===//
// String/variant/resource types

def TF_Str : Type<CPred<"$_self.isa<mlir::TF::StringType>()">,
                  "TensorFlow string type">,
             BuildableType<"getType<mlir::TF::StringType>()">;
def TF_StrTensor : TensorOf<[TF_Str]>;

def TF_Variant : Type<CPred<"$_self.isa<mlir::TF::VariantType>()">,
                      "TensorFlow variant type">,
                 BuildableType<"getType<mlir::TF::VariantType>()">;
def TF_VariantTensor : TensorOf<[TF_Variant]>;

def TF_Resource : Type<CPred<"$_self.isa<mlir::TF::ResourceType>()">,
                       "TensorFlow variant type">,
                  BuildableType<"getType<mlir::TF::ResourceType>()">;
def TF_ResourceTensor : TensorOf<[TF_Resource]>;

//===----------------------------------------------------------------------===//
// Multi-category type constraints

def TF_IntOrF32OrF64Tensor: TensorOf<[TF_Int, TF_F32Or64]>;

def TF_FpOrI32OrI64Tensor : TensorOf<[AnyFloat, TF_I32Or64]>;

// Any integer or floating-point tensor types
def TF_IntOrFpTensor : TensorOf<[TF_Int, AnyFloat]>;

def TF_FpOrComplexTensor : TensorOf<[AnyFloat, TF_AnyComplex]>;

def TF_AnyNumber : AnyTypeOf<[TF_Int, AnyFloat, TF_AnyComplex], "number">;

def TF_NumberTensor : TensorOf<[TF_AnyNumber]>;

def TF_NumberOrStrTensor : TensorOf<[TF_AnyNumber, TF_Str]>;

//===----------------------------------------------------------------------===//
// TensorFlow attribute definitions
//===----------------------------------------------------------------------===//

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

// A derived attribute that returns the element type of `idx`-th ODS-declared
// operand. If the `idx`-th operand is a variadic operand, then this attribute
// just returns the element type of its first tensor, which is only meaningful
// when the variadic operand has at least one tensor and the tensors all have
// the same element type.
class TF_DerivedOperandTypeAttr<int idx> : DerivedTypeAttr<
  "return mlir::getElementTypeOrSelf(*getODSOperands(" # idx # ").begin());">;

// A derived attribute that returns the element types of the tensors in the
// dynamic value pack that corresponds to the `idx`-th ODS-declared variadic
// operand. This returns a list of element types so it is used for variadic
// operands that can have different element types.
class TF_DerivedOperandTypeListAttr<int idx> : DerivedAttr<
  "mlir::OperandElementTypeRange",
  "auto values = getODSOperands(" # idx # ");\n"
  "return {mlir::OperandElementTypeIterator(values.begin()), "
          "mlir::OperandElementTypeIterator(values.end())};"
>;

// A derived attribute that returns the element type of `idx`-th ODS-declared
// result. If the `idx`-th result is a variadic result, then this attribute
// just returns the element type of its first tensor, which is only meaningful
// when the variadic result has at least one tensor and the tensors all have
// the same element type.
class TF_DerivedResultTypeAttr<int idx> : DerivedTypeAttr<
  "return mlir::getElementTypeOrSelf(*getODSResults(" # idx # ").begin());">;

// A derived attribute that returns the element types of the tensors in the
// dynamic value pack that corresponds to the `idx`-th ODS-declared variadic
// result. This returns a list of element types so it is used for variadic
// results that can have different element types.
class TF_DerivedResultTypeListAttr<int idx> : DerivedAttr<
  "mlir::ResultElementTypeRange",
  "auto values = getODSResults(" # idx # ");\n"
  "return {mlir::ResultElementTypeIterator(values.begin()), "
          "mlir::ResultElementTypeIterator(values.end())};"
>;

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
"Builder *builder, OperationState *result, Value* x, Value* y",
[{
  auto resultType =
      OpTrait::util::getBroadcastedType(x->getType(), y->getType());
  if (!resultType)
    mlir::emitError(result->location, "non-broadcastable operands");
  return build(builder, result, resultType, x, y);
}]
  >];
}

// Mixin class defining a builder for comparison ops supporting broadcast
// behavior. The result type has bool element type.
class WithBroadcastableCmpOpBuilder {
  list<OpBuilder> builders = [OpBuilder<
"Builder *builder, OperationState *result, Value* x, Value* y",
[{
  Type resultType;
  if (x->getType().isa<UnrankedTensorType>() ||
      y->getType().isa<UnrankedTensorType>()) {
    resultType = builder->getTensorType(builder->getI1Type());
  } else {
    SmallVector<int64_t, 4> resultShape;
    if (!OpTrait::util::getBroadcastedShape(
            x->getType().cast<ShapedType>().getShape(),
            y->getType().cast<ShapedType>().getShape(), resultShape)) {
      mlir::emitError(result->location,
                      "operands have no broadcastable shapes");
    }

    resultType = builder->getTensorType(resultShape, builder->getI1Type());
  }
  return build(builder, result, resultType, x, y);
}]
  >];
}

#endif // TF_OP_BASE
