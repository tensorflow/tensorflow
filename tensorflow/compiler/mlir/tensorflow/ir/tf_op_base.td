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

def TensorFlowDialect : Dialect {
  let name = "tf";

  let description = [{
The TensorFlow dialect.

This dialect maps to TensorFlow operations.

Invariants:

* All values are of Tensor type (in particular, scalars are
  represented using zero-dimensional tensors);

TODO: Make invariants more structured so that we can reference them in ops.
  }];

  let cppNamespace = "::mlir::TF";

  let emitAccessorPrefix = kEmitAccessorPrefix_Raw;
}

//===----------------------------------------------------------------------===//
// TensorFlow traits
//===----------------------------------------------------------------------===//

// Specify this trait if the op requires all outputs to have the same type and
// the inputs either have the same type as result or a ref type corresponding to
// the result type.
def TF_OperandsSameAsResultsTypeOrRef : NativeOpTrait<
  "TF::OperandsSameAsResultsTypeOrRef">;

// Op has the same operand and result element types (or type itself, if scalar)
// after resolving reference types (i.e., after converting reference types to
// their corresponding TensorFlow or standard types).
def TF_SameOperandsAndResultElementTypeResolveRef : NativeOpTrait<
  "TF::SameOperandsAndResultElementTypeResolveRef">;

// Op has the same operand and result types after resolving reference types
// (i.e., after converting reference types to their corresponding TensorFlow or
// standard types).
def TF_SameOperandsAndResultTypeResolveRef : NativeOpTrait<
  "TF::SameOperandsAndResultTypeResolveRef">;

// Layout agnostic operations do not depend on the operands data layout (data
// format), as an example all element wise operations are layout agnostic.
def TF_LayoutAgnostic : NativeOpTrait<"TF::LayoutAgnostic">;

// Trait to indicate operations that cannot be duplicated as they might carry
// certain state around within their implementations.
def TF_CannotDuplicate : NativeOpTrait<"TF::CannotDuplicate">;

// Trait to indicate an operation cannot be constant folded.
def TF_NoConstantFold : NativeOpTrait<"TF::NoConstantFold">;

// Coefficient wise binary operation with implicit broadcasting support, for
// example tf.Sub operation.
def TF_CwiseBinary : NativeOpTrait<"TF::CwiseBinary">;

// Coefficient wise unary operation, for example tf.Sqrt operation.
def TF_CwiseUnary : NativeOpTrait<"TF::CwiseUnary">;

// Variant of broadcastable trait that considers TF's subtype behavior.
class TF_OpIsBroadcastableToRes<int opId, int resId> : And<[
    TCOpResIsShapedTypePred<opId, resId>,
    CPred<"mlir::tf_type::BroadcastCompatible("
              "$_op.getOperand(" # opId # ").getType(), "
              "$_op.getResult(" # resId # ").getType())">]>;


class TF_AllTypesMatchPred<list<string> values> :
    CPred<"tf_type::AreCastCompatible(llvm::makeArrayRef({" #
      !interleave(values, ", ") # "}))">;

class TF_AllTypesMatch<list<string> names> :
    PredOpTrait<
        "all of {" # !interleave(names, ", ") #
          "} have dynamically equal types ",
        TF_AllTypesMatchPred<
            !foreach(n, names, !subst("$_self", "$" # n, "$_self.getType()"))>>;

// This trait indicates that all returned resources are unique for a
// resource-allocating op (i.e. op with `MemAlloc` side effect).
//
// Note that if the trait is used where this invariant is not true, then this
// might lead to incorrect execution order, while if not used where it should
// be, it can only lead to reduced performance due to conservative ordering.
// Example op where the invariant is not true: `TF_VarHandleOp`.
def TF_UniqueResourceAllocation: TraitList<[
    TF_ResourceHandleAllocatorInterface,
    NativeOpTrait<"TF::UniqueResourceAllocation">
]>;

//===----------------------------------------------------------------------===//
// Rank/Shape helpers.
//===----------------------------------------------------------------------===//

class TF_OperandIsUnrankedPred<int n> :
  CPred<"$_op.getOperand(" # n # ").getType().isa<UnrankedTensorType>()">;

class TF_ResultIsUnrankedPred<int n> :
  CPred<"$_op.getResult(" # n # ").getType().isa<UnrankedTensorType>()">;

// Returns true if the n-th operand has unknown rank or has rank m.
class TF_OperandHasRank<int n, int m> :
  PredOpTrait<"operand " # n # " is " # m # "-D",
    Or<[TF_OperandIsUnrankedPred<n>,
      CPred<"$_op.getOperand(" # n #
      ").getType().cast<ShapedType>().getRank() == " # m>]>>;

// Returns true if the n-th result has unknown rank or has rank m.
class TF_ResultHasRank<int n, int m> :
  PredOpTrait<"result " # n # " is " # m # "-D",
    Or<[TF_ResultIsUnrankedPred<n>,
      CPred<"$_op.getResult(" # n #
      ").getType().cast<ShapedType>().getRank() == " # m>]>>;

//===----------------------------------------------------------------------===//
// TensorFlow resources and side effects
//===----------------------------------------------------------------------===//

class TF_ResourceBase<string resourceKind> :
  Resource<!strconcat("::mlir::TF::ResourceEffects::", resourceKind)> {
}

// Resource types
def TF_VariableResource : TF_ResourceBase<"Variable">;
def TF_StackResource : TF_ResourceBase<"Stack">;
def TF_TensorArrayResource : TF_ResourceBase<"TensorArray">;
def TF_SummaryResource : TF_ResourceBase<"Summary">;
def TF_LookupTableResource : TF_ResourceBase<"LookupTable">;
def TF_DatasetSeedGeneratorResource : TF_ResourceBase<"DatasetSeedGenerator">;
def TF_DatasetMemoryCacheResource : TF_ResourceBase<"DatasetMemoryCache">;
def TF_DatasetIteratorResource : TF_ResourceBase<"DatasetIterator">;
def TF_TPUEmbeddingResource : TF_ResourceBase<"TPUEmbedding">;
def TF_GeneratorOpResource : TF_ResourceBase<"GeneratorOp">;
def TF_SendResource : TF_ResourceBase<"Send">;
def TF_RecvResource : TF_ResourceBase<"Recv">;
def TF_TPUExecuteResource : TF_ResourceBase<"TPUExecute">;
def TF_RandomGeneratorResource : TF_ResourceBase<"RandomGenerator">;
def TF_XlaHostComputeResource : TF_ResourceBase<"XlaHostCompute">;
def TF_CollectiveReduceOrderingResource : TF_ResourceBase<"CollectiveReduceOrdering">;

// Fake resource, see `TF_MustExecute` below.
def TF_MustExecuteResource : TF_ResourceBase<"MustExecute">;

// Value-based side effects
//
// Value-based side effect traits are attached to op operands or results to
// signal what type of resource is accessed and in which way.
def TF_VariableRead : MemRead<TF_VariableResource>;
def TF_StackRead : MemRead<TF_StackResource>;
def TF_TensorArrayRead : MemRead<TF_TensorArrayResource>;
def TF_LookupTableRead : MemRead<TF_LookupTableResource>;
def TF_DatasetSeedGeneratorRead : MemRead<TF_DatasetSeedGeneratorResource>;
def TF_DatasetMemoryCacheRead : MemRead<TF_DatasetMemoryCacheResource>;
def TF_DatasetIteratorRead : MemRead<TF_DatasetIteratorResource>;

def TF_VariableWrite : MemWrite<TF_VariableResource>;
def TF_StackWrite : MemWrite<TF_StackResource>;
def TF_TensorArrayWrite : MemWrite<TF_TensorArrayResource>;
def TF_SummaryWrite : MemWrite<TF_SummaryResource>;
def TF_LookupTableWrite : MemWrite<TF_LookupTableResource>;
def TF_DatasetSeedGeneratorWrite : MemWrite<TF_DatasetSeedGeneratorResource>;
def TF_DatasetMemoryCacheWrite : MemWrite<TF_DatasetMemoryCacheResource>;
def TF_DatasetIteratorWrite : MemWrite<TF_DatasetIteratorResource>;

def TF_VariableAlloc : MemAlloc<TF_VariableResource>;
def TF_StackAlloc : MemAlloc<TF_StackResource>;
def TF_TensorArrayAlloc : MemAlloc<TF_TensorArrayResource>;
def TF_SummaryAlloc : MemAlloc<TF_SummaryResource>;
def TF_LookupTableAlloc : MemAlloc<TF_LookupTableResource>;
def TF_DatasetSeedGeneratorAlloc : MemAlloc<TF_DatasetSeedGeneratorResource>;
def TF_DatasetMemoryCacheAlloc : MemAlloc<TF_DatasetMemoryCacheResource>;
def TF_DatasetIteratorAlloc : MemAlloc<TF_DatasetIteratorResource>;

def TF_StackFree : MemFree<TF_StackResource>;
def TF_TensorArrayFree : MemFree<TF_TensorArrayResource>;
def TF_SummaryFree : MemFree<TF_SummaryResource>;
def TF_DatasetSeedGeneratorFree : MemFree<TF_DatasetSeedGeneratorResource>;
def TF_DatasetMemoryCacheFree : MemFree<TF_DatasetMemoryCacheResource>;
def TF_DatasetIteratorFree : MemFree<TF_DatasetIteratorResource>;

// Op-based side effects

// Op-based side effect traits can be used to enforce certain execution order
// constraints, in particular for ops that don't use resource handles (those
// typically have value-based side effects). For a `read` effect, all instances
// of ops with the trait keep their order to all ops with unknown side effects
// (e.g. `stateful` ops). For a `write` effect, all instances of ops with the
// trait stay in order, and they also keep their order to all unknown side-
// effecting ops. Note that for `read` effects ops might be pruned if nothing
// depends on them.
def TF_GeneratorOpSideEffect : MemoryEffects<[MemWrite<TF_GeneratorOpResource>]>;

def TF_TPUEmbeddingWriteEffect : MemoryEffects<[MemWrite<TF_TPUEmbeddingResource>]>;
def TF_TPUEmbeddingReadEffect : MemoryEffects<[MemRead<TF_TPUEmbeddingResource>]>;

def TF_SendSideEffect : MemoryEffects<[MemWrite<TF_SendResource>]>;
def TF_RecvSideEffect : MemoryEffects<[MemWrite<TF_RecvResource>]>;
def TF_XlaHostComputeSideEffect : MemoryEffects<[MemWrite<TF_XlaHostComputeResource>]>;

def TF_TPUExecuteSideEffect : MemoryEffects<[MemWrite<TF_TPUExecuteResource>]>;

def TF_RandomGeneratorSideEffect : MemoryEffects<[MemWrite<TF_RandomGeneratorResource>]>;

// Special effect for keeping `CollectiveReduce` ops in order.
def TF_CollectiveReduceOrderingEffect : MemoryEffects<[MemWrite<TF_CollectiveReduceOrderingResource>]>;

// Trait for enforcing that a side-effecting op is executed, even if it would be
// considered dead by MLIR (see b/195782952).
// The trait is implemented as a write effect for a fake resource which is
// ignored by side effect analysis, so it does not affect execution order
// constraints and control dependencies at all (for example, multiple ops with
// this trait do not have to execute in order).
def TF_MustExecute : MemoryEffects<[MemWrite<TF_MustExecuteResource>]>;

//===----------------------------------------------------------------------===//
// TensorFlow op definitions
//===----------------------------------------------------------------------===//

class TF_Op<string mnemonic, list<Trait> traits = []> :
    Op<TensorFlowDialect, mnemonic, traits>;

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

//===----------------------------------------------------------------------===//
// Reference types

// Float reference types
def TF_Float16Ref : TF_TensorFlowType<"HalfRef", "f16ref">;
def TF_Float32Ref : TF_TensorFlowType<"FloatRef", "f32ref">;
def TF_Float64Ref : TF_TensorFlowType<"DoubleRef", "f64ref">;
def TF_Bfloat16Ref : TF_TensorFlowType<"Bfloat16Ref", "bf16ref">;

// Complex reference types
def TF_Complex64Ref : TF_TensorFlowType<"Complex64Ref", "complex64ref">;
def TF_Complex128Ref : TF_TensorFlowType<"Complex128Ref", "complex128ref">;

// Integer reference types
def TF_Int8Ref : TF_TensorFlowType<"Int8Ref", "i8ref">;
def TF_Int16Ref : TF_TensorFlowType<"Int16Ref", "i16ref">;
def TF_Int32Ref : TF_TensorFlowType<"Int32Ref", "i32ref">;
def TF_Int64Ref : TF_TensorFlowType<"Int64Ref", "i64ref">;

def TF_Uint8Ref : TF_TensorFlowType<"Uint8Ref", "ui8ref">;
def TF_Uint16Ref : TF_TensorFlowType<"Uint16Ref", "ui16ref">;
def TF_Uint32Ref : TF_TensorFlowType<"Uint32Ref", "ui32ref">;
def TF_Uint64Ref : TF_TensorFlowType<"Uint64Ref", "ui64ref">;

// Quantized reference types
def TF_Qint8Ref : TF_TensorFlowType<"Qint8Ref", "qint8ref">;
def TF_Qint16Ref : TF_TensorFlowType<"Qint16Ref", "qint16ref">;
def TF_Qint32Ref : TF_TensorFlowType<"Qint32Ref", "qint32ref">;
def TF_Quint8Ref : TF_TensorFlowType<"Quint8Ref", "quint8ref">;
def TF_Quint16Ref : TF_TensorFlowType<"Quint16Ref", "quint16ref">;

// Other reference types
def TF_BoolRef : TF_TensorFlowType<"BoolRef", "boolref">;
def TF_ResourceRef : TF_TensorFlowType<"ResourceRef", "resourceref">;
def TF_StrRef : TF_TensorFlowType<"StringRef", "stringref">;
def TF_VariantRef : TF_TensorFlowType<"VariantRef", "variantref">;

//===----------------------------------------------------------------------===//
// Integer types (including corresponding reference types)

def TF_Bool : AnyTypeOf<[I<1>, TF_BoolRef], "bool">;

def TF_Int8 : AnyTypeOf<[I8, TF_Int8Ref], "8-bit integer">;
def TF_Int16 : AnyTypeOf<[I16, TF_Int16Ref], "16-bit integer">;
def TF_Int32 : AnyTypeOf<[I32, TF_Int32Ref], "32-bit integer">;
def TF_Int64 : AnyTypeOf<[I64, TF_Int64Ref], "64-bit integer">;
def TF_I32OrI64 : AnyTypeOf<[I32, I64, TF_Int32Ref, TF_Int64Ref],
                           "32/64-bit signed integer">;

def TF_Uint8 : AnyTypeOf<[UI<8>, TF_Uint8Ref], "8-bit unsigned integer">;
def TF_Uint16 : AnyTypeOf<[UI<16>, TF_Uint16Ref], "16-bit unsigned integer">;
def TF_Uint32 : AnyTypeOf<[UI<32>, TF_Uint32Ref], "32-bit unsigned integer">;
def TF_Uint64 : AnyTypeOf<[UI<64>, TF_Uint64Ref], "64-bit unsigned integer">;

// Any unsigned integer type
def TF_UInt : AnyTypeOf<[TF_Uint8, TF_Uint16, TF_Uint32, TF_Uint64],
                        "unsigned integer">;

// Any signed integer type
def TF_SInt : AnyTypeOf<[TF_Int8, TF_Int16, TF_Int32, TF_Int64],
                        "signed integer">;

// Any integer type
def TF_Int : AnyTypeOf<[TF_SInt, TF_UInt], "integer">;

// Tensor types
def TF_BoolTensor : TensorOf<[TF_Bool]>;

def TF_IntTensor : TensorOf<[TF_Int]>;
def TF_Int8Tensor : TensorOf<[TF_Int8]>;
def TF_Int16Tensor : TensorOf<[TF_Int16]>;
def TF_Int32Tensor : TensorOf<[TF_Int32]>;
def TF_Int64Tensor : TensorOf<[TF_Int64]>;
def TF_I32OrI64Tensor : TensorOf<[TF_I32OrI64]>;

def TF_Uint8Tensor : TensorOf<[TF_Uint8]>;
def TF_Uint16Tensor : TensorOf<[TF_Uint16]>;
def TF_Uint32Tensor : TensorOf<[TF_Uint32]>;
def TF_Uint64Tensor : TensorOf<[TF_Uint64]>;

//===----------------------------------------------------------------------===//
// Quantized types (including corresponding reference types)

def TF_Qint8   : AnyTypeOf<
  [TF_TensorFlowType<"Qint8", "qint8">, TF_Qint8Ref],
  "8-bit quantized integer">;
def TF_Qint16  : AnyTypeOf<
  [TF_TensorFlowType<"Qint16", "qint16">, TF_Qint16Ref],
  "16-bit quantized integer">;
def TF_Qint32  : AnyTypeOf<
  [TF_TensorFlowType<"Qint32", "qint32">, TF_Qint32Ref],
  "32-bit quantized integer">;
def TF_Quint8  : AnyTypeOf<
  [TF_TensorFlowType<"Quint8", "quint8">, TF_Quint8Ref],
  "8-bit quantized unsigned integer">;
def TF_Quint16 : AnyTypeOf<
  [TF_TensorFlowType<"Quint16", "quint16">, TF_Quint16Ref],
  "16-bit quantized unsigned integer">;

// Any quantized type
def TF_Quantized : AnyTypeOf<
  [TF_Qint8, TF_Qint16, TF_Qint32, TF_Quint8, TF_Quint16], "quantized">;

//===----------------------------------------------------------------------===//
// Floating-point types (including corresponding reference types)

def TF_Float16 : AnyTypeOf<[F16, TF_Float16Ref], "16-bit float">;
def TF_Float32 : AnyTypeOf<[F32, TF_Float32Ref], "32-bit float">;
def TF_Float64 : AnyTypeOf<[F64, TF_Float64Ref], "64-bit float">;
def TF_Bfloat16 : AnyTypeOf<[BF16, TF_Bfloat16Ref], "bfloat16">;

def TF_F32OrF64 : AnyTypeOf<[TF_Float32, TF_Float64], "32/64-bit float">;

def TF_Float : AnyTypeOf<
  [TF_Float16, TF_Float32, TF_Float64, TF_Bfloat16],
  "floating-point">;

// Tensor types
def TF_FloatTensor : TensorOf<[TF_Float]>;
def TF_F32OrF64Tensor : TensorOf<[TF_F32OrF64]>;
def TF_Float16Tensor : TensorOf<[TF_Float16]>;
def TF_Float32Tensor : TensorOf<[TF_Float32]>;
def TF_Float64Tensor : TensorOf<[TF_Float64]>;
def TF_Bfloat16Tensor : TensorOf<[TF_Bfloat16]>;

//===----------------------------------------------------------------------===//
// Complex types (including corresponding reference types)

// TODO(suderman): Remove TF_Complex64 and use a standard ops declaration, along
// with the associated cleanup.
def TF_Complex64 : AnyTypeOf<[Complex<F<32>>, TF_Complex64Ref],
  "64-bit complex">;
def TF_Complex128 : AnyTypeOf<[Complex<F<64>>, TF_Complex128Ref],
  "128-bit complex">;
def TF_Complex : AnyTypeOf<[TF_Complex64, TF_Complex128], "complex">;

// Tensor types
def TF_ComplexTensor : TensorOf<[TF_Complex]>;
def TF_Complex64Tensor : TensorOf<[TF_Complex64]>;
def TF_Complex128Tensor : TensorOf<[TF_Complex128]>;

//===----------------------------------------------------------------------===//
// String/variant/resource types (including corresponding reference types)

def TF_Str : AnyTypeOf<
  [TF_TensorFlowType<"String", "str">, TF_StrRef], "string">;
def TF_StrTensor : TensorOf<[TF_Str]>;

def TF_Variant : AnyTypeOf<
  [TF_TensorFlowType<"Variant", "var">, TF_VariantRef], "variant">;
def TF_VariantTensor : TensorOf<[TF_Variant]>;

def TF_Resource : AnyTypeOf<
  [TF_TensorFlowType<"Resource", "res">, TF_ResourceRef], "resource">;
def TF_ResourceTensor : TensorOf<[TF_Resource]>;

//===----------------------------------------------------------------------===//
// Multi-category type constraints

def TF_IntOrF32OrF64Tensor: TensorOf<[TF_Int, TF_F32OrF64]>;
def TF_FpOrI32OrI64Tensor : TensorOf<[TF_Float, TF_I32OrI64]>;
def TF_IntOrFpTensor : TensorOf<[TF_Int, TF_Float]>;
def TF_SintOrFpTensor : TensorOf<[TF_SInt, TF_Float]>;
def TF_FpOrComplexTensor : TensorOf<[TF_Float, TF_Complex]>;

def TF_Number : AnyTypeOf<
  [TF_Int, TF_Float, TF_Quantized, TF_Complex], "number">;
def TF_NumberTensor : TensorOf<[TF_Number]>;

def TF_NumberNotQuantizedTensor : TensorOf<
  [TF_Float, TF_SInt, TF_Complex, TF_Uint8]>;

def TF_NumberNotQuantizedOrStr :
  AnyTypeOf<[TF_Float, TF_SInt, TF_Complex, TF_Uint8, TF_Str]>;
def TF_NumberNotQuantizedOrStrTensor : TensorOf<[TF_NumberNotQuantizedOrStr]>;

//===----------------------------------------------------------------------===//
// Tensor and tensor element types

// Any tensor element type allowed in TensorFlow ops
// (see https://www.tensorflow.org/api_docs/python/tf/dtypes/DType)
def TF_ElementType : Type<Or<[TF_Float.predicate,
                              TF_Complex.predicate,
                              TF_Int.predicate,
                              TF_Bool.predicate,
                              TF_TFDialectType.predicate]>,
                          "tf.dtype">;

// Any TensorFlow tensor type
def TF_Tensor : TensorOf<[TF_ElementType]>;

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
    ArrayAttr::get($_ctx, 
    [&]() {
      llvm::SmallVector<Attribute, 4> ret;
      for (auto t : $_self)
        ret.push_back(TypeAttr::get(t));
      return ret;
    }())
  }]
>;

// A derived attribute that returns the shapes of the tensors in the actual
// value pack that corresponds to the `idx`-th ODS-declared variadic operand.
// This returns a list of shapes so it is used for variadic operands that
// can have different shapes.
class TF_DerivedOperandShapeListAttr<int idx> : DerivedAttr<
  "::mlir::TF::OperandShapeRange",
  "auto values = getODSOperands(" # idx # ");\n"
  "return {mlir::TF::OperandShapeIterator(values.begin()), "
          "mlir::TF::OperandShapeIterator(values.end())};",
  [{
    ArrayAttr::get($_ctx,
      [&](){
        llvm::SmallVector<Attribute, 4> ret;
        for (auto shape : $_self)
          ret.push_back(mlir::TF::ShapeAttr::get($_ctx, shape));
        return ret;
      }())
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
    ArrayAttr::get($_ctx, 
    [&]() {
      llvm::SmallVector<Attribute, 4> ret;
      for (auto t : $_self)
        ret.push_back(TypeAttr::get(t));
      return ret;
    }())
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
    ArrayAttr::get($_ctx,
      [&](){
        llvm::SmallVector<Attribute, 4> ret;
        for (auto shape : $_self)
          ret.push_back(mlir::TF::ShapeAttr::get($_ctx, shape));
        return ret;
      }())
  }]
>;

// A derived attribute that returns the shape of the first result type.
def TF_DerivedResultShapeAttr : DerivedAttr<"ShapedType",
  "return (*getOperation()->result_type_begin()).cast<ShapedType>();",
  [{ mlir::TF::ShapeAttr::get($_ctx, $_self) }]>;

def TF_IntTypeAttr : TypeAttrBase<"IntegerType", "integer type"> {
  let returnType = "Type";
}

//===----------------------------------------------------------------------===//
// TensorFlow common builders
//===----------------------------------------------------------------------===//

// Mixin class defining a builder for binary ops supporting broadcast
// behavior. The result type has the same element type as both operands.
class WithBroadcastableBinOpBuilder {
  list<OpBuilder> builders = [
    OpBuilder<(ins "Value":$x, "Value":$y),
    [{
  auto resultType =
      OpTrait::util::getBroadcastedType(x.getType(), y.getType());
  if (!resultType)
    mlir::emitError($_state.location, "non-broadcastable operands");
  return build($_builder, $_state, resultType, x, y);
}]>];
}

// Mixin class defining a builder for comparison ops supporting broadcast
// behavior. The result type has bool element type.
class WithBroadcastableCmpOpBuilder {
  list<OpBuilder> builders = [
    OpBuilder<(ins "Value":$x, "Value":$y),
    [{
  Type resultType;
  if (x.getType().isa<UnrankedTensorType>() ||
      y.getType().isa<UnrankedTensorType>()) {
    resultType = UnrankedTensorType::get($_builder.getI1Type());
  } else {
    SmallVector<int64_t, 4> resultShape;
    if (!OpTrait::util::getBroadcastedShape(
            x.getType().cast<ShapedType>().getShape(),
            y.getType().cast<ShapedType>().getShape(), resultShape)) {
      mlir::emitError($_state.location,
                      "operands have no broadcastable shapes");
    }

    resultType = RankedTensorType::get(resultShape, $_builder.getI1Type());
  }
  return build($_builder, $_state, resultType, x, y);
}]>];
}

#endif // TF_OP_BASE
