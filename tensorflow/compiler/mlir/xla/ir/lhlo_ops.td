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

// This is the operation definition file for LXLA.

#ifndef LHLO_OPS
#define LHLO_OPS

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/ViewLikeInterface.td"
include "tensorflow/compiler/mlir/xla/ir/hlo_ops_base.td"

def LHLO_Dialect : Dialect {
  let name = "xla_lhlo";
  let cppNamespace = "xla_lhlo";
}

//===----------------------------------------------------------------------===//
// XLA type definitions.
//===----------------------------------------------------------------------===//

// Any integer tensor types
def LHLO_IntBuffer : MemRefOf<[HLO_Int]>;

// Any floating-point tensor types
def LHLO_FpBuffer : MemRefOf<[AnyFloat]>;

def LHLO_PredBuffer : MemRefOf<[HLO_Pred]>;

// Any integer or floating-point tensor types
def LHLO_IntOrFpBuffer : MemRefOf<[HLO_Int, AnyFloat]>;

def LHLO_Buffer : MemRefOf<[AnyFloat, AnySignlessInteger, AnyComplex]>;

def LHLO_TupleBuffer : NestedTupleOf<[LHLO_Buffer]>;

def LHLO_BufferOrTuple : AnyTypeOf<[LHLO_Buffer, LHLO_TupleBuffer]>;

//===----------------------------------------------------------------------===//
// XLA nullary op definitions.
//===----------------------------------------------------------------------===//

class LHLO_Op<string mnemonic, list<OpTrait> traits> :
  Op<LHLO_Dialect, mnemonic,
    !listconcat([MemoryEffects<[MemRead, MemWrite]>], traits)>;

def LHLO_ConstOp : LHLO_Op<"constant", []>, BASE_HLO_ConstOp {
  let arguments = (ins
    ElementsAttr:$value,
    Arg<LHLO_Buffer, "", [MemWrite]>:$output
  );
}

def LHLO_IotaOp : LHLO_Op<"iota", []>, BASE_HLO_IotaOp {
  let arguments = (ins I64Attr:$iota_dimension,
                   Arg<LHLO_Buffer, "", [MemWrite]>:$output);
}

//===----------------------------------------------------------------------===//
// XLA unary elementwise op definitions.
//===----------------------------------------------------------------------===//
// See https://www.tensorflow.org/xla/operation_semantics#element-wise_unary_functions

class LHLO_UnaryElementwiseOp<string mnemonic> :
    LHLO_Op<mnemonic, [SameTypeOperands]> {
  let arguments = (ins Arg<LHLO_Buffer, "", [MemRead]>:$input,
                       Arg<LHLO_Buffer, "", [MemWrite]>:$output);
}

def LHLO_AbsOp: LHLO_UnaryElementwiseOp<"abs">, BASE_HLO_AbsOp;

def LHLO_CeilOp: LHLO_UnaryElementwiseOp<"ceil">, BASE_HLO_CeilOp;

def LHLO_ConvertOp :  LHLO_Op<"convert", [SameOperandsShape]>, BASE_HLO_ConvertOp {
  let arguments = (ins Arg<LHLO_Buffer, "", [MemRead]>:$input,
                       Arg<LHLO_Buffer, "", [MemWrite]>:$output);
}

def LHLO_CosOp: LHLO_UnaryElementwiseOp<"cosine">, BASE_HLO_CosOp;

def LHLO_ExpOp: LHLO_UnaryElementwiseOp<"exponential">, BASE_HLO_ExpOp;

def LHLO_ImagOp: LHLO_Op<"imag", [SameOperandsShape]>, BASE_HLO_ImagOp {
  let arguments = (ins Arg<LHLO_Buffer, "", [MemRead]>:$input,
                       Arg<LHLO_Buffer, "", [MemWrite]>:$output);
}

def LHLO_LogOp: LHLO_UnaryElementwiseOp<"log">, BASE_HLO_LogOp;

def LHLO_NegOp: LHLO_UnaryElementwiseOp<"negate">, BASE_HLO_NegOp;

def LHLO_RealOp: LHLO_Op<"real", [SameOperandsShape]>, BASE_HLO_RealOp {
  let arguments = (ins Arg<LHLO_Buffer, "", [MemRead]>:$input,
                       Arg<LHLO_Buffer, "", [MemWrite]>:$output);
}

def LHLO_RsqrtOp: LHLO_UnaryElementwiseOp<"rsqrt">, BASE_HLO_RsqrtOp;

def LHLO_SqrtOp: LHLO_UnaryElementwiseOp<"sqrt">, BASE_HLO_SqrtOp;

def LHLO_SignOp: LHLO_UnaryElementwiseOp<"sign">, BASE_HLO_SignOp;

def LHLO_SinOp: LHLO_UnaryElementwiseOp<"sine">, BASE_HLO_SinOp;

def LHLO_TanhOp: LHLO_UnaryElementwiseOp<"tanh">, BASE_HLO_TanhOp;

//===----------------------------------------------------------------------===//
// XLA binary elementwise op definitions.
//===----------------------------------------------------------------------===//
// See https://www.tensorflow.org/xla/operation_semantics#element-wise_binary_arithmetic_operations

class LHLO_BinaryElementwiseOp<string mnemonic, list<OpTrait> traits> :
        LHLO_Op<mnemonic, traits> {
  let arguments = (ins
      Arg<LHLO_Buffer, "", [MemRead]>:$lhs,
      Arg<LHLO_Buffer, "", [MemRead]>:$rhs,
      Arg<LHLO_Buffer, "", [MemWrite]>:$out,
      OptionalAttr<BroadcastDimAttr>:$broadcast_dimensions
  );
}

def LHLO_AddOp : LHLO_BinaryElementwiseOp<"add", []>, BASE_HLO_AddOp;

def LHLO_ComplexOp: LHLO_Op<"complex", [SameOperandsShape]>, BASE_HLO_ComplexOp {
  let arguments = (ins Arg<LHLO_Buffer, "", [MemRead]>:$lhs,
                       Arg<LHLO_Buffer, "", [MemRead]>:$rhs,
                       Arg<LHLO_Buffer, "", [MemWrite]>:$output);
}

def LHLO_DivOp : LHLO_BinaryElementwiseOp<"divide", []>, BASE_HLO_DivOp;

def LHLO_MaxOp : LHLO_BinaryElementwiseOp<"maximum", []>, BASE_HLO_MaxOp;

def LHLO_MinOp : LHLO_BinaryElementwiseOp<"minimum", []>, BASE_HLO_MinOp;

def LHLO_MulOp : LHLO_BinaryElementwiseOp<"multiply", []>, BASE_HLO_MulOp;

def LHLO_RemOp :
      LHLO_BinaryElementwiseOp<"remainder", []>, BASE_HLO_RemOp;

def LHLO_SubOp : LHLO_BinaryElementwiseOp<"subtract", []>, BASE_HLO_SubOp;

def LHLO_AndOp: LHLO_BinaryElementwiseOp<"and", []>, BASE_HLO_AndOp;

def LHLO_OrOp: LHLO_BinaryElementwiseOp<"or", []>, BASE_HLO_OrOp;

//===----------------------------------------------------------------------===//
// XLA control flow op definitions.
//===----------------------------------------------------------------------===//

// TODO(b/139813999): specify required function signature in a type-safe way.
def LHLO_ReduceOp: LHLO_Op<"reduce", [
      SameVariadicOperandSize,
      SingleBlockImplicitTerminator<"TerminatorOp">
    ]>, BASE_HLO_ReduceOp {
  let arguments = (ins
    Arg<Variadic<LHLO_BufferOrTuple>, "", [MemRead]>:$operands,
    Arg<Variadic<LHLO_BufferOrTuple>, "", [MemRead]>:$init_values,
    Arg<Variadic<LHLO_BufferOrTuple>, "", [MemWrite]>:$out,
    I64ElementsAttr:$dimensions
  );

  let regions = (region SizedRegion<1>:$body);
}

def LHLO_ReduceWindowOp: LHLO_Op<"reduce_window", [
      SingleBlockImplicitTerminator<"TerminatorOp">
    ]>, BASE_HLO_ReduceWindowOp {

  let arguments = (ins
    Arg<LHLO_Buffer, "", [MemRead]>:$operand,
    Arg<LHLO_Buffer, "", [MemRead]>:$init_value,
    Arg<LHLO_Buffer, "", [MemWrite]>:$out,
    I64ElementsAttr:$window_dimensions,
    // If strides or dilations attributes are missing then the default value is
    // one for each of the input dimensions. Similarly, padding values are zero
    // for both low and high in each of the dimensions, if not specified.
    OptionalAttr<I64ElementsAttr>:$window_strides,
    OptionalAttr<I64ElementsAttr>:$base_dilations,
    OptionalAttr<I64ElementsAttr>:$window_dilations,
    OptionalAttr<I64ElementsAttr>:$padding
  );

  let regions = (region SizedRegion<1>:$body);
}

def LHLO_CaseOp: LHLO_Op<"case", [
      SingleBlockImplicitTerminator<"TerminatorOp">
    ]>, BASE_HLO_CaseOp {

  let arguments = (ins
    Arg<LHLO_Buffer, "", [MemRead]>:$index,
    Arg<Variadic<LHLO_BufferOrTuple>, "", [MemRead]>:$branch_operands,
    Arg<LHLO_BufferOrTuple, "", [MemWrite]>:$out
  );

  let regions = (region VariadicRegion<SizedRegion<1>>:$branches);
}

//===----------------------------------------------------------------------===//
// XLA tuple op definitions.
//===----------------------------------------------------------------------===//

def LHLO_GetTupleElementOp: LHLO_Op<"get_tuple_element", []>, BASE_HLO_GetTupleElementOp {
  let arguments = (ins
    Arg<LHLO_TupleBuffer, "", [MemRead]>:$input,
    Arg<LHLO_BufferOrTuple, "", [MemWrite]>:$out,
    I32Attr:$index
  );
}

def LHLO_TupleOp : LHLO_Op<"tuple", []>, BASE_HLO_TupleOp {
   let arguments = (ins
     Arg<Variadic<LHLO_BufferOrTuple>, "", [MemRead]>:$val,
     Arg<LHLO_TupleBuffer, "", [MemWrite]>:$out);
}

def LHLO_CompareOp: LHLO_Op<"compare", []>, BASE_HLO_CompareOp {
  let arguments = (ins
    Arg<LHLO_Buffer, "", [MemRead]>:$lhs,
    Arg<LHLO_Buffer, "", [MemRead]>:$rhs,
    Arg<LHLO_PredBuffer, "", [MemWrite]>:$out,
    OptionalAttr<BroadcastDimAttr>:$broadcast_dimensions,
    HLO_ComparisonDirectionAttr:$comparison_direction
  );
}

//===----------------------------------------------------------------------===//
// XLA Slice definitions.
//===----------------------------------------------------------------------===//

def LHLO_SliceOp: LHLO_Op<
      "slice",
      [AllTypesMatch<["start_indices", "limit_indices", "strides"]>]> {
  let arguments = (ins
    Arg<LHLO_Buffer, "", [MemRead]>:$operand,
    Arg<LHLO_Buffer, "", [MemWrite]>:$output,
    I64ElementsAttr:$start_indices,
    I64ElementsAttr:$limit_indices,
    I64ElementsAttr:$strides
  );
}

def HLO_DynamicUpdateSliceOp: LHLO_Op<"dynamic-update-slice", []> {
  let arguments = (ins
    Arg<LHLO_Buffer, "", [MemRead]>:$operand,
    Arg<LHLO_Buffer, "", [MemRead]>:$update,
    Arg<LHLO_Buffer, "", [MemWrite]>:$output,
    Arg<Variadic<LHLO_Buffer>, "", [MemRead]>:$start_indices
  );
}

//===----------------------------------------------------------------------===//
// StaticMemRefCastOp
//===----------------------------------------------------------------------===//

def HLO_StaticMemRefCastOp: Op<LHLO_Dialect, "static_memref_cast",
    [NoSideEffect, DeclareOpInterfaceMethods<ViewLikeOpInterface>]> {
  let summary = [{
    "modifies the offset, sizes and strides of a statically shaped memref.
  }];
  let description = [{
    Allows to modify the offset, sizes and strides of a statically shaped memref.

    Example:
    ```mlir
    %buf_transformed =
        xla_lhlo.static_memref_cast %buf
        : memref<1x5xf32> -> memref<5xf32, offset: 2, strides: [1]>

    // The result of the op is a rank-1 memref with `[5]` shape, stride 1 and
    // offset 2.
    ```
  }];

  let arguments = (ins Arg<LHLO_Buffer, "", []>:$operand);
  let results = (outs Res<LHLO_Buffer, "", []>:$result);

  let builders = [OpBuilder<
    "OpBuilder &builder, OperationState &result, MemRefType resultType, " #
    "Value operand", [{
       result.addOperands(operand);
       result.types.push_back(resultType);
     }]>];

  let extraClassDeclaration = [{
    MemRefType getType() { return getResult().getType().cast<MemRefType>(); }
  }];

  let verifier = [{ return Verify(*this); }];
  let assemblyFormat = [{
    $operand attr-dict `:` type($operand) `->` type($result)
  }];
}

//===----------------------------------------------------------------------===//
// DynamicMemRefCastOp
//===----------------------------------------------------------------------===//

def HLO_DynamicMemRefCastOp: Op<LHLO_Dialect, "dynamic_memref_cast",
    [SameVariadicOperandSize, NoSideEffect,
     DeclareOpInterfaceMethods<ViewLikeOpInterface>]> {
  let summary = "dynamic memref cast operation";
  let description = [{
    Change sizes and strides of a memref using the values computed in runtime.

    Example:
    ```mlir
    %buf_transformed =
        xla_lhlo.dynamic_memref_cast %buf(%size_X, %size_Y)[%step_X, %step_Y]
        : memref<?x?xf32> -> memref<?x?xf32, offset: 0, strides: [?, ?]>
    // The result of the op is a type-erased memref with `[%size_X, %size_Y]`
    // shape and `[%step_X, %step_Y]` strides. The offset will be inherited
    // from the input.
    ```
  }];

  let arguments = (ins
    Arg<LHLO_Buffer, "", []>:$operand,
    Variadic<Index>:$sizes,
    Variadic<Index>:$strides
  );
  let results = (outs Res<LHLO_Buffer, "", []>:$result);

  let builders = [OpBuilder<
    "OpBuilder &builder, OperationState &result, MemRefType resultType, " #
    "Value operand, ValueRange sizes, ValueRange strides", [{
       result.addOperands(operand);
       result.addOperands(sizes);
       result.addOperands(strides);
       result.types.push_back(resultType);
     }]>];

  let extraClassDeclaration = [{
    MemRefType getType() { return getResult().getType().cast<MemRefType>(); }
  }];

  let verifier = [{ return Verify(*this); }];
  let assemblyFormat = [{
    $operand `(` $sizes `)` `[` $strides `]` attr-dict `:` type($operand) `->`
    type($result)
  }];
}

//===----------------------------------------------------------------------===//
// XLA Other op definitions.
//===----------------------------------------------------------------------===//

def HLO_BatchNormInferenceOp : LHLO_Op<"batch_norm_inference", []>,
    BASE_HLO_BatchNormInferenceOp {

  let arguments = (ins
    Arg<LHLO_Buffer, "", [MemRead]>:$operand,
    Arg<LHLO_Buffer, "", [MemRead]>:$scale,
    Arg<LHLO_Buffer, "", [MemRead]>:$offset,
    Arg<LHLO_Buffer, "", [MemRead]>:$mean,
    Arg<LHLO_Buffer, "", [MemRead]>:$variance,
    Arg<LHLO_Buffer, "", [MemWrite]>:$output,
    F32Attr:$epsilon,
    I64Attr:$feature_index
  );
}

def LHLO_BroadcastOp : LHLO_Op<"broadcast",
      []>, BASE_HLO_BroadcastOp {
  let arguments = (ins
    Arg<LHLO_Buffer, "", [MemRead]>:$operand,
    Arg<LHLO_Buffer, "", [MemWrite]>:$output,
    I64ElementsAttr:$broadcast_sizes
  );
}

def LHLO_BroadcastInDimOp : LHLO_Op<"broadcast_in_dim",
      []>, BASE_HLO_BroadcastInDimOp {
  let arguments = (ins
    Arg<LHLO_Buffer, "", [MemRead]>:$operand,
    Arg<LHLO_Buffer, "", [MemWrite]>:$output,
    BroadcastDimAttr:$broadcast_dimensions
  );
}

def LHLO_ClampOp : LHLO_Op<"clamp", []>, BASE_HLO_ClampOp {
  let arguments = (ins
    Arg<LHLO_Buffer, "", [MemRead]>:$min,
    Arg<LHLO_Buffer, "", [MemRead]>:$operand,
    Arg<LHLO_Buffer, "", [MemRead]>:$max,
    Arg<LHLO_Buffer, "", [MemWrite]>:$output
  );
}

def LHLO_ConcatenateOp : LHLO_Op<"concatenate", []>, BASE_HLO_ConcatenateOp {
   let arguments = (ins
     Arg<Variadic<LHLO_Buffer>, "", [MemRead]>:$val,
     Arg<LHLO_Buffer, "", [MemWrite]>:$output,
     I64Attr:$dimension
   );
}

// TODO(bondhugula): Make this struct dialect independent so that it can be
// shared between the HLO and LHLO dialects.
def ConvDimensionNumbers : StructAttr<"ConvDimensionNumbers", LHLO_Dialect, [
  StructFieldAttr<"input_batch_dimension",I64Attr>,
  StructFieldAttr<"input_feature_dimension", I64Attr>,
  StructFieldAttr<"input_spatial_dimensions", I64ElementsAttr>,
  StructFieldAttr<"kernel_input_feature_dimension", I64Attr>,
  StructFieldAttr<"kernel_output_feature_dimension", I64Attr>,
  StructFieldAttr<"kernel_spatial_dimensions", I64ElementsAttr>,
  StructFieldAttr<"output_batch_dimension", I64Attr>,
  StructFieldAttr<"output_feature_dimension", I64Attr>,
  StructFieldAttr<"output_spatial_dimensions", I64ElementsAttr>] > {

  let description = "Structure of dimension information for conv op";
}

def LHLO_ConvOp : LHLO_Op<"convolution", []>, BASE_HLO_ConvOp {
  let arguments = (ins
    Arg<LHLO_Buffer, "", [MemRead]>:$lhs,
    Arg<LHLO_Buffer, "", [MemRead]>:$rhs,
    Arg<LHLO_Buffer, "", [MemWrite]>:$output,
    // Default value: one for each of the spatial dimension.
    OptionalAttr<I64ElementsAttr>:$window_strides,
    // Default value: zero for each of the spatial dimension.
    OptionalAttr<I64ElementsAttr>:$padding,
    // Default value: one for each of the spatial dimension.
    OptionalAttr<I64ElementsAttr>:$lhs_dilation,
    // Default value: one for each of the spatial dimension.
    OptionalAttr<I64ElementsAttr>:$rhs_dilation,
    ConvDimensionNumbers:$dimension_numbers,
    I64Attr:$feature_group_count,
    I64Attr:$batch_group_count,
    HLO_PrecisionConfigAttr:$precision_config
  );
}

def LHLO_CopyOp: LHLO_Op<"copy", []>, BASE_HLO_CopyOp {
  let arguments = (ins
    Arg<LHLO_Buffer, "", [MemRead]>:$operand,
    Arg<LHLO_Buffer, "", [MemWrite]>:$output
  );
}

def LHLO_DotOp: LHLO_Op<"dot", []>, BASE_HLO_DotOp {
  let arguments = (ins
    Arg<LHLO_Buffer, "", [MemRead]>:$lhs,
    Arg<LHLO_Buffer, "", [MemRead]>:$rhs,
    HLO_PrecisionConfigAttr:$precision_config,
    Arg<LHLO_Buffer, "", [MemWrite]>:$output
  );
}

def LHLO_GatherOp: LHLO_Op<"gather", []>, BASE_HLO_GatherOp {
  let arguments = (ins
    Arg<LHLO_Buffer, "", [MemRead]>:$operand,
    Arg<LHLO_IntBuffer, "", [MemRead]>:$start_indices,
    I64Attr:$index_vector_dim,
    I64ElementsAttr:$offset_dims,
    I64ElementsAttr:$slice_sizes,
    I64ElementsAttr:$collapsed_slice_dims,
    I64ElementsAttr:$start_index_map,
    Arg<LHLO_Buffer, "", [MemWrite]>:$output
  );
}

def LHLO_ReshapeOp: LHLO_Op<"reshape", []>, BASE_HLO_ReshapeOp {
  let arguments = (ins
    Arg<LHLO_Buffer, "", [MemRead]>:$operand,
    Arg<LHLO_Buffer, "", [MemWrite]>:$output
  );
}


def LHLO_SelectOp: LHLO_Op<"select", []>, BASE_HLO_SelectOp {
  let arguments = (ins
    Arg<LHLO_PredBuffer, "", [MemRead]>:$pred,
    Arg<LHLO_Buffer, "", [MemRead]>:$on_true,
    Arg<LHLO_Buffer, "", [MemRead]>:$on_false,
    Arg<LHLO_Buffer, "", [MemWrite]>:$output
  );
}

def LHLO_SelectAndScatterOp: LHLO_Op<"select_and_scatter", []>,
      BASE_HLO_SelectAndScatterOp {
  let arguments = (ins
    Arg<LHLO_Buffer, "", [MemRead]>:$operand,
    Arg<LHLO_Buffer, "", [MemRead]>:$source,
    Arg<LHLO_Buffer, "", [MemRead]>:$init_value,
    Arg<LHLO_Buffer, "", [MemWrite]>:$out,
    OptionalAttr<I64ElementsAttr>:$window_dimensions,
    OptionalAttr<I64ElementsAttr>:$window_strides,
    OptionalAttr<I64ElementsAttr>:$padding
  );

  let regions = (region SizedRegion<1>:$select, SizedRegion<1>:$scatter);
}

def LHLO_ReverseOp: LHLO_Op<"reverse", []>, BASE_HLO_ReverseOp {
  let arguments = (ins
    Arg<LHLO_Buffer, "", [MemRead]>:$operand,
    I64ElementsAttr:$dimensions,
    Arg<LHLO_Buffer, "", [MemWrite]>:$output
  );
}

def LHLO_PadOp: LHLO_Op<"pad", []>, BASE_HLO_PadOp {
  let arguments = (ins
    Arg<LHLO_Buffer, "", [MemRead]>:$operand,
    Arg<LHLO_Buffer, "", [MemRead]>:$padding_value,
    I64ElementsAttr:$edge_padding_low,
    I64ElementsAttr:$edge_padding_high,
    I64ElementsAttr:$interior_padding,
    Arg<LHLO_Buffer, "", [MemWrite]>:$output
  );
}

def LHLO_TransposeOp: LHLO_Op<"transpose", []>, BASE_HLO_TransposeOp {
  let arguments = (ins
    Arg<LHLO_Buffer, "", [MemRead]>:$operand,
    I64ElementsAttr:$permutation,
    Arg<LHLO_Buffer, "", [MemWrite]>:$output
  );
}

//===----------------------------------------------------------------------===//
// Late operations
//===----------------------------------------------------------------------===//

def FusionOp : LHLO_Op<"fusion", [SingleBlockImplicitTerminator<"TerminatorOp">]> {
  let summary = "Fusion operator";
  let description = [{
    Models the fusion instruction generated by the XLA compiler's fusion pass.

    Fusion instructions are generated by the fusion pass of the XLA compiler.
    They serve as a hint to the backend that it is beneficial to emit the
    contained instructions into a single loop nest or kernel. The XLA fusion
    pass is designed such that it only generates fusion nodes that can be
    handled by the XLA compilers backends.
    The XLA runtime expects this hint to be followed, as it expects a single
    kernel per HLO instruction. This restriction might be lifted in the future.
  }];
  let regions = (region SizedRegion<1>:$region);

  let skipDefaultBuilders = 1;
  let builders = [
     OpBuilder<"OpBuilder &builder, OperationState &result, "
               "ArrayRef<NamedAttribute> attributes">
   ];
}

def TerminatorOp :
    LHLO_Op<"terminator", [Terminator]> {
  let summary = "LHLO termination operation";
  let description = [{
    Terminator operation for the LHLO dialect.
  }];
  let builders = [OpBuilder<
    "OpBuilder &b, OperationState &result, ValueRange operands",
    [{ build(b, result, llvm::None, operands, llvm::None); }]
  >];
}

#endif // LHLO_OPS
