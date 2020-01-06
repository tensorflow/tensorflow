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

def LHLO_Buffer : MemRefOf<[AnyFloat, AnyInteger]>;

def LHLO_TupleBuffer : NestedTupleOf<[LHLO_Buffer]>;

def LHLO_BufferOrTuple : AnyTypeOf<[LHLO_Buffer, LHLO_TupleBuffer]>;

//===----------------------------------------------------------------------===//
// XLA nullary op definitions.
//===----------------------------------------------------------------------===//

class LHLO_Op<string mnemonic, list<OpTrait> traits> : Op<LHLO_Dialect,
                                                          mnemonic, traits>;

def LHLO_ConstOp : BASE_HLO_ConstOp, LHLO_Op<"constant", []> {
  let arguments = (ins
    ElementsAttr:$value,
    LHLO_Buffer:$output
  );
}

def LHLO_IotaOp : BASE_HLO_IotaOp, LHLO_Op<"iota", []> {
  let arguments = (ins I64Attr:$iota_dimension,
                   LHLO_Buffer:$output);
}

//===----------------------------------------------------------------------===//
// XLA unary elementwise op definitions.
//===----------------------------------------------------------------------===//
// See https://www.tensorflow.org/xla/operation_semantics#element-wise_unary_functions

class LHLO_UnaryElementwiseOp<string mnemonic> :
    LHLO_Op<mnemonic, [SameTypeOperands]> {
  let arguments = (ins LHLO_Buffer:$input,
                       LHLO_Buffer:$output);
}

def LHLO_AbsOp: LHLO_UnaryElementwiseOp<"abs">, BASE_HLO_AbsOp;

def LHLO_CeilOp: LHLO_UnaryElementwiseOp<"ceil">, BASE_HLO_CeilOp;

def LHLO_ConvertOp : LHLO_UnaryElementwiseOp<"convert">, BASE_HLO_ConvertOp;

def LHLO_CosOp: LHLO_UnaryElementwiseOp<"cos">, BASE_HLO_CosOp;

def LHLO_ExpOp: LHLO_UnaryElementwiseOp<"exp">, BASE_HLO_ExpOp;

def LHLO_NegOp: LHLO_UnaryElementwiseOp<"neg">, BASE_HLO_NegOp;

def LHLO_SignOp: LHLO_UnaryElementwiseOp<"sign">, BASE_HLO_SignOp;

def LHLO_TanhOp: LHLO_UnaryElementwiseOp<"tanh">, BASE_HLO_TanhOp;

//===----------------------------------------------------------------------===//
// XLA binary elementwise op definitions.
//===----------------------------------------------------------------------===//

class LHLO_BinaryElementwiseOp<string mnemonic, list<OpTrait> traits> :
        LHLO_Op<mnemonic, traits> {
  let arguments = (ins
      LHLO_Buffer:$lhs,
      LHLO_Buffer:$rhs,
      LHLO_Buffer:$out,
      BroadcastDimAttr:$broadcast_dimensions
  );
}

def LHLO_AddOp : LHLO_BinaryElementwiseOp<"add", []>, BASE_HLO_AddOp;

def LHLO_DivOp : LHLO_BinaryElementwiseOp<"div", []>, BASE_HLO_DivOp;

def LHLO_MaxOp : LHLO_BinaryElementwiseOp<"max", []>, BASE_HLO_MaxOp;

def LHLO_MinOp : LHLO_BinaryElementwiseOp<"min", []>, BASE_HLO_MinOp;

def LHLO_MulOp : LHLO_BinaryElementwiseOp<"mul", []>, BASE_HLO_MulOp;

def LHLO_RemOp :
      LHLO_BinaryElementwiseOp<"remainder", []>, BASE_HLO_RemOp;

def LHLO_SubOp : LHLO_BinaryElementwiseOp<"sub", []>, BASE_HLO_SubOp;

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
    Variadic<LHLO_BufferOrTuple>:$operands,
    Variadic<LHLO_BufferOrTuple>:$init_values,
    Variadic<LHLO_BufferOrTuple>:$out,
    I64ElementsAttr:$dimensions
  );

  let regions = (region SizedRegion<1>:$body);
}
//===----------------------------------------------------------------------===//
// XLA tuple op definitions.
//===----------------------------------------------------------------------===//

def LHLO_GetTupleElementOp: LHLO_Op<"get_tuple_element", []>, BASE_HLO_GetTupleElementOp {
  let arguments = (ins
    LHLO_TupleBuffer:$input,
    LHLO_BufferOrTuple:$out,
    I32Attr:$index
  );
}

def LHLO_TupleOp : LHLO_Op<"tuple", []>, BASE_HLO_TupleOp {
   let arguments = (ins
     Variadic<LHLO_BufferOrTuple>:$val,
                 LHLO_TupleBuffer:$out);
}

def LHLO_CompareOp: LHLO_Op<"compare", []>, BASE_HLO_CompareOp {
  let arguments = (ins
    LHLO_Buffer:$lhs,
    LHLO_Buffer:$rhs,
    LHLO_PredBuffer:$out,
    BroadcastDimAttr:$broadcast_dimensions,
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
    LHLO_Buffer:$operand,
    LHLO_Buffer:$output,
    I64ElementsAttr:$start_indices,
    I64ElementsAttr:$limit_indices,
    I64ElementsAttr:$strides
  );
}

def HLO_DynamicUpdateSliceOp: LHLO_Op<"dynamic-update-slice", []> {
  let arguments = (ins
    LHLO_Buffer:$operand,
    LHLO_Buffer:$update,
    LHLO_Buffer:$output,
    Variadic<LHLO_Buffer>:$start_indices
  );
}

//===----------------------------------------------------------------------===//
// XLA Other op definitions.
//===----------------------------------------------------------------------===//

def HLO_BatchNormInferenceOp : LHLO_Op<"batch_norm_inference", []>,
    BASE_HLO_BatchNormInferenceOp {

  let arguments = (ins
    LHLO_Buffer:$operand,
    LHLO_Buffer:$scale,
    LHLO_Buffer:$offset,
    LHLO_Buffer:$mean,
    LHLO_Buffer:$variance,
    LHLO_Buffer:$output,
    F32Attr:$epsilon,
    I64Attr:$feature_index
  );
}

def LHLO_BroadcastOp : LHLO_Op<"broadcast",
      []>, BASE_HLO_BroadcastOp {
  let arguments = (ins
    LHLO_Buffer:$operand,
    LHLO_Buffer:$output,
    I64ElementsAttr:$broadcast_sizes
  );
}

def LHLO_BroadcastInDimOp : LHLO_Op<"broadcast_in_dim",
      []>, BASE_HLO_BroadcastInDimOp {
  let arguments = (ins
    LHLO_Buffer:$operand,
    LHLO_Buffer:$output,
    BroadcastDimAttr:$broadcast_dimensions
  );
}

def LHLO_ClampOp : LHLO_Op<"clamp", []>, BASE_HLO_ClampOp {
  let arguments = (ins
    LHLO_Buffer:$min,
    LHLO_Buffer:$operand,
    LHLO_Buffer:$max,
    LHLO_Buffer:$output
  );
}

def LHLO_ConcatenateOp : LHLO_Op<"concatenate", []>, BASE_HLO_ConcatenateOp {
   let arguments = (ins
     Variadic<LHLO_Buffer>:$val,
     LHLO_Buffer:$output,
     I64Attr:$dimension
   );
}

def LHLO_ConvOp : LHLO_Op<"conv", []>, BASE_HLO_ConvOp {
  let arguments = (ins
    LHLO_Buffer:$lhs,
    LHLO_Buffer:$rhs,
    LHLO_Buffer:$output
  );
}

def LHLO_DotOp: LHLO_Op<"dot", []>, BASE_HLO_DotOp {
  let arguments = (ins
    LHLO_Buffer:$lhs,
    LHLO_Buffer:$rhs,
    HLO_PrecisionConfigAttr:$precision_config,
    LHLO_Buffer:$output
  );
}

def LHLO_GatherOp: LHLO_Op<"gather", []>, BASE_HLO_GatherOp {
  let arguments = (ins
    LHLO_Buffer:$operand,
    LHLO_IntBuffer:$start_indices,
    I64Attr:$index_vector_dim,
    I64ElementsAttr:$offset_dims,
    I64ElementsAttr:$slice_sizes,
    I64ElementsAttr:$collapsed_slice_dims,
    I64ElementsAttr:$start_index_map,
    LHLO_Buffer:$output
  );
}

def LHLO_ReshapeOp: LHLO_Op<"reshape", []>, BASE_HLO_ReshapeOp {
  let arguments = (ins
    LHLO_Buffer:$operand,
    LHLO_Buffer:$output
  );
}


def LHLO_SelectOp: LHLO_Op<"select", []>, BASE_HLO_SelectOp {
  let arguments = (ins
    LHLO_PredBuffer:$pred,
    LHLO_Buffer:$on_true,
    LHLO_Buffer:$on_false,
    LHLO_Buffer:$output
  );
}

def LHLO_ReverseOp: LHLO_Op<"reverse", []>, BASE_HLO_ReverseOp {
  let arguments = (ins
    LHLO_Buffer:$operand,
    I64ElementsAttr:$dimensions,
    LHLO_Buffer:$output
  );
}

def LHLO_PadOp: LHLO_Op<"pad", []>, BASE_HLO_PadOp {
  let arguments = (ins
    LHLO_Buffer:$operand,
    LHLO_Buffer:$padding_value,
    I64ElementsAttr:$edge_padding_low,
    I64ElementsAttr:$edge_padding_high,
    I64ElementsAttr:$interior_padding,
    LHLO_Buffer:$output
  );
}

def LHLO_TransposeOp: LHLO_Op<"transpose", []>, BASE_HLO_TransposeOp {
  let arguments = (ins
    LHLO_Buffer:$operand,
    I64ElementsAttr:$permutation,
    LHLO_Buffer:$output
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
     OpBuilder<"Builder *builder, OperationState &result, "
               "ArrayRef<NamedAttribute> attributes">
   ];
}

def TerminatorOp :
    LHLO_Op<"terminator", [Terminator]> {
  let summary = "LHLO termination operation";
  let description = [{
    Terminator operation for the LHLO dialect.
  }];
}

#endif // LHLO_OPS
