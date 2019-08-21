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

#ifdef LXLA_OPS
#else
#define LXLA_OPS

#ifdef OP_BASE
#else
include "mlir/IR/OpBase.td"
#endif // OP_BASE

#ifdef XLA_OPS_BASE
#else
include "tensorflow/compiler/mlir/xla/ir/xla_ops_base.td"
#endif

def LXLA_Dialect : Dialect {
  let name = "lxla_hlo";
  let cppNamespace = "LXLA";
}

//===----------------------------------------------------------------------===//
// XLA type definitions.
//===----------------------------------------------------------------------===//

// Any integer tensor types
def LXLA_IntBuffer : StaticShapeMemRefOf<[XLA_Int]>;

// Any floating-point tensor types
def LXLA_FpBuffer : StaticShapeMemRefOf<[AnyFloat]>;


def LXLA_PredBuffer : StaticShapeMemRefOf<[XLA_Pred]>;

// Any integer or floating-point tensor types
def LXLA_IntOrFpBuffer : StaticShapeMemRefOf<[XLA_Int, AnyFloat]>;

def LXLA_Buffer : StaticShapeMemRefOf<[AnyFloat, AnyInteger]>;

def LXLA_TupleBuffer : NestedTupleOf<[LXLA_Buffer]>;

def LXLA_BufferOrTuple : AnyTypeOf<[LXLA_Buffer, LXLA_TupleBuffer]>;

//===----------------------------------------------------------------------===//
// XLA nullary op definitions.
//===----------------------------------------------------------------------===//

class LXLA_Op<string mnemonic, list<OpTrait> traits> : Op<LXLA_Dialect,
                                                          mnemonic, traits>;

def LXLA_ConstOp : BXLA_ConstOp, LXLA_Op<"constant", []> {
  let arguments = (ins
    ElementsAttr:$value,
    LXLA_Buffer:$output
  );
}

def LXLA_IotaOp : BXLA_IotaOp, LXLA_Op<"iota", []> {
  let arguments = (ins I64Attr:$iota_dimension,
                   LXLA_Buffer:$output);
}

//===----------------------------------------------------------------------===//
// XLA unary elementwise op definitions.
//===----------------------------------------------------------------------===//
// See https://www.tensorflow.org/xla/operation_semantics#element-wise_unary_functions

class LXLA_UnaryElementwiseOp<string mnemonic> :
    LXLA_Op<mnemonic, [SameTypeOperands]> {
  let arguments = (ins LXLA_Buffer:$input,
                       LXLA_Buffer:$output);
}

def LXLA_AbsOp: LXLA_UnaryElementwiseOp<"abs">, BXLA_AbsOp;

def LXLA_ConvertOp : LXLA_UnaryElementwiseOp<"convert">, BXLA_ConvertOp;

def LXLA_ExpOp: LXLA_UnaryElementwiseOp<"exp">, BXLA_ExpOp;

def LXLA_NegOp: LXLA_UnaryElementwiseOp<"neg">, BXLA_NegOp;

def LXLA_SignOp: LXLA_UnaryElementwiseOp<"sign">, BXLA_SignOp;

def LXLA_TanhOp: LXLA_UnaryElementwiseOp<"tanh">, BXLA_TanhOp;

//===----------------------------------------------------------------------===//
// XLA binary elementwise op definitions.
//===----------------------------------------------------------------------===//

class LXLA_BinaryElementwiseOp<string mnemonic, list<OpTrait> traits> :
        LXLA_Op<mnemonic, traits> {
  let arguments = (ins
      LXLA_Buffer:$lhs,
      LXLA_Buffer:$rhs,
      LXLA_Buffer:$out,
      BroadcastDimAttr:$broadcast_dimensions
  );
}

def LXLA_AddOp : LXLA_BinaryElementwiseOp<"add", []>, BXLA_AddOp;

def LXLA_DivOp : LXLA_BinaryElementwiseOp<"div", []>, BXLA_DivOp;

def LXLA_MaxOp : LXLA_BinaryElementwiseOp<"max", []>, BXLA_MaxOp;

def LXLA_MinOp : LXLA_BinaryElementwiseOp<"min", []>, BXLA_MinOp;

def LXLA_MulOp : LXLA_BinaryElementwiseOp<"mul", []>, BXLA_MulOp;

def LXLA_SubOp : LXLA_BinaryElementwiseOp<"sub", []>, BXLA_SubOp;

def LXLA_AndOp: LXLA_BinaryElementwiseOp<"and", []>, BXLA_AndOp;

//===----------------------------------------------------------------------===//
// XLA control flow op definitions.
//===----------------------------------------------------------------------===//

// TODO(b/139813999): specify required function signature in a type-safe way.
def LXLA_ReduceOp: LXLA_Op<"reduce", [SameVariadicOperandSize]>, BXLA_ReduceOp {
  let arguments = (ins
    Variadic<LXLA_BufferOrTuple>:$operands_and_init,
    Variadic<LXLA_BufferOrTuple>:$out,
    SymbolRefAttr:$computation,
    ElementsAttr:$dimensions
  );
}
//===----------------------------------------------------------------------===//
// XLA tuple op definitions.
//===----------------------------------------------------------------------===//

def LXLA_GetTupleElementOp: LXLA_Op<"get_tuple_element", []>, BXLA_GetTupleElementOp {
  let arguments = (ins
    LXLA_TupleBuffer:$input,
    LXLA_BufferOrTuple:$out,
    I32Attr:$index
  );
}

def LXLA_TupleOp : LXLA_Op<"tuple", []>, BXLA_TupleOp {
   let arguments = (ins
     Variadic<LXLA_BufferOrTuple>:$val,
                 LXLA_TupleBuffer:$out);
}

def LXLA_CompareOp: LXLA_Op<"compare", []>, BXLA_CompareOp {
  let arguments = (ins
    LXLA_Buffer:$lhs,
    LXLA_Buffer:$rhs,
    LXLA_PredBuffer:$out,
    BroadcastDimAttr:$broadcast_dimensions,
    XLA_ComparisonDirectionAttr:$comparison_direction
  );
}

//===----------------------------------------------------------------------===//
// XLA Slice definitions.
//===----------------------------------------------------------------------===//

def LXLA_SliceOp: LXLA_Op<
      "slice",
      [AllTypesMatch<["start_indices", "limit_indices"]>]> {
  let arguments = (ins
    LXLA_Buffer:$operand,
    LXLA_Buffer:$output,
    ElementsAttr:$start_indices,
    ElementsAttr:$limit_indices
  );
}

def XLA_DynamicUpdateSliceOp: LXLA_Op<"dynamic-update-slice", []> {
  let arguments = (ins
    LXLA_Buffer:$operand,
    LXLA_Buffer:$update,
    LXLA_Buffer:$output,
    Variadic<LXLA_Buffer>:$start_indices
  );
}

//===----------------------------------------------------------------------===//
// XLA Other op definitions.
//===----------------------------------------------------------------------===//

def XLA_BatchNormInferenceOp : LXLA_Op<"batch_norm_inference", []>,
    BXLA_BatchNormInferenceOp {

  let arguments = (ins
    LXLA_Buffer:$operand,
    LXLA_Buffer:$scale,
    LXLA_Buffer:$offset,
    LXLA_Buffer:$mean,
    LXLA_Buffer:$variance,
    LXLA_Buffer:$output,
    F32Attr:$epsilon,
    I64Attr:$feature_index
  );
}

def LXLA_BroadcastOp : LXLA_Op<"broadcast",
      []>, BXLA_BroadcastOp {
  let arguments = (ins
    LXLA_Buffer:$operand,
    LXLA_Buffer:$output,
    ElementsAttr:$broadcast_sizes
  );
}

def LXLA_BroadcastInDimOp : LXLA_Op<"broadcast_in_dim",
      []>, BXLA_BroadcastInDimOp {
  let arguments = (ins
    LXLA_Buffer:$operand,
    LXLA_Buffer:$output,
    BroadcastDimAttr:$broadcast_dimensions
  );
}

def LXLA_ClampOp : LXLA_Op<"clamp", []>, BXLA_ClampOp {
  let arguments = (ins
    LXLA_Buffer:$min,
    LXLA_Buffer:$operand,
    LXLA_Buffer:$max,
    LXLA_Buffer:$output
  );
}

def LXLA_ConcatenateOp : LXLA_Op<"concatenate", []>, BXLA_ConcatenateOp {
   let arguments = (ins
     Variadic<LXLA_Buffer>:$val,
     LXLA_Buffer:$output,
     I64Attr: $dimension
   );
}

def LXLA_ConvOp : LXLA_Op<"conv", []>, BXLA_ConvOp {
  let arguments = (ins
    LXLA_Buffer:$lhs,
    LXLA_Buffer:$rhs,
    LXLA_Buffer:$output
  );
}

def LXLA_DotOp: LXLA_Op<"dot", []>, BXLA_DotOp {
  let arguments = (ins
    LXLA_Buffer:$lhs,
    LXLA_Buffer:$rhs,
    XLA_PrecisionConfigAttr:$precision_config,
    LXLA_Buffer:$output
  );
}

def LXLA_GatherOp: LXLA_Op<"gather", []>, BXLA_GatherOp {
  let arguments = (ins
    LXLA_Buffer:$operand,
    LXLA_IntBuffer:$start_indices,
    I64Attr: $index_vector_dim,
    ElementsAttr: $offset_dims,
    ElementsAttr: $slice_sizes,
    ElementsAttr: $collapsed_slice_dims,
    ElementsAttr: $start_index_map,
    LXLA_Buffer:$output
  );
}

def LXLA_ReshapeOp: LXLA_Op<"reshape", []>, BXLA_ReshapeOp {
  let arguments = (ins
    LXLA_Buffer:$operand,
    LXLA_Buffer:$output
  );
}


def LXLA_SelectOp: LXLA_Op<"select", []>, BXLA_SelectOp {
  let arguments = (ins
    LXLA_PredBuffer:$pred,
    LXLA_Buffer:$on_true,
    LXLA_Buffer:$on_false,
    LXLA_Buffer:$output
  );
}

def LXLA_ReverseOp: LXLA_Op<"reverse", []>, BXLA_ReverseOp {
  let arguments = (ins
    LXLA_Buffer:$operand,
    ElementsAttr:$dimensions,
    LXLA_Buffer:$output
  );
}

def LXLA_PadOp: LXLA_Op<"pad", []>, BXLA_PadOp {
  let arguments = (ins
    LXLA_Buffer:$operand,
    LXLA_Buffer:$padding_value,
    ElementsAttr: $edge_padding_low,
    ElementsAttr: $edge_padding_high,
    ElementsAttr: $interior_padding,
    LXLA_Buffer: $output
  );
}

def LXLA_TransposeOp: LXLA_Op<"transpose", []>, BXLA_TransposeOp {
  let arguments = (ins
    LXLA_Buffer:$operand,
    ElementsAttr:$permutation,
    LXLA_Buffer:$output
  );
}


#endif // LXLA_OPS
