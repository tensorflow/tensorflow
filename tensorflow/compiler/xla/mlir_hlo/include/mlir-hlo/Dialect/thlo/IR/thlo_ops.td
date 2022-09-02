/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#ifndef THLO_OPS
#define THLO_OPS

include "mlir-hlo/Dialect/gml_st/transforms/fusion_interface.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def TensorOrMemref : AnyTypeOf<[AnyMemRef, AnyRankedTensor]>;

def THLO_Dialect : Dialect {
  let name = "thlo";
  let cppNamespace = "::mlir::thlo";

  let emitAccessorPrefix = kEmitAccessorPrefix_Raw;
}

class THLO_Op<string mnemonic, list<Trait> traits> :
    Op<THLO_Dialect, mnemonic, traits> {
  let hasVerifier = 1;
}


def THLO_YieldOp : THLO_Op<"yield", [NoSideEffect, ReturnLike, Terminator,
    ParentOneOf<["MapOp", "ReductionOp"]>]>,
    Arguments<(ins Variadic<AnyType>:$values)> {
  let summary = "Yield operation for thlo.map and thlo.reduce.";
  let assemblyFormat = "attr-dict $values `:` type($values)";
  let hasVerifier = 1;
}

class THLO_DstStyleOp<string mnemonic, list<Trait> traits> : THLO_Op<mnemonic, [
    NoSideEffect] # traits> {
  let hasCustomAssemblyFormat = 1;
  code extraBaseClassDeclaration = [{
    unsigned getNumInputs() {
      return this->getOperation()->getNumOperands() - getNumOutputs();
    };
    unsigned getNumOutputs() { return 1; };
  }];
}

def THLO_ConcatenateOp : THLO_DstStyleOp<"concatenate", [
    SameOperandsAndResultElementType,
    DeclareOpInterfaceMethods<FusionInterface>]> {
  let summary = "Destination-style twin for `mhlo.concatenate`";
  let arguments = (ins
    Variadic<AnyTensor>:$operands,
    AnyTensor:$init,
    I64Attr:$dimension
  );
  let results = (outs AnyTensor:$result);

  let extraClassDeclaration = extraBaseClassDeclaration;
}

def THLO_DynamicBroadcastInDimOp : THLO_DstStyleOp<"dynamic_broadcast_in_dim", [
    DeclareOpInterfaceMethods<FusionInterface>,
    SameOperandsAndResultElementType]> {
  let summary = "Destination-style twin for `mhlo.dynamic_broadcast_in_dim`";

  let arguments = (ins
    // Input args
    TensorOrMemref:$operand,
    // Output arg
    TensorOrMemref:$init,

    DenseI64ArrayAttr:$broadcast_dimensions,
    OptionalAttr<DenseI64ArrayAttr>:$known_expanding_dimensions,
    OptionalAttr<DenseI64ArrayAttr>:$known_nonexpanding_dimensions
  );
  let results = (outs Variadic<AnyTensor>:$result);

  let extraClassDeclaration = extraBaseClassDeclaration;
}

def THLO_GatherOp : THLO_DstStyleOp<"gather", []> {
  let summary = "Destination-style twin for `mhlo.gather`";
  let description = [{
    Does not currently support the full interface of mhlo.gather. In particular:
    - index_vector_dim is start_indices.shape.rank - 1
    - slice_sizes is [1,1,...]
    - offset_dims is []
    - collapsed_slice_dims is range(operand.shape.rank)
    - start_index_map is range(slice_sizes.shape[index_vector_dim])
  }];
  let arguments = (ins
    // Input args
    TensorOrMemref:$operand,
    I64Tensor:$start_indices,
    // Output arg
    TensorOrMemref:$init
  );
  let results = (outs Variadic<AnyTensor>:$result);

  let extraClassDeclaration = extraBaseClassDeclaration;
}

def THLO_ScatterOp : THLO_DstStyleOp<"scatter", []> {
  let summary = "Destination-style twin for `mhlo.scatter`";
  let description = [{
    Caveats:
    - the variadic case is not supported.
    - update_computation is sum.
    - Only point updates are supported
      - update_window_dims is []
      - inserted_window_dims is range(operand.shape.rank)
      - scatter_dims_to_operand_dims is range(indices.shape.rank)
      - index_vector_dim is indices.shape.rank-1
  }];
  let arguments = (ins
    // Input args
    TensorOf<[I32, I64]>:$indices,
    TensorOrMemref:$updates,
    // Output arg
    TensorOrMemref:$init
  );
  let results = (outs Variadic<AnyTensor>:$result);

  let extraClassDeclaration = extraBaseClassDeclaration;
}

def THLO_TransposeOp : THLO_DstStyleOp<"transpose", []> {
  let summary = "Destination-style twin for `mhlo.transpose`";
  let description = [{
    Permutes the dimensions of `operand` according to the given `permutation`.
      `dim(result, i) = dim(operand, permutation[i])`

    Example:
    ```
      %transpose = thlo.transpose
          ins(%input:tensor<16x64xf32>)
          outs(%init:tensor<64x16xf32>)
          { permutation =  [:i64 1, 0] }
    ```
  }];

  let arguments = (ins
    // Input arg
    TensorOrMemref:$input,
    // Output arg
    TensorOrMemref:$init,

    DenseI64ArrayAttr:$permutation
  );
  let results = (outs TensorOrMemref:$result);

  let extraClassDeclaration = extraBaseClassDeclaration;
}

def THLO_ReductionOp : THLO_DstStyleOp<"reduction", [
      SameVariadicOperandSize, SingleBlockImplicitTerminator<"YieldOp">
    ]> {
  let summary = "Destination-style twin for `mhlo.reduce`";
  let description = [{
    Executes `combiner` on the `dimensions` of `inputs` and returns the
    reduced result. The `dimensions` attribute needs to list the reduction
    dimensions in increasing order.

    Example:
    ```
      %reduction = thlo.reduction
          ins(%input:tensor<16x32x64xf32>)
          outs(%init:tensor<16x64xf32>)
          dimensions = [1]
          (%in: f32, %out: f32) {
            %0 = arith.addf %in, %out: f32
            thlo.yield %0: f32
          }
    ```
  }];

  let arguments = (ins
    // Input arg
    Variadic<TensorOrMemref>:$inputs,
    // Output arg
    Variadic<TensorOrMemref>:$inits,

    DenseI64ArrayAttr:$dimensions
  );
  let results = (outs Variadic<TensorOrMemref>);
  let regions = (region SizedRegion<1>:$combiner);

  let extraClassDeclaration = [{
    unsigned getNumInputs() {
      return this->getOperation()->getNumOperands() - getNumOutputs();
    };
    unsigned getNumOutputs() { return inits().size(); };
  }];
}

def THLO_MapOp : THLO_DstStyleOp<"map", [
    SameOperandsAndResultShape,
    SingleBlockImplicitTerminator<"YieldOp">
]> {
  let summary = "Destination-style twin for elementwise operations";
  let description = [{
    Models elementwise operations on tensors in terms of arithmetic operations
    on the corresponding elements.

    Example:
    ```
      %add = thlo.map
          ins(%lhs: tensor<64xf32>, %rhs: tensor<64xf32>)
          outs(%init: tensor<64xf32>)
          (%lhs_elem: f32, %rhs_elem: f32) {
            %0 = arith.addf %lhs_elem, %rhs_elem: f32
            thlo.yield %0: f32
          }
    ```
  }];

  let arguments = (ins
    // Input args
    Variadic<TensorOrMemref>:$inputs,

    // Output arg
    TensorOrMemref:$init
  );
  let results = (outs AnyTensor:$result);
  let regions = (region AnyRegion:$mapper);

  let extraClassDeclaration = extraBaseClassDeclaration;
}

#endif // THLO_OPS
