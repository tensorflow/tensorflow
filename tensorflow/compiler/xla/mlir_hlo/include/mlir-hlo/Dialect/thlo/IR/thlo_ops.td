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

include "mlir-hlo/Dialect/gml_st/transforms/tiling_interface.td"
include "mlir/Dialect/Linalg/IR/LinalgInterfaces.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def TensorOrMemref :
  AnyTypeOf<[AnyMemRef, AnyRankedTensor], "", "::mlir::ShapedType">;

class TensorOrMemrefOf<list<Type> allowedTypes> :
  AnyTypeOf<[MemRefOf<allowedTypes>, RankedTensorOf<allowedTypes>],
  "", "::mlir::ShapedType">;

def THLO_Dialect : Dialect {
  let name = "thlo";
  let cppNamespace = "::mlir::thlo";

  let emitAccessorPrefix = kEmitAccessorPrefix_Prefixed;
}

class THLO_Op<string mnemonic, list<Trait> traits> :
    Op<THLO_Dialect, mnemonic, traits> {
  let hasVerifier = 1;
}

class THLO_DstStyleOp<string mnemonic, list<Trait> traits> : THLO_Op<mnemonic, [
    DestinationStyleOpInterface, Pure] # traits> {
  let hasCustomAssemblyFormat = 1;
  code extraBaseClassDeclaration = [{
    unsigned getNumInputs() {
      return this->getOperation()->getNumOperands() - getNumOutputs();
    };
    unsigned getNumOutputs() { return 1; };
  }];
}

def THLO_YieldOp : THLO_Op<"yield", [Pure, ReturnLike, Terminator,
    ParentOneOf<["MapOp", "ReductionOp", "ScatterOp", "SortOp",
                 "TransposeOp"]>]>,
    Arguments<(ins Variadic<AnyType>:$values)> {
  let summary = "Yield operation for tHLO ops with regions.";
  let assemblyFormat = "attr-dict $values `:` type($values)";
  let hasVerifier = 1;
}

def THLO_ConcatenateOp : THLO_DstStyleOp<"concatenate", [
    DeclareOpInterfaceMethods<TilingInterface>]> {
  let summary = "Destination-style twin for `mhlo.concatenate`";
  let arguments = (ins
    Variadic<TensorOrMemref>:$inputs,
    TensorOrMemref:$init,
    I64Attr:$dimension
  );
  let results = (outs Variadic<AnyTensor>:$result);

  let extraClassDeclaration = extraBaseClassDeclaration # [{
    // Implement functions necessary for DestinationStyleOpInterface.
    mlir::ValueRange inputs() { return getInputs(); }
    mlir::ValueRange getOutputs() { return getOperands().take_back(1); }
  }];
}

def THLO_DynamicBroadcastInDimOp : THLO_DstStyleOp<"dynamic_broadcast_in_dim", [
    DeclareOpInterfaceMethods<TilingInterface>]> {
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

  let extraClassDeclaration = extraBaseClassDeclaration # [{
    // Implement functions necessary for DestinationStyleOpInterface.
    mlir::ValueRange inputs() { return getOperands().take_front(1); }
    mlir::ValueRange getInputs() { return inputs(); }
    mlir::ValueRange getOutputs() { return getOperands().take_back(1); }
  }];
}

def THLO_GatherOp : THLO_DstStyleOp<"gather", [
    DeclareOpInterfaceMethods<TilingInterface>]> {
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
    TensorOrMemrefOf<[I64]>:$start_indices,
    // Output arg
    TensorOrMemref:$init
  );
  let results = (outs Variadic<AnyTensor>:$result);

  let extraClassDeclaration = extraBaseClassDeclaration # [{
    // Implement functions necessary for DestinationStyleOpInterface.
    mlir::ValueRange inputs() {
      return getOperands().drop_back(getNumOutputs());
    }
    mlir::ValueRange getInputs() { return inputs(); }
    mlir::ValueRange getOutputs() { return getOperands().take_back(1); }
  }];
}

def THLO_ScatterOp : THLO_DstStyleOp<"scatter", [
    DeclareOpInterfaceMethods<TilingInterface>,
    SingleBlockImplicitTerminator<"YieldOp">]> {
  let summary = "Destination-style twin for `mhlo.scatter`";
  let description = [{
    tHLO ScatterOp corresponds to the canonicalized mHLO ScatterOp, i.e.

    - update_window_dims is range(1, rank(update_window_dims))
    - inserted_window_dims is []
    - scatter_dims_to_operand_dims is range(0, rank(indices))
    - index_vector_dim is rank(indices) - 1

    At the moment, the variadic case is not supported.
  }];
  let arguments = (ins
    // Input args
    TensorOrMemrefOf<[I32, I64]>:$indices,
    TensorOrMemref:$updates,
    // Output arg
    TensorOrMemref:$init
  );

  let results = (outs Variadic<AnyTensor>:$result);

  let regions = (region SizedRegion<1>:$update_computation);

  let extraClassDeclaration = extraBaseClassDeclaration # [{
    // Implement functions necessary for DestinationStyleOpInterface.
    mlir::ValueRange inputs() {
      return getOperands().drop_back(getNumOutputs());
    }
    mlir::ValueRange getInputs() { return inputs(); }
    mlir::ValueRange getOutputs() { return getOperands().take_back(1); }

    // Returns index vector dimension size, which is always statically-known.
    int64_t getIndexVectorDimSize() {
      return getIndices().getType().getDimSize(1);
    }

    // Returns the number of indices, i.e. number of scalar/tensor updates.
    int64_t getIndicesCount() { return getIndices().getType().getDimSize(0); }
  }];
}

def THLO_TransposeOp : THLO_DstStyleOp<"transpose", [
    DeclareOpInterfaceMethods<LinalgStructuredInterface, ["getIteratorTypesArray"]>,
    SingleBlockImplicitTerminator<"YieldOp">,
]> {
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
  let results = (outs Variadic<AnyTensor>:$result);

  let regions = (region SizedRegion<1>:$region);

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins "Type":$resultType, "Value":$input, "Value":$init,
        "DenseI64ArrayAttr":$permutation,
        CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes)>,
    OpBuilder<(ins "Type":$resultType, "Value":$input, "Value":$init,
        "ArrayRef<int64_t>":$permutation,
        CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes)>,
  ];

  let extraClassDeclaration = extraBaseClassDeclaration # [{
    // Implement functions necessary for DestinationStyleOpInterface.
    mlir::ValueRange inputs() { return getOperands().take_front(1); }
    mlir::ValueRange getInputs() { return inputs(); }
    mlir::ValueRange getOutputs() { return getOperands().take_back(1); }

    static std::function<void(mlir::ImplicitLocOpBuilder &, mlir::Block &,
        mlir::ArrayRef<mlir::NamedAttribute>)>
      getRegionBuilder();

    static void createRegion(::mlir::OpBuilder &opBuilder,
                             ::mlir::OperationState & odsState);
  }];
}

def THLO_ReductionOp : THLO_DstStyleOp<"reduction", [
      DeclareOpInterfaceMethods<LinalgStructuredInterface, ["getIteratorTypesArray"]>,
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
  let results = (outs Variadic<AnyTensor>);
  let regions = (region SizedRegion<1>:$combiner);

  let extraClassDeclaration = [{
    // Implement functions necessary for DestinationStyleOpInterface.
    mlir::ValueRange inputs() { return getInputs(); }
    mlir::ValueRange getOutputs() { return getInits(); }
    unsigned getNumInputs() { return getInputs().size(); };
    unsigned getNumOutputs() { return getInits().size(); };
    static std::function<void(mlir::ImplicitLocOpBuilder &, mlir::Block &,
                              mlir::ArrayRef<mlir::NamedAttribute>)>
    getRegionBuilder() {
      return nullptr;
    }
  }];
}

def THLO_MapOp : THLO_DstStyleOp<"map", [
    SingleBlockImplicitTerminator<"YieldOp">,
    DeclareOpInterfaceMethods<LinalgStructuredInterface, ["getIteratorTypesArray"]>
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
  let results = (outs Variadic<AnyTensor>:$result);
  let regions = (region SizedRegion<1>:$mapper);

  let extraClassDeclaration = extraBaseClassDeclaration # [{
    // Implement functions necessary for DestinationStyleOpInterface.
    mlir::ValueRange inputs() { return getInputs(); }
    mlir::ValueRange getOutputs() { return getOperands().take_back(1); }
    linalg::OpOperandVector getOpOperandsMatchingBBargs() { return getInputOperands(); }

    static std::function<void(mlir::ImplicitLocOpBuilder &, mlir::Block &,
                              mlir::ArrayRef<mlir::NamedAttribute>)>
    getRegionBuilder() {
      return nullptr;
    }

  }];
}

def THLO_SortOp : THLO_DstStyleOp<"sort", [RecursiveMemoryEffects,
        DeclareOpInterfaceMethods<TilingInterface>, SameVariadicOperandSize,
        SingleBlockImplicitTerminator<"YieldOp">]> {
  let summary = "Destination-style twin for the `mhlo.sort`";
  let description = [{
    Sorts the given `operands` along the given `dimension` using the given
    `comparator`.

    Example:
    ```
      %sorted1, %sorted2 = thlo.sort
          ins(%input1: tensor<?x?xf32>, %input2: tensor<?x?xi32>)
          outs(%init1: tensor<?x?xf32>, %init2: tensor<?x?xi32>)
          { dimension = 0 : i64, is_stable = true }
          (%e11: f32, %e12: f32, %e21: i32, %e22: i32) {
            %gt = arith.cmpf ogt, %e11, %e12: f32
            thlo.yield %gt : i1
          }
    ```
    See https://www.tensorflow.org/xla/operation_semantics#sort.
  }];

  let arguments = (ins
    // Input args
    Variadic<TensorOrMemref>:$inputs,
    // Output args
    Variadic<TensorOrMemref>:$inits,

    I64Attr:$dimension,
    BoolAttr:$is_stable
  );

  let results = (outs Variadic<AnyTensor>:$result);
  let regions = (region SizedRegion<1>:$comparator);

  let extraClassDeclaration = [{
    // Implement functions necessary for DestinationStyleOpInterface.
    mlir::ValueRange getOutputs() { return getInits(); }
    unsigned getNumInputs() { return getInputs().size(); };
    unsigned getNumOutputs() { return getInits().size(); };
  }];
}

#endif // THLO_OPS
