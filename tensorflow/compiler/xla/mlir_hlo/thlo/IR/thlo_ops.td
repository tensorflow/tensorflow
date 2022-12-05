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

include "gml_st/interfaces/tiling_interface.td"
include "mlir/Dialect/Linalg/IR/LinalgInterfaces.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/DestinationStyleOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def TensorOrMemref :
  AnyTypeOf<[AnyMemRef, AnyRankedTensor], "", "::mlir::ShapedType">;

class TensorOrMemrefOf<list<Type> allowedTypes> :
  AnyTypeOf<[MemRefOf<allowedTypes>, RankedTensorOf<allowedTypes>],
  "", "::mlir::ShapedType">;

def THLO_Dialect : Dialect {
  let name = "thlo";
  let cppNamespace = "::mlir::thlo";
}

class THLO_Op<string mnemonic, list<Trait> traits> :
    Op<THLO_Dialect, mnemonic, traits> {
  let hasVerifier = 1;
}

class THLO_DstStyleOp<string mnemonic, list<Trait> traits> : THLO_Op<mnemonic, [
    DestinationStyleOpInterface] # traits> {
  let hasCustomAssemblyFormat = 1;
}

def THLO_ConcatenateOp : THLO_DstStyleOp<"concatenate", [
    DeclareOpInterfaceMethods<TilingInterface>]> {
  let summary = "Destination-style twin for `mhlo.concatenate`";
  let description = [{
    tHLO ConcatenateOp composes a tensor or a memref from multiple tensors or
    memrefs.

    Example:
    ```
      %concat = thlo.concatenate
        ins(%T1 : tensor<100x?xf32>, %T2 : tensor<300x?xf32>)
        outs(%init : tensor<400x?xf32>)
        dimension = 0
    ```

    See https://www.tensorflow.org/xla/operation_semantics#concatenate
  }];

  let arguments = (ins
    Variadic<TensorOrMemref>:$inputs,
    TensorOrMemref:$init,
    IndexAttr:$dimension
  );
  let results = (outs Variadic<AnyTensor>:$result);

  let extraClassDeclaration = [{
    // Implement method necessary for DestinationStyleOpInterface.
    std::pair<int64_t, int64_t> getDpsInitsPositionRange() {
      int64_t getNumOperands = this->getNumOperands();
      return {getNumOperands - 1, getNumOperands};
    }
  }];
}

def THLO_DynamicBroadcastInDimOp : THLO_DstStyleOp<"dynamic_broadcast_in_dim", [
    DeclareOpInterfaceMethods<TilingInterface>]> {
  let summary = "Destination-style twin for `mhlo.dynamic_broadcast_in_dim`";
  let description = [{
    tHLO DynamicBroadcastInDimOp specifies a map how to broadcast input
    dimensions. It also supports broadcasting size-1 dimensions.

    Example:
    ```
      %dyn_bcast = thlo.dynamic_broadcast_in_dim
        ins(%input : tensor<?x?xf32>)
        outs(%init : tensor<?x?x?xf32>)
        broadcast_dimensions = [0, 2]
    ```

    See https://www.tensorflow.org/xla/operation_semantics#broadcastindim
  }];

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

  let extraClassDeclaration = [{
    // Implement method necessary for DestinationStyleOpInterface.
    std::pair<int64_t, int64_t> getDpsInitsPositionRange() {
      int64_t getNumOperands = this->getNumOperands();
      return {getNumOperands - 1, getNumOperands};
    }
  }];
}

def THLO_GatherOp : THLO_DstStyleOp<"gather", [
    DeclareOpInterfaceMethods<TilingInterface>]> {
  let summary = "Destination-style twin for `mhlo.gather`";
  let description = [{
    tHLO GatherOp corresponds to the canonicalized mHLO GatherOp, i.e.

    - start_indices is a two-dimensional tensor.
    - index_vector_dim is 1
    - offset_dims is [1, 2, ...]
    - collapsed_slice_dims is []
    - start_index_map is range(start_indices.shape[1])

    Example:
    ```
      %gathered = thlo.gather
        ins(%input : tensor<100xf32>, %indices : tensor<42x1xindex>)
        outs(%init : tensor<42xf32>)
    ```

    See https://www.tensorflow.org/xla/operation_semantics#gather.
  }];
  let arguments = (ins
    // Input args
    TensorOrMemref:$operand,
    TensorOrMemrefOf<[Index]>:$start_indices,
    // Output arg
    TensorOrMemref:$init
  );
  let results = (outs Variadic<AnyTensor>:$result);

  let extraClassDeclaration = [{
    // Implement method necessary for DestinationStyleOpInterface.
    std::pair<int64_t, int64_t> getDpsInitsPositionRange() {
      int64_t getNumOperands = this->getNumOperands();
      return {getNumOperands - 1, getNumOperands};
    }
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

    Example:
    ```
      %scattered = thlo.scatter
        ins(%indices : tensor<2x2xindex>, %input : tensor<2x1x3xf32>)
        outs(%init : tensor<3x3xf32>)
        (%arg3: f32, %arg4: f32) {
          %0 = arith.addf %arg3, %arg4 : f32
          thlo.yield %0 : f32
        }
    ```

    See https://www.tensorflow.org/xla/operation_semantics#scatter.
  }];

  let arguments = (ins
    // Input args
    TensorOrMemrefOf<[Index]>:$indices,
    TensorOrMemref:$updates,
    // Output arg
    TensorOrMemref:$init
  );

  let results = (outs Variadic<AnyTensor>:$result);

  let regions = (region SizedRegion<1>:$update_computation);

  let extraClassDeclaration = [{
    // Returns index vector dimension size, which is always statically-known.
    int64_t getIndexVectorDimSize() {
      return getIndices().getType().getDimSize(1);
    }

    // Returns the number of indices, i.e. number of scalar/tensor updates.
    int64_t getIndicesCount() { return getIndices().getType().getDimSize(0); }

    // Implement method necessary for DestinationStyleOpInterface.
    std::pair<int64_t, int64_t> getDpsInitsPositionRange() {
      int64_t getNumOperands = this->getNumOperands();
      return {getNumOperands - 1, getNumOperands};
    }
  }];
}

def THLO_SortOp : THLO_DstStyleOp<"sort", [
        DeclareOpInterfaceMethods<OpAsmOpInterface,
          ["getAsmResultNames", "getAsmBlockArgumentNames"]>,
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
        dimension = 0
        is_stable = true
        (%lhs0: f32, %rhs0: f32, %lhs1: i32, %rhs1: i32) {
          %0 = arith.cmpf ogt, %lhs0, %rhs0 : f32
          thlo.yield %0 : i1
        }
    ```

    See https://www.tensorflow.org/xla/operation_semantics#sort.
  }];

  let arguments = (ins
    // Input args
    Variadic<TensorOrMemref>:$inputs,
    // Output args
    Variadic<TensorOrMemref>:$inits,

    IndexAttr:$dimension,
    BoolAttr:$is_stable
  );

  let results = (outs Variadic<AnyTensor>:$result);
  let regions = (region SizedRegion<1>:$comparator);

  let extraClassDeclaration = [{
    // Implement method necessary for DestinationStyleOpInterface.
    std::pair<int64_t, int64_t> getDpsInitsPositionRange() {
      int64_t getNumOperands = this->getNumOperands();
      return {getNumOperands - getInits().size(), getNumOperands};
    }
  }];
}

def THLO_YieldOp : THLO_Op<"yield", [Pure, ReturnLike, Terminator,
    ParentOneOf<["ScatterOp", "SortOp"]>]>,
    Arguments<(ins Variadic<AnyType>:$values)> {
  let summary = "Yield operation for tHLO ops with regions.";
  let assemblyFormat = "attr-dict $values `:` type($values)";
  let hasVerifier = 1;
}


#endif // THLO_OPS
