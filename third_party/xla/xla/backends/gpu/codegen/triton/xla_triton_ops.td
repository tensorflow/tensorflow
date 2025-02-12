/* Copyright 2024 The OpenXLA Authors.

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


#ifndef XLA_BACKENDS_GPU_CODEGEN_TRITON_XLA_TRITON_OPS_TD_
#define XLA_BACKENDS_GPU_CODEGEN_TRITON_XLA_TRITON_OPS_TD_

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td" // Pure
include "mlir/Interfaces/InferTypeOpInterface.td" // SameOperandsAndResultType
include "mlir/Interfaces/ViewLikeInterface.td" // OffsetSizeAndStrideOpInterface
include "xla/backends/gpu/codegen/triton/xla_triton_dialect.td"
include "xla/backends/gpu/codegen/triton/xla_triton_types.td"
include "triton/Dialect/Triton/IR/TritonInterfaces.td"
include "triton/Dialect/TritonGPU/IR/TritonGPUTypeInterfaces.td"
include "triton/Dialect/Triton/IR/TritonTypes.td"

// -----------------------------------------------------------------------------
// Triton XLA Ops
// -----------------------------------------------------------------------------

class TTXLA_Op<string mnemonic, list<Trait> traits = []> :
      Op<XlaTritonDialect, mnemonic, traits> {
}

def TTXLA_SparseDotOp : TTXLA_Op<"sparse_dot", [
        Pure, DotLike, DeclareOpInterfaceMethods<InferTypeOpInterface>,
        TypesMatchWith<"result's type matches accumulator's type", "d", "c", "$_self">]> {
    let summary = "sparse dot";

    let arguments = (ins
      TTG_TensorOrMemDesc:$a,
      TTG_TensorOrMemDesc:$b,
      TT_FpIntTensor:$c,
      TT_IntTensor: $aMeta);
    let results = (outs TT_FpIntTensor:$d);
    let assemblyFormat = [{
      $a`,` $b`,` $c`,` $aMeta attr-dict
        `:` type($a) `meta` type($aMeta) `*` type($b) `->` type($d)
    }];
    let hasVerifier = 1;
}



def TTXLA_TileOp : TTXLA_Op<"tile", [Pure]> {
  let summary = "Capture the tiling information of a tensor.";
  let description = [{
    Usage:
      This op is used to capture the tiling information of a tensor. The tiling
      information can later be used with triton_xla.extract and
      triton_xla.insert.

    Example:
      ```
      %arg0: tensor<128x320xbf16>
      ...
      %tiled_arg0 = triton_xla.tile %arg0 [0, 0] [1, 1] [16, 64]
        : tensor<120x320xbf16> -> !triton_xla.tiled_tensor<16x64xbf16>
      ```
  }];

  let arguments = (ins
    AnyRankedTensor:$tensor,
    DenseI64ArrayAttr:$shape,
    DenseI64ArrayAttr:$strides,
    DenseI64ArrayAttr:$offsets
  );

  let results = (outs TTXLA_TiledTensorType:$tiled_tensor);

  let assemblyFormat = [{
    $tensor $offsets $strides $shape attr-dict `:` type($tensor) `->` qualified(type($tiled_tensor))
  }];
}

def TTXLA_ExtractOp : TTXLA_Op<"extract", [Pure]> {
  let summary = "Extract a tile from a tensor.";
  let description = [{
    Usage:
      This op is used to extract a tile from a tensor. The tiling information
      can be captured using triton_xla.tile and passed to this op.

    Example:
      ```
      %tiled_arg0 = triton_xla.tile %arg0 [0, 0] [1, 1] [16, 64]
        : tensor<120x320xbf16> -> !triton_xla.tiled_tensor<16x64xbf16>
      ...
      %extracted_tensor = triton_xla.extract %tiled_arg0 [%cst, %cst]
        : !triton_xla.tiled_tensor<16x64xbf16> -> tensor<16x64xbf16>
  }];

  let arguments = (ins
    TTXLA_TiledTensorType:$tiled_tensor,
    Variadic<Index>:$offsets
  );

  let results = (outs AnyRankedTensor:$extracted_tensor);

  let assemblyFormat = [{
    $tiled_tensor `[` $offsets `]` attr-dict `:`
      qualified(type($tiled_tensor)) `->` type($extracted_tensor)
  }];
}

def TTXLA_InsertOp : TTXLA_Op<"insert", [Pure]> {
  let summary = "Insert a tile into a tensor.";
  let description = [{
    Usage:
      This op is used to insert a tile into a tensor. The tiling information
      can be captured using triton_xla.tile and passed to this op.

    Example:
      ```
      %tiled_arg2 = triton_xla.tile %tiled_arg2 [0, 0] [1, 1] [16, 64]
      : !triton_xla.tiled_tensor<16x64xbf16>
      ...
      %inserted_tensor = triton_xla.insert %arg0 into %tiled_arg2 [%cst, %cst]
        : tensor<16x64xbf16> into !triton_xla.tiled_tensor<16x64xbf16>
        -> tensor<16x64xbf16>
      ```
  }];

  let arguments = (ins
    AnyRankedTensor:$source_tensor,
    TTXLA_TiledTensorType:$dest_tiled_tensor,
    Variadic<Index>:$offsets
  );

  let results = (outs AnyRankedTensor:$dest_tensor);

  let assemblyFormat = [{
    $source_tensor `into` $dest_tiled_tensor `[` $offsets  `]` attr-dict `:`
       type($source_tensor) `into` qualified(type($dest_tiled_tensor))
       `->` type($dest_tensor)
  }];
}

#endif // XLA_BACKENDS_GPU_CODEGEN_TRITON_XLA_TRITON_OPS_TD_
