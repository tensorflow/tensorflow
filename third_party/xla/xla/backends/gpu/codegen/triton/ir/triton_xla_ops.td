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


#ifndef XLA_BACKENDS_GPU_CODEGEN_TRITON_IR_TRITON_XLA_OPS_TD_
#define XLA_BACKENDS_GPU_CODEGEN_TRITON_IR_TRITON_XLA_OPS_TD_

include "mlir/IR/OpBase.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td" // Pure
include "mlir/Interfaces/InferTypeOpInterface.td" // SameOperandsAndResultType
include "mlir/Interfaces/ViewLikeInterface.td" // OffsetSizeAndStrideOpInterface
include "xla/backends/gpu/codegen/triton/ir/triton_xla_dialect.td"
include "xla/backends/gpu/codegen/triton/ir/triton_xla_types.td"
include "triton/Dialect/Triton/IR/TritonInterfaces.td"
include "triton/Dialect/TritonGPU/IR/TritonGPUTypeInterfaces.td"
include "triton/Dialect/Triton/IR/TritonTypes.td"

// -----------------------------------------------------------------------------
// Triton XLA Ops
// -----------------------------------------------------------------------------

class TTXLA_Op<string mnemonic, list<Trait> traits = []> :
      Op<XlaTritonDialect, mnemonic, traits> {
}

def TTXLA_TileOp : TTXLA_Op<"tile", [Pure,
    DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>]> {
  let summary = "Capture the tiling information of a tensor.";
  let description = [{
    This op captures the tiling information of a tensor, which can be used to
    extract/insert tiles with the `triton_xla.extract` and `triton_xla.insert`
    ops.

    Example:
      ```
      %tensor: tensor<120x320xbf16>
      ...
      %tiled_tensor = triton_xla.tile %tensor [0, 0][16, 64][120, 1]
      : !triton_xla.tiled_tensor<16x64|120x320xbf16>
      ```
  }];

  let arguments = (ins
    AnyRankedTensor:$tensor,
    DenseI32ArrayAttr:$offsets,
    DenseI32ArrayAttr:$sizes,
    DenseI64ArrayAttr:$strides
  );
  let results = (outs TTXLA_TiledTensorType:$tiled_tensor);
  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

def TTXLA_ExtractOp : TTXLA_Op<"extract", [Pure,
    DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>]> {
  let summary = "Extract a tile from a tensor.";
  let description = [{
    This op is used to extract a tile from a tiled tensor.

    Example:
      ```
      %tiled_tensor = triton_xla.tile %tensor [0, 0][16, 64][120, 1]
      : !triton_xla.tiled_tensor<16x64|120x320xbf16>
      ...
      %extracted_tensor = triton_xla.extract %tiled_tensor [%cst, %cst]
        : tensor<120x320xbf16> to tensor<16x64xbf16>
      ```
  }];

  let arguments = (ins
    TTXLA_TiledTensorType:$src,
    Variadic<I32>:$offsets
  );
  let results = (outs AnyRankedTensor:$result);
  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

def TTXLA_InsertOp : TTXLA_Op<"insert", [Pure,
    DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>]> {
  let summary = "Insert a tile into a tensor.";
  let description = [{
    This op is used to insert a tile into a tensor. The tiling information can
    be captured using triton_xla.tile and passed to this op.

    Example:
      ```
      %tiled_tensor = triton_xla.tile %tensor [0, 0][16, 64][120, 1]
      : !triton_xla.tiled_tensor<16x64|120x320xbf16>
      ...
      %updated_tensor = triton_xla.insert %arg0 into %tiled_tensor[%cst, %cst]
        : tensor<16x64xbf16> into tensor<120x320xbf16>
      ```
  }];

  let arguments = (ins
    AnyRankedTensor:$src,
    TTXLA_TiledTensorType:$dst,
    Variadic<I32>:$offsets
  );
  let results = (outs AnyRankedTensor:$result);
  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

#endif // XLA_BACKENDS_GPU_CODEGEN_TRITON_IR_TRITON_XLA_OPS_TD_
