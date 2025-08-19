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
include "triton/Dialect/Triton/IR/TritonInterfaces.td"
include "triton/Dialect/Triton/IR/TritonOpInterfaces.td"
include "triton/Dialect/Triton/IR/TritonTypes.td"
include "triton/Dialect/TritonGPU/IR/TritonGPUTypeInterfaces.td"

// -----------------------------------------------------------------------------
// Triton XLA Ops
// -----------------------------------------------------------------------------

class TTXLA_Op<string mnemonic, list<Trait> traits = []> :
      Op<XlaTritonDialect, mnemonic, traits> {
}

// Base class for ops with static/dynamic offset, sizes and strides
// attributes/arguments.
class TTXLA_OpWithOffsetSizesAndStrides<string mnemonic,
                                        list<Trait> traits = []>
    : TTXLA_Op<mnemonic, !listconcat(traits, [
      AttrSizedOperandSegments,
      OffsetSizeAndStrideOpInterface
  ])> {
  code extraBaseClassDeclaration = [{
    SmallVector<Value> getOffsetsAsValues(::mlir::ImplicitLocOpBuilder &b) {
      return ::mlir::getValueOrCreateConstantIndexOp(b, b.getLoc(),
        getMixedOffsets());
    }

    SmallVector<Value> getSizesAsValues(::mlir::ImplicitLocOpBuilder &b) {
      return ::mlir::getValueOrCreateConstantIndexOp(b, b.getLoc(),
        getMixedSizes());
    }

    SmallVector<Value> getStridesAsValues(::mlir::ImplicitLocOpBuilder &b) {
      return ::mlir::getValueOrCreateConstantIndexOp(b, b.getLoc(),
        getMixedStrides());
    }
  }];
  let hasVerifier = 1;
  let hasCanonicalizer = 1;
}

def TTXLA_ExtractOp : TTXLA_OpWithOffsetSizesAndStrides<"extract", [
      DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>]> {
  let summary = "Extract a tile from a pointer.";
  let description = [{
    Offsets, strides, and sizes operands specify the result tensor tile
    to extract. Shape and minor-to-major layout specify the layout of the source
    pointer.

    Assembly format:
      from %src as memref<...> [%offsets] [%sizes] [%strides] : tensor_type

    Example:
      ```
      %extracted_tensor = triton_xla.extract from %src
        as memref<512x128xbf16, #triton_xla.layout<[1, 0]>>
        [0, 0] [16, 64] [128, 1] : tensor<16x64xbf16>
      ```
  }];
  let builders = [
    OpBuilder<(ins "RankedTensorType":$result_type, "Value":$src,
      "ValueRange":$offsets,  "ValueRange":$strides,
      CArg<"ArrayRef<int64_t>">:$src_shape,
      CArg<"ArrayRef<int64_t>">:$src_layout)>,
    OpBuilder<(ins "RankedTensorType":$result_type, "Value":$src,
      "ArrayRef<OpFoldResult>":$offsets,  "ArrayRef<OpFoldResult>":$strides,
      CArg<"ArrayRef<int64_t>">:$src_shape,
      CArg<"ArrayRef<int64_t>">:$src_layout)>
  ];

  let arguments = (ins
    TT_Ptr:$src,
    Variadic<Index>:$offsets,
    Variadic<Index>:$sizes,
    Variadic<Index>:$strides,
    DenseI64ArrayAttr:$static_offsets,
    DenseI64ArrayAttr:$static_sizes,
    DenseI64ArrayAttr:$static_strides,
    DenseI64ArrayAttr:$src_shape,
    DenseI64ArrayAttr:$src_layout
  );
  let results = (outs AnyRankedTensor:$result);

  let assemblyFormat = [{
    `from` $src `as` custom<AsMemRefType>(type($src), $src_shape, $src_layout)
    custom<DynamicIndexList>($offsets, $static_offsets)
    custom<DynamicIndexList>($sizes, $static_sizes)
    custom<DynamicIndexList>($strides, $static_strides)
    attr-dict `:` type($result)
  }];
  let extraClassDeclaration = extraBaseClassDeclaration # [{
    /// Return the expected rank of each of the `static_offsets`, `static_sizes`
    /// and `static_strides` attributes.
    std::array<unsigned, 3> getArrayAttrMaxRanks() {
      unsigned rank = getType().getRank();
      return {rank, rank, rank};
    }
    /// Return the number of leading operands before the `offsets`, `sizes` and
    /// and `strides` operands.
    static unsigned getOffsetSizeAndStrideStartOperandIndex() { return 1; }
  }];
}

def TTXLA_InsertOp : TTXLA_OpWithOffsetSizesAndStrides<"insert"> {
  let summary = "Insert a tile into a pointer.";
  let description = [{
    Offsets, strides, and source tensor size are used to specify the tile to
    insert. Shape and minor-to-major layout specify the layout of the
    destination pointer.

    Assembly format:
      %src into %dst as memref<...> [%offsets] [%sizes] [%strides] : tensor_type

    Example:
      ```
      triton_xla.insert %src into %dst
        as memref<512x128xbf16, #triton_xla.layout<[1, 0]>>
        [0, 0] [8, 8] [1, 1] : tensor<8x8xbf16>
      ```
  }];
  let builders = [
    OpBuilder<(ins "Value":$src, "Value":$dst,
      "ValueRange":$offsets,  "ValueRange":$strides,
      CArg<"ArrayRef<int64_t>">:$dst_shape,
      CArg<"ArrayRef<int64_t>">:$dst_layout)>,
    OpBuilder<(ins "Value":$src, "Value":$dst,
      "ArrayRef<OpFoldResult>":$offsets,  "ArrayRef<OpFoldResult>":$strides,
      CArg<"ArrayRef<int64_t>">:$dst_shape,
      CArg<"ArrayRef<int64_t>">:$dst_layout)>
  ];

  let arguments = (ins
    AnyStaticShapeTensor:$src,
    TT_Ptr:$dst,
    Variadic<Index>:$offsets,
    Variadic<Index>:$sizes,
    Variadic<Index>:$strides,
    DenseI64ArrayAttr:$static_offsets,
    DenseI64ArrayAttr:$static_sizes,
    DenseI64ArrayAttr:$static_strides,
    DenseI64ArrayAttr:$dst_shape,
    DenseI64ArrayAttr:$dst_layout
  );

  let assemblyFormat = [{
    $src `into`
    $dst `as` custom<AsMemRefType>(type($dst), $dst_shape, $dst_layout)
    custom<DynamicIndexList>($offsets, $static_offsets)
    custom<DynamicIndexList>($sizes, $static_sizes)
    custom<DynamicIndexList>($strides, $static_strides)
    attr-dict `:` type($src)
  }];

  let extraClassDeclaration = extraBaseClassDeclaration # [{
    /// Return the expected rank of each of the `static_offsets`, `static_sizes`
    /// and `static_strides` attributes.
    std::array<unsigned, 3> getArrayAttrMaxRanks() {
      unsigned rank = getSrc().getType().getRank();
      return {rank, rank, rank};
    }
    /// Return the number of leading operands before the `offsets`, `sizes` and
    /// and `strides` operands.
    static unsigned getOffsetSizeAndStrideStartOperandIndex() { return 2; }
  }];
}

def TTXLA_SqueezeDimsOp : TTXLA_Op<"squeeze_dims", [
    Pure, SameOperandsAndResultElementType
  ]> {
    let summary = "Remove a size-1 dimension from a tensor.";
    let arguments = (ins TT_Tensor:$src, I32Attr:$axis);
    let results = (outs TT_Tensor:$result);
    let assemblyFormat = "$src attr-dict `:` type($src) `->` type($result)";
}

#endif // XLA_BACKENDS_GPU_CODEGEN_TRITON_IR_TRITON_XLA_OPS_TD_
