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
      Pure,
      AttrSizedOperandSegments,
      OffsetSizeAndStrideOpInterface,
      DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>
  ])> {
  let results = (outs AnyRankedTensor:$result);
  code extraBaseClassDeclaration = [{
    /// Return the type of the base tensor operand.
    ::mlir::RankedTensorType getSrcType() {
      return ::llvm::cast<RankedTensorType>(getSrc().getType());
    }

    /// Return the type of the result tensor.
    ::mlir::RankedTensorType getResultType() {
      return ::llvm::cast<RankedTensorType>(getResult().getType());
    }

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

    /// Return the expected rank of each of the `static_offsets`, `static_sizes`
    /// and `static_strides` attributes.
    std::array<unsigned, 3> getArrayAttrMaxRanks() {
      unsigned rank = getResultType().getRank();
      return {rank, rank, rank};
    }
  }];
  let hasVerifier = 1;
  let hasCanonicalizer = 0;
}

def TTXLA_ExtractOp : TTXLA_OpWithOffsetSizesAndStrides<"extract", []> {
  let summary = "Extract a tile from a tensor.";
  let description = [{
    Offsets, strides, and destination tensor size are used to specify the tile
    to extract. Minor-to-major layout should also be specified.

    Example:
      ```
      %tensor: tensor<120x320xbf16>
      %of = arith.constant 0 : index
      %st1 = arith.constant 120 : index
      %st2 = arith.constant 1 : index
      ...
      %extracted_tensor = triton_xla.extract %tensor [%of, %of][%st1, %st2]
        {layout = array<i64:1, 0>} : tensor<120x320xbf16> to tensor<16x64xbf16>
      ```
  }];
  let builders = [
    OpBuilder<(ins "RankedTensorType":$result_type, "Value":$src,
      "ValueRange":$offsets,  "ValueRange":$strides,
      CArg<"ArrayRef<int64_t>", "{}">:$layout,
      CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs)>,
    OpBuilder<(ins "RankedTensorType":$result_type, "Value":$src,
      "ArrayRef<OpFoldResult>":$offsets,  "ArrayRef<OpFoldResult>":$strides,
      CArg<"ArrayRef<int64_t>", "{}">:$layout,
      CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs)>
  ];

  let arguments = (ins
    AnyRankedTensor:$src,
    Variadic<Index>:$offsets,
    Variadic<Index>:$sizes,
    Variadic<Index>:$strides,
    DenseI64ArrayAttr:$static_offsets,
    DenseI64ArrayAttr:$static_sizes,
    DenseI64ArrayAttr:$static_strides,
    DenseI64ArrayAttr:$layout
  );
  let assemblyFormat = [{
    $src ``
    custom<DynamicIndexList>($offsets, $static_offsets)
    custom<DynamicIndexList>($sizes, $static_sizes)
    custom<DynamicIndexList>($strides, $static_strides)
    attr-dict `:` type($src) `to` type($result)
  }];
  let extraClassDeclaration = extraBaseClassDeclaration # [{
    /// Return the number of leading operands before the `offsets`, `sizes` and
    /// and `strides` operands.
    static unsigned getOffsetSizeAndStrideStartOperandIndex() { return 1; }
  }];
}

def TTXLA_InsertOp : TTXLA_OpWithOffsetSizesAndStrides<"insert", [
    TypesMatchWith<"expected result type to match dest type",
                   "dst", "result", "$_self">
  ]> {
  let summary = "Insert a tile into a tensor.";
  let description = [{
    Offsets, strides, and source tensor size are used to specify the tile to
    insert into. Minor-to-major layout should also be specified.

    Example:
      ```
      %src: tensor<16x64xbf16>
      %dst: tensor<120x320xbf16>
      %of = arith.constant 0 : index
      %st1 = arith.constant 120 : index
      %st2 = arith.constant 1 : index
      ...
      %updated_tensor = triton_xla.insert %src into %dst [%of, %of][%st1, %st2]
        {layout = array<i64:1, 0>} : tensor<16x64xbf16> into tensor<120x320xbf16>
      ```
  }];
  let builders = [
    OpBuilder<(ins "Value":$src, "Value":$dst,
      "ValueRange":$offsets,  "ValueRange":$strides,
      CArg<"ArrayRef<int64_t>", "{}">:$layout,
      CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs)>,
    OpBuilder<(ins "Value":$src, "Value":$dst,
      "ArrayRef<OpFoldResult>":$offsets,  "ArrayRef<OpFoldResult>":$strides,
      CArg<"ArrayRef<int64_t>", "{}">:$layout,
      CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs)>
  ];

  let arguments = (ins
    AnyRankedTensor:$src,
    AnyRankedTensor:$dst,
    Variadic<Index>:$offsets,
    Variadic<Index>:$sizes,
    Variadic<Index>:$strides,
    DenseI64ArrayAttr:$static_offsets,
    DenseI64ArrayAttr:$static_sizes,
    DenseI64ArrayAttr:$static_strides,
    DenseI64ArrayAttr:$layout
  );

  let assemblyFormat = [{
    $src `into` $dst ``
    custom<DynamicIndexList>($offsets, $static_offsets)
    custom<DynamicIndexList>($sizes, $static_sizes)
    custom<DynamicIndexList>($strides, $static_strides)
    attr-dict `:` type($src) `into` type($dst)
  }];

  let extraClassDeclaration = extraBaseClassDeclaration # [{
    /// Return the number of leading operands before the `offsets`, `sizes` and
    /// and `strides` operands.
    static unsigned getOffsetSizeAndStrideStartOperandIndex() { return 2; }
  }];
}

#endif // XLA_BACKENDS_GPU_CODEGEN_TRITON_IR_TRITON_XLA_OPS_TD_
