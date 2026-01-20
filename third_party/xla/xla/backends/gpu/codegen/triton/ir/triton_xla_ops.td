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
include "xla/backends/gpu/codegen/triton/ir/triton_xla_attrs.td"
include "xla/backends/gpu/codegen/triton/ir/triton_xla_dialect.td"
include "triton/Dialect/Triton/IR/TritonAttrDefs.td" // TT_MemSemanticAttr, TT_MemSyncScopeAttr
include "triton/Dialect/Triton/IR/TritonInterfaces.td"
include "triton/Dialect/Triton/IR/TritonOpInterfaces.td"
include "triton/Dialect/Triton/IR/TritonTypes.td"
include "triton/Dialect/TritonGPU/IR/TritonGPUTypeInterfaces.td"


// -----------------------------------------------------------------------------
// Interfaces
// -----------------------------------------------------------------------------
// Copied over from TritonOps.td which cannot be directly included here.
def GlobalMemory : Resource<"::mlir::triton::GlobalMemory">;

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
        as memref<512x128xbf16, #xtile.layout<[1, 0]>>
        [0, 0] [16, 64] [128, 1] : tensor<16x64xbf16>
      ```
  }];
  let builders = [
    OpBuilder<(ins "RankedTensorType":$result_type, "Value":$src,
      "ValueRange":$offsets, "ArrayRef<int64_t>":$sizes,
      "ArrayRef<int64_t>":$strides, "ArrayRef<int64_t>":$src_shape,
      "ArrayRef<int64_t>":$src_layout)>,
    OpBuilder<(ins "RankedTensorType":$result_type, "Value":$src,
      "ArrayRef<OpFoldResult>":$offsets, "ArrayRef<int64_t>":$sizes,
      "ArrayRef<int64_t>":$strides, "ArrayRef<int64_t>":$src_shape,
      "ArrayRef<int64_t>":$src_layout)>
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
      unsigned rank = getSrcShape().size();
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
        as memref<512x128xbf16, #xtile.layout<[1, 0]>>
        [0, 0] [8, 8] [1, 1] : tensor<8x8xbf16>
      ```
  }];
  let builders = [
    OpBuilder<(ins "Value":$src, "Value":$dst,
      "ValueRange":$offsets, "ArrayRef<int64_t>":$sizes,
      "ArrayRef<int64_t>":$strides, "ArrayRef<int64_t>":$dst_shape,
      "ArrayRef<int64_t>":$dst_layout)>,
    OpBuilder<(ins "Value":$src, "Value":$dst,
      "ArrayRef<OpFoldResult>":$offsets, "ArrayRef<int64_t>":$sizes,
      "ArrayRef<int64_t>":$strides, "ArrayRef<int64_t>":$dst_shape,
      "ArrayRef<int64_t>":$dst_layout)>
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
      unsigned rank = getDstShape().size();
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

def TTXLA_GetTidOp : TTXLA_Op<"get_tid", [Pure]> {
  let summary = "Get the thread ID within a triton kernel.";
  let description = [{
    This operation returns the thread ID within a kernel.
    Only the X dimension is supported. The result is a 32-bit
    integer which in CUDA terms is threadIdx.x.
  }];
  let arguments = (ins);
  let results = (outs I32:$result);
  let assemblyFormat = "attr-dict `:` functional-type(operands, $result)";
}

def TTXLA_AtomicWriteOp : TTXLA_Op<"atomic_write", []> {
  let summary = "Atomically write u32 value(s) to memory location(s).";

  let description = [{
      Atomically write u32 value(s) to the memory location(s) specified by
      $ptr. The memory ordering guarantees are specified by the
      $mem_sync_(scope|semantic) attribute.

      An optional $mask argument can be provided to specify which of the
      sublanes/threads should be updated. The mask must be the same shape as
      the $ptr.

      Memory Synchronization Scope:
        - cta: The write is visible to all threads within the same compute
          thread-block.
        - gpu: The write is visible to all threads within the same GPU.
        - sys: The write is visible to all threads within the system.

      The following memory synchronization semantics are allowed:
        - relaxed: No memory ordering guarantees.
        - release: All writes that are issued before this one will be
          completed before the store is visible to other threads.

      Internally this operation expands into st.global.<scope>.<semantic>.u32
  }];

  let arguments = (ins
    Arg<TT_PtrLike, "",
      [MemWrite<GlobalMemory>]>:$ptr,
    TT_I32Like:$value,
    Optional<TT_BoolLike>:$mask,
    TT_MemSyncScopeAttr:$mem_sync_scope,
    TT_MemSemanticAttr:$mem_sync_semantic
  );

  // An atomic write has no results.
  let results = (outs);

  // We keep the mem_sync_scope and mem_sync_semantic attributes in the
  // assembly format to allow parsing them from strings.
  // The alternative through attr-dict makes the string form look like:
  // { mem_sync_scope = 1, mem_sync_semantic = 3 } which is not very readable.
  let assemblyFormat = [{
    $mem_sync_scope `,` $mem_sync_semantic `,`
    $ptr `,` $value (`,` $mask^) ? attr-dict `:`
    functional-type(operands, results)
  }];
}

// NB: MemoryWrite is a workaround to prevent this op from being cleaned up
// during dead code elimination.
def TTXLA_AtomicSpinWaitOp : TTXLA_Op<"atomic_spin_wait",
  [MemoryEffects<[MemWrite]>]> {
    let summary = "Spin-wait while reading u32 value(s) atomically.";

    let description = [{
      Atomically read u32 value(s) from the memory location(s) specified by
      $ptr. The memory ordering guarantees are specified by the
      $mem_sync_(scope|semantic) attributes.

      The loaded value is compared to $expected using the $comparator.
      As long as the comparison returns true, the spin-wait loop continues.
      If the comparison returns false, the spin-wait loop is finished.

      NB: In the end only a single lane/thread within a program will
      write to (or read from) a given memory location so the synchronization
      happens between a single writing lane with a single reading lane. However,
      the operation will wait until all lanes within the block have synchronized
      so the end result is a block level memory synchronization.

      For Memory Synchronization Scope, see comment on TTXLA_AtomicWriteOp.
      The following memory synchronization semantics are allowed:
        - relaxed: No memory ordering guarantees.
        - acquire: No reads or writes in the current block/program can be
          reordered before this wait. All writes in other programs that
          `release` the same memory location are visible in the current block.

      Internally this operation expands into elementwise
      `ld.global.<semantic>.<scope>.u32` and a labelled loop.
    }];

    let arguments = (ins
      Arg<TT_PtrLike, "", [MemRead<GlobalMemory>]>:$ptr,
      TT_I32Like:$expected,
      Optional<TT_BoolLike>:$mask,
      TT_MemSyncScopeAttr:$mem_sync_scope,
      TT_MemSemanticAttr:$mem_sync_semantic,
      TTXLA_ComparatorAttr:$comparator
    );

    // An atomic spin-wait returns no results.
    let results = (outs);

    // See comment on TTXLA_AtomicWriteOp for why we explicitly add the mem sync
    // attributes to the assembly format.
    let assemblyFormat = [{
      $mem_sync_scope `,` $mem_sync_semantic `,`
      $ptr `,` $comparator `,` $expected (`,` $mask^) ?  attr-dict `:`
      functional-type(operands, results)
    }];
}

def TTXLA_BlockBarrierOp : TTXLA_Op<"block_barrier", []> {
  let summary = "A block-level barrier for inter-GPU synchronization.";

  let description = [{
    This operation implements a block-level barrier between multiple GPUs.
    It takes a pointer of pointers to signal buffers (one from each
    participating GPU), the rank of the current device, and a signal value to
    write/wait for. The signal values on consequent calls to this operation must
    increase monotonically for the same signal buffers pointer. A block on one
    GPU synchronizes with blocks on other GPUs with the same block id.

    *NOTE* Block orchestration order is not guaranteed across GPUs,
    so there is a chance of a deadlock should the number of blocks be greater
    than the number of SMs on a GPU. That is to say that if BlockN is launched
    on GPU0 and BlockM on GPU1 and they both wait for each other to finish,
    and there is no place to schedule more blocks on either GPU, then we
    will have a deadlock. In general, the user of this operation should avoid
    this by not oversubscribing the number of available SMs on a GPU. There is
    also a secondary restriction that the number of threads per block must
    exceed the number of participating GPUs. In the nominal case, where we are
    within a single NV Link domain, this should not be an issue since the number
    of participating GPUs would usually be much smaller than the number of
    threads per block.

    The synchronization logic is as follows:
    1. Each GPU (identified by `device_rank`) writes the `signal` value to an
       index `blockId * world_size + device_rank` in *every* buffer provided in
       `$signal_buffers`.
    2. After writing, GPU `r` polls its own buffer (`$signal_buffers[r]`).
    3. It waits until the values at locations corresponding to all other ranks
       (i.e., `blockId * world_size + [0..world_size)` are greater than or equal
       to the `signal` value.
    4. When the wait is over, all GPUs working on the same `blockId` have
       successfully synchronized.

    This operation ensures that GPUs in the same synchronization group do not
    get more than one "wave" (or chunk) of data processing ahead of each other.
    For clarity, a wave here is a set of work items that are processed together.
    A single thread block for instance can process multiple waves. In such a
    scenario, any given GPU can be at most 1 wave ahead of the other GPUs for
    the same block id.

    Internally, this operation uses TTXLA_Atomic(Write|SpinWait)Op on signal
    buffers to synchronize all participating GPUs.
  }];

  let arguments = (ins
    // Pointers of signal buffers (one from each participating GPU).
    TT_PtrOf<[TT_PtrOf<[I32]>]>:$signal_buffers,
    // The rank of the current device.
    I32:$device_rank,
    // The signal value to write/wait for.
    I32:$signal_value,
    // The number of devices participating in the barrier.
    // The length of the first dimension of the signal buffers should be the
    // same as this value.
    I32Attr:$world_size
  );

  // This is a barrier op and produces no results.
  let results = (outs);

  let assemblyFormat = [{
    $signal_buffers `,` $device_rank `,` $signal_value `,` attr-dict `:`
    functional-type(operands, results)
  }];
}

def TTXLA_GetRankOp : TTXLA_Op<"get_rank", [Pure]> {
  let summary = "Extract the device rank from the collectives metadata.";
  let arguments = (ins
    Arg<TT_PtrLike, "",
      [MemRead<GlobalMemory>]>:$metadata);
  let results = (outs I64:$result);
  let assemblyFormat = "$metadata attr-dict `:` type($metadata) `->` type($result)";
}

def TTXLA_GetPeerPtrOp : TTXLA_Op<"get_peer_ptr", [Pure]> {
  let summary = [{
    Extract the pointer to the given symmetric memory `address` on the given
    `peer` device. An `address` should point to the memory of the given kernel
    argument with `argument_index`. The result is calculated using the symmetric
    memory `metadata` constructed at the runtime.
    To calculate offsets operation also need to know the number of devices
    participating in the collective operation (`world_size`).
  }];
  let arguments = (ins
    Arg<TT_PtrLike, "",
      [MemRead<GlobalMemory>]>:$address,
    I64:$peer_id,
    Arg<TT_PtrLike, "",
      [MemRead<GlobalMemory>]>:$metadata,
    I32Attr:$argument_index,
    // The number of devices participating in the collective operation.
    I32Attr:$world_size);

  let results = (outs Arg<TT_PtrLike, "", [MemRead<GlobalMemory>]>:$result);

  let assemblyFormat = [{
    $address `,` $peer_id `,` $metadata `,` attr-dict `:`
    functional-type(operands, results)
  }];
}

def TTXLA_MemrefToPtrOp : TTXLA_Op<"memref_to_ptr", [Pure]> {
  let summary = [{
    A specialized version of unrealized_conversion_cast that converts a
    memref to a pointer.
  }];

  let arguments = (ins AnyMemRef:$src);

  let results = (outs TT_Ptr:$result);

  let assemblyFormat = [{
    $src `from` type($src) `to` type($result) attr-dict
  }];

  let hasFolder = 1;
  let hasVerifier = 1;
}

def TTXLA_PtrToMemrefOp : TTXLA_Op<"ptr_to_memref", [Pure]> {
  let summary = [{
    A specialized version of unrealized_conversion_cast that converts a
    pointer to a memref.
  }];

  let arguments = (ins TT_Ptr:$src);

  let results = (outs AnyMemRef:$result);

  let assemblyFormat = [{
    $src `from` type($src) `to` type($result) attr-dict
  }];

  let hasVerifier = 1;
}


#endif // XLA_BACKENDS_GPU_CODEGEN_TRITON_IR_TRITON_XLA_OPS_TD_

