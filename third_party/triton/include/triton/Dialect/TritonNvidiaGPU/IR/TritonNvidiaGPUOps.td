// Copyright (c) 2023 NVIDIA Corporation & Affiliates. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining
// a copy of this software and associated documentation files
// (the "Software"), to deal in the Software without restriction,
// including without limitation the rights to use, copy, modify, merge,
// publish, distribute, sublicense, and/or sell copies of the Software,
// and to permit persons to whom the Software is furnished to do so,
// subject to the following conditions:
//
// The above copyright notice and this permission notice shall be
// included in all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
// EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
// MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
// IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
// CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
// TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
// SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef TRITONNVIDIAGPU_OPS
#define TRITONNVIDIAGPU_OPS

include "triton/Dialect/TritonNvidiaGPU/IR/TritonNvidiaGPUDialect.td"
include "triton/Dialect/TritonNvidiaGPU/IR/TritonNvidiaGPUOpInterfaces.td"
include "mlir/Dialect/Arith/IR/ArithBase.td"
include "triton/Dialect/Triton/IR/TritonTypes.td"
include "triton/Dialect/Triton/IR/TritonAttrDefs.td"
include "triton/Dialect/Triton/IR/TritonInterfaces.td"
include "triton/Dialect/Triton/IR/TritonOpInterfaces.td"
include "triton/Dialect/TritonGPU/IR/TritonGPUTypes.td"
include "triton/Dialect/TritonGPU/IR/TritonGPUTypeInterfaces.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td" // Pure
include "mlir/Interfaces/InferTypeOpInterface.td" // SameOperandsAndResultType
include "mlir/Interfaces/DestinationStyleOpInterface.td"
include "mlir/Interfaces/ViewLikeInterface.td"

def GlobalMemory : Resource<"::mlir::triton::GlobalMemory">;
def SharedMemory : Resource<"::mlir::triton::gpu::SharedMemory">;

class TTNG_Op<string mnemonic, list<Trait> traits = []> :
    Op<TritonNvidiaGPU_Dialect, mnemonic,
       !listconcat(traits, [VerifyTensorLayoutsTrait])> {
}

def TTNG_FenceAsyncSharedOp : TTNG_Op<"fence_async_shared"> {
  let arguments = (ins BoolAttr:$bCluster);

  let summary = "fence proxy async";

  let assemblyFormat = "attr-dict";

  let extraClassDeclaration = [{
    static bool isSupported(int computeCapability) {
      return computeCapability >= 90;
    }
  }];
}

def TTNG_ClusterArriveOp : TTNG_Op<"cluster_arrive", []> {
  let arguments = (ins I1Attr:$relaxed);
  let assemblyFormat = "attr-dict";
}

def TTNG_ClusterWaitOp : TTNG_Op<"cluster_wait", []> {
  let assemblyFormat = "attr-dict";
}

//
// WarpGroupDot Op
//
def TTNG_WarpGroupDotOp : TTNG_Op<"warp_group_dot", [DeclareOpInterfaceMethods<InferTypeOpInterface>,
                                                     DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
                                                     DeclareOpInterfaceMethods<DotOpInterface>,
                                                     TypesMatchWith<"result's type matches accumulator's type",
                                                                     "d", "c", "$_self">]> {
    let summary = "warp group dot";

    let description = [{
        $d = matrix_multiply($a, $b) + $c. For docs on InputPrecisionAttr, see TT_DotOp
    }];

    let arguments = (ins TTG_TensorOrMemDesc:$a,
                         TTG_TensorOrMemDesc:$b,
                         TT_FpIntTensor:$c,
                         Optional<I1>:$useC,
                         DefaultValuedAttr<TT_InputPrecisionAttr, "::mlir::triton::InputPrecision::IEEE">:$inputPrecision,
                         DefaultValuedAttr<I32Attr, "0">:$maxNumImpreciseAcc,
                         DefaultValuedAttr<BoolAttr, "false">:$isAsync);

    let results = (outs TT_FpIntTensor:$d);

    let assemblyFormat = "$a`,` $b`,` $c (`,` $useC^)? attr-dict `:` type($a) `*` type($b) `->` type($d)";

    let extraClassDeclaration = [{
      bool needsPartialAccumulator();
    }];
}

def TTNG_WarpGroupDotWaitOp : TTNG_Op<"warp_group_dot_wait", [DeclareOpInterfaceMethods<InferTypeOpInterface>,
                                                              AllTypesMatch<["inputs", "outputs"]>]> {
  let summary = "warp group dot wait";
  let arguments = (ins Variadic<TTG_TensorOrMemDesc>:$inputs, I32Attr:$pendings);
  let results = (outs Variadic<TTG_TensorOrMemDesc>:$outputs);
  let description = [{
    Waits until there are $pendings or fewer outstanding async dot operations.

    $inputs must be the tensors corresponding to the async dot ops that we're
    waiting on.  For example, if there are N pending async dot ops and we call
    `warp_group_dot_wait 1`, then $inputs must be the result of the first dot op.
  }];

  let assemblyFormat = "$inputs attr-dict `:` type($inputs)";
}

def TTNG_InitBarrierOp : TTNG_Op<"init_barrier", [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
    let summary = "Initialize a barrier in the given shared memory allocation.";

    let description = [{
        Initializes a shared memory allocation with mbarrier information.
        `alloc` is a descriptor to the shared memory allocation. `count` is the
        number of arrives expected by the barrier.

        This lowers to PTX mbarrier.init.shared::cta.b64.
    }];

    let hasVerifier = 1;
    let arguments = (ins TTG_MemDescType:$alloc,
                         I32Attr:$count);
    let assemblyFormat = "$alloc `,` $count attr-dict `:` qualified(type($alloc))";
}

def TTNG_InvalBarrierOp : TTNG_Op<"inval_barrier", [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
    let summary = "Invalidate a barrier allocation.";

    let description = [{
      Invalidate a barrier allocation so that it can be re-used. According to PTX
      spec this has to be done before any reuse of the memory used by mbarrier.

      https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-mbarrier-inval
    }];

    let hasVerifier = 1;
    let arguments = (ins TTG_MemDescType:$alloc);
    let assemblyFormat = "$alloc attr-dict `:` qualified(type($alloc))";
}

def TTNG_BarrierExpectOp : TTNG_Op<"barrier_expect", [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "Signal a barrier of an expected number of bytes to be copied.";

  let description = [{
    This signal the barrier that `size` bytes are expected to be copied. The
    associated barrier wait will block until the expected number of bytes are copied.
  }];

  let hasVerifier = 1;
  let arguments = (
    ins TTG_MemDescType:$alloc,
    I32Attr:$size,
    I1:$pred
  );

  let assemblyFormat = [{
    $alloc `,` $size attr-dict `,` $pred `:` qualified(type($alloc))
  }];
}

def TTNG_WaitBarrierOp : TTNG_Op<"wait_barrier", [
      DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
      AttrSizedOperandSegments]> {
    let summary = "wait until the mbarrier phase completes.";

    let description = [{
      Blocks the program progress until the mbarrier object in `alloc` completes
      its current phase.

      This lowers a waitloop using PTX instruction
      mbarrier.try_wait.parity.shared.b64.

      Accepts optional list of memory. If present, it is assumed that any of the
      dependencies may be accessed until the barrier completes.

      The barrier behavior is described here:
      https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-asynchronous-copy-completion-mechanisms
    }];

    let hasVerifier = 1;
    let arguments = (ins TTG_MemDescType:$alloc,
                         I32:$phase,
                         Optional<I1>:$pred,
                         Variadic<TTG_MemDescType>:$deps);
    let builders = [
      OpBuilder<(ins "Value":$alloc, "Value":$phase),
      [{
      build($_builder, $_state, alloc, phase, /*pred=*/static_cast<mlir::Value>(nullptr), /*deps=*/{});
      }]>,
      OpBuilder<(ins "Value":$alloc, "Value":$phase, "Value":$pred),
      [{
      build($_builder, $_state, alloc, phase, pred, /*deps=*/{});
      }]>,
      OpBuilder<(ins "Value":$alloc, "Value":$phase, "ValueRange":$deps),
      [{
      build($_builder, $_state, alloc, phase, /*pred=*/static_cast<mlir::Value>(nullptr), deps);
      }]>,
    ];
    let assemblyFormat = "$alloc `,` $phase attr-dict (`,` $pred^)? (`deps` $deps^)? `:` qualified(type($alloc)) (`,` type($deps)^)?";
}

def TTNG_ArriveBarrierOp : TTNG_Op<"arrive_barrier", [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "perform the arrive operation on an mbarrier";
  let description = [{
    The `ttng.arrive_barrier` operation performs the "arrive" operation on an
    mbarrier object in shared memory. The operation requires a `count` attribute
    of at least 1, and decreasing the pending arrival count of the mbarrier by
    the specific count.

    The operation accepts an optional predicate.

    Example:

    ```mlir
    ttng.arrive_barrier %barrier, 2 : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttng.arrive_barrier %barrier, 1, %pred : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ```
  }];

  let arguments = (ins
    TTG_MemDescType:$alloc,
    I32Attr:$count,
    Optional<I1>:$pred
  );

  let assemblyFormat = [{
    $alloc `,` $count (`,` $pred^)? attr-dict `:` qualified(type($alloc))
  }];

  let builders = [
    OpBuilder<(ins "Value":$alloc, "uint32_t":$count), [{
      return build($_builder, $_state, alloc, count, /*pred=*/Value());
    }]>
  ];

  let hasVerifier = 1;
}

def TTNG_TensorDescToTMAPtrOp : TTNG_Op<"tensor_desc_to_tma_ptr", [Pure]> {
  let summary = "Convert tensor descriptor to pointer to tma descriptor";

  let arguments = (ins TT_TensorDescType:$desc);
  let results = (outs TT_Ptr:$ptr);

  let assemblyFormat = [{
    $desc attr-dict `:` qualified(type($desc)) `to` qualified(type($ptr))
  }];

  let builders = [
    OpBuilder<(ins "Value":$desc), [{
      auto ptrTy = triton::PointerType::get($_builder.getI8Type(), 1);
      build($_builder, $_state, ptrTy, desc);
    }]>
  ];

  let hasCanonicalizeMethod = 1;
}


def TTNG_AsyncTMACopyGlobalToLocalOp : TTNG_Op<"async_tma_copy_global_to_local", [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "copy data based on descriptor from global memory to local memory asynchronously";

  let description = [{
    This operation copies data from global memory to local memory
    asynchronously.  This is analogue to tt.load except the data are copied to
    local memory pointed by the memory descriptor instead of a distributed
    tensor. The data copied depends on the global memory descriptor pointed to
    by `desc_ptr`.
  }];

  let hasVerifier = 1;
  let arguments = (
    ins TT_PtrType:$desc_ptr,
    Variadic<I32>:$coord,
    TTG_MemDescType:$barrier,
    TTG_MemDescType:$result,
    I1:$pred,
    DefaultValuedAttr<TT_CacheModifierAttr, "triton::CacheModifier::NONE">:$cache,
    DefaultValuedAttr<TT_EvictionPolicyAttr, "triton::EvictionPolicy::NORMAL">:$evict,
    DefaultValuedAttr<BoolAttr, "false">:$isVolatile
  );

  let assemblyFormat = [{
    $desc_ptr `[` $coord `]` $result `,` $barrier `,` $pred
    oilist(`cacheModifier` `=` $cache | `evictionPolicy` `=` $evict)
    attr-dict `:` qualified(type($desc_ptr)) `,` qualified(type($barrier)) `->` qualified(type($result))
  }];
}

def TTNG_AsyncTMACopyLocalToGlobalOp : TTNG_Op<"async_tma_copy_local_to_global", [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "copy data based on descriptor from local memory to global memory asynchronously";

  let description = [{
    This operation copies data from local memory to global memory
    asynchronously.  This is analogue to tt.store except the data are copied from
    local memory pointed by the memory descriptor instead of a distributed
    tensor. The data copied depends on the global memory descriptor pointed to
    by `desc_ptr`.
  }];

  let arguments = (
    ins TT_PtrType:$desc_ptr,
    Variadic<I32>:$coord,
    TTG_MemDescType:$src);

  let assemblyFormat = [{
    $desc_ptr `[` $coord `]` $src
    attr-dict `:` qualified(type($desc_ptr)) `,` qualified(type($src))
  }];
}

def TTNG_AsyncTMAGatherOp : TTNG_Op<"async_tma_gather", [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "gather data based on descriptor from global memory to local memory asynchronously";

  let description = [{
    This operation gathers multiple rows of data from global memory matrix to
    local memory asynchronously.  This is similar to
    async_tma_copy_global_to_local except that each row is indexed independently.
  }];

  let arguments = (ins
    TT_PtrType:$desc_ptr,
    RankedTensorOf<[I32]>:$x_offsets,
    I32:$y_offset,
    TTG_MemDescType:$barrier,
    TTG_MemDescType:$result,
    I1:$pred
  );

  let assemblyFormat = [{
    $desc_ptr `[` $x_offsets `,` $y_offset `]` $result `,` $barrier `,` $pred
    attr-dict `:` type(operands)
  }];

  let hasVerifier = 1;
}

def TTNG_AsyncTMAScatterOp : TTNG_Op<"async_tma_scatter", [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "scatter data from local memory into global memory based on a descriptor asynchronously";

  let description = [{
    The `ttng.async_tma_scatter` operation scatters multiple separately-indexed
    rows of data from local memory into global memory asynchronously. The
    operation scatters a 2D tensor in shared memory, laid out by core tensor
    tiles nvmma_shared layout into separately indexed rows in global
    memory at a given `y` offset.
  }];

  let arguments = (ins
    TT_PtrType:$desc_ptr,
    RankedTensorOf<[I32]>:$x_offsets,
    I32:$y_offset,
    TTG_MemDescType:$src
  );

  let assemblyFormat = [{
    $desc_ptr `[` $x_offsets `,` $y_offset `]` $src
    attr-dict `:` type(operands)
  }];
}

def TTNG_TMAStoreWaitOp : TTNG_Op<"async_tma_store_wait"> {
  let summary = "wait until all the inputs are read.";
  let arguments = (ins I32Attr:$pendings);
  let description = [{
    Wait until all the read operations are done from the associated store operations.
    This is needed before the shared memory can be written to.
  }];

  let assemblyFormat = "attr-dict";
}

def TTNG_TCGen5MMAOp : TTNG_Op<"tc_gen5_mma", [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>, DeclareOpInterfaceMethods<DotOpInterface>, DeclareOpInterfaceMethods<MMAv5OpInterface>]> {
    let summary = "block level op mapping to tensorcore gen5 mma";

    let description = [{
        $d += matrix_multiply($a, $b).
        If no barrier is given the op is assumed to be synchronous otherwise the op will trigger a commit/arrive on the given barrier.
        If there is a barrier the result will be safe to read after a barrier wait.
        If $two_ctas is set the op will execute a matmul across two contiguous CTAs, it will read the data distributed across the two CTAs.
        and syncronize both CTAs if the op is synchronous.
    }];

    let arguments = (ins TTG_MemDescType:$a,
                         TTG_MemDescType:$b,
                         TTG_MemDescType:$d,
                         I1:$useD,
                         I1:$pred,
                         Optional<TTG_MemDescType>:$barrier,
                         OptionalAttr<UnitAttr>:$two_ctas);

    // TODO: improve printing format.
    let assemblyFormat = "$a`,` $b`,` $d`,` $useD`,` $pred (`,` $barrier^)? attr-dict `:` functional-type(operands, results)";
}

def TTNG_TCGen5MMAScaledOp : TTNG_Op<"tc_gen5_mma_scaled", [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>, DeclareOpInterfaceMethods<DotOpInterface>, DeclareOpInterfaceMethods<MMAv5OpInterface>]> {
    let summary = "block level op mapping to tensorcore gen5 mma";

    let description = [{
        $d += matrix_multiply(scale($lhs, $lhs_scale), scale(rlhs, $rhs_scale))
        If no barrier is given the op is assumed to be synchronous otherwise the op will trigger a commit/arrive on the given barrier.
        If there is a barrier the result will be safe to read after a barrier wait.
    }];

    let arguments = (ins TTG_MemDescType:$a,
                         TTG_MemDescType:$b,
                         TTG_MemDescType:$d,
                         TTG_MemDescType:$a_scale,
                         TTG_MemDescType:$b_scale,
                         TT_ScaleDotElemTypeAttr:$a_type,
                         TT_ScaleDotElemTypeAttr:$b_type,
                         I1:$useD,
                         I1:$pred,
                         Optional<TTG_MemDescType>:$barrier);

    // TODO: improve printing format.
    let assemblyFormat = "$a `,` $b `,` $d `,` $a_scale `,` $b_scale `,` $useD`,` $pred `lhs` `=` $a_type `rhs` `=` $b_type (`,` $barrier^)? attr-dict `:` functional-type(operands, results)";
}

def TTNG_TMEMLoadOp : TTNG_Op<"tmem_load", [MemoryEffects<[MemRead]>]> {
  let summary = "Load a buffer from tensor memory into a distributed tensor";

  let description = [{
    This is similar to ttg.local_load except the result layout is restricted to only few possibility.
    Therefore we cannot combine this op with any convert layout like local_load.
  }];
  let arguments = (ins TTG_MemDescType:$src);

  let assemblyFormat = [{$src attr-dict `:` qualified(type($src)) `->` type($result)}];
  let results = (outs TT_Tensor:$result);
  let hasVerifier = 1;
}

def TTNG_TMEMStoreOp : TTNG_Op<"tmem_store", [MemoryEffects<[MemWrite]>]> {
  let summary = "Store a distributed tensor into a buffer in tensor memory";

  let description = [{
    This is similar to ttg.local_local except the source layout is restricted to only few possibility.
  }];
  let arguments = (ins TTG_MemDescType:$dst, TT_Tensor:$src, I1:$pred);

  let assemblyFormat = [{$src `,` $dst `,` $pred attr-dict `:` type($src) `->` qualified(type($dst))}];
  let hasVerifier = 1;
}

def TTNG_TMEMAllocOp : TTNG_Op<"tmem_alloc", [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "allocate tensor memory";
  let description = [{
    This operation allocates buffer in tensor memory and return a descriptor
    containing the address and a view of the buffer.
    This is similar to ttg.local_alloc except the buffer is allocated in tensor memory.

    Explicitly deallocating a buffer is optional; see local_dealloc.
  }];
  let arguments = (ins Optional<TT_Tensor>:$src);

  let assemblyFormat = [{
    ($src^)? attr-dict `:` functional-type(operands, results)
  }];

  let results = (outs TTG_MemDescType:$result);
  let hasVerifier = 1;
}

def TTNG_TMEMCopyOp : TTNG_Op<"tmem_copy", [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "Initiate an asynchronous copy operation from shared memory to the Tensor Memory.";

  let description = [{
    2D blocks stored contiguously in SMEM are copied into TMEM as specified by the destination address.
    The completion of the copy can be observed by waiting on the optional barrier. If this op is used
    together with an MMA op, one barrier can be used to wait for both copy and MMA. We do not need to wait
    for the completion of the copy before MMA, since tcgen05.cp followed by tcgen05.mma is guaranteed to
    execute in that order.

    This op lowers to the PTX instruction tcgen05.cp. Right now, we only support 1CTA and the warpx4.32x128b
    variant of the instruction. Each 32x128b block in SMEM is duplicated over 4 warps and stored into 128 rows
    and 4 columns of TMEM. The primary use case of this op is to copy blocked scales from SMEM to TMEM.

    The shape of the input SMEM can be flexibily chosen depending on use cases. In the simplest case (e.g. unit test),
    the source SMEM can be of shape (32 x num_blocks, 16), and the destination TMEM should be of shape (128, 16 x num_blocks),
    for copying 8 bit values. For scaled GEMM, rep_m x rep_k copies of a 32x128b block need to be stored in SMEM, where
    rep_m = BLOCK_M / 128, rep_k = BLOCK_K / scale_vec_size / 4, and scale_vec_size = 32 for MXFP.
    Conceptually, the SMEM is organized in a high-dimensional layout, (rep_m, rep_k, 32, 4, 4B).
    Some of axes can be flattened into one, to reduce the rank of the load. For example, the following patterns are supported:
     * (rep_m, rep_k * 32 x 4 x 4B), 2D scale load with cp.async
     * (rep_m, rep_k, 32, 16B), 4D scale load with TMA
     * (rep_m, rep_k, 32, 4, 4B), 5D scale load with cp.async
    Since rep_m blocks are not contiguous in SMEM, this axis cannot be flattened into inner ones.

    In Triton, the TMEM memdesc for blocked scales must be of the following form:
    * Its shape must be (BLOCK_MN, BLOCK_K / scale_vec_size), representing the logical shape of blocked scales.
    * It must be attached with `tensor_memory_scales_encoding` to indicate the chunk-based layout and its duplication over 4 warps.

    In contrast, the src SMEM must be in the explicit chunk-based layout as described above. So the IR might look like this:

    %0 = ttng.tmem_alloc : () -> !ttg.memdesc<128x4xi8, #tmem_scales, #ttng.tensor_memory>
    ttng.tmem_copy %1, %0 : (!ttg.memdesc<1x1x32x4x4xi8, #shared1, #smem>, !ttg.memdesc<128x4xi8, #tmem_scales, #ttng.tensor_memory>) -> ()

    We interpret the semantics of this copy operation as follows. The chunk-based layout in SMEM implies that
    the logical shape (BLOCK_MN, BLOCK_K / scale_vec_size) in TMEM is the result of certain reshape and transpose operations.
    In practice, to take an advantage of the native scale layout and the TMEM copy op,  users need to do
    `scales5D.trans(0, 3, 2, 1, 4).reshape(BLOCK_M, BLOCK_K // scale_vec_size)` before feeding scales into dot_scaled.
    When we use tmem_copy in the IR, such reshape and transpose operations are removed. But the change in the logical shape they have caused on
    registers is now understood to be incorporated into tmem_copy itself. Ideally, we would lift reshape / transpose done on registers onto
    the SMEM memdesc, making tmem_copy a straightforward 2D copy operation: (BLOCK_MN, BLOCK_K / scale_vec_size) -> (BLOCK_MN, BLOCK_K / scale_vec_size).
    In the absence of such operations on memdesc, we resort to implicitly encoding the reshape/transpose semantics in tmem_copy.

  }];
  let arguments = (ins TTG_MemDescType:$src, TTG_MemDescType:$dst, Optional<TTG_MemDescType>:$barrier);

  let assemblyFormat = [{$src `,` $dst `,` $barrier attr-dict `:` functional-type(operands, results)}];
  let hasVerifier = 1;
}

#endif
