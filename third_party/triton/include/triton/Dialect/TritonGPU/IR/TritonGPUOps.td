#ifndef TRITONGPU_OPS
#define TRITONGPU_OPS

include "triton/Dialect/TritonGPU/IR/TritonGPUDialect.td"
include "triton/Dialect/TritonGPU/IR/TritonGPUTypes.td"
include "triton/Dialect/TritonGPU/IR/TritonGPUTypeInterfaces.td"
include "triton/Dialect/TritonGPU/IR/TritonGPUAttrDefs.td"
include "mlir/Dialect/Arith/IR/ArithBase.td"
include "triton/Dialect/Triton/IR/TritonTypes.td"
include "triton/Dialect/Triton/IR/TritonAttrDefs.td"
include "triton/Dialect/Triton/IR/TritonOpInterfaces.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/ControlFlowInterfaces.td" // RegionBranchOpInterface
include "mlir/Interfaces/DestinationStyleOpInterface.td"
include "mlir/Interfaces/InferTypeOpInterface.td"  // SameOperandsAndResultType
include "mlir/Interfaces/SideEffectInterfaces.td"  // Pure
include "mlir/Interfaces/ViewLikeInterface.td"

//
// Interfaces
//
def GlobalMemory : Resource<"::mlir::triton::GlobalMemory">;
def SharedMemory : Resource<"::mlir::triton::gpu::SharedMemory">;

class TTG_Op<string mnemonic, list<Trait> traits = []> :
    Op<TritonGPU_Dialect, mnemonic,
       !listconcat(traits, [VerifyTensorLayoutsTrait])> {
}

def TTG_ConvertLayoutOp : TTG_Op<"convert_layout",
                                 [SameOperandsAndResultShape,
                                  SameOperandsAndResultElementType,
                                  Pure]> {
  let summary = "convert layout";

  let arguments = (ins TT_Tensor:$src);

  let results = (outs TT_Tensor:$result);

  let hasCanonicalizer = 1;

  let assemblyFormat = "$src attr-dict `:` type($src) `->` type($result)";
}

def TTG_AsyncWaitOp : TTG_Op<"async_wait"> {
  let summary = "async wait";

  let arguments = (ins Variadic<TTG_AsyncToken>:$asyncToken, I32Attr:$num);

  let results = (outs TTG_AsyncToken:$retToken);

  let assemblyFormat = "$asyncToken attr-dict";

  let extraClassDeclaration = [{
    static bool isSupported(int computeCapability) {
      return computeCapability >= 80;
    }
  }];
}

def TTG_AsyncCommitGroupOp : TTG_Op<"async_commit_group"> {
  let summary = "async commit group";

  let results = (outs TTG_AsyncToken:$asyncToken);
  let arguments = (ins Variadic<TTG_AsyncToken>:$inputTokens);

  let assemblyFormat = [{
    $inputTokens attr-dict
  }];

  let extraClassDeclaration = [{
    static bool isSupported(int computeCapability) {
      return computeCapability >= 80;
    }
  }];
}

def TTG_AsyncCopyGlobalToLocalOp : TTG_Op<"async_copy_global_to_local", [
  AttrSizedOperandSegments,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
  TypesMatchWith<"infer mask type from src type",
                 "src", "mask", "getI1SameShape($_self)",
                 "($_op.getOperands().size() <= 3) || std::equal_to<>()">,
  TypesMatchWith<"infer other type from src type",
                 "src", "other", "getPointeeType($_self)",
                 "($_op.getOperands().size() <= 4) || std::equal_to<>()">
]> {
  let summary = "copy data from global memory to local memory asynchronously";

  let hasVerifier = 1;
  let description = [{
    This operation copies data from global memory to local memory asynchronously.
    This is analogue to tt.load except the data are copied to local memory pointed
    by by the memory descriptor instead of a distributed tensor. The rest of the
    operands are the same as tt.load.
  }];

  let arguments = (
    ins TT_PtrTensor:$src,
    TTG_MemDescType:$result,
    Optional<I1Tensor>:$mask,
    Optional<TT_Type>:$other,
    DefaultValuedAttr<TT_CacheModifierAttr, "triton::CacheModifier::NONE">:$cache,
    DefaultValuedAttr<TT_EvictionPolicyAttr, "triton::EvictionPolicy::NORMAL">:$evict,
    DefaultValuedAttr<BoolAttr, "false">:$isVolatile
  );

  let builders = [
      OpBuilder<(ins "Value":$src, "Value":$result,
                     "triton::CacheModifier":$cache,
                     "triton::EvictionPolicy":$evict, "bool":$isVolatile)>,
  ];

  let results = (outs TTG_AsyncToken:$token);

  let extraClassDeclaration = [{
    static DenseSet<unsigned> getEligibleLoadByteWidth(int computeCapability) {
      DenseSet<unsigned> validLoadBytes;
      if (computeCapability >= 80) {
        validLoadBytes = {4, 8, 16};
      }
      return validLoadBytes;
    }
  }];

  // Specify cacheModifier and evictionPolicy explicitly, instead of leaving
  // them in attr-dict, because this way their values get printed as strings,
  // rather than as opaque integers.
  //
  // Note there are no commas between other, cacheModifier, and evictionPolicy,
  // due to limitations in MLIR's asm parser.
  let assemblyFormat = [{
    $src `,` $result (`mask` $mask^)? (`other` $other^)?
    oilist(`cacheModifier` `=` $cache | `evictionPolicy` `=` $evict)
    attr-dict `:` type($src) `->` type($result)
  }];
}


// Allocate shared memory
def TTG_LocalAllocOp : TTG_Op<"local_alloc", [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "allocate tensor";
  let description = [{
    This operation allocates buffer in shared memory and return a descriptor
    containing the address and a view of the buffer.

    Explicitly deallocating a buffer is optional; see local_dealloc.
  }];
  let arguments = (
    ins
    Optional<TT_Tensor>:$src,
    OptionalAttr<I32Attr>:$alignment
  );

  let builders = [
    OpBuilder<(ins "Type":$result),
              [{ build($_builder, $_state, result, Value(), IntegerAttr()); }]>,
    OpBuilder<(ins "Type":$result, "Value":$src),
              [{ build($_builder, $_state, result, src, IntegerAttr()); }]>,
    OpBuilder<(ins "Type":$result, "Value":$src, "int32_t":$alignment),
              [{ build($_builder, $_state, result, src, $_builder.getI32IntegerAttr(alignment)); }]>
  ];

  let extraClassDeclaration = [{
    bool isSharedMemoryAlloc() {
      return getType().getMemorySpace() &&
             isa<SharedMemorySpaceAttr>(getType().getMemorySpace());
    }
    int32_t getAlignmentOrDefault();
  }];
  let assemblyFormat = [{
    ($src^)? attr-dict `:` functional-type(operands, results)
  }];

  let results = (outs TTG_MemDescType:$result);
  let hasFolder = 1;
  let hasVerifier = 1;
}

// Deallocate shared memory
def TTG_LocalDeallocOp : TTG_Op<"local_dealloc", [MemoryEffects<[MemFree<SharedMemory>]>]> {
  let summary = "dealloc buffer";

  let description = [{
    This operation deallocates a buffer explicitly. Using the buffer after this
    operation is undefined.

    This operation is optional.  If you don't explicitly dealloc a buffer, the
    compiler assumes it's deallocated at the first point that post-dominates all
    uses of the alloc.

    Because we assume a memdesc is dead at the first point that post-dominates
    its uses, ops that wait for an async operation on a memdesc to complete
    (such as ttng.warp_group_dot_wait) should also take the memdesc as an
    operand.
  }];

  let arguments = (ins TTG_MemDescType:$src);

  // Use qualified() otherwise "!ttg.memdesc<X>" is printed as "<X>".
  let assemblyFormat = [{$src attr-dict `:` qualified(type($src))}];
}

def TTG_MemDescSubviewOp : TTG_Op<"memdesc_subview", [Pure]> {
  let summary = "take a subview of the descriptor.";

  let description = [{
    This operation returns a new descriptor representing a subview of the buffer.
    It doesn't affect the underlying memory. The subview can be rank-reduced.

    For example, suppose that
     - the input shape is 2x4x16xf16,
     - the output shape is 4x4xf16, and
     - offsets = [1, 0, 4].

    Then in Python syntax, the subview covers input[1][0:4][4:8].
  }];
  let arguments = (
    ins TTG_MemDescType:$src, Variadic<I32>:$offsets);

  // Use qualified() otherwise "!ttg.memdesc<X>" is printed as "<X>".
  let assemblyFormat = [{$src `[` $offsets `]` attr-dict `:` qualified(type($src)) `->` qualified(type($result))}];

  let results = (outs TTG_MemDescType:$result);

  let hasVerifier = 1;
}

def TTG_MemDescTransOp : TTG_Op<"memdesc_trans", [Pure,
                                                  TransposeOpInterface,
                                                  InferTypeOpWithLayoutEquivalence,
                                                  SameOperandsAndResultElementType]> {
  let summary = "transpose the descriptor";

  let description = [{
    This operation returns a new descriptor
    representing a transposed view of the buffer.
  }];

  let arguments = (ins TTG_MemDescType:$src, Variadic<I32>:$order);

  let arguments = (
    ins TTG_MemDescType:$src,
    DenseI32ArrayAttr:$order
  );

  let results = (outs TTG_MemDescType:$result);

  let assemblyFormat = "$src attr-dict `:` qualified(type($src)) `->` qualified(type($result))";

  let hasFolder = 1;
}

def TTG_LocalLoadOp : TTG_Op<"local_load", [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "Load a buffer from local memory into a distributed tensor";

  let description = [{
    Load a tensor from the local memory descriptor into a distributed tensor.
  }];
  let arguments = (ins TTG_MemDescType:$src, Optional<TTG_AsyncToken> :$token);

  let builders = [
      OpBuilder<(ins "Type":$retType, "Value":$src),
      [{
      build($_builder, $_state, retType, src, /*token=*/static_cast<mlir::Value>(nullptr));
      }]>];

  // Use qualified() otherwise "!ttg.memdesc<X>" is printed as "<X>".
  let assemblyFormat = [{$src (`token` $token^)? attr-dict `:` qualified(type($src)) `->` type($result)}];

  let results = (outs TT_Tensor:$result);
}

def TTG_LocalStoreOp : TTG_Op<"local_store", [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "Store a distributed tensor into a buffer in local memory";

  let description = [{
    Store a distributed tensor into a buffer in local memory.
  }];
  let arguments = (ins TT_Tensor:$src, TTG_MemDescType:$dst);

  let hasVerifier = 1;
  // Use qualified() otherwise "!ttg.memdesc<X>" is printed as "<X>".
  let assemblyFormat = [{
    $src `,` $dst attr-dict `:` type($src) `->` qualified(type($dst))
  }];
}

def TTG_Fp4ToFpOp : TTG_Op<"fp4_to_fp", [Pure]> {
  let summary = "Upcast fp4 (e2m1) to fp";

  let hasVerifier = 1;

  let description = [{
    Upcast fp4 (e2m1) represented packed as i8s to fp.

    The lower 4 bits of the i8s represent the first fp4 element, and the upper 4 bits
    the second fp4 element.

    The `axis` attribute specifies the axis along which the fp4 elements are packed.
  }];

  let builders = [
      OpBuilder<(ins "TypedValue<RankedTensorType>":$src, "Type":$elemType, "int32_t":$axis)>
    ];

  let arguments = (ins RankedTensorOf<[I8]>:$src, I32Attr:$axis);
  let results = (outs TT_FloatTensor:$result);

  let assemblyFormat = [{
    $src attr-dict `:` type($src) `->` type($result)
  }];
}

// Allocate global memory
def TTG_GlobalScratchAllocOp : TTG_Op<"global_scratch_alloc", [MemoryEffects<[MemAlloc<GlobalMemory>]>]> {
  let summary = "allocate a global memory buffer";
  let description = [{
    This operation allocates a buffer in global memory that is private to the current program.
  }];
  let arguments = (
    ins
    I32Attr:$nbytes,
    I32Attr:$alignment
  );
  let results = (outs TT_Ptr:$result);

  let builders = [
    OpBuilder<(ins "Type":$result, "int32_t":$nbytes, "int32_t":$alignment),
              [{ build($_builder, $_state, result,
                       $_builder.getI32IntegerAttr(nbytes), $_builder.getI32IntegerAttr(alignment)); }]>
  ];

  let assemblyFormat = [{attr-dict `:` qualified(type($result))}];
}

def TTG_WarpSpecializeOp : TTG_Op<"warp_specialize", [
  RecursiveMemoryEffects, RecursivelySpeculatable, AsyncRegions,
  DeclareOpInterfaceMethods<RegionBranchOpInterface>
]> {
  let summary = "asynchronously execute code on multiple warpgroups";
  let description = [{
    The `ttg.warp_specialize` op represents executing different code
    simultaneously on different warp groups. A warp group is a group of
    power-of-2 warps, which can be a different number of warps than in the
    enclosing region.

    The "default" region of the op represents the code executed by the currently
    executing warp group. This region is allowed to implicitly capture. The op
    contains a number of "partition" regions that are isolated from above. They
    must be isolated because these regions represent different layout domains,
    as the number of warps is different.

    Semantically, execution of each region starts simultaneously for each warp
    group, and all warp groups are joined at the end of the op.

    Example:

    ```mlir
    %0 = ttg.warp_specialize(%a, %b)
    default {
      %out = some_operation(%a) // implicit capture of `%a`
      ttg.warp_yield %out : i32
    }
    partition0(%arg0: i32, %arg1: i32) num_warps(8) {
      some_async_dispatch(%arg0, %arg1)
      ttg.warp_return
    }
    partition1(%arg0: i32, %arg1: i32) num_warps(1) {
      some_async_dispatch(%arg0, %arg1)
      ttg.warp_return
    } : (i32, i32) -> i32
    ```
  }];

  let arguments = (ins
    Variadic<AnyType>:$explicitCaptures,
    DenseI32ArrayAttr:$partitionNumWarps,
    OptionalAttr<DenseI32ArrayAttr>:$warpGroupStartIds
  );
  let results = (outs Variadic<AnyType>:$defaultPassthrough);

  let regions = (region
    MinSizedRegion<1>:$defaultRegion,
    SizedRegion<1>:$partitionOpHolder
  );

  let extraClassDeclaration = [{
    RegionRange getPartitionRegions();

    // Get the size and alignment of the capture list.
    std::pair<uint64_t, uint64_t> getCaptureSizeAlign();
    // Get the total number of extra warps required.
    unsigned getTotalPartitionWarps();
  }];

  let builders = [
    OpBuilder<(ins "TypeRange":$resultTypes,
                   "ArrayRef<int32_t>":$partitionNumWarps,
                   "unsigned":$numPartitionRegions)>
  ];

  let hasVerifier = 1;
  let hasCustomAssemblyFormat = 1;
  let hasCanonicalizeMethod = 1;
}

def TTG_WarpSpecializePartitionsOp : TTG_Op<"warp_specialize.partitions", [
  IsolatedFromAbove, RecursiveMemoryEffects, RecursivelySpeculatable,
  Terminator, HasParent<"WarpSpecializeOp">
]> {
  let summary = "container op for `ttg.warp_specialize`";
  let description = [{
    Because MLIR requires entire operations be isolated from above, this op
    contains the actual isolated from above regions of `ttg.warp_specialize`.
  }];

  let regions = (region VariadicRegion<MinSizedRegion<1>>:$partitionRegions);
}

def TTG_WarpYieldOp : TTG_Op<"warp_yield", [
  Pure, Terminator, ReturnLike, HasParent<"WarpSpecializeOp">,
  DeclareOpInterfaceMethods<RegionBranchTerminatorOpInterface>
]> {
  let summary = "yield from the default region of `ttg.warp_specialize`";
  let description = [{
    The `ttg.warp_yield` operation is the terminator for the "default" region of
    a `ttg.warp_specialize` operation. The operands are passed transparently as
    the SSA results of the `ttg.warp_specialize` operation.

    Example:

    ```mlir
    ttg.warp_yield %a, %b : i32, tensor<32xbf16, #blocked>
    ```
  }];

  let arguments = (ins Variadic<AnyType>:$values);

  let assemblyFormat = "($values^)? attr-dict (`:` type($values)^)?";
  let hasVerifier = 1;
}

def TTG_WarpReturnOp : TTG_Op<"warp_return", [
  Pure, Terminator, ReturnLike, HasParent<"WarpSpecializePartitionsOp">
]> {
  let summary = "implicit terminator from partition regions";
  let description = [{
    The `ttg.warp_return` operation is the implicit terminator that ends the
    partition regions of a `ttg.warp_specialize` op. It has no operands as these
    regions cannot return anything.

    TODO: Support returning uniform values from partition regions.
  }];

  let assemblyFormat = "attr-dict";
}

#endif // TRITONGPU_OPS
