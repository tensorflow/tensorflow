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

#ifndef NVGPU_OPS
#define NVGPU_OPS

include "mlir/IR/OpBase.td"
include "mlir/IR/EnumAttr.td"
include "mlir/Dialect/LLVMIR/LLVMOpBase.td"
include "mlir/Interfaces/InferTypeOpInterface.td" // SameOperandsAndResultType
include "NVGPUDialect.td"
include "NVGPUAttrDefs.td"

def LLVM_PointerGlobal : LLVM_PointerInAddressSpace<1>;
def LLVM_PointerShared : LLVM_PointerInAddressSpace<3>;
def LLVM_PointerTensorMemory : LLVM_PointerInAddressSpace<6>;


def NVGPU_Float : AnyTypeOf<[F8E4M3FN, F8E4M3FNUZ, F8E5M2, F8E5M2FNUZ, F16, BF16, F32, F64], "floating-point">;
def NVGPU_Int : AnyTypeOf<[I1, I8, I16, I32, I64], "integer">;
def NVGPU_ScalarLike : AnyTypeOf<[NVGPU_Float, NVGPU_Int]>;


def NVGPU_MemSemanticAttr : I32EnumAttr<
    "MemSemantic", "",
    [
      I32EnumAttrCase<"RELAXED", 1, "relaxed">,
      I32EnumAttrCase<"ACQUIRE", 2, "acquire">,
      I32EnumAttrCase<"RELEASE", 3, "release">,
      I32EnumAttrCase<"ACQUIRE_RELEASE", 4, "acq_rel">,
    ]> {
    let cppNamespace = "::mlir::triton::nvgpu";
}

def NVGPU_MemSyncScopeAttr : I32EnumAttr<
    "MemSyncScope", "",
    [
      I32EnumAttrCase<"GPU", 1, "gpu">,
      I32EnumAttrCase<"CTA", 2, "cta">,
      I32EnumAttrCase<"SYSTEM", 3, "sys">,
    ]> {
    let cppNamespace = "::mlir::triton::nvgpu";
}

class NVGPU_Op<string mnemonic, list<Trait> traits = []> :
    LLVM_OpBase<NVGPU_Dialect, mnemonic, traits>;

def NVGPU_WGMMAFenceOp : NVGPU_Op<"wgmma_fence", []> {
  let assemblyFormat = "attr-dict";
}

def NVGPU_WGMMACommitGroupOp : NVGPU_Op<"wgmma_commit_group", []> {
  let assemblyFormat = "attr-dict";
}

def NVGPU_WGMMAWaitGroupOp : NVGPU_Op<"wgmma_wait_group", [DeclareOpInterfaceMethods<InferTypeOpInterface>,
                                                           AllTypesMatch<["input", "output"]>]> {
  let arguments = (ins LLVM_AnyStruct:$input, I32Attr:$pendings);
  let results = (outs LLVM_AnyStruct:$output);
  let assemblyFormat = "$input attr-dict `:` type($input)";
}

def WGMMA_LayoutAttr : I32EnumAttr<"WGMMALayout",
    "wgmma layout, either 'row' or 'col'",
    [
      I32EnumAttrCase<"row", 0>,
      I32EnumAttrCase<"col", 1>
    ]>{
  let cppNamespace = "::mlir::triton::nvgpu";
}

def WGMMA_EltTypeAttr : I32EnumAttr<"WGMMAEltType",
    "wgmma operand type, either 's8', 's32', 'e4m3', 'e5m2', 'f16', 'bf16', 'tf32', or 'f32'",
    [
      I32EnumAttrCase<"s8", 0>,
      I32EnumAttrCase<"s32", 1>,
      I32EnumAttrCase<"e4m3", 2>,
      I32EnumAttrCase<"e5m2", 3>,
      I32EnumAttrCase<"f16", 4>,
      I32EnumAttrCase<"bf16", 5>,
      I32EnumAttrCase<"tf32", 6>,
      I32EnumAttrCase<"f32", 7>
    ]>{
  let cppNamespace = "::mlir::triton::nvgpu";
}

def WGMMA_OperandType : AnyTypeOf<[LLVM_AnyStruct, I64], "wgmma operand A/B type">;

def NVGPU_WGMMAOp : NVGPU_Op<"wgmma", []> {
  let arguments = (ins WGMMA_OperandType:$opA, WGMMA_OperandType:$opB, I1:$useC, Optional<LLVM_AnyStruct>:$opC,
                   I32Attr:$m, I32Attr:$n, I32Attr:$k,
                   WGMMA_EltTypeAttr:$eltTypeC, WGMMA_EltTypeAttr:$eltTypeA, WGMMA_EltTypeAttr:$eltTypeB,
                   WGMMA_LayoutAttr:$layoutA, WGMMA_LayoutAttr:$layoutB);
  let results = (outs LLVM_AnyStruct:$res);
  let assemblyFormat = "$opA `,` $opB `,` $useC (`,` $opC^)? attr-dict `:` functional-type(operands, $res)";
}

def NVGPU_FenceAsyncSharedOp : NVGPU_Op<"fence_async_shared", []> {
  let arguments = (ins BoolAttr:$bCluster);
  let assemblyFormat = "attr-dict";
}

def NVGPU_ClusterArriveOp : NVGPU_Op<"cluster_arrive", []> {
  let arguments = (ins I1Attr:$relaxed);

  let assemblyFormat = "attr-dict";
}

def NVGPU_ClusterWaitOp : NVGPU_Op<"cluster_wait", []> {
  let assemblyFormat = "attr-dict";
}

def NVGPU_StoreMatrixOp : NVGPU_Op<"stmatrix", [MemoryEffects<[MemWrite]>]> {
  let arguments = (
    ins LLVM_PointerShared:$addr,
    Variadic<I32>:$vals,
    UnitAttr:$trans
  );
  let assemblyFormat = "operands attr-dict `:` type(operands)";
}

def NVGPU_LoadMatrixOp : NVGPU_Op<"ldmatrix", [MemoryEffects<[MemRead]>]> {
  let arguments = (
    ins LLVM_PointerShared:$addr,
    UnitAttr:$trans
  );
  let results = (outs LLVM_AnyStruct:$result);
  let assemblyFormat = "$addr attr-dict `:` functional-type($addr, $result)";
}

def NVGPU_ClusterCTAIdOp : NVGPU_Op<"cluster_id", [Pure]> {
  let results = (outs I32:$result);
  let assemblyFormat = "attr-dict";
}

def NVGPU_LoadAcquireOp : NVGPU_Op<"ld_acquire", [MemoryEffects<[MemRead]>]> {
  let arguments = (
    ins LLVM_PointerGlobal:$addr,
    Optional<I1>:$mask,
    NVGPU_MemSemanticAttr:$sem,
    NVGPU_MemSyncScopeAttr:$scope
  );
  let results = (outs NVGPU_ScalarLike:$result);
  let assemblyFormat = "$sem `,` $scope `,` $addr (`,` $mask^)? attr-dict `:` functional-type($addr, $result)";
}

def NVGPU_WarpIdOp : NVGPU_Op<"warp_id", [Pure]> {
  let results = (outs I32:$result);
  let assemblyFormat = "attr-dict";
}

def NVGPU_TensorMemoryBaseAddress : NVGPU_Op<"tensor_memory_base", [Pure]> {
  let description = [{
    Op to represent base address of tensor memory in a kernel.
    This is used to simplify lowering from TritonGPU to LLVM.
  }];
  let results = (outs LLVM_PointerTensorMemory:$result);
  let assemblyFormat = "attr-dict";
}


#endif
