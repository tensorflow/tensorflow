#ifndef TRITONNVIDIAGPU_ATTRDEFS
#define TRITONNVIDIAGPU_ATTRDEFS

include "mlir/IR/AttrTypeBase.td"
include "triton/Dialect/TritonNvidiaGPU/IR/TritonNvidiaGPUDialect.td"
include "triton/Dialect/Triton/IR/TritonInterfaces.td"

def TTG_TensorMemorySpace : AttrDef<TritonNvidiaGPU_Dialect, "TensorMemorySpace"> {
  let mnemonic = "tensor_memory";
  let description = [{
    Attribute to indicate that the memory descriptor points to tensor memory.
    The memory is laid out in blocks of size blockM x blockN. Each block is distributed
    across TMEM 128 rows.

    Blocks are distributed along M dimension first and then N dimension. This is an arbitrary
    convention that need to be followed operations reading/writing to TMEM.

    a tensor <128x128xf32> with blockM = 64 and blockN = 64 will be distributed as follows:

        \ col    0        1           31       32            64             96           127
    rows: 0  ( 0,  0) ( 0,  1) ... ( 0, 31)  (64,  0) ...   (0, 64) ...  (64, 64) ...  (64, 96)
          1
         ...
          15 (15,  0) (15,  1) ... (15, 31)  (79,  0) ...   (15, 64) ... (79, 64) ...  (79, 96)
          16 ( 0, 32) ( 0, 33) ... ( 0, 63)  (64, 32) ...   ( 0, 96) ... (64, 96) ...  (64, 127)
         ...
          31 (15, 32) (15, 33) ... (15, 63)  (79, 32) ...   (15, 96) ... (79, 96) ...  (79, 127)
          32 (16,  0) (16,  1) ... (16, 31)  (80,  0) ...   (16, 64) ... (80, 64) ...  (80, 96)
         ...
         127 (63, 32) (63, 33) ... (63, 63)  (127, 32) ...  (63, 96) ... (127, 96)...  (127, 127)
  }];
}

def TTG_TensorMemoryEncodingAttr : AttrDef<TritonNvidiaGPU_Dialect, "TensorMemoryEncoding"> {
  let mnemonic = "tensor_memory_encoding";
  let attrName = "triton.gpu.tensor_memory_encoding";
  let description = [{
    An encoding to represent the different way the tensor memory is laid out.
    `unpacked` attributes indicates whether types smaller than 32bits are unpacked (take full 32bits)
    or are packed (N elements are stored within one 32bits row).
  }];
  let parameters = (
    ins
    "unsigned":$blockM,
    "unsigned":$blockN,
    "bool":$unpacked,
    DefaultValuedParameter<"unsigned", "1">:$CTASplitM,
    DefaultValuedParameter<"unsigned", "1">:$CTASplitN
  );
  let assemblyFormat = "`<` struct(params) `>`";
}

def TTG_TensorMemoryScalesEncodingAttr : AttrDef<TritonNvidiaGPU_Dialect, "TensorMemoryScalesEncoding"> {
  let mnemonic = "tensor_memory_scales_encoding";
  let attrName = "triton.gpu.tensor_memory_scales_encoding";
  let description = [{
    An encoding to represent the layout of tensor memory scales.
    As described in the PTX doc, blocked scales in TMEM must be in a special layout. They are organized
    as a multiple copies of "chunk", each of which having the size 32x4x4B. Moreover, such chunks are duplicated
    over 4 warps to fill entire 128 rows of TMEM. This encoding indicates that a tensor in TMEM is in such a special
    layout.
  }];
  let parameters = (
    ins
    DefaultValuedParameter<"unsigned", "1">:$CTASplitM,
    DefaultValuedParameter<"unsigned", "1">:$CTASplitN
  );
  let assemblyFormat = "`<` struct(params) `>`";
}

#endif
