/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

// Operation definition file for GPU thunks.

#ifndef XLA_THUNKS_OPS
#define XLA_THUNKS_OPS

include "mlir/Dialect/LLVMIR/LLVMOpBase.td"
include "mlir/IR/OpBase.td"

def XLAThunks_Dialect : Dialect {
  let name = "xla_thunks";
  let cppNamespace = "xla_thunks";
}

class ThunkOp<string mnemonic, list<OpTrait> traits = []> :
    Op<XLAThunks_Dialect, mnemonic, traits>;

def AllocationSlice : StructAttr<"AllocationSlice", XLAThunks_Dialect, [
    StructFieldAttr<"allocation_index", I64Attr>,
    StructFieldAttr<"offset", I64Attr>,
    StructFieldAttr<"size", I64Attr>,
  ]> {
  let summary = "Defines a slice of an allocation for XLA thunk ops";
}

def MemzeroThunkOp : ThunkOp<"execute_memzero_thunk"> {
  let arguments = (ins
    LLVM_PointerTo<I8>:$execute_params,
    AllocationSlice:$allocation_slice
  );
  let results = (outs
    I<1>:$ok,
    LLVM_PointerTo<I8>:$error_message
  );
}

#endif // XLA_THUNKS_OPS
