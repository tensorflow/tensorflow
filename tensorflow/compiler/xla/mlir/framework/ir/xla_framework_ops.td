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

// This is the operation definition file for XLA Framework ops.

#ifndef TENSORFLOW_COMPILER_MLIR_XLA_XLA_FRAMEWORK_OPS_TD_
#define TENSORFLOW_COMPILER_MLIR_XLA_XLA_FRAMEWORK_OPS_TD_

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/OpBase.td"

def XLAFramework_Dialect : Dialect {
  let name = "xla_framework";

  let summary = "Types and operations for xla_framework dialect";
  let description = [{
    This dialect contains operations and types that correspond to XLA compiled C
    functions in tensorflow/compiler/xla/service/cpu/ir_function.cc.
  }];
  let cppNamespace = "::mlir::xla_framework";

  let extraClassDeclaration = [{
    // The attribute string tied to entry point functions honoring XLA's calling
    // convention.
    static constexpr StringRef kXLAEntryAttrName = "xla_entry";
  }];

  let useDefaultTypePrinterParser = 1;
}

def XLAFramework_BufferType : TypeDef<XLAFramework_Dialect, "Buffer"> {
  let description = [{
    The BufferType corresponds to the shaped memrefs that XLA operates on. This
    will be lowered to a caller supplied pointer. This type has no additional,
    unique functionality that a memref doesn't have, but is presented uniquely
    to ensure that no undesired transformations affect it, and to ease
    transformations specifically intended for this dialect.
  }];

  let mnemonic = "buffer";
}

// Base class for XLA Framework dialect ops.
class XLAFramework_Op<string mnemonic, list<Trait> traits = []> :
    Op<XLAFramework_Dialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// XLABufferToMemOp
//===----------------------------------------------------------------------===//
def XLAFramework_XLABufferToMemOp : XLAFramework_Op<"buffer_to_mem",
    []> {
  let summary = "Creation of memref from XLA buffer pointer";
  let description = [{
    This convert's between XLA's representation of a buffer and standard MLIR's
    representation of memrefs.

    This will be lowered to LLVM logic that will take a pointer and create a
    memref descriptor based on the static information for the memref.
  }];

  let arguments = (ins XLAFramework_BufferType:$buffer);
  let results = (outs Res<AnyRankedOrUnrankedMemRef>:$result);

  let builders = [
    OpBuilder<(ins "MemRefType":$memref_type, "Value":$buffer),
    [{
      $_state.addOperands(buffer);
      $_state.types.push_back(memref_type);
    }]>];

  let extraClassDeclaration = [{
    MemRefType getType() { return getResult().getType().cast<MemRefType>(); }
  }];
  let assemblyFormat = [{
    $buffer attr-dict `:` type($result)
  }];
}

//===----------------------------------------------------------------------===//
// XLABufferToMemOp
//===----------------------------------------------------------------------===//
def XLAFramework_MemToXLABufferOp : XLAFramework_Op<"mem_to_buffer",
    []> {
  let summary = "Creation of XLA buffer pointer from memref";
  let description = [{
    This convert's between XLA's and standard MLIR's representations of memrefs.

    This will be lowered to LLVM logic that will pass through a memref's
    pointer.
  }];

  let arguments = (ins Res<AnyRankedOrUnrankedMemRef>:$memref);
  let results = (outs XLAFramework_BufferType:$buffer);

  let assemblyFormat = [{
    $memref attr-dict `:` type($memref)
  }];
}

#endif // TENSORFLOW_COMPILER_MLIR_XLA_XLA_FRAMEWORK_OPS_TD_
