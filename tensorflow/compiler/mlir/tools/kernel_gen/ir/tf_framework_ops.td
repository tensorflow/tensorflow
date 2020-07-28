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

// This is the operation definition file for TF Framework ops.

#ifndef TF_FRAMEWORK_OPS
#define TF_FRAMEWORK_OPS

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def TFFramework_Dialect : Dialect {
  let name = "tf_framework";

  let summary = "Types and operations for tf_framework dialect";
  let description = [{
    This dialect contains operations and types for that correspond to
    TensorFlow C++ Framework.
  }];
  let cppNamespace = "kernel_gen::tf_framework";
}

def TFFramework_OpKernelContextType : DialectType<TFFramework_Dialect,
    CPred<"$_self.isa<::mlir::kernel_gen::tf_framework::OpKernelContextType>()">,
          "op_kernel_construction">,
    BuildableType<"$_builder.getType<::mlir::kernel_gen::tf_framework::OpKernelContextType>()"> {
  let typeDescription = [{
    OpKernelContextType corresponds to C++ class OpKernelContext defined in
    tensorflow/core/framework/op_kernel.h
  }];
}

def AllocatorAttributes : StructAttr<"AllocatorAttributes",
  TFFramework_Dialect, [
    StructFieldAttr<"on_host", BoolAttr>,
    StructFieldAttr<"nic_compatible", BoolAttr>,
    StructFieldAttr<"gpu_compatible", BoolAttr>]> {
  let description = "Equivalent to `tensorflow::AllocatorAttributes` in C++";
}

def AllocationAttributes : StructAttr<"AllocationAttributes",
  TFFramework_Dialect, [
    StructFieldAttr<"no_retry_on_failure",
                    DefaultValuedAttr<BoolAttr, "false">>,
    StructFieldAttr<"allocation_will_be_logged",
                    DefaultValuedAttr<BoolAttr, "false">>]> {
  let description = "Equivalent to `tensorflow::AllocationAttributes` in C++";
}


// Base class for TF Framework dialect ops.
class TFFramework_Op<string mnemonic, list<OpTrait> traits = []> :
    Op<TFFramework_Dialect, mnemonic, traits> {
  let verifier = "return Verify(*this);";
}


// Base class for TF Framework alloc ops.
class TFFramework_AllocLikeOp<string mnemonic,
                  Resource resource,
                  list<OpTrait> traits = []> :
    TFFramework_Op<mnemonic,
                !listconcat([MemoryEffects<[MemAlloc<resource>]>], traits)> {

  let arguments = (ins TFFramework_OpKernelContextType:$op_kernel_ctx,
                   Variadic<Index>:$dyn_sizes,
                   OptionalAttr<AllocatorAttributes>:$allocator_attrs,
                   OptionalAttr<AllocationAttributes>:$allocation_attrs);
  let results = (outs Res<AnyMemRef, "", [MemAlloc<resource>]>:$result);

  let builders = [
    OpBuilder<[{
      OpBuilder &builder, OperationState &result, MemRefType memref_type,
      Value op_kernel_ctx,
      AllocatorAttributes allocator_attrs = AllocatorAttributes(),
      AllocationAttributes allocation_attrs = AllocationAttributes()
    }], [{
      result.addOperands(op_kernel_ctx);
      result.types.push_back(memref_type);
      if (allocator_attrs)
        result.addAttribute("allocator_attrs", allocator_attrs);
      if (allocation_attrs)
        result.addAttribute("allocation_attrs", allocation_attrs);
    }]>,

    OpBuilder<[{
      OpBuilder &builder, OperationState &result, MemRefType memref_type,
      Value op_kernel_ctx, ValueRange dyn_sizes,
      AllocatorAttributes allocator_attrs = AllocatorAttributes(),
      AllocationAttributes allocation_attrs = AllocationAttributes()
    }], [{
      build(builder, result, memref_type, op_kernel_ctx, allocator_attrs,
            allocation_attrs);
      result.addOperands(dyn_sizes);
    }]>];

  let extraClassDeclaration = [{
    MemRefType getType() { return getResult().getType().cast<MemRefType>(); }
  }];
  let assemblyFormat = [{
    `(` $op_kernel_ctx (`,` $dyn_sizes^ )? `)` attr-dict `:` type($result)
  }];
}

//===----------------------------------------------------------------------===//
// AllocOutputOp
//===----------------------------------------------------------------------===//

def TFFramework_AllocOutputOp
    : TFFramework_AllocLikeOp<"alloc_output", DefaultResource> {
  let summary = "allocation of tensorsmemory allocation operation";
  let description = [{
    Allocation of output tensors during kernel execution in the Compute method.

    This should be used to allocate any tensor that is going to be used as an
    output from the kernel at the end of the current execution.

    Defined in third_party/tensorflow/core/framework/op_kernel.cc.
  }];
}

//===----------------------------------------------------------------------===//
// AllocTempOp
//===----------------------------------------------------------------------===//

def TFFramework_AllocTempOp
    : TFFramework_AllocLikeOp<"alloc_temp", DefaultResource> {
  let summary = "memory allocation operation";
  let description = [{
    Allocation of temp tensors during kernel execution in the Compute method.

    This should be used to allocate any scratch storage that is needed while
    the kernel is executing, and will not be retained.

    Defined in third_party/tensorflow/core/framework/op_kernel.cc.
  }];
}

#endif // TF_FRAMEWORK_OPS
