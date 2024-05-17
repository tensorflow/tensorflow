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
include "mlir/Dialect/Bufferization/IR/AllocationOpInterface.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "tensorflow/compiler/mlir/tools/kernel_gen/ir/tf_status.td"

def TFFramework_Dialect : Dialect {
  let name = "tf_framework";

  let summary = "Types and operations for tf_framework dialect";
  let description = [{
    This dialect contains operations and types for that correspond to
    TensorFlow C++ Framework.
  }];
  let cppNamespace = "::mlir::kernel_gen::tf_framework";

  let extraClassDeclaration = [{
    static constexpr StringRef kTFEntryAttrName = "tf_entry";
    static constexpr size_t kAllocationAlignment = 16;
  }];

  let useDefaultTypePrinterParser = 1;
}

def TFFramework_OpKernelContextType : DialectType<TFFramework_Dialect,
    CPred<"$_self.isa<::mlir::kernel_gen::tf_framework::OpKernelContextType>()">,
          "op_kernel_construction">,
    BuildableType<"$_builder.getType<::mlir::kernel_gen::tf_framework::OpKernelContextType>()"> {
  let description = [{
    OpKernelContextType corresponds to C++ class OpKernelContext defined in
    tensorflow/core/framework/op_kernel.h
  }];
}

def TFFramework_JITCallableType : DialectType<TFFramework_Dialect, CPred<
    "$_self.isa<::mlir::kernel_gen::tf_framework::JITCallableType>()">>,
    BuildableType<"$_builder.getType<::mlir::kernel_gen::tf_framework::JITCallableType>()"> {
  let description = [{
    A `callable` represents the result of JIT compilation. Conceptually, it
    corresponds to a function pointer.
  }];
}

// Base class for TF Framework dialect ops.
class TFFramework_Op<string mnemonic, list<Trait> traits = []> :
    Op<TFFramework_Dialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// TFAllocOp
//===----------------------------------------------------------------------===//
def TFFramework_TFAllocOp : TFFramework_Op<"alloc", [
    MemoryEffects<[MemAlloc<DefaultResource>]>,
    DeclareOpInterfaceMethods<AllocationOpInterface,
      ["buildDealloc", "buildClone"]>
  ]> {
  let summary = "allocation of tensors that uses TF Framework";
  let description = [{
    Allocation of tensors during kernel execution in the Compute method.

    This should be used to allocate any temporary or output memref. If
    `output_index` and `input_indices` are given, attempts to forward one of
    the input tensors to the output by calling `OpKernelContext::forward_input`.

    If the attributes are missing or the forwarding fails, calls
    `Allocator::AllocateRaw` in tensorflow/core/framework/allocator.h.
  }];

  let arguments = (ins
    TFFramework_OpKernelContextType:$ctx,
    Variadic<Index>:$dyn_sizes,
    OptionalAttr<I32ArrayAttr>:$input_indices,
    OptionalAttr<I32Attr>:$output_index
  );
  let results = (outs Res<AnyMemRef, "", [MemAlloc<DefaultResource>]>:$result);

  let builders = [
    OpBuilder<(ins "MemRefType":$memref_type, "Value":$ctx),
    [{
      $_state.addOperands(ctx);
      $_state.types.push_back(memref_type);
    }]>,
    OpBuilder<(ins "MemRefType":$memref_type, "Value":$ctx,
      "ValueRange":$dyn_sizes),
    [{
      build($_builder, $_state, memref_type, ctx);
      $_state.addOperands(dyn_sizes);
    }]>];

  let extraClassDeclaration = [{
    MemRefType getType() { return getResult().getType().cast<MemRefType>(); }
    static constexpr StringRef kReuseOutputAttrName = "reuse_output";
    static constexpr StringRef kReuseInputCandidatesAttrName =
        "reuse_input_candidates";
  }];
  let assemblyFormat = [{
    `(` $ctx (`,` $dyn_sizes^ )? `)` attr-dict `:` type($result)
  }];

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// TFAssertOp
//===----------------------------------------------------------------------===//
def TFFramework_TFAssertOp : TFFramework_Op<"assert"> {
  let summary = "Assert operation with message attribute and error code";
  let description = [{
    Assert operation that propagates the error message to TF Framework.
  }];

  let arguments = (ins
    TFFramework_OpKernelContextType:$ctx,
    I1:$arg,
    TFFramework_ErrorCodeAttr:$error_code,
    StrAttr:$msg
  );

  let assemblyFormat = "$ctx `,` $arg `,` $error_code `,` $msg attr-dict";
}

//===----------------------------------------------------------------------===//
// TFDeallocOp
//===----------------------------------------------------------------------===//
def TFFramework_TFDeallocOp : TFFramework_Op<"dealloc",
    [MemoryEffects<[MemFree]>]> {
  let summary = "deallocation of tensors that uses TF Framework";
  let description = [{
    Deallocation of tensors during kernel execution in the Compute method.

    This should be used to deallocate any temporary memref that was allocated
    with `tf_framework.alloc`.
    Corresponds to `Allocator::DeallocateRaw` in
    tensorflow/core/framework/allocator.h.
  }];

  let arguments = (ins TFFramework_OpKernelContextType:$ctx,
                   Arg<AnyRankedOrUnrankedMemRef, "", [MemFree]>:$memref);
  let assemblyFormat = "`(` $ctx `,` $memref `)` attr-dict `:` type($memref)";
}

//===----------------------------------------------------------------------===//
// IsValidMemRefOp
//===----------------------------------------------------------------------===//
def TFFramework_IsValidMemRefOp : TFFramework_Op<"is_valid_memref", []> {
  let summary = "Op to check if the memref is valid.";
  let description = [{
    The op checks if the allocation was successful so that the underlying ptr
    in the descriptor is not equal to NULL. Also, we check if the number of
    elements is not zero to support empty shapes inputs correctly.

    is_valid_memref(memref) = ptr(memref) != NULL || num_elements(memref) == 0
  }];

  let arguments = (ins AnyMemRef:$arg);
  let results = (outs I1:$result);
  let assemblyFormat = [{
    `(` $arg `)` attr-dict `:` type($arg) `->` type($result)
  }];
}

//===----------------------------------------------------------------------===//
// NullContextOp
//===----------------------------------------------------------------------===//
def TFFramework_NullContextOp : TFFramework_Op<"null_context",
    [Pure]> {
  let summary = "Creates a fake TF context that will be lowered to nullptr";
  let description = [{Needed for testing}];
  let results = (outs TFFramework_OpKernelContextType:$result);
  let assemblyFormat = "attr-dict `:` type($result)";
}

//===----------------------------------------------------------------------===//
// NullMemRefOp
//===----------------------------------------------------------------------===//
def TFFramework_NullMemRefOp : TFFramework_Op<"null_memref", []> {
  let summary = "Op to construct unranked memref with 0-rank";
  let description = [{
    The op is needed to construct a throw-away result after error reporting
    happened. It constructs a ranked memref descriptor with 0-rank and
    {NULL, NULL, 0} underlying ranked descriptor of type memref<elem_type>.
    In ranked case it constructs a ranked memref descriptor depending on the
    result type, but sets both allocated and aligned pointers to NULL.
  }];

  let results = (outs AnyRankedOrUnrankedMemRef:$result);
  let assemblyFormat = "attr-dict `:` type($result)";
}

//===----------------------------------------------------------------------===//
// ReportErrorOp
//===----------------------------------------------------------------------===//
def TFFramework_ReportErrorOp : TFFramework_Op<"report_error", []> {
  let summary = "Operation that propagates error message to TF Framework";
  let description = [{
    Error reporting operation that corresponds to
    `OpKernelContext::CtxFailureWithWarning`.
  }];

  let arguments = (ins
    TFFramework_OpKernelContextType:$ctx,
    TFFramework_ErrorCodeAttr:$error_code,
    StrAttr:$msg
  );

  let assemblyFormat = "$ctx `,` $error_code `,` $msg attr-dict";
}

//===----------------------------------------------------------------------===//
// JITCompileOp
//===----------------------------------------------------------------------===//

def TFFramework_JITCompileOp : TFFramework_Op<"jit_compile", [SingleBlock]> {
  let summary = "Invokes JIT compilation through the TF framework";
  let description = [{
    The op takes an optional TF context, so that it can be added at a later
    stage in the compilation pipeline. The op's body corresponds to a function
    body and the corresponding `tf_framework.jit_compile_yield` represents the
    result value(s).
  }];

  let arguments = (ins
    Optional<TFFramework_OpKernelContextType>:$ctx
  );
  let results = (outs TFFramework_JITCallableType:$result);
  let regions = (region SizedRegion<1>:$body);

  let assemblyFormat = "$ctx $body attr-dict";
}

//===----------------------------------------------------------------------===//
// JITCompileYieldOp
//===----------------------------------------------------------------------===//

def TFFramework_JITCompileYieldOp : TFFramework_Op<"jit_compile_yield", [
    Pure, ReturnLike, Terminator, HasParent<"JITCompileOp">]> {
  let summary = "Yields the results in a `tf_framework.jit_compile` op's body";
  let description = [{
    See `tf_framework.jit_compile`.
  }];

  let arguments = (ins
    AnyTypeOf<[AnyRankedOrUnrankedMemRef, AnyTensor]>:$result
  );

  let assemblyFormat = "$result attr-dict `:` type($result)";
}

//===----------------------------------------------------------------------===//
// JITCompileFromStrOp
//===----------------------------------------------------------------------===//

def TFFramework_JITCompileFromStrOp : TFFramework_Op<"jit_compile_from_str",
    []> {
  let summary = "Invokes JIT compilation through the TF framework";
  let description = [{
    This operation is similar to `tf_framework.jit_compile`. Instead of a body,
    the to-be-compiled function is represented as a string attribute. The string
    shall be the serialized form of a module with a single function named
    `main`.
  }];

  let arguments = (ins
    Optional<TFFramework_OpKernelContextType>:$ctx,
    StrAttr:$code,
    I64ArrayAttr:$tileSizes,
    I64ArrayAttr:$unrollFactors,
    BoolAttr:$enableFtz,
    BoolAttr:$index64Bit,
    BoolAttr:$cpuCodegen
  );
  let results = (outs TFFramework_JITCallableType:$result);

  let extraClassDeclaration = [{
    static constexpr StringRef kJITEntryFunctionName = "main";
  }];

  let assemblyFormat = "($ctx^ `,`)? $code attr-dict";
}

//===----------------------------------------------------------------------===//
// JITExecuteOp
//===----------------------------------------------------------------------===//

def TFFramework_JITExecuteOp : TFFramework_Op<"jit_execute", [
    AttrSizedOperandSegments,
    MemoryEffects<[MemAlloc<DefaultResource>]>,
    DeclareOpInterfaceMethods<AllocationOpInterface,
        ["buildDealloc", "buildClone"]>]> {
  let summary = "Executes a JIT-compiled function through the TF framework";
  let description = [{
    The op takes an optional TF context, so that it can be added at a later
    stage in the compilation pipeline. The callable must be a JIT-compiled
    function that is the result of either `tf_framework.jit_compile` or
    `tf_framework.jit_compile_from_str`. The remaining operands must be tensor
    or memref arguments and will be forwarded to the callable function. The
    result types must match those of the callable function. Otherwise, the
    execution behavior is undefined.
  }];

  let arguments = (ins
    Optional<TFFramework_OpKernelContextType>:$ctx,
    TFFramework_JITCallableType:$callable,
    Variadic<AnyTypeOf<[AnyRankedOrUnrankedMemRef, AnyTensor]>>:$inputs
  );
  let results = (outs
    Res<AnyTypeOf<[AnyRankedOrUnrankedMemRef, AnyTensor]> , "",
        [MemAlloc<DefaultResource>]>:$result);

  let assemblyFormat = [{
    (`ctx` `(` $ctx^ `)`)? $callable `(` $inputs `)` attr-dict
        `:` type($inputs) `->` type($result)
  }];
}

#endif // TF_FRAMEWORK_OPS
