// Copyright 2021 The TensorFlow Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//===- xlir_ops.td --------------------------------------------------------===//
//
// Operation definitions for XLIR functionality.
//
//===----------------------------------------------------------------------===//

#ifdef XLIR_OPS
#else
#define XLIR_OPS

include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "tfrt/gpu/kernels/gpu_ops_base.td"
include "tfrt/tfrt_op_base.td"

// "xlir" dialect
def XLIR_Dialect : Dialect {
  let name = "xlir";

  let description = [{
    The XLIR dialect.

    This dialect contains common GPU operations for CUDA and ROCm.
  }];

  let cppNamespace = "xla::gpu";
}

// Base class for XLIR dialect ops.
class XLIR_Op<string mnemonic, list<OpTrait> traits = []> :
    Op<XLIR_Dialect, mnemonic, traits> {
  let summary = !strconcat("xlir.", mnemonic, " operation");
  let assemblyFormat = "operands attr-dict";
}

def XLIR_ModuleLoadOp : XLIR_Op<"module.load"> {
  let description = [{
    xlir.module.load loads GPU device module from a binary blob provided by the
    execution context.

    Example:
      %module = tfrt_gpu.module.load %ctx
  }];
  let arguments = (ins GPU_ContextType:$ctx);
  let results = (outs GPU_ModuleType);
}

def XLIR_CclCreateOp : XLIR_Op<"ccl.create"> {
  let description = [{
    xlir.ccl.create returns a ccl.handle, which can be populated with individual
    collective ops (e.g., using tfrt_gpu.ccl.all_reduce) to be executed as a
    single fused operation.
  }];
  let arguments = (ins GPU_ContextType);
  let results = (outs GPU_CclHandleType);
}

def XLIR_CclCollectivePermuteOp : XLIR_Op<"ccl.collective_permute"> {
  let description = [{
    xlir.ccl.collective_permute inserts into the ccl.handle operations to send
    data at the input buffer and receive data into the output buffer depending
    on the source and target peers supplied in the ExecutionContext
    (XcclContext).
  }];
  let arguments = (ins GPU_CclHandleType:$handle, GPU_BufferType:$input,
                   GPU_BufferType:$output, GPU_CclDataTypeAttr:$data_type,
                   TFRT_ChainType:$chain);
  let results = (outs TFRT_ChainType);
  let assemblyFormat = [{
    $handle`,` $input`,` $output`,` custom<Enum>($data_type)`,` $chain attr-dict
  }];
}

def XLIR_CustomCallOp : XLIR_Op<"custom_call"> {
  let description = [{
    xlir.custom_call invokes a host function, supplied in the ExecutionContext
    (CustomCallContext), using 'buffers' for the purposes of enqueuing
    operations on the GPU device. 'opaque' is used to supply an arbitrary string
    of bytes to the host function call.

    Optionally specify an 'args_to_target_args' and/or
    'results_to_target_results' to rearrange 'buffers' supplied to the host
    function. Example:
        buffers = [ A, B ]
        args_to_target_args = [ 0, 2 ]
        target_args_count = 5
        (rearranged buffers = [ A, nullptr, B, nulltpr, nullptr ])
  }];
  let arguments = (ins GPU_StreamType:$stream, TFRT_ChainType:$chain,
                   Variadic<GPU_BufferType>:$buffers,
                   I64ArrayAttr:$args_to_target_args, StrAttr:$opaque,
                   I64ArrayAttr:$results_to_target_results,
                   I64Attr:$target_args_count, I64Attr:$target_results_count);
  let results = (outs TFRT_ChainType);
  let assemblyFormat = [{
    $stream`,` $chain`,` $buffers`,` $args_to_target_args`,`
    $results_to_target_results`,` $target_args_count`,` $target_results_count
    attr-dict
  }];
}

#endif  // XLIR_OPS
