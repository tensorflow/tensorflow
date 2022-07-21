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

  let emitAccessorPrefix = kEmitAccessorPrefix_Raw;
}

// Base class for XLIR dialect ops.
class XLIR_Op<string mnemonic, list<Trait> traits = []> :
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
  let arguments = (ins TFRTGPU_ContextType:$ctx);
  let results = (outs TFRTGPU_ModuleType);
}

def XLIR_ReplicaIdOp : XLIR_Op<"replica_id"> {
  let description = [{
    xlir.replica_id copies the 4-byte replica ID to the output buffer.
  }];
  let arguments = (ins TFRTGPU_StreamType:$stream, TFRTGPU_BufferType:$output,
                   TFRT_ChainType:$chain);
  let results = (outs TFRT_ChainType);
  let assemblyFormat = [{
    $stream`,` $output`,` $chain attr-dict
  }];
}

def XLIR_PartitionIdOp : XLIR_Op<"partition_id"> {
  let description = [{
    xlir.partition_id copies the 4-byte partition ID to the output buffer.
  }];
  let arguments = (ins TFRTGPU_StreamType:$stream, TFRTGPU_BufferType:$output,
                   TFRT_ChainType:$chain);
  let results = (outs TFRT_ChainType);
  let assemblyFormat = [{
    $stream`,` $output`,` $chain attr-dict
  }];
}

def XLIR_CclCreateOp : XLIR_Op<"ccl.create"> {
  let description = [{
    xlir.ccl.create returns a ccl.handle, which can be populated with individual
    collective ops (e.g., using tfrt_gpu.ccl.all_reduce) to be executed as a
    single fused operation.
    group_mode is the int64_t-cast CollectiveOpGroupMode (as described in
    tensorflow/compiler/xla/service/collective_ops_utils.h). op_id is the
    hlo.unique_id value if in cross-replica mode, or the op's channel ID if in
    cross-module mode.
  }];
  let arguments = (ins TFRTGPU_ContextType:$context, I64Attr:$group_mode,
                   I64Attr:$op_id, TFRT_ChainType:$chain);
  let results = (outs TFRTGPU_CclHandleType);
}

def XLIR_CclCollectivePermuteOp : XLIR_Op<"ccl.collective_permute"> {
  let description = [{
    xlir.ccl.collective_permute inserts into the ccl.handle operations to send
    data at the input buffer and receive data into the output buffer depending
    on the paired list of source and target peers supplied as attributes.
    group_mode is the int64_t-cast CollectiveOpGroupMode (as described in
    tensorflow/compiler/xla/service/collective_ops_utils.h).
  }];
  let arguments = (ins TFRTGPU_CclHandleType:$handle, TFRTGPU_BufferType:$input,
                   TFRTGPU_BufferType:$output, TFRTGPU_CclDataTypeAttr:$data_type,
                   I64Attr:$group_mode, I64ArrayAttr:$source_peers,
                   I64ArrayAttr:$target_peers, TFRT_ChainType:$chain);
  let results = (outs TFRT_ChainType);
  let assemblyFormat = [{
    $handle`,` $input`,` $output`,` custom<Enum>($data_type)`,` $group_mode`,`
    $source_peers`,` $target_peers`,` $chain attr-dict
  }];
}

def XLIR_CustomCallOp : XLIR_Op<"custom_call"> {
  let description = [{
    xlir.custom_call invokes the host function 'symbol' from the
    CustomCallTargetRegistry. The target must have the following signature:

    void target(void* stream, void* const pointers[],
                const char* opaque, size_t opaque_size,
                XlaCustomCallStatus* status);

    The 'indices' maps the 'pointers' array to 'buffers', or nullptr for
    negative values.

    'opaque' is an arbitrary array of bytes passed as extra arguments.
  }];
  let arguments = (ins TFRTGPU_StreamType:$stream,
                   Variadic<TFRTGPU_BufferType>:$buffers,
                   TFRT_ChainType:$chain, StrAttr:$symbol,
                   I32ArrayAttr:$indices, StrAttr:$opaque);
  let results = (outs TFRT_ChainType);
  let assemblyFormat = [{
    $stream`,` $buffers`,` $chain attr-dict
  }];
}

def XLIR_InfeedOp : XLIR_Op<"infeed"> {
  let description = [{
    xlir.infeed copies data from the streaming interface of the device to the
    output buffers.
  }];
  let arguments = (ins TFRTGPU_StreamType:$stream,
                   Variadic<TFRTGPU_BufferType>:$outputs,
                   TFRT_ChainType:$chain);
  let results = (outs TFRT_ChainType);
  let assemblyFormat = [{
    $stream `,` ($outputs^ `,`)?  $chain attr-dict
  }];
}

def XLIR_OutfeedOp : XLIR_Op<"outfeed"> {
  let description = [{
    xlir.outfeed outputs data from the input buffers to the streaming interface
    to the host.
  }];
  let arguments = (ins TFRTGPU_StreamType:$stream,
                   Variadic<TFRTGPU_BufferType>:$inputs,
                   TFRT_ChainType:$chain);
  let results = (outs TFRT_ChainType);
  let assemblyFormat = [{
    $stream `,` ($inputs^ `,`)? $chain attr-dict
  }];
}

#endif  // XLIR_OPS
