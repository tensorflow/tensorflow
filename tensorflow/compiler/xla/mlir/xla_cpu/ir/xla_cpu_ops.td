/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_XLA_MLIR_XLA_CPU_OPS_TD_
#define TENSORFLOW_COMPILER_XLA_MLIR_XLA_CPU_OPS_TD_

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "tensorflow/compiler/xla/mlir/xla_cpu/ir/xla_cpu_enums.td"
include "tensorflow/compiler/xla/mlir/xla_cpu/ir/xla_cpu_dialect.td"

// Base class for XLA CPU dialect ops.
class XlaCpu_Op<string mnemonic, list<Trait> traits = []> :
    Op<XlaCpuDialect, mnemonic, traits>;

class TensorOrMemrefOf<list<Type> allowedTypes> :
  AnyTypeOf<[MemRefOf<allowedTypes>, RankedTensorOf<allowedTypes>],
  "", "::mlir::ShapedType">;

def XlaCpuAllReduceOp : XlaCpu_Op<"all_reduce",
    [SameOperandsElementType, SameOperandsAndResultType]> {
  let summary = [{
    CPU-specific version of AllReduce.
  }];

  let description = [{
    The major differences between this and HLO's all_reduce are:
    - This bufferizes to itself
    - It has no region.
  }];

  let arguments = (ins
      Variadic<TensorOrMemrefOf<
        [I8, I16, I32, I64, F16, F32, F64, Complex<F<32>>, Complex<F<64>>]
      >>:$operand,
      I64ElementsAttr:$replica_groups,
      I64:$channel_handle,
      UnitAttr:$use_global_device_ids,
      XlaCpuReductionKind:$reduction_kind
  );
  let results = (outs
      Variadic<TensorOrMemrefOf<
        [I8, I16, I32, I64, F16, F32, F64, Complex<F<32>>, Complex<F<64>>]
      >>
  );
}

#endif  // TENSORFLOW_COMPILER_XLA_MLIR_XLA_CPU_OPS_TD_