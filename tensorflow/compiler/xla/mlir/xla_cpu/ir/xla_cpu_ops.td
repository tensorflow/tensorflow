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

include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "tensorflow/compiler/xla/mlir/xla_cpu/ir/xla_cpu_dialect.td"
include "tensorflow/compiler/xla/mlir/xla_cpu/ir/xla_cpu_enums.td"
include "tensorflow/compiler/xla/mlir_hlo/mhlo/IR/hlo_base.td"

// Base class for XLA CPU dialect ops.
class XlaCpu_Op<string mnemonic, list<Trait> traits = []> :
    Op<XlaCpuDialect, mnemonic, traits>;

def TensorOrMemref :
  AnyTypeOf<[AnyMemRef, AnyRankedTensor], "", "::mlir::ShapedType">;

def AllReduceOp : XlaCpu_Op<"all_reduce",
    [SameOperandsElementType,
     SameVariadicOperandSize,
     BufferizableOpInterface]> {
  let summary = [{
    CPU-specific version of AllReduce.
  }];

  let description = [{
    The major differences between this and HLO's all_reduce are:
    - It bufferizes to itself.
    - It has no region.
    - It uses destination passing style.
  }];

  let arguments = (ins
      Variadic<TensorOrMemref>:$operand,
      Variadic<TensorOrMemref>:$dsts,
      I64ElementsAttr:$replica_groups,
      I64Attr:$channel_handle,
      I32Attr:$use_global_device_ids,
      XlaCpuReductionKind:$reduction_kind
  );
  let results = (outs
      Variadic<TensorOrMemref>
  );
  let extraClassDeclaration = [{
    // Declarations for BufferizableOpInterface:
    bool bufferizesToMemoryRead(OpOperand &opOperand,
        const bufferization::AnalysisState &state);
    bool bufferizesToMemoryWrite(OpOperand &opOperand,
        const bufferization::AnalysisState &state);
    SmallVector<OpResult> getAliasingOpResult(
        OpOperand &opOperand, const bufferization::AnalysisState &state);
    LogicalResult bufferize(RewriterBase &rewriter,
        const bufferization::BufferizationOptions &options);
    bufferization::BufferRelation bufferRelation(OpResult opResult,
        const bufferization::AnalysisState &state);
  }];
}

def ReplicaIdOp : XlaCpu_Op<"replica_id"> {
  let summary = "CPU-specific version of ReplicaId";
  let description = [{
    ReplicaId, but returns a i32 instead of tensor<ui32>.
  }];
  let results = (outs I32);
}

def PartitionIdOp : XlaCpu_Op<"partition_id"> {
  let summary = "CPU-specific version of PartitionId";
  let description = [{
    PartitionId, but returns a i32 instead of tensor<ui32>.
  }];
  let results = (outs I32);
}

def CollectivePermuteOp : XlaCpu_Op<"collective_permute", [BufferizableOpInterface]> {
  let summary = "CPU-specific version of CollectivePermute";
  let description = [{
    The major differences between this and HLO's collective_permute are:
    - It bufferizes to itself.
    - It uses destination passing style.
  }];

  let arguments = (ins
      TensorOrMemref:$operand,
      TensorOrMemref:$dst,
      I64ElementsAttr:$source_target_pairs,
      I64Attr:$channel_handle
  );
  let results = (outs Variadic<TensorOrMemref>);
  let extraClassDeclaration = [{
    // Declarations for BufferizableOpInterface:
    bool bufferizesToMemoryRead(OpOperand &opOperand,
        const bufferization::AnalysisState &state) {
      return opOperand.getOperandNumber() == 0;
    }
    bool bufferizesToMemoryWrite(OpOperand   &opOperand,
        const bufferization::AnalysisState &state) {
      return opOperand.getOperandNumber() == 1;
    }
    SmallVector<OpResult> getAliasingOpResult(
        OpOperand &opOperand, const bufferization::AnalysisState &state) {
      if (opOperand.getOperandNumber() == 0) return {};
      return {getOperation()->getOpResult(0)};
    }
    LogicalResult bufferize(RewriterBase &rewriter,
        const bufferization::BufferizationOptions &options);
    bufferization::BufferRelation bufferRelation(OpResult opResult,
        const bufferization::AnalysisState &state) {
      return bufferization::BufferRelation::Equivalent;
    }
  }];
}

def AllToAllOp : XlaCpu_Op<"all_to_all",
  [SameOperandsElementType,
   SameVariadicOperandSize,
   BufferizableOpInterface]> {
  let summary = "CPU-specific version of AllToAll";
  let description = [{
    The major differences between this and HLO's all_to_all are:
    - It bufferizes to itself.
    - It uses destination passing style.
  }];

  let arguments = (ins
      Variadic<TensorOrMemref>:$operand,
      Variadic<TensorOrMemref>:$dst,
      I64ElementsAttr:$replica_groups,
      OptionalAttr<I64Attr>:$split_dimension,
      OptionalAttr<I64Attr>:$concat_dimension,
      OptionalAttr<I64Attr>:$split_count
  );
  let results = (outs Variadic<TensorOrMemref>);
  let extraClassDeclaration = [{
    // Declarations for BufferizableOpInterface:
    bool bufferizesToMemoryRead(OpOperand &opOperand,
        const bufferization::AnalysisState &state) {
      return opOperand.getOperandNumber() < getNumOperands() / 2;
    }
    bool bufferizesToMemoryWrite(OpOperand &opOperand,
        const bufferization::AnalysisState &state) {
      return opOperand.getOperandNumber() >= getNumOperands() / 2;
    }
    SmallVector<OpResult> getAliasingOpResult(
        OpOperand &opOperand, const bufferization::AnalysisState &state) {
      if (bufferizesToMemoryRead(opOperand, state)) return {};
      return {getOperation()->getOpResult(opOperand.getOperandNumber() - getNumOperands() / 2)};
    }
    LogicalResult bufferize(RewriterBase &rewriter,
        const bufferization::BufferizationOptions &options);
    bufferization::BufferRelation bufferRelation(OpResult opResult,
        const bufferization::AnalysisState &state) {
      return bufferization::BufferRelation::Equivalent;
    }
  }];
}

def FftOp : XlaCpu_Op<"fft", [BufferizableOpInterface]> {
  let summary = "CPU-specific version of FFT";
  let description = [{
    The major differences between this and HLO's fft are:
    - It bufferizes to itself.
    - It uses destination passing style.
  }];

  let arguments = (ins
      TensorOrMemref:$operand,
      TensorOrMemref:$dst,
      I32Attr:$fft_type,
      I64ArrayAttr:$fft_length
  );
  let results = (outs Variadic<TensorOrMemref>);
  let extraClassDeclaration = [{
    // Declarations for BufferizableOpInterface:
    bool bufferizesToMemoryRead(OpOperand &opOperand,
        const bufferization::AnalysisState &state) {
      return opOperand.getOperandNumber() == 0;
    }
    bool bufferizesToMemoryWrite(OpOperand &opOperand,
        const bufferization::AnalysisState &state) {
      return opOperand.getOperandNumber() == 1;
    }
    SmallVector<OpResult> getAliasingOpResult(
        OpOperand &opOperand, const bufferization::AnalysisState &state) {
      if (opOperand.getOperandNumber() == 0) return {};
      return {getOperation()->getOpResult(0)};
    }
    LogicalResult bufferize(RewriterBase &rewriter,
        const bufferization::BufferizationOptions &options);
    bufferization::BufferRelation bufferRelation(OpResult opResult,
        const bufferization::AnalysisState &state) {
      return bufferization::BufferRelation::Equivalent;
    }
  }];
}

def OutfeedOp : XlaCpu_Op<"outfeed", [BufferizableOpInterface]> {
  let summary = "CPU-specific version of Outfeed";
  let description = [{
    The major differences between this and HLO's outfeed are:
    - It bufferizes to itself.
    - It captures the output type to reinstate it after signless conversions.
  }];
  let arguments = (ins
    Variadic<TensorOrMemref>:$operand,
    DefaultValuedStrAttr<StrAttr, "">:$config,
    ArrayAttr:$result_type
  );
  let extraClassDeclaration = [{
    // Declarations for BufferizableOpInterface:
    bool bufferizesToMemoryRead(OpOperand &opOperand,
        const bufferization::AnalysisState &state) {
      return true;
    }
    bool bufferizesToMemoryWrite(OpOperand &opOperand,
        const bufferization::AnalysisState &state) {
      return false;
    }
    SmallVector<OpResult> getAliasingOpResult(
        OpOperand &opOperand, const bufferization::AnalysisState &state) {
      return {};
    }
    LogicalResult bufferize(RewriterBase &rewriter,
        const bufferization::BufferizationOptions &options);
    bufferization::BufferRelation bufferRelation(OpResult opResult,
        const bufferization::AnalysisState &state) {
      return bufferization::BufferRelation::Equivalent;
    }
  }];
}

def AddDependencyOp : XlaCpu_Op<"add_dependency", [BufferizableOpInterface]> {
  let summary = "CPU-specific version of AddDependency";
  let description = [{
    The major differences between this and HLO's add_dependency are:
    - It bufferizes itself.
  }];
  let arguments = (ins
    MHLO_TensorOrToken:$operand,
    MHLO_Token:$token
  );
  let results = (outs MHLO_TensorOrToken);
  let extraClassDeclaration = [{
    // Declarations for BufferizableOpInterface:
    bool bufferizesToMemoryRead(OpOperand &opOperand,
        const bufferization::AnalysisState &state) {
      return opOperand.getOperandNumber() == 0;
    }
    bool bufferizesToMemoryWrite(OpOperand &opOperand,
        const bufferization::AnalysisState &state) {
      return false;
    }
    SmallVector<OpResult> getAliasingOpResult(
        OpOperand &opOperand, const bufferization::AnalysisState &state) {
      if (opOperand.getOperandNumber() == 0 || opOperand.getOperandNumber() == 1)
        return {};
      return {getOperation()->getOpResult(0)};
    }
    LogicalResult bufferize(RewriterBase &rewriter,
        const bufferization::BufferizationOptions &options);
    bufferization::BufferRelation bufferRelation(OpResult opResult,
        const bufferization::AnalysisState &state) {
      return bufferization::BufferRelation::None;
    }
  }];
}

#endif  // TENSORFLOW_COMPILER_XLA_MLIR_XLA_CPU_OPS_TD_