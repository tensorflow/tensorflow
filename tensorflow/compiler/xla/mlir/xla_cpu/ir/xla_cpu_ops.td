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
include "tensorflow/compiler/xla/mlir_hlo/mhlo/IR/hlo_ops_common.td"

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
    bufferization::AliasingOpResultList getAliasingOpResults(
        OpOperand &opOperand, const bufferization::AnalysisState &state);
    LogicalResult bufferize(RewriterBase &rewriter,
        const bufferization::BufferizationOptions &options);
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
    bufferization::AliasingOpResultList getAliasingOpResults(
        OpOperand &opOperand, const bufferization::AnalysisState &state) {
      if (opOperand.getOperandNumber() == 0) return {};
      return {{getOperation()->getOpResult(0),
               bufferization::BufferRelation::Equivalent}};
    }
    LogicalResult bufferize(RewriterBase &rewriter,
        const bufferization::BufferizationOptions &options);
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
      I32Attr:$channel_id_present,
      I64Attr:$op_id,
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
    bufferization::AliasingOpResultList getAliasingOpResults(
        OpOperand &opOperand, const bufferization::AnalysisState &state) {
      if (bufferizesToMemoryRead(opOperand, state)) return {};
      return {{getOperation()->getOpResult(opOperand.getOperandNumber() - getNumOperands() / 2),
               bufferization::BufferRelation::Equivalent}};
    }
    LogicalResult bufferize(RewriterBase &rewriter,
        const bufferization::BufferizationOptions &options);
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
    bufferization::AliasingOpResultList getAliasingOpResults(
        OpOperand &opOperand, const bufferization::AnalysisState &state) {
      if (opOperand.getOperandNumber() == 0) return {};
      return {{getOperation()->getOpResult(0),
               bufferization::BufferRelation::Equivalent}};
    }
    LogicalResult bufferize(RewriterBase &rewriter,
        const bufferization::BufferizationOptions &options);
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
    bufferization::AliasingOpResultList getAliasingOpResults(
        OpOperand &opOperand, const bufferization::AnalysisState &state) {
      return {};
    }
    LogicalResult bufferize(RewriterBase &rewriter,
        const bufferization::BufferizationOptions &options);
  }];
}

def MemRefElementCastOp : XlaCpu_Op<"memref_element_cast",
    [SameOperandsAndResultShape]> {
  let summary = "MemRef reinterpret_cast on element types";
  let description = [{
    This op is the equivalent of C++'s reinterpret_cast on pointers. The element
    types' storage sizes must be the same. Does not cast shapes.
  }];
  let arguments = (ins
    MemRefOf<[I1, I8, I16, I32, I64, BF16, F16, F32, F64]>:$src
  );
  let results = (outs
    MemRefOf<[I1, I8, I16, I32, I64, BF16, F16, F32, F64]>:$dst
  );
  let assemblyFormat = "$src attr-dict `:` type($src) `to` type($dst)";
  let hasVerifier = 1;
}

def RngBitGeneratorOp : XlaCpu_Op<"rng_bit_generator", [BufferizableOpInterface]> {
  let summary = "CPU-specific version of rng_bit_generator";
  let description = [{
    The major differences between this and HLO's rng_bit_generator are:
    - It bufferizes to itself.
    - It uses destination passing style.
  }];
  let arguments = (ins
      TensorOrMemref:$state,
      TensorOrMemref:$dst_state,
      TensorOrMemref:$dst,
      AnyAttr:$rng_algorithm
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
      return opOperand.getOperandNumber() != 0;
    }
    bufferization::AliasingOpResultList getAliasingOpResults(
        OpOperand &opOperand, const bufferization::AnalysisState &state) {
      if (opOperand.getOperandNumber() == 0) return {};
      return {{getOperation()->getOpResult(opOperand.getOperandNumber()-1),
               bufferization::BufferRelation::Equivalent}};
    }
    LogicalResult bufferize(RewriterBase &rewriter,
        const bufferization::BufferizationOptions &options);
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
    bufferization::AliasingOpResultList getAliasingOpResults(
        OpOperand &opOperand, const bufferization::AnalysisState &state) {
      if (opOperand.getOperandNumber() == 0 || opOperand.getOperandNumber() == 1)
        return {};
      return {{getOperation()->getOpResult(0),
               bufferization::BufferRelation::Unknown, /*isDefinite=*/false}};
    }
    LogicalResult bufferize(RewriterBase &rewriter,
        const bufferization::BufferizationOptions &options);
  }];
}

def ConvolutionOp : XlaCpu_Op<"convolution", [BufferizableOpInterface]> {
  let summary = "CPU-specific version of convolution";
  let description = [{
    The major differences between this and HLO's convolution are:
    - It bufferizes to itself.
    - It uses destination passing style.
  }];

  let arguments = (ins
      TensorOrMemref:$input,
      TensorOrMemref:$kernel,
      TensorOrMemref:$dst,
      // Default value: one for each of the spatial dimension.
      OptionalAttr<I64ElementsAttr>:$window_strides,
      // Default value: two zeros for each of the spatial dimension.
      OptionalAttr<I64ElementsAttr>:$padding,
      // Default value: one for each of the spatial dimension.
      OptionalAttr<I64ElementsAttr>:$lhs_dilation,
      // Default value: one for each of the spatial dimension.
      OptionalAttr<I64ElementsAttr>:$rhs_dilation,
      // Default value: false for each of the spatial dimension.
      OptionalAttr<MHLO_BoolElementsAttr>:$window_reversal,
      I64Attr:$inputBatchDimension,
      I64Attr:$inputFeatureDimension,
      I64ArrayAttr:$inputSpatialDimensions,
      I64Attr:$kernelInputFeatureDimension,
      I64Attr:$kernelOutputFeatureDimension,
      I64ArrayAttr:$kernelSpatialDimensions,
      I64Attr:$outputBatchDimension,
      I64Attr:$outputFeatureDimension,
      I64ArrayAttr:$outputSpatialDimensions,
      I64Attr:$feature_group_count,
      I64Attr:$batch_group_count,
      MHLO_PrecisionConfigAttr:$precision_config
  );
  let results = (outs Variadic<TensorOrMemref>);
  let extraClassDeclaration = [{
    // Declarations for BufferizableOpInterface:
    bool bufferizesToMemoryRead(OpOperand &opOperand,
        const bufferization::AnalysisState &state) {
      return opOperand.getOperandNumber() < 2;
    }
    bool bufferizesToMemoryWrite(OpOperand &opOperand,
        const bufferization::AnalysisState &state) {
      return opOperand.getOperandNumber() == 2;
    }
    bufferization::AliasingOpResultList getAliasingOpResults(
        OpOperand &opOperand, const bufferization::AnalysisState &state) {
      if (opOperand.getOperandNumber() < 2) return {};
      return {{getOperation()->getOpResult(0),
               bufferization::BufferRelation::Equivalent}};
    }
    LogicalResult bufferize(RewriterBase &rewriter,
        const bufferization::BufferizationOptions &options);
  }];
}

#endif  // TENSORFLOW_COMPILER_XLA_MLIR_XLA_CPU_OPS_TD_
