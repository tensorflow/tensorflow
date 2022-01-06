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

// This is the operation definition file for LHMLO level GPU operations.
// Because these are LMHLO level operations, they operate on memrefs.

#ifndef LHLO_GPU_OPS
#define LHLO_GPU_OPS

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir-hlo/Dialect/mhlo/IR/hlo_ops_base.td"
include "mlir-hlo/Dialect/lhlo/IR/lhlo_ops_base.td"
include "mlir-hlo/Dialect/lhlo_gpu/IR/lhlo_gpu_ops_base.td"
include "mlir-hlo/Dialect/lhlo_gpu/IR/lhlo_gpu_ops_enums.td"
include "mlir-hlo/Dialect/lhlo_gpu/IR/lhlo_gpu_ops_structs.td"

class LHLOGPU_Op<string mnemonic, list<OpTrait> traits = []> :
  Op<LHLO_GPU_Dialect, mnemonic,
    !listconcat([MemoryEffects<[MemRead, MemWrite]>], traits)>;

// Type for scratch buffers used by GPU library calls (memref<?xi8>)
def UntypedBuffer : MemRefRankOf<[I8], [1]>;

// Cholesky info output buffer type.
def I32Buffer : MemRefOf<[I32]>;

//===----------------------------------------------------------------------===//
// LMHLO ops representing batch norm library functions.
//===----------------------------------------------------------------------===//

// Note: these are semantically different from similar LHLO as the GPU library
// calls generate or consume standard deviation, whereas LHLO ops generate or
// consume variance (= std-dev ^ 2).

def LHLOGPU_BatchNormGradOp : LHLOGPU_Op<"batch_norm_grad"> {
  let summary = "Batch Normalization Gradient";
  let description = [{
    Calculates gradients of batch norm.

    See https://www.tensorflow.org/xla/operation_semantics#batchnormgrad
  }];
  let arguments = (ins
    Arg<LHLO_FpBuffer, "", [MemRead]>:$operand,
    Arg<LHLO_FpBuffer,      "", [MemRead]>:$scale,
    Arg<LHLO_FpBuffer,      "", [MemRead]>:$mean,
    Arg<LHLO_FpBuffer,      "", [MemRead]>:$stddev,
    Arg<LHLO_FpBuffer, "", [MemRead]>:$grad_output,
    Arg<LHLO_FpBuffer, "", [MemWrite]>:$grad_operand,  // gradient of $operand.
    Arg<LHLO_FpBuffer,      "", [MemWrite]>:$grad_scale,
    Arg<LHLO_FpBuffer,      "", [MemWrite]>:$grad_offset,
    F32Attr:$epsilon,
    I64Attr:$feature_index
  );
}

def LHLOGPU_BatchNormInferenceOp : LHLOGPU_Op<"batch_norm_inference"> {
  let summary = "Batch Normalization for Inference";
  let description = [{
    Normalizes an array across batch and spatial dimensions.

    See https://www.tensorflow.org/xla/operation_semantics#batchnorminference
  }];
  let arguments = (ins
    Arg<LHLO_FpBuffer, "", [MemRead]>:$operand,
    Arg<LHLO_FpBuffer,      "", [MemRead]>:$scale,
    Arg<LHLO_FpBuffer,      "", [MemRead]>:$offset,
    Arg<LHLO_FpBuffer,      "", [MemRead]>:$mean,
    Arg<LHLO_FpBuffer,      "", [MemRead]>:$stddev,
    Arg<LHLO_FpBuffer, "", [MemWrite]>:$output,
    F32Attr:$epsilon,
    I64Attr:$feature_index);
}

def LHLOGPU_BatchNormTrainingOp : LHLOGPU_Op<"batch_norm_training"> {
  let summary = "Batch Normalization for Training";
  let description = [{
    Normalizes an array across batch and spatial dimensions.

    See https://www.tensorflow.org/xla/operation_semantics#batchnormtraining
  }];

  let arguments = (ins
    Arg<LHLO_FpBuffer, "", [MemRead]>:$operand,
    Arg<LHLO_FpBuffer,      "", [MemRead]>:$scale,
    Arg<LHLO_FpBuffer,      "", [MemRead]>:$offset,
    Arg<LHLO_FpBuffer, "", [MemWrite]>:$output,
    Arg<LHLO_FpBuffer,      "", [MemWrite]>:$batch_mean,
    Arg<LHLO_FpBuffer,      "", [MemWrite]>:$batch_stddev,
    F32Attr:$epsilon,
    I64Attr:$feature_index
  );
}

//===----------------------------------------------------------------------===//
// LMHLO ops representing convolution library functions.
//===----------------------------------------------------------------------===//

class GpuConvolutionAttributes<dag extraAttribs> {
  dag attributes = !con(
    ConvolutionAttributes.attributes,
    (ins F64Attr:$result_scale),
    extraAttribs,
    (ins ConvolutionBackendConfigAttr:$backend_config));
}

// Provide a custom assembly format for all LHLO_GPU convolution operations.
class LHLOGPU_ConvBaseOp<string mnemonic> : LHLOGPU_Op<mnemonic> {
 let assemblyFormat = [{
    `(`operands`)`
       `dim_numbers` `=` custom<ConvolutionDimensions>($dimension_numbers) `,`
       `window` `=` `{` custom<WindowAttributes>($window_strides, $padding,
                                                 $lhs_dilation, $rhs_dilation,
                                                 $window_reversal) `}`
       attr-dict `:` functional-type(operands, results)
  }];
}

def LHLOGPU_ConvForwardOp : LHLOGPU_ConvBaseOp<"conv_forward"> {
  let arguments = !con(
    (ins
       Arg<LHLO_Buffer, "", [MemRead]>:$input,
       Arg<LHLO_Buffer, "", [MemRead]>:$filter,
       Arg<LHLO_Buffer, "", [MemWrite]>:$output,
       Arg<LHLO_Buffer, "", [MemWrite]>:$scratch),
     GpuConvolutionAttributes<(ins)>.attributes);
}

def LHLOGPU_ConvBackwardInputOp : LHLOGPU_ConvBaseOp<"conv_backwardinput"> {
  let arguments = !con(
    (ins
       Arg<LHLO_Buffer, "", [MemRead]>:$d_output,
       Arg<LHLO_Buffer, "", [MemRead]>:$filter,
       Arg<LHLO_Buffer, "", [MemWrite]>:$d_input,
       Arg<LHLO_Buffer, "", [MemWrite]>:$scratch),
     GpuConvolutionAttributes<(ins)>.attributes);
}

def LHLOGPU_ConvBackwardFilterOp : LHLOGPU_ConvBaseOp<"conv_backwardfilter"> {
  let arguments = !con(
    (ins
       Arg<LHLO_Buffer, "", [MemRead]>:$input,
       Arg<LHLO_Buffer, "", [MemRead]>:$d_output,
       Arg<LHLO_Buffer, "", [MemWrite]>:$d_filter,
       Arg<LHLO_Buffer, "", [MemWrite]>:$scratch),
     GpuConvolutionAttributes<(ins)>.attributes);
}

// output = activation(result_scale * conv(input, filter) + bias)
def LHLOGPU_ConvForwardFusedOp : LHLOGPU_ConvBaseOp<"conv_forward_fused"> {
  let arguments = !con(
    (ins
       Arg<LHLO_Buffer, "", [MemRead]>:$input,
       Arg<LHLO_Buffer, "", [MemRead]>:$filter,
       Arg<LHLO_Buffer, "", [MemRead]>:$bias,
       Arg<LHLO_Buffer, "", [MemWrite]>:$output,
       Arg<LHLO_Buffer, "", [MemWrite]>:$scratch),
     GpuConvolutionAttributes<(ins
         ActivationAttr:$activation_mode)>.attributes);
}

// output = activation(result_scale * conv(input, filter) +
//                     side_input * side_input_scale +
//                     bias)
def LHLOGPU_ConvForwardFusedSideInputOp :
      LHLOGPU_ConvBaseOp<"conv_forward_fused_with_side_input"> {
  let arguments = !con(
    (ins
       Arg<LHLO_Buffer, "", [MemRead]>:$input,
       Arg<LHLO_Buffer, "", [MemRead]>:$filter,
       Arg<LHLO_Buffer, "", [MemRead]>:$bias,
       Arg<LHLO_Buffer, "", [MemRead]>:$side_input,
       Arg<LHLO_Buffer, "", [MemWrite]>:$output,
       Arg<LHLO_Buffer, "", [MemWrite]>:$scratch),
     GpuConvolutionAttributes<(ins
         ActivationAttr:$activation_mode,
         F64Attr:$side_input_scale)>.attributes);
}

//===----------------------------------------------------------------------===//
// LMHLO ops representing other library functions.
//===----------------------------------------------------------------------===//

// output = alpha * (lhs * rhs)
// Verify: beta = 0.0
def LHLOGPU_GEMMOp : LHLOGPU_Op<"gemm"> {
  let arguments = (ins
    Arg<LHLO_Buffer, "", [MemRead]>:$lhs,
    Arg<LHLO_Buffer, "", [MemRead]>:$rhs,
    Arg<LHLO_Buffer, "", [MemRead]>:$output,
    DotDimensionNumbers:$dot_dimension_numbers,
    F64Attr:$alpha_real,
    F64Attr:$alpha_imag,
    I64Attr:$batch_size,
    I64Attr:$lhs_stride,
    I64Attr:$rhs_stride,
    OptionalAttr<I64Attr>:$algorithm);
}

// output = alpha(lhs * rhs) + beta * bias
def LHLOGPU_GEMM_BiasOp : LHLOGPU_Op<"gemm_bias"> {
  let arguments = (ins
    Arg<LHLO_Buffer, "", [MemRead]>:$lhs,
    Arg<LHLO_Buffer, "", [MemRead]>:$rhs,
    Arg<LHLO_Buffer, "", [MemRead]>:$bias,
    Arg<LHLO_Buffer, "", [MemRead]>:$output,
    DotDimensionNumbers:$dot_dimension_numbers,
    F64Attr:$alpha_real,
    F64Attr:$alpha_imag,
    F64Attr:$beta,
    I64Attr:$batch_size,
    I64Attr:$lhs_stride,
    I64Attr:$rhs_stride,
    OptionalAttr<I64Attr>:$algorithm);
}

def LHLOGPU_CholeskyOp : LHLOGPU_Op<"cholesky"> {
  let arguments = (ins
    Arg<LHLO_Buffer, "", [MemRead]>:$input,
    Arg<LHLO_Buffer, "", [MemWrite]>:$output,
    Arg<LHLO_Buffer, "", [MemWrite]>:$scratch,
    Arg<I32Buffer, "", [MemWrite]>:$info,
    BoolAttr:$is_lower);
}

def LHLOGPU_AllReduceStartOp :
  LHLOGPU_Op<"all_reduce_start", [SameOperandsElementType, SameVariadicOperandSize]> {
  let summary = "AllReduceStart operator";
  let description = [{
    Performs an asynchronous custom reduction across replicas.
  }];
  let arguments = (ins
    Arg<Variadic<LHLO_Buffer>, "", [MemRead]>:$operands,
    Arg<Variadic<LHLO_Buffer>, "", [MemWrite]>:$results,
    I64ElementsAttr:$replica_groups,
    DefaultValuedAttr<BoolAttr, "false">:$constrain_layout,
    OptionalAttr<ChannelHandle>:$channel_id,
    DefaultValuedAttr<BoolAttr, "false">:$use_global_device_ids
  );
  let results = (outs HLO_Token:$token);
  let regions = (region SizedRegion<1>:$computation);
  let verifier = [{ return Verify(*this); }];
}

def LHLOGPU_AllReduceDoneOp:
  LHLOGPU_Op<"all_reduce_done", [SameVariadicOperandSize]> {
  let summary = "AllReduceDone operator";
  let arguments = (ins
    HLO_Token:$token,
    Arg<Variadic<LHLO_Buffer>, "", [MemRead]>:$operands,
    Arg<Variadic<LHLO_Buffer>, "", [MemWrite]>:$results
  );
}

#endif // LHLO_GPU_OPS
