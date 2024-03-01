/* Copyright 2020 The OpenXLA Authors.

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
include "lhlo/IR/lhlo_ops_base.td"
include "lhlo_gpu/IR/lhlo_gpu_ops_base.td"
include "lhlo_gpu/IR/lhlo_gpu_ops_enums.td"
include "stablehlo/dialect/Base.td"

class LHLOGPU_Op<string mnemonic, list<Trait> traits = []> :
  Op<LmhloGpuDialect, mnemonic,
    !listconcat([MemoryEffects<[MemRead, MemWrite]>], traits)>;

// Type for scratch buffers used by GPU library calls (memref<?xi8>)
def UntypedBuffer : MemRefRankOf<[I8], [1]>;

// Cholesky info output buffer type.
def I32Buffer : MemRefOf<[I32]>;

//===----------------------------------------------------------------------===//
// LMHLO ops representing convolution library functions.
//===----------------------------------------------------------------------===//

class GpuConvolutionAttributes<dag extraAttribs> {
  dag attributes = !con(
    MHLO_ConvolutionAttributes.attributes,
    (ins F64Attr:$result_scale),
    extraAttribs,
    (ins ConvolutionBackendConfigAttr:$backend_config));
}

// Provide a custom assembly format for all LHLO_GPU convolution operations.
class LHLOGPU_ConvBaseOp<string mnemonic, list<Trait> traits = []> : LHLOGPU_Op<mnemonic, traits> {
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
         ActivationAttr:$activation_mode,
         F64Attr:$leakyrelu_alpha)>.attributes);
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

// Reordering helpers for int8x32 cuDNN convolutions.
def LHLOGPU_CudnnConvReorderFilterOp : LHLOGPU_Op<"cudnn_conv_reorder_filter"> {
  let arguments = (ins
    Arg<LHLO_Buffer, "", [MemRead]>:$filter_input,
    Arg<LHLO_Buffer, "", [MemWrite]>:$filter_output,
    I64ElementsAttr:$filter_dims);
}

def LHLOGPU_CudnnConvReorderFilterAndBiasOp :
      LHLOGPU_Op<"cudnn_conv_reorder_filter_and_bias"> {
  let arguments = (ins
    Arg<LHLO_Buffer, "", [MemRead]>:$filter_input,
    Arg<LHLO_Buffer, "", [MemRead]>:$bias_input,
    Arg<LHLO_Buffer, "", [MemWrite]>:$filter_output,
    Arg<LHLO_Buffer, "", [MemWrite]>:$bias_output,
    I64ElementsAttr:$filter_dims);
}

def LHLOGPU_ConvForwardGraphOp :
      LHLOGPU_ConvBaseOp<"conv_forward_graph", [AttrSizedOperandSegments]> {
  let arguments = !con(
    (ins
       Arg<LHLO_Buffer, "", [MemRead]>:$input,
       Arg<LHLO_Buffer, "", [MemRead]>:$filter,
       Arg<Variadic<LHLO_Buffer>, "", [MemRead]>:$binary_operands,
       Arg<LHLO_Buffer, "", [MemWrite]>:$output,
       Arg<Variadic<LHLO_Buffer>, "", [MemWrite]>:$aux_outputs,
       Arg<LHLO_Buffer, "", [MemWrite]>:$scratch),
     GpuConvolutionAttributes<(ins
          I32Attr:$n_aux_outputs,
          StrAttr:$serialized_graph)>.attributes);
}

//===----------------------------------------------------------------------===//
// LMHLO ops representing other library functions.
//===----------------------------------------------------------------------===//

// c = alpha * (a @ b) + beta * c
def LHLOGPU_GEMMOp : LHLOGPU_Op<"gemm"> {
  let arguments = (ins
    Arg<LHLO_Buffer, "", [MemRead]>:$a,
    Arg<LHLO_Buffer, "", [MemRead]>:$b,
    Arg<LHLO_Buffer, "", [MemRead, MemWrite]>:$c,
    Arg<Optional<LHLO_Buffer>, "", [MemRead, MemWrite]>:$workspace,
    MHLO_DotDimensionNumbers:$dot_dimension_numbers,
    MHLO_PrecisionConfigAttr:$precision_config,
    F64Attr:$alpha_real,
    F64Attr:$alpha_imag,
    F64Attr:$beta,
    OptionalAttr<I64Attr>:$algorithm,
    OptionalAttr<BoolAttr>:$grad_x,
    OptionalAttr<BoolAttr>:$grad_y);
}

def LHLOGPU_CublasLtMatmulOp : LHLOGPU_Op<"cublas.lt.matmul", [AttrSizedOperandSegments]> {
  let arguments = (ins
    Arg<LHLO_Buffer, "", [MemRead]>:$a,
    Arg<LHLO_Buffer, "", [MemRead]>:$b,
    Arg<LHLO_Buffer, "", [MemRead]>:$c,
    Arg<LHLO_Buffer, "", [MemWrite]>:$d,
    Arg<Optional<LHLO_Buffer>, "", [MemRead]>:$bias,
    Arg<Optional<LHLO_Buffer>, "", [MemRead, MemWrite]>:$aux,
    MHLO_DotDimensionNumbers:$dot_dimension_numbers,
    MHLO_PrecisionConfigAttr:$precision_config,
    F64Attr:$alpha_real,
    F64Attr:$alpha_imag,
    F64Attr:$beta,
    CublasLtMatmulEpilogueAttr:$epilogue,
    I64Attr:$algorithm,
    OptionalAttr<BoolAttr>:$grad_x,
    OptionalAttr<BoolAttr>:$grad_y);
}

def LHLOGPU_CublasLtMatmulF8Op : LHLOGPU_Op<"cublas.lt.matmul.f8", [AttrSizedOperandSegments]> {
  let arguments = (ins
    Arg<LHLO_Buffer, "", [MemRead]>:$a,
    Arg<LHLO_Buffer, "", [MemRead]>:$b,
    Arg<LHLO_Buffer, "", [MemRead]>:$c,
    Arg<LHLO_Buffer, "", [MemRead]>:$a_scale,
    Arg<LHLO_Buffer, "", [MemRead]>:$b_scale,
    Arg<LHLO_Buffer, "", [MemRead]>:$c_scale,
    Arg<LHLO_Buffer, "", [MemRead]>:$d_scale,
    Arg<LHLO_Buffer, "", [MemWrite]>:$d,
    Arg<Optional<LHLO_Buffer>, "", [MemRead]>:$bias,
    Arg<Optional<LHLO_Buffer>, "", [MemWrite]>:$d_amax,
    MHLO_DotDimensionNumbers:$dot_dimension_numbers,
    MHLO_PrecisionConfigAttr:$precision_config,
    F64Attr:$alpha_real,
    F64Attr:$alpha_imag,
    F64Attr:$beta,
    CublasLtMatmulEpilogueAttr:$epilogue,
    I64Attr:$algorithm,
    OptionalAttr<BoolAttr>:$grad_x,
    OptionalAttr<BoolAttr>:$grad_y);
}

def LHLOGPU_CholeskyOp : LHLOGPU_Op<"cholesky"> {
  let arguments = (ins
    Arg<LHLO_Buffer, "", [MemRead]>:$input,
    Arg<LHLO_Buffer, "", [MemWrite]>:$output,
    Arg<LHLO_Buffer, "", [MemWrite]>:$scratch,
    Arg<I32Buffer, "", [MemWrite]>:$info,
    BoolAttr:$is_lower);
}

// Base class for all async collective communication operations.
class LHLOGPU_AsyncCollectiveCommunicationOpBase<string name, list<Trait> traits = []> :
  LHLOGPU_Op<name, traits> {
  let results = (outs MHLO_Token:$token);
  let hasVerifier = 1;
}

// Base class for async all-reduce & all-gather.
class LHLOGPU_AsyncCollectiveCommunicationOp<string name, list<Trait> traits = []> :
    LHLOGPU_AsyncCollectiveCommunicationOpBase<name, !listconcat(traits, [SameVariadicOperandSize])> {
  dag arguments_base = (ins
    Arg<Variadic<LHLO_Buffer>, "", [MemRead]>:$inputs,
    Arg<Variadic<LHLO_Buffer>, "", [MemWrite]>:$outputs,
    I64ElementsAttr:$replica_groups,
    DefaultValuedOptionalAttr<BoolAttr, "false">:$constrain_layout,
    OptionalAttr<MHLO_ChannelHandle>:$channel_id,
    DefaultValuedOptionalAttr<BoolAttr, "false">:$use_global_device_ids,
    BoolAttr:$is_sync,
    BoolAttr:$no_parallel_custom_call
  );
}

def LHLOGPU_AllReduceStartOp :
  LHLOGPU_AsyncCollectiveCommunicationOp<"all_reduce_start", [SameOperandsElementType]> {
  let summary = "AllReduceStart operator";
  let description = [{
    Performs an asynchronous custom reduction across replicas.
  }];
  let arguments = arguments_base;
  let regions = (region SizedRegion<1>:$computation);
}

def LHLOGPU_AllReduceDoneOp: LHLOGPU_Op<"all_reduce_done"> {
  let summary = "AllReduceDone operator";
  let arguments = (ins MHLO_Token:$token);
}

def LHLOGPU_CollectiveBroadcastStartOp :
  LHLOGPU_AsyncCollectiveCommunicationOp<"collective_broadcast_start"> {
  let summary = "CollectiveBroadcastStart operator";
  let arguments = (ins
    Arg<Variadic<LHLO_Buffer>, "", [MemRead]>:$inputs,
    Arg<Variadic<LHLO_Buffer>, "", [MemWrite]>:$outputs,
    I64ElementsAttr:$replica_groups,
    DefaultValuedOptionalAttr<BoolAttr, "false">:$constrain_layout,
    OptionalAttr<MHLO_ChannelHandle>:$channel_id,
    DefaultValuedOptionalAttr<BoolAttr, "false">:$use_global_device_ids,
    BoolAttr:$is_sync,
    BoolAttr:$no_parallel_custom_call
  );
}

def LHLOGPU_CollectiveBroadcastDoneOp: LHLOGPU_Op<"collective_broadcast_done"> {
  let summary = "CollectiveBroadcastDone operator";
  let arguments = (ins MHLO_Token:$token);
}

def LHLOGPU_CollectivePermuteStartOp :
  LHLOGPU_AsyncCollectiveCommunicationOpBase<"collective_permute_start"> {
  let summary = "CollectivePermuteStart operator";
  let arguments = (ins
    Arg<LHLO_Buffer, "", [MemRead]>:$operand,
    Arg<LHLO_Buffer, "", [MemWrite]>:$output,
    I64ElementsAttr:$source_target_pairs,
    OptionalAttr<MHLO_ChannelHandle>:$channel_id,
    BoolAttr:$is_sync,
    BoolAttr:$no_parallel_custom_call
  );
}

def LHLOGPU_CollectivePermuteDoneOp: LHLOGPU_Op<"collective_permute_done"> {
  let summary = "CollectivePermuteDone operator";
  let arguments = (ins MHLO_Token:$token);
}

def LHLOGPU_AllGatherStartOp :
  LHLOGPU_AsyncCollectiveCommunicationOp<"all_gather_start"> {
  let summary = "AllGatherStart operator";
  let description = [{
     Performs asynchronous concatenation across replicas.
  }];
  let arguments = !con(
    arguments_base,
    (ins I64Attr:$all_gather_dimension));
}

def LHLOGPU_AllGatherDoneOp: LHLOGPU_Op<"all_gather_done"> {
  let summary = "AllGatherDone operator";
  let arguments = (ins MHLO_Token:$token);
}

def LHLOGPU_ReduceScatterStartOp :
  LHLOGPU_AsyncCollectiveCommunicationOp<"reduce_scatter_start", [SameOperandsElementType]> {
  let summary = "ReduceScatter start operator";
  let description = [{
     Performs all_reduce followed by a scatter.
  }];
  let arguments = !con(
    arguments_base,
    (ins I64Attr:$scatter_dimension));
  let regions = (region SizedRegion<1>:$computation);
}

def LHLOGPU_ReduceScatterDoneOp: LHLOGPU_Op<"reduce_scatter_done"> {
  let summary = "ReduceScatterDone operator";
  let arguments = (ins MHLO_Token:$token);
}

def LHLOGPU_AllToAllStartOp :
  LHLOGPU_AsyncCollectiveCommunicationOp<"all_to_all_start", [SameOperandsElementType]> {
  let summary = "All2AllStart operator";
  let description = [{
     Send data from all cores to all cores.
  }];
  let arguments = !con(
    arguments_base,
    (ins OptionalAttr<I64Attr>:$split_dimension));
}

def LHLOGPU_AllToAllDoneOp: LHLOGPU_Op<"all_to_all_done"> {
  let summary = "All2AllDone operator";
  let arguments = (ins MHLO_Token:$token);
}

def LHLOGPU_CopyStartOp : LHLOGPU_Op<"copy_start"> {
  let summary = "CopyStart operator";
  let description = [{
     Performs memory copies followed by a copy_done instruction.
  }];
  let arguments = (ins
    Arg<LHLO_Buffer, "", [MemRead]>:$input,
    Arg<LHLO_Buffer, "", [MemWrite]>:$output,
    I64Attr:$memory_space);
  let results = (outs MHLO_Token:$token);
}

def LHLOGPU_CopyDoneOp : LHLOGPU_Op<"copy_done"> {
  let summary = "CopyDone operator";
  let arguments = (ins MHLO_Token:$token);
}


def LHLOGPU_CudnnNormOp : LHLOGPU_Op<"Norm", [AttrSizedOperandSegments]> {
  let arguments = (ins
    Arg<LHLO_Buffer, "", [MemRead]>:$x,
    Arg<LHLO_Buffer, "", [MemRead]>:$scale,
    Arg<LHLO_Buffer, "", [MemWrite]>:$y_or_dx,
    Arg<Optional<LHLO_Buffer>, "", [MemRead]>:$bias,
    Arg<Optional<LHLO_Buffer>, "", [MemRead]>:$dy,
    Arg<Optional<LHLO_Buffer>, "", [MemRead, MemWrite]>:$expectation,
    Arg<Optional<LHLO_Buffer>, "", [MemRead, MemWrite]>:$norm_factor,
    Arg<Optional<LHLO_Buffer>, "", [MemWrite]>:$dscale,
    Arg<Optional<LHLO_Buffer>, "", [MemWrite]>:$dbias,
    Arg<LHLO_Buffer, "", [MemWrite]>:$scratch,
    NormAlgorithmConfigAttr:$algorithm_config,
    F64Attr:$epsilon,
    I64ArrayAttr:$operand_layouts,
    CudnnNormKindAttr:$kind
    );
}

def LHLOGPU_fusedMHAOp : LHLOGPU_Op<"fMHA", [AttrSizedOperandSegments]> {
  let arguments = (ins
    Arg<LHLO_Buffer, "", [MemRead]>:$lhs_bmm1,
    Arg<LHLO_Buffer, "", [MemRead]>:$rhs_bmm1,
    Arg<LHLO_Buffer, "", [MemRead]>:$rhs_bmm2,
    Arg<Optional<LHLO_Buffer>, "", [MemRead]>:$mask,
    Arg<Optional<LHLO_Buffer>, "", [MemRead]>:$bias,
    Arg<LHLO_Buffer, "", [MemWrite]>:$output,
    Arg<LHLO_Buffer, "", [MemWrite]>:$scratch,
    Arg<Optional<LHLO_Buffer>, "", [MemWrite]>:$activation,
    MHLO_DotDimensionNumbers:$bmm1_dot_dimension_numbers,
    MHLO_DotDimensionNumbers:$bmm2_dot_dimension_numbers,
    I64ArrayAttr:$intermediate_tensor_dimensions,
    I64ArrayAttr:$intermediate_tensor_layout,
    F64Attr:$fmha_scale,
    FusedMhaDagSignatureAttr:$fused_mha_dag,
    FusedMHAAlgorithmConfigAttr:$algorithm_config,
    OptionalAttr<F64Attr>:$dropout_rate,
    OptionalAttr<I64Attr>:$seed,
    BoolAttr:$is_flash_attention,
    BoolAttr:$is_causal_mask
    );
}

def LHLOGPU_fusedMHABackwardOp : LHLOGPU_Op<"fMHABackward", [AttrSizedOperandSegments]> {
  let arguments = (ins
    Arg<LHLO_Buffer, "", [MemRead]>:$bmm1_grad_gemm1_rhs,
    Arg<LHLO_Buffer, "", [MemRead]>:$bmm1_grad_gemm2_rhs,
    Arg<LHLO_Buffer, "", [MemRead]>:$bmm2_grad_gemm2_rhs,
    Arg<LHLO_Buffer, "", [MemRead]>:$bmm2_grad_gemm1_lhs,
    Arg<LHLO_Buffer, "", [MemRead]>:$d_output,
    Arg<Optional<LHLO_Buffer>, "", [MemRead]>:$mask,
    Arg<Optional<LHLO_Buffer>, "", [MemRead]>:$bias,
    Arg<Optional<LHLO_Buffer>, "", [MemRead]>:$fwd_output,
    Arg<LHLO_Buffer, "", [MemWrite]>:$d_bmm1_lhs,
    Arg<LHLO_Buffer, "", [MemWrite]>:$d_bmm1_rhs,
    Arg<LHLO_Buffer, "", [MemWrite]>:$d_bmm2_rhs,
    Arg<Optional<LHLO_Buffer>, "", [MemWrite]>:$d_S,
    Arg<Optional<LHLO_Buffer>, "", [MemWrite]>:$softmax_sum,
    Arg<Optional<LHLO_Buffer>, "", [MemWrite]>:$d_Q_accum,
    Arg<LHLO_Buffer, "", [MemWrite]>:$scratch,
    Arg<Optional<LHLO_Buffer>, "", [MemWrite]>:$d_bias,
    MHLO_DotDimensionNumbers:$bmm1_grad_gemm1_dot_dimension_numbers,
    MHLO_DotDimensionNumbers:$bmm1_grad_gemm2_dot_dimension_numbers,
    MHLO_DotDimensionNumbers:$bmm2_grad_gemm1_dot_dimension_numbers,
    MHLO_DotDimensionNumbers:$bmm2_grad_gemm2_dot_dimension_numbers,
    I64ArrayAttr:$intermediate_tensor_dimensions,
    I64ArrayAttr:$intermediate_tensor_layout,
    F64Attr:$fmha_scale,
    FusedMhaBackwardDagSignatureAttr:$fused_mha_dag,
    FusedMHAAlgorithmConfigAttr:$algorithm_config,
    OptionalAttr<F64Attr>:$dropout_rate,
    OptionalAttr<I64Attr>:$seed,
    BoolAttr:$is_flash_attention,
    BoolAttr:$is_causal_mask
    );
}

def LHLOGPU_RadixSortOp: LHLOGPU_Op<"radix_sort", [SameVariadicOperandSize]> {
  let arguments = (ins
    Arg<Variadic<LHLO_Buffer>, "", [MemRead]>:$inputs,
    Arg<Variadic<LHLO_Buffer>, "", [MemWrite]>:$output,
    Arg<LHLO_Buffer, "", [MemWrite]>:$scratch,
    DefaultValuedOptionalAttr<BoolAttr, "false">:$descending
  );
}

#endif // LHLO_GPU_OPS
