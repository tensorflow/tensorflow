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
// This file contains operator definitions for TF uniform quantized ops.

include "tensorflow/compiler/mlir/tensorflow/ir/tf_op_base.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/IR/OpAsmInterface.td"

// Uniform quantization operators
def TF_UniformDequantizeOp : TF_Op<"UniformDequantize", [NoSideEffect]> {
  // TODO(b/230804708): Improve the operator description.
  let summary = "Dequantize the given uniform quantized tensor.";

  let arguments = (ins
    TensorOf<[TF_Qint32, TF_Qint8]>:$input,
    TF_Float32Tensor:$scales,
    TF_Int32Tensor:$zps,

    DefaultValuedAttr<I64Attr, "-1">:$quantization_axis,
    DefaultValuedAttr<I64Attr, "-128">:$quantization_min_val,
    DefaultValuedAttr<I64Attr, "127">:$quantization_max_val
  );

  let results = (outs
    TF_Float32Tensor:$output
  );

  TF_DerivedOperandTypeAttr Tin = TF_DerivedOperandTypeAttr<0>;
  TF_DerivedResultTypeAttr Tout = TF_DerivedResultTypeAttr<0>;
}

def TF_UniformQuantizeOp : TF_Op<"UniformQuantize", [NoSideEffect]> {
  // TODO(b/230804708): Improve the operator description.
  let summary = "Convert the given tensor to uniform quantized tensor.";

  let arguments = (ins
    TensorOf<[TF_Float32, TF_Qint32, TF_Qint8]>:$input,
    TF_Float32Tensor:$input_scales,
    TF_Int32Tensor:$input_zps,
    TF_Float32Tensor:$output_scales,
    TF_Int32Tensor:$output_zps,

    DefaultValuedAttr<I64Attr, "-1">:$input_quantization_axis,
    DefaultValuedAttr<I64Attr, "-128">:$input_quantization_min_val,
    DefaultValuedAttr<I64Attr, "127">:$input_quantization_max_val,
    DefaultValuedAttr<I64Attr, "-1">:$output_quantization_axis,
    DefaultValuedAttr<I64Attr, "-128">:$output_quantization_min_val,
    DefaultValuedAttr<I64Attr, "127">:$output_quantization_max_val
  );

  let results = (outs
    TensorOf<[TF_Qint32, TF_Qint8]>:$output
  );

  TF_DerivedOperandTypeAttr Tin = TF_DerivedOperandTypeAttr<0>;
  TF_DerivedResultTypeAttr Tout = TF_DerivedResultTypeAttr<0>;
}

def TF_UniformQuantizedAddOp : TF_Op<"UniformQuantizedAdd", [NoSideEffect]> {
  // TODO(b/230804708): Improve the operator description.
  let summary = "Quantized add operator.";

  let arguments = (ins
    TensorOf<[TF_Qint32]>:$lhs,
    TensorOf<[TF_Qint32]>:$rhs,
    TF_Float32Tensor:$lhs_scales,
    TF_Int32Tensor:$lhs_zps,
    TF_Float32Tensor:$rhs_scales,
    TF_Int32Tensor:$rhs_zps,
    TF_Float32Tensor:$output_scales,
    TF_Int32Tensor:$output_zps,

    DefaultValuedAttr<I64Attr, "-1">:$lhs_quantization_axis,
    DefaultValuedAttr<I64Attr, "-128">:$lhs_quantization_min_val,
    DefaultValuedAttr<I64Attr, "127">:$lhs_quantization_max_val,
    DefaultValuedAttr<I64Attr, "-1">:$rhs_quantization_axis,
    DefaultValuedAttr<I64Attr, "-128">:$rhs_quantization_min_val,
    DefaultValuedAttr<I64Attr, "127">:$rhs_quantization_max_val,
    DefaultValuedAttr<I64Attr, "-1">:$output_quantization_axis,
    DefaultValuedAttr<I64Attr, "-128">:$output_quantization_min_val,
    DefaultValuedAttr<I64Attr, "127">:$output_quantization_max_val
  );

  let results = (outs
    TensorOf<[TF_Qint32]>:$output
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
}

def TF_UniformQuantizedClipByValueOp : TF_Op<"UniformQuantizedClipByValue", [NoSideEffect]> {
  // TODO(b/230804708): Improve the operator description.
  let summary = "Quantized clipping operator.";

  let arguments = (ins
    TensorOf<[TF_Qint32]>:$operand,
    TensorOf<[TF_Qint32]>:$min,
    TensorOf<[TF_Qint32]>:$max,
    TF_Float32Tensor:$scales,
    TF_Int32Tensor:$zps,

    DefaultValuedAttr<I64Attr, "-1">:$quantization_axis,
    DefaultValuedAttr<I64Attr, "-128">:$quantization_min_val,
    DefaultValuedAttr<I64Attr, "127">:$quantization_max_val
  );

  let results = (outs
    TensorOf<[TF_Qint32]>:$output
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
}

// TODO(b/230804708): Add hybrid convolution.
def TF_UniformQuantizedConvolutionOp : TF_Op<"UniformQuantizedConvolution", [NoSideEffect]> {
  // TODO(b/230804708): Improve the operator description.
  let summary = "Quantized conv2d operator.";

  let arguments = (ins
    TensorOf<[TF_Float32, TF_Qint8]>:$lhs,
    TensorOf<[TF_Qint8]>:$rhs,
    TF_Float32Tensor:$lhs_scales,
    TF_Int32Tensor:$lhs_zps,
    TF_Float32Tensor:$rhs_scales,
    TF_Int32Tensor:$rhs_zps,
    TF_Float32Tensor:$output_scales,
    TF_Int32Tensor:$output_zps,
    TF_Int32Tensor:$window_strides,
    TF_Int32Tensor:$padding,
    TF_Int32Tensor:$lhs_dilation,
    TF_Int32Tensor:$rhs_dilation,
    TF_Int32Tensor:$feature_group_count,

    StrAttr:$dimension_numbers,
    DefaultValuedAttr<I64Attr, "1">:$batch_group_count,
    DefaultValuedAttr<I64Attr, "-1">:$lhs_quantization_axis,
    DefaultValuedAttr<I64Attr, "-128">:$lhs_quantization_min_val,
    DefaultValuedAttr<I64Attr, "127">:$lhs_quantization_max_val,
    DefaultValuedAttr<I64Attr, "-1">:$rhs_quantization_axis,
    DefaultValuedAttr<I64Attr, "-128">:$rhs_quantization_min_val,
    DefaultValuedAttr<I64Attr, "127">:$rhs_quantization_max_val,
    DefaultValuedAttr<I64Attr, "-1">:$output_quantization_axis,
    DefaultValuedAttr<I64Attr, "-128">:$output_quantization_min_val,
    DefaultValuedAttr<I64Attr, "127">:$output_quantization_max_val
  );

  let results = (outs
    TensorOf<[TF_Qint32]>:$output
  );

  TF_DerivedOperandTypeAttr LhsT = TF_DerivedOperandTypeAttr<0>;
  TF_DerivedOperandTypeAttr RhsT = TF_DerivedOperandTypeAttr<1>;
  TF_DerivedOperandTypeAttr Tindices = TF_DerivedOperandTypeAttr<8>;
  TF_DerivedResultTypeAttr Tout = TF_DerivedResultTypeAttr<0>;
}

def TF_UniformQuantizedDotOp : TF_Op<"UniformQuantizedDot", [NoSideEffect]> {
  // TODO(b/230804708): Improve the operator description.
  let summary = "Quantized dot operator.";

  let arguments = (ins
    TensorOf<[TF_Qint8]>:$lhs,
    TensorOf<[TF_Qint8]>:$rhs,
    TF_Float32Tensor:$lhs_scales,
    TF_Int32Tensor:$lhs_zps,
    TF_Float32Tensor:$rhs_scales,
    TF_Int32Tensor:$rhs_zps,
    TF_Float32Tensor:$output_scales,
    TF_Int32Tensor:$output_zps,

    DefaultValuedAttr<I64Attr, "-1">:$lhs_quantization_axis,
    DefaultValuedAttr<I64Attr, "-128">:$lhs_quantization_min_val,
    DefaultValuedAttr<I64Attr, "127">:$lhs_quantization_max_val,
    DefaultValuedAttr<I64Attr, "-1">:$rhs_quantization_axis,
    DefaultValuedAttr<I64Attr, "-128">:$rhs_quantization_min_val,
    DefaultValuedAttr<I64Attr, "127">:$rhs_quantization_max_val,
    DefaultValuedAttr<I64Attr, "-1">:$output_quantization_axis,
    DefaultValuedAttr<I64Attr, "-128">:$output_quantization_min_val,
    DefaultValuedAttr<I64Attr, "127">:$output_quantization_max_val
  );

  let results = (outs
    TensorOf<[TF_Qint32]>:$output
  );

  TF_DerivedOperandTypeAttr LhsT = TF_DerivedOperandTypeAttr<0>;
  TF_DerivedOperandTypeAttr RhsT = TF_DerivedOperandTypeAttr<1>;
  TF_DerivedResultTypeAttr Tout = TF_DerivedResultTypeAttr<0>;
}

// TODO(b/230804708): Add hybrid dot general.
def TF_UniformQuantizedDotGeneralOp : TF_Op<"UniformQuantizedDotGeneral", [NoSideEffect]> {
  // TODO(b/230804708): Improve the operator description.
  let summary = "Quantized dot general operator.";

  let arguments = (ins
    TensorOf<[TF_Float32, TF_Qint8]>:$lhs,
    TensorOf<[TF_Qint8]>:$rhs,
    TF_Float32Tensor:$lhs_scales,
    TF_Int32Tensor:$lhs_zps,
    TF_Float32Tensor:$rhs_scales,
    TF_Int32Tensor:$rhs_zps,
    TF_Float32Tensor:$output_scales,
    TF_Int32Tensor:$output_zps,

    StrAttr:$dimension_numbers,
    DefaultValuedAttr<I64Attr, "-1">:$lhs_quantization_axis,
    DefaultValuedAttr<I64Attr, "-128">:$lhs_quantization_min_val,
    DefaultValuedAttr<I64Attr, "127">:$lhs_quantization_max_val,
    DefaultValuedAttr<I64Attr, "-1">:$rhs_quantization_axis,
    DefaultValuedAttr<I64Attr, "-128">:$rhs_quantization_min_val,
    DefaultValuedAttr<I64Attr, "127">:$rhs_quantization_max_val,
    DefaultValuedAttr<I64Attr, "-1">:$output_quantization_axis,
    DefaultValuedAttr<I64Attr, "-128">:$output_quantization_min_val,
    DefaultValuedAttr<I64Attr, "127">:$output_quantization_max_val
  );

  let results = (outs
    TensorOf<[TF_Qint32]>:$output
  );

  TF_DerivedOperandTypeAttr LhsT = TF_DerivedOperandTypeAttr<0>;
  TF_DerivedOperandTypeAttr RhsT = TF_DerivedOperandTypeAttr<1>;
  TF_DerivedResultTypeAttr Tout = TF_DerivedResultTypeAttr<0>;
}

def TF_UniformQuantizedDotHybridOp : TF_Op<"UniformQuantizedDotHybrid", [NoSideEffect]> {
  // TODO(b/230804708): Improve the operator description.
  let summary = "Quantized hybrid dot operator.";

  let arguments = (ins
    TF_Float32Tensor:$lhs,
    TensorOf<[TF_Qint8]>:$rhs,
    TF_Float32Tensor:$rhs_scales,
    TF_Int32Tensor:$rhs_zps,

    DefaultValuedAttr<I64Attr, "-1">:$rhs_quantization_axis,
    DefaultValuedAttr<I64Attr, "-128">:$rhs_quantization_min_val,
    DefaultValuedAttr<I64Attr, "127">:$rhs_quantization_max_val
  );

  let results = (outs
    TF_Float32Tensor:$output
  );

  TF_DerivedOperandTypeAttr LhsT = TF_DerivedOperandTypeAttr<0>;
  TF_DerivedOperandTypeAttr RhsT = TF_DerivedOperandTypeAttr<1>;
  TF_DerivedResultTypeAttr Tout = TF_DerivedResultTypeAttr<0>;
}

