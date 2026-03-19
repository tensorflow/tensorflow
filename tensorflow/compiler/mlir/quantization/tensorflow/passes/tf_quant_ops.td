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

// TODO(b/230804708): Add hybrid dot general.
def TF_UniformQuantizedDotGeneralOp : TF_Op<"UniformQuantizedDotGeneral", [Pure]> {
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
    DefaultValuedOptionalAttr<I64Attr, "-1">:$lhs_quantization_axis,
    DefaultValuedOptionalAttr<I64Attr, "-128">:$lhs_quantization_min_val,
    DefaultValuedOptionalAttr<I64Attr, "127">:$lhs_quantization_max_val,
    DefaultValuedOptionalAttr<I64Attr, "-1">:$rhs_quantization_axis,
    DefaultValuedOptionalAttr<I64Attr, "-128">:$rhs_quantization_min_val,
    DefaultValuedOptionalAttr<I64Attr, "127">:$rhs_quantization_max_val,
    DefaultValuedOptionalAttr<I64Attr, "-1">:$output_quantization_axis,
    DefaultValuedOptionalAttr<I64Attr, "-128">:$output_quantization_min_val,
    DefaultValuedOptionalAttr<I64Attr, "127">:$output_quantization_max_val
  );

  let results = (outs
    TensorOf<[TF_Qint32]>:$output
  );

  TF_DerivedOperandTypeAttr LhsT = TF_DerivedOperandTypeAttr<0>;
  TF_DerivedOperandTypeAttr RhsT = TF_DerivedOperandTypeAttr<1>;
  TF_DerivedResultTypeAttr Tout = TF_DerivedResultTypeAttr<0>;
}

def TF_CustomAggregatorOp : TF_Op<"CustomAggregator", [Pure]> {
  let summary = "Gathers min and max statistics of a given tensor.";

  let arguments = (ins
    TensorOf<[TF_Float32]>:$input,

    // The unique id of this `CustomAggregator` op.
    StrAttr:$id,
    // The integer value of the enforcing `CalibrationMethod`.
    I32Attr:$calibration_method,
    // The number of histogram bins.
    I32Attr:$num_bins,
    // Min percentile to be included in the selected range, only used in the
    // `HISTOGRAM_PERCENTILE` method.
    F32Attr:$min_percentile,
    // Max percentile to be included in the selected range, only used in the
    // `HISTOGRAM_PERCENTILE` method.
    F32Attr:$max_percentile
  );

  let results = (outs
    TensorOf<[TF_Float32]>:$output,
    TensorOf<[TF_Float32]>:$min,
    TensorOf<[TF_Float32]>:$max,
    TensorOf<[TF_Int64]>:$histogram
  );
}

def TF_CalibrationStatisticsSaverOp : TF_Op<"CalibrationStatisticsSaver", []> {
  let summary = "Aggregates and saves calibration statistics.";

  let arguments = (ins
    Variadic<TensorOf<[TF_Float32, TF_Int64]>>:$inputs,

    StrAttr:$output_file_path,
    StrArrayAttr:$ids,
    I32ArrayAttr:$calibration_methods
  );

  TF_DerivedOperandTypeListAttr Tin = TF_DerivedOperandTypeListAttr<0>;
}

def TF_DumpTensorOp : TF_Op<"DumpTensor", []> {
  let summary = "Dump tensor proto.";

  let arguments = (ins
    Arg<TF_Tensor>:$input,

    StrAttr:$log_dir_path,
    StrAttr:$file_name,
    BoolAttr:$enabled,
    StrAttr:$func_name,
    StrAttr:$node_name
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
}