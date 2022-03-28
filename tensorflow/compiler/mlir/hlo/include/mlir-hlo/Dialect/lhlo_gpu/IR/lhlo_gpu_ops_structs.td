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

#ifndef LHLO_GPU_OPS_STRUCTS
#define LHLO_GPU_OPS_STRUCTS

include "mlir-hlo/Dialect/lhlo_gpu/IR/lhlo_gpu_ops_base.td"

def ConvolutionBackendConfigAttr : StructAttr<"ConvolutionBackendConfig",
                                          LmhloGpuDialect, [

   // These six fields are a TableGen transliteration of AlgorithmProto.
   StructFieldAttr<"algorithm", I64Attr>,
   StructFieldAttr<"tensor_ops_enabled", BoolAttr>,
   // The next two fields are aligned arrays of knob IDs and values to
   // represent the knob_id -> knob_value map.
   StructFieldAttr<"knob_ids", I64ArrayAttr>,
   StructFieldAttr<"knob_values", I64ArrayAttr>,
   StructFieldAttr<"is_cudnn_frontend", BoolAttr>,
   StructFieldAttr<"workspace_size", I64Attr>,

   // The following 3 attributes describe the layout as an array of integers
   // that list the dimensions in minor-to-major order similar to XLA's layout
   // representation. operand_0_layout and operand_0_layout described the layout
   // of the first 2 operands of the convolution, and result_layout describes
   // the layout of the primary output operand of the convolution.
   // Note: Not using names like input_layout or filter_layout as `input` may be
   // an input operand (for ConvForward) but output for ConvBackward.
   StructFieldAttr<"operand_0_layout", I64ArrayAttr>,
   StructFieldAttr<"operand_1_layout", I64ArrayAttr>,
   StructFieldAttr<"result_layout", I64ArrayAttr>]> {
   let summary = "GPU Convolution backend configuration";
}

#endif // LHLO_GPU_OPS_STRUCTS
