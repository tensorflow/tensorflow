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

#ifndef LHLO_GPU_OPS_ENUMS
#define LHLO_GPU_OPS_ENUMS

include "mlir/IR/OpBase.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/AttrTypeBase.td"

include "mlir-hlo/Dialect/lhlo_gpu/IR/lhlo_gpu_ops_base.td"

def ActivationModeNone : I32EnumAttrCase<"None", 0>;
def ActivationModeSigmoid : I32EnumAttrCase<"Sigmoid", 1>;
def ActivationModeTanh : I32EnumAttrCase<"Tanh", 2>;
def ActivationModeRelu : I32EnumAttrCase<"Relu", 3>;
def ActivationModeRelu6 : I32EnumAttrCase<"Relu6", 4>;
def ActivationModeReluX : I32EnumAttrCase<"ReluX", 5>;
def ActivationModeBandPass : I32EnumAttrCase<"BandPass", 6>;

def Activation: I32EnumAttr<"Activation",
    "Activation applied with fused convolution",
    [ActivationModeNone,  ActivationModeSigmoid, ActivationModeTanh,
     ActivationModeRelu, ActivationModeRelu6, ActivationModeReluX,
     ActivationModeBandPass]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::lmhlo_gpu";
}

def ActivationAttr : EnumAttr<LmhloGpuDialect, Activation, "activation">;

def BoolParameter : AttrOrTypeParameter<"bool", ""> {
  let parser = "::mlir::lmhlo_gpu::parseBool($_parser)";
}

def I64ArrayParameter :
    AttrOrTypeParameter<"::llvm::ArrayRef<int64_t>", ""> {
  let allocator = [{$_dst = $_allocator.copyInto($_self);}];
  let cppStorageType = "::llvm::SmallVector<int64_t>";
  let parser = "::mlir::lmhlo_gpu::parseI64Array($_parser)";
  let printer = "$_printer << '[' << $_self << ']'";
}

def ConvolutionBackendConfigAttr : AttrDef<
  LmhloGpuDialect, "ConvolutionBackendConfig"> {
  let mnemonic = "convolution_backend_config";
  let parameters = (ins
   // These six fields are a TableGen transliteration of AlgorithmProto.
   "int64_t":$algorithm,
   BoolParameter:$tensor_ops_enabled,
   // The next two fields are aligned arrays of knob IDs and values to
   // represent the knob_id -> knob_value map.
   I64ArrayParameter:$knob_ids,
   I64ArrayParameter:$knob_values,
   BoolParameter:$is_cudnn_frontend,
   "int64_t":$workspace_size,

   // The following 3 attributes describe the layout as an array of integers
   // that list the dimensions in minor-to-major order similar to XLA's layout
   // representation. operand_0_layout and operand_0_layout described the layout
   // of the first 2 operands of the convolution, and result_layout describes
   // the layout of the primary output operand of the convolution.
   // Note: Not using names like input_layout or filter_layout as `input` may be
   // an input operand (for ConvForward) but output for ConvBackward.
   I64ArrayParameter:$operand_0_layout,
   I64ArrayParameter:$operand_1_layout,
   I64ArrayParameter:$result_layout
 );
  let assemblyFormat = "`<` struct(params) `>`";
 let summary = "GPU Convolution backend configuration";
}

def CublasLtMatmulEpilogueDefault : I32EnumAttrCase<"Default", 0>;
def CublasLtMatmulEpilogueBias : I32EnumAttrCase<"Bias", 1>;

def CublasLtMatmulEpilogue: I32EnumAttr<"CublasLtMatmulEpilogue",
    "Epilogue for cublasLt matmul",
    [CublasLtMatmulEpilogueDefault, CublasLtMatmulEpilogueBias]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::lmhlo_gpu";
}

def CublasLtMatmulEpilogueAttr : EnumAttr<LmhloGpuDialect, CublasLtMatmulEpilogue, "epilogue">;

#endif // LHLO_GPU_OPS_ENUMS
