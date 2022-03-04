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

#endif // LHLO_GPU_OPS_ENUMS
