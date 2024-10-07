/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

// This is the quantization definition file for TensorFlow.

#ifdef TF_Quantization
#else
#define TF_Quantization

include "mlir/IR/OpBase.td"
include "mlir/Dialect/Quant/IR/QuantBase.td"

//===----------------------------------------------------------------------===//
// QuantizedType definitions.
//===----------------------------------------------------------------------===//

// The base class of a quantized type. Signed quantized types may be expressed
// as signless integers (i.e. up to op interpretation), but we include an
// explicit signedness check to differentiate the signed/unsigned constraints
// predicates from one another at the TD level.
class QuantizedType<string n, list<int> params, bit signed>
  : Type<And<[CPred<"$_self.isa<mlir::quant::QuantizedType>()">,
              CPred<"$_self.cast<mlir::quant::QuantizedType>()" #
                    ".getStorageTypeIntegralWidth() == " # !head(params)>,
              Or<[CPred<"$_self.cast<mlir::quant::QuantizedType>()" #
                    ".getStorageType().isSignlessInteger()">,
                  CPred<"$_self.cast<mlir::quant::QuantizedType>()" #
                    ".getStorageType().isSignedInteger() == " # signed>]>]>,
    "Q" # !if (signed, "I", "UI") # !head(params) # " type"> {
  string name = n;
  string asTraitArgsStr =
    !interleave(params, ", ") # !if(signed, ", true", ", false");
}

// Uniform quantized types. Two integers "smantissa" and "sexp" are used to
// express the Mantissa and Exponent components of the floating-point scale so
// the scale of the quantized type is "smantissa * 10 ^ sexp".
class UInt8UniformQuantizedType<int zero_pt, int smantissa, int sexp>
    : QuantizedType<"Uniform",
                        [8, zero_pt, smantissa, sexp, 0, 255], 0>;
class Int8UniformQuantizedType<int zero_pt, int smantissa, int sexp>
    : QuantizedType<"Uniform",
                        [8, zero_pt, smantissa, sexp, -128, 127], 1>;

// General uniform quantized types. The definitions can be used to specify
// operand's tensor types.
def QI4 : QuantizedType<"Uniform", [4], 1>;
def QUI8 : QuantizedType<"Uniform", [8], 0>;
def QI8 : QuantizedType<"Uniform", [8], 1>;
def QUI16 : QuantizedType<"Uniform", [16], 0>;
def QI16 : QuantizedType<"Uniform", [16], 1>;
def QUI32 : QuantizedType<"Uniform", [32], 0>;
def QI32 : QuantizedType<"Uniform", [32], 1>;

//===----------------------------------------------------------------------===//
// TFL native op traits (for quantization).
//
// Ops in this link should have those traits specified:
// https://www.tensorflow.org/lite/performance/quantization_spec
//===----------------------------------------------------------------------===//

def FixedOutputRangeInterface : OpInterface<
  "FixedOutputRangeInterface"> {
  let description = [{
    Interface for defining the fixed output range.
  }];

  let methods = [
    InterfaceMethod<
      [{Returns the fixed output range.}],
      "UniformQuantizedType", "GetFixedOutputRange",
      (ins "bool":$sign, "int":$bit_width)
    >,
  ];
}

def AffineQuantizedOpInterface : OpInterface<
  "AffineQuantizedOpInterface"> {
  let description = [{
    Interface for affine quantized ops (conv2d, fully_connected, etc.)
  }];

  let methods = [
    InterfaceMethod<
      [{Returns the affine operand index.}],
      "int", "GetAffineOperandIndex",
      (ins), [{}], [{return 1;}]>,
    InterfaceMethod<
      [{Returns whether narrow range is required for the affine operand.}],
      "bool", "RequiredNarrowRangeAffineOperand",
      (ins), [{}], [{return true;}]>,
    InterfaceMethod<
      [{Returns quantization dim for the affine operand.}],
      "int", "GetQuantizationDimIndex",
      (ins)>,
    InterfaceMethod<
      [{Returns the dimension index of the output channels.}],
      "int", "GetChannelDimIndex", (ins)
    >,
  ];
}

def SameOperandsAndResultsScale : OpInterface<"SameScalesOpInterface"> {
  let description = [{
    Interface for ops potentially have same operands and results scales.
  }];

  let methods = [
    InterfaceMethod<
      [{Returns whether same operands and results scales are required.}],
      "bool", "RequiredSameOperandsAndResultsScale",
      (ins "bool":$sign, "int":$bit_width), [{}], [{return true;}]
    >,
    InterfaceMethod<
      [{Returns whether operands and results must have the same quantized axis.}],
      "bool", "RequiredSameQuantizedAxes",
      (ins), [{}], [{return true;}]
    >,
  ];

  let verify = [{
    return quant::VerifySameScales($_op);
  }];
}

def DynamicRangeQuantizedOpInterface : OpInterface<
  "DynamicRangeQuantizedOpInterface"> {
  let description = [{
    Interface for ops dynamic range quantization is supported.

    If the op has the kernel support for dynamic range quantization, Q/DQ op
    pairs connected to the op are rewritten by its quantized alternatives where
    a new op uses Q ops for its operands instead of DQ op. Otherwise, it is
    left as is for weight-only which means the weight is dequantized at runtime.

    For example, if the kernel does not support dynamic range quantization the
    graph will be converted into the following IR:

    %q_w = "tfl.pseudo_qconst"() {
         qtype = tensor<64x3x3x3x!quant.uniform<i8<-127:127>:f32, 1.000000e+00>>
    %w = "tfl.dequantize"(%q_w) :
         (tensor<64x3x3x3x!quant.uniform<i8<-127:127>:f32, 1.000000e+00>>) ->
         tensor<64x3x3x3xf32>
    %conv = "tfl.conv_2d"(%input_act, %w, %bias)

    but if it is supported, it will be rewritten as:

    %q_w = "tfl.pseudo_qconst"() {
         qtype = tensor<64x3x3x3x!quant.uniform<i8<-127:127>:f32, 1.000000e+00>>
    %conv = "tfl.conv_2d"(%input_act, %q_w, %bias)

    Note that this is part of reaching feature parity with the old quantizer for
    dynamic range quantization except:
    - Only use_updated_hybrid_scheme=True is supported which means the ops with
    the asymmetrically quantizing input support is enabled to use this feature
    during MLIR graph rewriting passes while it is configurable in the old
    quantizer. So when those ops are matched during graph rewriting passes,
    MLIR quantizer will always ignore the pre-set value of the attribute, if
    there's any, and set it to True. The reason behind this decision is that
    generally activations of these ops show better accuracy with asymmetric
    input quantization so we want to deprecate symmetric activation quantization
    for those ops eventually.
    - Unlike to the old quantizer, per-channel quantization is supported for
    weight-only TransposeConvOp.
  }];

  let methods = [
    InterfaceMethod<
      [{Returns the quantizable operand indices of the op.}],
      "std::vector<int>", "GetQuantizableOperandIndices",
      (ins), [{}], [{return {};}]>,
    InterfaceMethod<
      [{Returns whether the op has the kernel support for dynamic range
      quantization.}],
      "bool", "GetDynamicRangeQuantKernelSupport",
      (ins), [{}], [{return false;}]>,
    InterfaceMethod<
      [{Returns whether the op requires asymmetric quantize input attribute
      setting.}],
      "bool", "RequireAsymmetricQuantizeInputsAttr",
      (ins), [{}], [{return false;}]>,
  ];
}

// Specify this trait if the op has a fixed output value range.
class FixedResultScale<QuantizedType qt> : NativeOpTrait<!strconcat(
  "quant::FixedResult", qt.name, "Scale<", qt.asTraitArgsStr, ">::Impl")>;

// Specify this trait if the bias-th input of the op is a bias input, which
// needs a scale based on the scales of op1 and op2.
class AccumulatorUniformScale<int bias, int op1, int op2> : NativeOpTrait<
  !strconcat("quant::AccumulatorUniformScale<",
             !interleave([bias, op1, op2], ", "),
             ">::Impl")>;

// Specify the operand index of the coefficient operand for an affine op
// and also the quantization dimension if per-axis quantization is support.
// If the quantization dimension is -1, per-axis quantization isn't supported.
class AffineOpCoefficient<int dim, int index> : NativeOpTrait<
  !strconcat("quant::AffineOpCoefficient<",
             !interleave([dim, index], ", "),
             ">::Impl")>;

// Specify this trait if the op does have quantizable output. Quantizers will
// apply quantization on this op.
def QuantizableResult : NativeOpTrait<"quant::QuantizableResult">;
#endif // TF_Quantization
