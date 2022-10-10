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
//
//===----------------------------------------------------------------------===//
//
// This is the operation definition file for Quantization.
//
//===----------------------------------------------------------------------===//

#ifndef QUANT_FORK_OPS
#define QUANT_FORK_OPS

include "tensorflow/compiler/mlir/lite/quantization/ir/QuantOpsBase.td"
include "mlir/Dialect/Quant/QuantOpsBase.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// Base classes
//===----------------------------------------------------------------------===//

class quantfork_Op<string mnemonic, list<Trait> traits> :
    Op<QuantizationFork_Dialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// Quantization casts
//===----------------------------------------------------------------------===//
// A QuantizeCast (qcast) represents a potential type shift from a quantizable
// type to a quantized type.
//
// At runtime, a qcast will apply the transformation expressed by its
// operand and result type. For flexibility during transformation, it is also
// possible to have a qcast that performs no transformation (both its
// operand and result type are quantizable).
//
// A qcast will typically originate from either:
//   a) An expressed or implied constraint in the source dialect which signals
//      that a certain level of quantization is possible or required.
//   b) An inference made by a quantization algorithm indicating that a
//      quantized representation may be acceptable.
//
// Especially early in transformation, it is common to have pairs of
// qcast/dcast at points where a transition to a quantized type is
// required. In addition, it is also common to have an identity qcast
// (where the operand and result type are not quantized) at all points where
// it is legal to use a quantized representation (but is not known to be
// acceptable).
def quantfork_QuantizeCastOp : quantfork_Op<"qcast", [NoSideEffect]> {
  let arguments = (ins quant_RealValueType:$arg);
  let results = (outs quant_RealValueType);
}

// A DequantizeCast op (dcast) represents the inverse of a qcast,
// converting back from a quantized to quantizable (expressed) type.
//
// Like qcasts, a dcast is allowed to have both its operand and result
// as non quantized types. This facilitates transformations and marks edges
// where the computation must be carried out in the expressed type.
//
// Especially early in transformation, it is common to have dcasts on
// all operands to ops that must operate with the expressed type (typically
// math ops prior to lowering to target-specific, quantized kernels).
def quantfork_DequantizeCastOp : quantfork_Op<"dcast", [NoSideEffect]> {
  let arguments = (ins quant_RealValueType:$arg);
  let results = (outs quant_RealValueType);
}

// A StorageCast (scast) represents a cast from or to a type based on the
// storage type and a type based on a corresponding quantized type.
//
// This op exists to ensure type coherency for between parts of the computation
// which are operating directly on an underlying storage type and those which
// operate on quantized values.
//
// Examples from storage to quantized type:
//   i8 -> !quant<"uniform[i8:f32]{1.0}">
//   tensor<4xi8> -> tensor<4x!quant<"uniform[i8:f32]{1.0}">>
//   vector<4xi8> -> vector<4x!quant<"uniform[i8:f32]{1.0}">>
def quantfork_StorageCastOp : quantfork_Op<"scast", [NoSideEffect]> {
  let arguments = (ins quant_RealOrStorageValueType:$arg);
  let results = (outs quant_RealOrStorageValueType);
  let hasFolder = 1;
}

// A QuantizeRegion (region) represents a quantization unit which wraps
// high-precision ops with quantization specifications for all the inputs
// and outputs. Some quantization specifications can be undetermined and
// derived from other ports by the target specification of the kernel.
def quantfork_QuantizeRegionOp : quantfork_Op<"region", [
    NoSideEffect,
    IsolatedFromAbove,
    SingleBlockImplicitTerminator<"ReturnOp">]> {
  let summary = [{
    The `region` operation wraps high-precision ops as a logical low-precision
    quantized kernel.
  }];

  let arguments = (ins Variadic<AnyType>:$inputs,
                    TypeArrayAttr:$input_specs,
                    TypeArrayAttr:$output_specs,
                    StrAttr:$logical_kernel);
  let results = (outs Variadic<AnyType>:$outputs);
  let regions = (region SizedRegion<1>:$body);
  let hasVerifier = 1;
}

def quantfork_ReturnOp : quantfork_Op<"return", [Terminator]> {
  let summary = [{
    The `return` operation terminates a quantize region and returns values.
  }];

  let arguments = (ins Variadic<AnyTensor>:$results);
}

//===----------------------------------------------------------------------===//
// Training integration and instrumentation ops
//===----------------------------------------------------------------------===//

def quantfork_ConstFakeQuant : quantfork_Op<"const_fake_quant",
                                    [SameOperandsAndResultType, NoSideEffect]> {
  let summary = [{
    Simulates the effect of uniform quantization with const range.
  }];

  let description = [{
    Given a const min, max, num_bits and narrow_range attribute, applies the
    same uniform quantization simulation as is done by the TensorFlow
    fake_quant_with_min_max_args op. See the fakeQuantAttrsToType() utility
    method and the quant-convert-simulated-quantization pass for further details.
  }];

  let arguments = (ins
    F32Tensor:$inputs,
    F32Attr:$min,
    F32Attr:$max,
    // The bitwidth of the quantization; between 2 and 16, inclusive.
    I64Attr:$num_bits,
    // Quantization range starts from 0 or 1; starts from 1 if true.
    DefaultValuedOptionalAttr<BoolAttr, "false">:$narrow_range,
    // The sign of the quantization.
    DefaultValuedOptionalAttr<BoolAttr, "false">:$is_signed
  );

  let results = (outs
    F32Tensor:$outputs
  );
}

def quantfork_ConstFakeQuantPerAxis : quantfork_Op<"const_fake_quant_per_axis",
                                    [SameOperandsAndResultType, NoSideEffect]> {
  let summary = [{
    Simulates the effect of per axis uniform quantization with const range.
  }];

  let description = [{
    Given a const min, max, num_bits and narrow_range attribute, applies the
    same per axis uniform quantization simulation as is done by the TensorFlow
    fake_quant_with_min_max_vars_per_channel op. See the fakeQuantAttrsToType()
    utility method and the quant-convert-simulated-quantization pass for further
    details.
  }];

  let arguments = (ins
    F32Tensor:$inputs,
    F32ArrayAttr:$min,
    F32ArrayAttr:$max,
    // The quantized dimension of the inputs tensor.
    I64Attr:$axis,
    // The bitwidth of the quantization; between 2 and 16, inclusive.
    I64Attr:$num_bits,
    // Quantization range starts from 0 or 1; starts from 1 if true.
    DefaultValuedOptionalAttr<BoolAttr, "false">:$narrow_range,
    // The sign of the quantization.
    DefaultValuedOptionalAttr<BoolAttr, "false">:$is_signed
  );

  let results = (outs
    F32Tensor:$outputs
  );
}

def quantfork_StatisticsRefOp : quantfork_Op<"stats_ref", [SameOperandsAndResultType]> {
  let summary = "Indicates that statistics are resolved by reference.";

  let description = [{
    This op acts as an identity that, when encountered at runtime, should result
    in statistics being collected about about the value of its operand/result.
    Such statistics will be stored with the provided key, allowing this node
    to later be converted to a 'stats' op if statistics with that key have been
    encountered.
  }];

  let arguments = (ins
    quant_RealValueType:$arg,
    StrAttr:$statsKey
  );
  let results = (outs quant_RealValueType);
}

def quantfork_StatisticsOp : quantfork_Op<"stats", [SameOperandsAndResultType]> {
  let summary = "Identity op which associates statistics with the value.";

  let description = [{
    Associates statistics about the runtime ranges of values observed for
    evaluations of this node.

    Statistics about the entire type are reported in the 'layerStats' attribute
    and those for each axis, in the (optional) `axisStats` attribute. The
    interpretation of each is determined by the last dimension of its shape.
    Currently, only dim=2 is supported, which is interpreted as [min, max].

    `layerStats` must be a rank 1 tensor: [2]
    `axisStats` must be a rank 2 tensor: [N, 2], where N=the slice size
      splitted by the `axis` dimension. For example:

    ```
    <?x?x3x2>, axis=3 => N=2
    <?x?x3x2>, axis=2 => N=6
    ```
  }];

  let arguments = (ins
    quant_RealValueType:$arg,
    ElementsAttr:$layerStats,
    OptionalAttr<ElementsAttr>:$axisStats,
    OptionalAttr<I64Attr>:$axis);
  let results = (outs quant_RealValueType);
  let hasVerifier = 1;
}

def quantfork_CoupledRefOp : quantfork_Op<"coupled_ref", [SameOperandsAndResultType]> {
  let summary = [{
    Indicates that one point of the computation is coupled to another.
  }];

  let description = [{
    Ordinarily, relationships between ops for the purposes of determining
    compatible quantized types is explicit based on the use-def chain. However,
    in some situations, a use may be separated from its def by arbitrary
    external connections. In such a case, during analysis, all coupled_ref
    nodes in a module which share a coupledKey will be considered to be
    directly connected as via an identity op for the purpose of type inference.
  }];

  let arguments = (ins
    quant_RealValueType:$arg,
    StrAttr:$coupledKey);
  let results = (outs quant_RealValueType);
}

#endif // QUANT_FORK_OPS
