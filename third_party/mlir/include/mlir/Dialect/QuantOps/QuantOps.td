//===- QuantOps.td - Quantization operation definition -----*- tablegen -*-===//
//
// Copyright 2019 The MLIR Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================
//
// This is the operation definition file for Quantization.
//
//===----------------------------------------------------------------------===//

#ifndef DIALECT_QUANTOPS_QUANT_OPS_
#define DIALECT_QUANTOPS_QUANT_OPS_

include "mlir/IR/OpBase.td"
include "mlir/Dialect/QuantOps/QuantPredicates.td"

def quant_Dialect : Dialect {
  let name = "quant";
}

//===----------------------------------------------------------------------===//
// Base classes
//===----------------------------------------------------------------------===//

class quant_Op<string mnemonic, list<OpTrait> traits> :
    Op<quant_Dialect, mnemonic, traits>;

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
def quant_QuantizeCastOp : quant_Op<"qcast", [NoSideEffect]> {
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
def quant_DequantizeCastOp : quant_Op<"dcast", [NoSideEffect]> {
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
def quant_StorageCastOp : quant_Op<"scast", [NoSideEffect]> {
  let arguments = (ins quant_RealOrStorageValueType:$arg);
  let results = (outs quant_RealOrStorageValueType);
  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// Training integration and instrumentation ops
//===----------------------------------------------------------------------===//

def quant_ConstFakeQuant : quant_Op<"const_fake_quant",
                                    [SameOperandsAndResultType, NoSideEffect]> {
  let summary =
      "Simulates the effect of uniform quantization with const range.";

  let description = [{
    Given a const min, max, num_bits and narrow_range attribute, applies the
    same uniform quantization simulation as is done by the TensorFlow
    fake_quant_with_min_max_args op. See the fakeQuantAttrsToType() utility
    method and the quant-convert-simulated-quantization pass for futher details.
  }];

  let arguments = (ins
    F32Tensor:$inputs,
    F32Attr:$min,
    F32Attr:$max,
    // The bitwidth of the quantization; between 2 and 16, inclusive.
    I64Attr:$num_bits,
    // Quantization range starts from 0 or 1; starts from 1 if true.
    DefaultValuedAttr<BoolAttr, "false">:$narrow_range,
    // The sign of the quantization.
    DefaultValuedAttr<BoolAttr, "false">:$is_signed
  );

  let results = (outs
    F32Tensor:$outputs
  );
}

def quant_ConstFakeQuantPerAxis : quant_Op<"const_fake_quant_per_axis",
                                    [SameOperandsAndResultType, NoSideEffect]> {
  let summary =
      "Simulates the effect of per axis uniform quantization with const range.";

  let description = [{
    Given a const min, max, num_bits and narrow_range attribute, applies the
    same per axis uniform quantization simulation as is done by the TensorFlow
    fake_quant_with_min_max_vars_per_channel op. See the fakeQuantAttrsToType()
    utility method and the quant-convert-simulated-quantization pass for futher
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
    DefaultValuedAttr<BoolAttr, "false">:$narrow_range,
    // The sign of the quantization.
    DefaultValuedAttr<BoolAttr, "false">:$is_signed
  );

  let results = (outs
    F32Tensor:$outputs
  );
}

def quant_StatisticsRefOp : quant_Op<"stats_ref", [SameOperandsAndResultType]> {
  let summary =
      "Indicates that statistics are resolved by reference.";

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

def quant_StatisticsOp : quant_Op<"stats", [SameOperandsAndResultType]> {
  let summary =
      "Identity op which associates statistics with the value.";

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
      <?x?x3x2>, axis=3 => N=2
      <?x?x3x2>, axis=2 => N=6
  }];

  let arguments = (ins
    quant_RealValueType:$arg,
    ElementsAttr:$layerStats,
    OptionalAttr<ElementsAttr>:$axisStats,
    OptionalAttr<I64Attr>:$axis);
  let results = (outs quant_RealValueType);

  let verifier = [{
    auto tensorArg = arg()->getType().dyn_cast<TensorType>();
    if (!tensorArg) return emitOpError("arg needs to be tensor type.");

    // Verify layerStats attribute.
    {
      auto layerStatsType = layerStats().getType();
      if (!layerStatsType.getElementType().isa<FloatType>()) {
        return emitOpError(
            "layerStats must have a floating point element type");
      }
      if (layerStatsType.getRank() != 1 || layerStatsType.getDimSize(0) != 2) {
        return emitOpError("layerStats must have shape [2]");
      }
    }
    // Verify axisStats (optional) attribute.
    if (axisStats()) {
      if (!axis()) return emitOpError("axis must be specified for axisStats");

      auto shape = tensorArg.getShape();
      auto argSliceSize = std::accumulate(std::next(shape.begin(),
        axis()->getSExtValue()), shape.end(), 1, std::multiplies<int64_t>());

      auto axisStatsType = axisStats()->getType();
      if (!axisStatsType.getElementType().isa<FloatType>()) {
        return emitOpError("axisStats must have a floating point element type");
      }
      if (axisStatsType.getRank() != 2 ||
          axisStatsType.getDimSize(1) != 2 ||
          axisStatsType.getDimSize(0) != argSliceSize) {
        return emitOpError("axisStats must have shape [N,2] "
                           "where N = the slice size defined by the axis dim");
      }
    }
    return success();
  }];
}

def quant_CoupledRefOp : quant_Op<"coupled_ref", [SameOperandsAndResultType]> {
  let summary =
      "Indicates that one point of the computation is coupled to another.";

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

#endif // DIALECT_QUANTOPS_QUANT_OPS_
