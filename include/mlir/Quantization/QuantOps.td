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

#ifdef QUANTIZATION_OPS
#else

#ifdef OP_BASE
#else
include "mlir/IR/OpBase.td"
include "mlir/Quantization/QuantPredicates.td"
#endif // OP_BASE

//===----------------------------------------------------------------------===//
// Base classes
//===----------------------------------------------------------------------===//

class quant_Op<string mnemonic, list<OpTrait> traits> :
    Op<!strconcat("quant.", mnemonic), traits>;

//===----------------------------------------------------------------------===//
// Quantization barriers
//===----------------------------------------------------------------------===//
class quant_BarrierOp<string mnemonic, list<OpTrait> traits> :
    quant_Op<mnemonic, traits>, Arguments<(ins quant_RealValueType:$arg)>,
    Results<(outs quant_RealValueType)>;

// A QuantizeBarrier (qbarrier) represents a potential type shift from a
// quantizable type to a quantized type.
//
// At runtime, a qbarrier will apply the transformation expressed by its
// operand and result type. For flexibility during transformation, it is also
// possible to have a qbarrier that performs no transformation (both its
// operand and result type are quantizable).
//
// A qbarrier will typically originate from either:
//   a) An expressed or implied constraint in the source dialect which signals
//      that a certain level of quantization is possible or required.
//   b) An inference made by a quantization algorithm indicating that a
//      quantized representation may be acceptable.
//
// Especially early in transformation, it is common to have pairs of
// qbarrier/dbarrier at points where a transition to a quantized type is
// required. In addition, it is also common to have an identity qbarrier
// (where the operand and result type are not quantized) at all points where
// it is legal to use a quantized representation (but is not known to be
// acceptable).
def quant_QuantizeBarrierOp : quant_BarrierOp<"qbarrier", [NoSideEffect]>;

// A DequantizeBarrier (dbarrier) represents the inverse of a qbarrier,
// converting back from a quantized to quantizable (expressed) type.
//
// Like qbarriers, a dbarrier is allowed to have both its operand and result
// as non quantized types. This facilitates transformations and marks edges
// where the computation must be carried out in the expressed type.
//
// Especially early in transformation, it is common to have dbarriers on
// all operands to ops that must operate with the expressed type (typically
// math ops prior to lowering to target-specific, quantized kernels).
def quant_DequantizeBarrierOp : quant_BarrierOp<"dbarrier", [NoSideEffect]>;

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
def quant_StorageCastOp :
    quant_Op<"scast", [NoSideEffect]>,
    Arguments<(ins quant_RealOrStorageValueType:$arg)>,
    Results<(outs quant_RealOrStorageValueType)>;

//===----------------------------------------------------------------------===//
// Training integration ops
//===----------------------------------------------------------------------===//

def quant_ConstFakeQuant : quant_Op<"const_fake_quant",
                                    [NoSideEffect]> {
  let summary =
      "Simulates the effect of uniform quantization with const range.";

  let description = [{
Given a const min, max, num_bits and narrow_range attribute, applies the same
uniform quantization simulation as is done by the TensorFlow
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
    DefaultValuedAttr<BoolAttr, "false">:$narrow_range
  );

  let results = (outs
    F32Tensor:$outputs
  );
}

#endif // QUANT_OPS
