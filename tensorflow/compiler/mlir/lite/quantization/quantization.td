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

// This is the operation definition file for TensorFlow Lite.

#ifdef TFL_Quantization
#else
#define TFL_Quantization

#ifdef OP_BASE
#else
include "mlir/IR/OpBase.td"
#endif // OP_BASE

include "mlir/Dialect/QuantOps/QuantPredicates.td"


//===----------------------------------------------------------------------===//
// Min-max range pair definitions.
//===----------------------------------------------------------------------===//

// A pair of floating point values which defines the min and max of a value
// range for quantization. The attribute is allowed to be empty or
// have 2 elements.
def MinMaxAttr : Attr<Or<[CPred<"$_self.cast<ArrayAttr>().size() == 0">,
                             CPred<"$_self.cast<ArrayAttr>().size() == 2">]>,
                      "min-max range pair"> {
  let storageType = [{ ArrayAttr }];
  let returnType = [{ ArrayRef<Attribute> }];
}

//===----------------------------------------------------------------------===//
// QuantizedType definitions.
//===----------------------------------------------------------------------===//

// The base class of a quantized type.
class TFL_QuantizedType<string n, list<int> params, bit signed>
  : Type<And<[CPred<"$_self.isa<mlir::quant::QuantizedType>()">,
              CPred<"$_self.cast<mlir::quant::QuantizedType>()" #
                    ".getStorageTypeIntegralWidth() == " # !head(params)>]>,
    "Q" # !if (signed, "I", "UI") # !head(params) # " type"> {
  string name = n;
  string asTraitArgsStr =
    StrJoinInt<params>.result # !if(signed, ", true", ", false");
}

// Uniform quantized types. Two integers "smantissa" and "sexp" are used to
// express the Mantissa and Exponent components of the floating-point scale so
// the scale of the quantized type is "smantissa * 10 ^ sexp".
class TFL_UInt8UniformQuantizedType<int zero_pt, int smantissa, int sexp>
    : TFL_QuantizedType<"Uniform",
                        [8, zero_pt, smantissa, sexp, 0, 255], 0>;
class TFL_Int8UniformQuantizedType<int zero_pt, int smantissa, int sexp>
    : TFL_QuantizedType<"Uniform",
                        [8, zero_pt, smantissa, sexp, -128, 127], 1>;

// General uniform quantized types. The definitions can be used to specify
// operand's tensor types.
def TFL_QUI8 : TFL_QuantizedType<"Uniform", [8], 0>;
def TFL_QI8 : TFL_QuantizedType<"Uniform", [8], 1>;
def TFL_QUI16 : TFL_QuantizedType<"Uniform", [16], 0>;
def TFL_QI16 : TFL_QuantizedType<"Uniform", [16], 1>;
def TFL_QUI32 : TFL_QuantizedType<"Uniform", [32], 0>;
def TFL_QI32 : TFL_QuantizedType<"Uniform", [32], 1>;

//===----------------------------------------------------------------------===//
// TFL native op traits (for quantization).
//
// Ops in this link should have those traits specified:
// https://www.tensorflow.org/lite/performance/quantization_spec
//===----------------------------------------------------------------------===//

// Specify this trait if the op has a fixed output value range.
class TFL_FixedResultScale<TFL_QuantizedType qt> : NativeOpTrait<!strconcat(
  "TFL::FixedResult", qt.name, "Scale<", qt.asTraitArgsStr, ">::Impl")>;

// Specify this trait if the op requires same inputs and outputs quantization
// scales.
def TFL_SameOperandsAndResultsScale : NativeOpTrait<
  "TFL::SameOperandsAndResultsScale">;

// Specify this trait if the b-th input of the op is a bias input, which needs
// a scale based on the scales of op1 and op2.
class TFL_AccumulatorUniformScale<int bias, int op1, int op2> : NativeOpTrait<
  !strconcat("TFL::AccumulatorUniformScale<",
             StrJoinInt<[bias, op1, op2]>.result,
             ">::Impl")>;

// Specify this trait if the op doesn't have quantizable ouput. We shouldn't
// apply quantization on this op.
def TFL_NoQuantizableResult : NativeOpTrait<"TFL::NoQuantizableResult">;

#endif // TFL_Quantization
