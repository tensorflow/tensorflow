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

// This is the legalization pattern that converts complex operations into
// equivalent real value operations.

include "mlir/IR/OpBase.td"
include "mlir/Dialect/Func/IR/FuncOps.td"
include "mlir-hlo/Dialect/mhlo/IR/hlo_ops.td"

//===----------------------------------------------------------------------===//
// Binary op patterns.
//===----------------------------------------------------------------------===//

// Add and subtraction are elementwise and can be distributed across the real
// and imaginary components.
foreach elementwiseOp = [HLO_AddOp, HLO_SubtractOp] in
  def : Pat<(elementwiseOp HLO_ComplexTensor:$lhs,
             HLO_ComplexTensor:$rhs),
            (HLO_ComplexOp
              (elementwiseOp (HLO_RealOp $lhs), (HLO_RealOp $rhs)),
              (elementwiseOp (HLO_ImagOp $lhs), (HLO_ImagOp $rhs)))>;

// Complex multiplication results in a cross product multiplication between the
// real and imaginary components such that:
//   result.real = lhs.real * rhs.real - lhs.imag * rhs.imag
//   result.imag = lhs.imag * rhs.real + lhs.real * rhs.imag
def : Pat<(HLO_MulOp HLO_ComplexTensor:$lhs,
           HLO_ComplexTensor:$rhs),
          (HLO_ComplexOp
           (HLO_SubtractOp
            (HLO_MulOp
             (HLO_RealOp:$lhs_real $lhs),
             (HLO_RealOp:$rhs_real $rhs)),
            (HLO_MulOp
             (HLO_ImagOp:$lhs_imag $lhs),
             (HLO_ImagOp:$rhs_imag $rhs))),
           (HLO_AddOp
            (HLO_MulOp $lhs_real, $rhs_imag),
            (HLO_MulOp $lhs_imag, $rhs_real)))>;


// Division is performed by normalizing the denominator by multiplying by the
// conjugate of the rhs.
//   numerator = lhs * conj(rhs)
//   denominator = rhs * conj(rhs)
def : Pat<(HLO_DivOp HLO_ComplexTensor:$lhs, HLO_ComplexTensor:$rhs),
          (HLO_ComplexOp
            (HLO_DivOp
             (HLO_RealOp (HLO_MulOp:$num $lhs,
                          (HLO_ComplexOp:$conj
                           (HLO_RealOp $rhs),
                           (HLO_NegOp (HLO_ImagOp $rhs))))),
              (HLO_AddOp:$den
               (HLO_MulOp (HLO_RealOp $rhs), (HLO_RealOp $rhs)),
               (HLO_MulOp (HLO_ImagOp $rhs), (HLO_ImagOp $rhs)))),
            (HLO_DivOp (HLO_ImagOp $num), $den))>;

// Absolute value is evaluated as:
//   result = sqrt(val.real * val.real + val.imag * val.imag)
def : Pat<(HLO_AbsOp HLO_ComplexTensor:$val),
           (HLO_SqrtOp
             (HLO_AddOp
              (HLO_MulOp (HLO_RealOp:$real $val), $real),
              (HLO_MulOp (HLO_ImagOp:$imag $val), $imag)))>;

// Can deconstruct sin(a + ib) as follows:
//   sin(a) * cosh(b) + icos(a) * sinh(b)
//   sinh(b) = (e^x - e^-x) / 2
//   cosh(b) = (e^x + e^-x) / 2
def : Pat<(HLO_SineOp HLO_ComplexTensor:$val),
            (HLO_ComplexOp
              (HLO_DivOp
                (HLO_MulOp
                  (HLO_SineOp (HLO_RealOp:$real $val)),
                  (HLO_AddOp
                    (HLO_ExpOp:$exp (HLO_ImagOp:$imag $val)),
                    (HLO_ExpOp:$nexp (HLO_NegOp $imag)))),
                 (HLO_ConstantOp : $two (ConstantSplat<"2.0"> $real))),
              (HLO_DivOp
                (HLO_MulOp
                  (HLO_CosineOp $real),
                  (HLO_SubtractOp $exp, $nexp)), $two))>;

// Can deconstruct cos(a + ib) as follows:
//   cos(a) * cosh(b) - isin(a) * sinh(b)
//   sinh(b) = (e^x - e^-x) / 2
//   cosh(b) = (e^x + e^-x) / 2
def : Pat<(HLO_CosineOp HLO_ComplexTensor:$val),
            (HLO_ComplexOp
              (HLO_DivOp
                (HLO_MulOp
                  (HLO_CosineOp (HLO_RealOp:$real $val)),
                  (HLO_AddOp
                    (HLO_ExpOp:$exp (HLO_ImagOp:$imag $val)),
                    (HLO_ExpOp:$nexp (HLO_NegOp $imag)))),
                 (HLO_ConstantOp : $two (ConstantSplat<"2.0"> $real))),
              (HLO_DivOp
                (HLO_MulOp
                  (HLO_SineOp $real),
                  (HLO_SubtractOp $nexp, $exp)), $two))>;

// Exponential can be lowered to an exponential on the real component and a
// sum of sinusoids of the imaginary component, which equates to a normal
// exponential operator multiplied by Euler's formula.
//
// Exp(a + ib) = Exp(a) * Exp(ib) = Exp(a) * Cos(b) + Exp(a) * iSin(b))
class HLO_ComparisonDirectionValue<string enumStr> :
  ConstantAttr<HLO_ComparisonDirectionAttr, "::mlir::mhlo::ComparisonDirection::" # enumStr>;

def : Pat<(HLO_ExpOp HLO_ComplexTensor:$val),
          (HLO_ComplexOp
           (HLO_MulOp
            (HLO_CosineOp (HLO_ImagOp:$imag $val)),
            (HLO_ExpOp:$exp (HLO_RealOp:$real $val))),
           (HLO_MulOp (HLO_SineOp $imag), $exp))>;

foreach pair = [[HLO_ComparisonDirectionValue<"NE">, HLO_OrOp],
                [HLO_ComparisonDirectionValue<"EQ">, HLO_AndOp]] in {
def : Pat<(HLO_CompareOp HLO_ComplexTensor:$lhs, HLO_ComplexTensor:$rhs, pair[0], $compare_type),
            (pair[1]
             (HLO_CompareOp (HLO_RealOp $lhs), (HLO_RealOp $rhs), pair[0], $compare_type),
             (HLO_CompareOp (HLO_ImagOp $lhs), (HLO_ImagOp $rhs), pair[0], $compare_type))>;
}
