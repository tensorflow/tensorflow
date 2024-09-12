/* Copyright 2019 The OpenXLA Authors.

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
include "mhlo/IR/hlo_ops.td"

//===----------------------------------------------------------------------===//
// Binary op patterns.
//===----------------------------------------------------------------------===//

// Add and subtraction are elementwise and can be distributed across the real
// and imaginary components.
foreach elementwiseOp = [MHLO_AddOp, MHLO_SubtractOp] in
  def : Pat<(elementwiseOp MHLO_ComplexTensor:$lhs,
             MHLO_ComplexTensor:$rhs),
            (MHLO_ComplexOp
              (elementwiseOp (MHLO_RealOp $lhs), (MHLO_RealOp $rhs)),
              (elementwiseOp (MHLO_ImagOp $lhs), (MHLO_ImagOp $rhs)))>;

// Complex multiplication results in a cross product multiplication between the
// real and imaginary components such that:
//   result.real = lhs.real * rhs.real - lhs.imag * rhs.imag
//   result.imag = lhs.imag * rhs.real + lhs.real * rhs.imag
def : Pat<(MHLO_MulOp MHLO_ComplexTensor:$lhs,
           MHLO_ComplexTensor:$rhs),
          (MHLO_ComplexOp
           (MHLO_SubtractOp
            (MHLO_MulOp
             (MHLO_RealOp:$lhs_real $lhs),
             (MHLO_RealOp:$rhs_real $rhs)),
            (MHLO_MulOp
             (MHLO_ImagOp:$lhs_imag $lhs),
             (MHLO_ImagOp:$rhs_imag $rhs))),
           (MHLO_AddOp
            (MHLO_MulOp $lhs_real, $rhs_imag),
            (MHLO_MulOp $lhs_imag, $rhs_real)))>;


// Division is performed by normalizing the denominator by multiplying by the
// conjugate of the rhs.
//   numerator = lhs * conj(rhs)
//   denominator = rhs * conj(rhs)
def : Pat<(MHLO_DivOp MHLO_ComplexTensor:$lhs, MHLO_ComplexTensor:$rhs),
          (MHLO_ComplexOp
            (MHLO_DivOp
             (MHLO_RealOp (MHLO_MulOp:$num $lhs,
                          (MHLO_ComplexOp:$conj
                           (MHLO_RealOp $rhs),
                           (MHLO_NegOp (MHLO_ImagOp $rhs))))),
              (MHLO_AddOp:$den
               (MHLO_MulOp (MHLO_RealOp $rhs), (MHLO_RealOp $rhs)),
               (MHLO_MulOp (MHLO_ImagOp $rhs), (MHLO_ImagOp $rhs)))),
            (MHLO_DivOp (MHLO_ImagOp $num), $den))>;

// Absolute value is evaluated as:
//   result = sqrt(val.real * val.real + val.imag * val.imag)
def : Pat<(MHLO_AbsOp MHLO_ComplexTensor:$val),
           (MHLO_SqrtOp
             (MHLO_AddOp
              (MHLO_MulOp (MHLO_RealOp:$real $val), $real),
              (MHLO_MulOp (MHLO_ImagOp:$imag $val), $imag)))>;

// Can deconstruct sin(a + ib) as follows:
//   sin(a) * cosh(b) + icos(a) * sinh(b)
//   sinh(b) = (e^x - e^-x) / 2
//   cosh(b) = (e^x + e^-x) / 2
def : Pat<(MHLO_SineOp MHLO_ComplexTensor:$val),
            (MHLO_ComplexOp
              (MHLO_DivOp
                (MHLO_MulOp
                  (MHLO_SineOp (MHLO_RealOp:$real $val)),
                  (MHLO_AddOp
                    (MHLO_ExpOp:$exp (MHLO_ImagOp:$imag $val)),
                    (MHLO_ExpOp:$nexp (MHLO_NegOp $imag)))),
                 (MHLO_ConstantOp : $two (ConstantSplat<"2.0"> $real))),
              (MHLO_DivOp
                (MHLO_MulOp
                  (MHLO_CosineOp $real),
                  (MHLO_SubtractOp $exp, $nexp)), $two))>;

// Can deconstruct cos(a + ib) as follows:
//   cos(a) * cosh(b) - isin(a) * sinh(b)
//   sinh(b) = (e^x - e^-x) / 2
//   cosh(b) = (e^x + e^-x) / 2
def : Pat<(MHLO_CosineOp MHLO_ComplexTensor:$val),
            (MHLO_ComplexOp
              (MHLO_DivOp
                (MHLO_MulOp
                  (MHLO_CosineOp (MHLO_RealOp:$real $val)),
                  (MHLO_AddOp
                    (MHLO_ExpOp:$exp (MHLO_ImagOp:$imag $val)),
                    (MHLO_ExpOp:$nexp (MHLO_NegOp $imag)))),
                 (MHLO_ConstantOp : $two (ConstantSplat<"2.0"> $real))),
              (MHLO_DivOp
                (MHLO_MulOp
                  (MHLO_SineOp $real),
                  (MHLO_SubtractOp $nexp, $exp)), $two))>;

// Exponential can be lowered to an exponential on the real component and a
// sum of sinusoids of the imaginary component, which equates to a normal
// exponential operator multiplied by Euler's formula.
//
// Exp(a + ib) = Exp(a) * Exp(ib) = Exp(a) * Cos(b) + Exp(a) * iSin(b))
class MHLO_ComparisonDirectionValue<string enumStr> :
  ConstantAttr<MHLO_ComparisonDirectionAttr, "::mlir::mhlo::ComparisonDirection::" # enumStr>;

def : Pat<(MHLO_ExpOp MHLO_ComplexTensor:$val),
          (MHLO_ComplexOp
           (MHLO_MulOp
            (MHLO_CosineOp (MHLO_ImagOp:$imag $val)),
            (MHLO_ExpOp:$exp (MHLO_RealOp:$real $val))),
           (MHLO_MulOp (MHLO_SineOp $imag), $exp))>;

foreach pair = [[MHLO_ComparisonDirectionValue<"NE">, MHLO_OrOp],
                [MHLO_ComparisonDirectionValue<"EQ">, MHLO_AndOp]] in {
def : Pat<(MHLO_CompareOp MHLO_ComplexTensor:$lhs, MHLO_ComplexTensor:$rhs, pair[0], $compare_type),
            (pair[1]
             (MHLO_CompareOp (MHLO_RealOp $lhs), (MHLO_RealOp $rhs), pair[0], $compare_type),
             (MHLO_CompareOp (MHLO_ImagOp $lhs), (MHLO_ImagOp $rhs), pair[0], $compare_type))>;
}
