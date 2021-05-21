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

// This is the legalization pattern definition file for CHLO to MHLO.
// These are included in the PopulateDecomposeChloPatterns factory
// and should only include canonical expansions which are not actually
// ambiguous/different for various backends. Avoid patterns that are actually
// lowering to non-canonical forms.

include "mlir/IR/OpBase.td"
include "mlir-hlo/Dialect/mhlo/IR/hlo_ops.td"
include "mlir-hlo/Dialect/mhlo/IR/chlo_ops.td"

//===----------------------------------------------------------------------===//
// Unary op patterns.
//===----------------------------------------------------------------------===//

// Expand acos to MHLO dialect as follows:
//   acos(x) = 2 * atan2(sqrt(1 - x^2), (1 + x))  if x != -1
//           = pi                                 if x == -1
//
// TODO(hinsu): Support operands with complex element types separately using
// the following formula.
//   acos(x) = -(i * log(x + i * sqrt((1 + x) * (1 - x))))
def : Pat<(HLOClient_AcosOp NonComplexElementType:$input),
  (HLO_SelectOp
    (HLO_CompareOp
      $input,
      (HLO_ConstantLike<"-1"> $input),
      HLO_COMPARISON_DIRECTION_NE,
      (HLO_DEFAULT_COMPARISON_TYPE)
    ),
    (HLO_MulOp
      (HLO_ConstantLike<"2"> $input),
      (HLO_Atan2Op
        (HLO_SqrtOp
          (HLO_SubOp
            (HLO_ConstantLike<"1"> $input),
            (HLO_MulOp $input, $input)
          )
        ),
        (HLO_AddOp
          (HLO_ConstantLike<"1"> $input),
          $input
        )
      )
    ),
    (HLO_ConstantLike<"M_PI"> $input)
  )>;

// Expand acosh to MHLO dialect as follows:
//   acosh(x) = log(x + sqrt(x^2 - 1))      if x >= -1
//            = log(x + sqrt((x+1)*(x-1)))
//   acosh(x) = nan                         if x < -1
//
// If x^2 will overflow, we approximate sqrt(x^2 - 1) == x and compute as
// log(2*x) = log(2) + log(x).  (Note this works because negative x never
// overflows; x < -1 simply yields nan.
def : Pat<(HLOClient_AcoshOp NonComplexElementType:$input),
  (HLO_SelectOp
    (HLO_CompareOp
      $input,
      (HLO_ConstantLike<"-1"> $input),
      HLO_COMPARISON_DIRECTION_LT,
      (HLO_DEFAULT_COMPARISON_TYPE)
    ),
    (HLO_ConstantLike<"NAN"> $input),
    (HLO_SelectOp
      (HLO_CompareOp
        $input,
        (HLO_SqrtOp
          (HLO_ConstantLikeMaxFiniteValue $input)
        ),
        HLO_COMPARISON_DIRECTION_GE,
        (HLO_DEFAULT_COMPARISON_TYPE)
      ),
      (HLO_AddOp
        (HLO_LogOp $input),
        (HLO_LogOp
          (HLO_ConstantLike<"2"> $input)
        )
      ),
      (HLO_LogOp
        (HLO_AddOp
          $input,
          (HLO_SqrtOp
            (HLO_MulOp
              (HLO_AddOp
                (HLO_ConstantLike<"1"> $input),
                $input
              ),
              (HLO_AddOp
                (HLO_ConstantLike<"-1"> $input),
                $input
              )
            )
          )
        )
      )
    )
  )>;

// Expand asin to MHLO dialect as follows:
//   asin(x) = 2 * atan(x / (1 + sqrt(1 - x^2)))
def : Pat<(HLOClient_AsinOp NonComplexElementType:$input),
  (HLO_MulOp
    (HLO_ConstantLike<"2"> $input),
    (HLO_Atan2Op
      $input,
      (HLO_AddOp
        (HLO_ConstantLike<"1"> $input),
        (HLO_SqrtOp
          (HLO_SubOp
            (HLO_ConstantLike<"1"> $input),
            (HLO_MulOp $input, $input)
          )
        )
      )
    )
  )>;

// Expand asinh to MHLO dialect as
//   asinh(x) = log(x + sqrt(x^2 + 1))
//
// If x^2 will overflow and x is positive, we can approximate x + sqrt(x^2 + 1)
// as 2*x and return log(2) + log(x).
//
// For small x, sqrt(x^2 + 1) will evaluate to 1 due to floating point
// arithmetic. However, we would like to retain the low order term of this,
// which is around 0.5 * x^2 using a binomial expansion.
// Let z = sqrt(a^2 + 1)
// The following rewrite retains the lower order term.
// log(a + sqrt(a^2 + 1))
//   = log((a + sqrt(a^2 + 1)) * (1 + sqrt(a^2 + 1)) / (1 + sqrt(a^2 + 1)))
//   = log((a + a^2 + 1 + a * z + z) / (1 + z))
//   = log(1 + a + a^2 / (1 + z))
//   = log(1 + a + a^2 / (1 + sqrt(a^2 + 1)))
//
// If x is negative, the above would give us some trouble; we can't approximate
// the result as x + abs(x) = 0 but we are saved by the fact that asinh(-x) =
// -asinh(x).
def : Pat<(HLOClient_AsinhOp NonComplexElementType:$input),
  (HLO_MulOp
    (HLO_SignOp $input),
    (HLO_SelectOp
      (HLO_CompareOp
        (HLO_AbsOp $input),
        (HLO_SqrtOp
          (HLO_ConstantLikeMaxFiniteValue $input)
        ),
        HLO_COMPARISON_DIRECTION_GE,
        (HLO_DEFAULT_COMPARISON_TYPE)
      ),
      (HLO_AddOp
        (HLO_LogOp
          (HLO_AbsOp $input)
        ),
        (HLO_LogOp
          (HLO_ConstantLike<"2"> $input)
        )
      ),
      (HLO_SelectOp
        (HLO_CompareOp
          (HLO_AbsOp $input),
          (HLO_ConstantLike<"1"> $input),
          HLO_COMPARISON_DIRECTION_LE,
          (HLO_DEFAULT_COMPARISON_TYPE)
        ),
        (HLO_Log1pOp
          (HLO_AddOp
            (HLO_AbsOp $input),
            (HLO_MulOp
              (HLO_AbsOp $input),
              (HLO_DivOp
                (HLO_AbsOp $input),
                (HLO_AddOp
                  (HLO_ConstantLike<"1"> $input),
                  (HLO_SqrtOp
                    (HLO_AddOp
                      (HLO_MulOp
                        (HLO_AbsOp $input),
                        (HLO_AbsOp $input)
                      ),
                      (HLO_ConstantLike<"1"> $input)
                    )
                  )
                )
              )
            )
          )
        ),
        (HLO_LogOp
          (HLO_AddOp
            (HLO_AbsOp $input),
            (HLO_SqrtOp
              (HLO_AddOp
                (HLO_MulOp
                  (HLO_AbsOp $input),
                  (HLO_AbsOp $input)
                ),
                (HLO_ConstantLike<"1"> $input)
              )
            )
          )
        )
      )
    )
  )>;

// Express `atan` as
//   atan(x) = atan2(x, 1)
def : Pat<(HLOClient_AtanOp $input),
  (HLO_Atan2Op
    $input,
    (HLO_ConstantLike<"1"> $input)
  )>;

// Express `atanh` as follows:
//   atanh(x) = 0.5 * log((1 + x) / (1 - x)) if abs(x) <= 1
//   atanh(x) = nan                          otherwise
def : Pat<(HLOClient_AtanhOp NonComplexElementType:$input),
  (HLO_SelectOp
    (HLO_CompareOp
      (HLO_AbsOp $input),
      (HLO_ConstantLike<"1"> $input),
      HLO_COMPARISON_DIRECTION_GT,
      (HLO_DEFAULT_COMPARISON_TYPE)
    ),
    (HLO_ConstantLike<"NAN"> $input),
    (HLO_MulOp
      (HLO_SubOp
        (HLO_Log1pOp $input),
        (HLO_Log1pOp
          (HLO_NegOp $input)
        )
      ),
      (HLO_ConstantLike<"0.5"> $input)
    )
  )>;

// Express `conj` as
//   conj(x) = (re(x), -im(x)).
def : Pat<(HLOClient_ConjOp $v),
          (HLO_ComplexOp (HLO_RealOp $v), (HLO_NegOp (HLO_ImagOp $v)))>;

// Express `cosh` as
//   cosh(x) = (e^x + e^-x) / 2
//           = e^(x + log(1/2)) + e^(-x + log(1/2))
//
// The second formulation avoids overflowing when e^x = inf but (e^x)/2 is not.
//
// This incorrectly overflows to inf for two f32 input values, namely
// +/-89.4159851, due to rounding error when computing x +/- log(1/2).  The
// correct answer of 3.40281961e+38 (0x7f7fffec) is very close to max-float, so
// we deem this acceptable.
def : Pat<(HLOClient_CoshOp NonComplexElementType:$input),
  (HLO_AddOp
    (HLO_ExpOp
      (HLO_AddOp
        $input,
        (HLO_LogOp
          (HLO_ConstantLike<"0.5"> $input)
        )
      )
    ),
    (HLO_ExpOp
      (HLO_AddOp
        (HLO_NegOp $input),
        (HLO_LogOp
          (HLO_ConstantLike<"0.5"> $input)
        )
      )
    )
  )>;

// Express `is_inf` as
//   is_inf(x) = is_pos_inf(|x|)
def : Pat<(HLOClient_IsInfOp NonComplexElementType:$input),
  (HLOClient_IsPosInfOp
    (HLO_AbsOp $input)
  )>;

// Express `is_pos_inf` as
//   is_pos_inf(x) = (x == +inf)
def : Pat<(HLOClient_IsPosInfOp NonComplexElementType:$input),
  (HLO_CompareOp
    $input,
    (HLO_ConstantLikePosInfValue $input),
    HLO_COMPARISON_DIRECTION_EQ,
    (HLO_DEFAULT_COMPARISON_TYPE)
  )>;

// Express `is_neg_inf` as
//   is_neg_inf(x) = (x == -inf)
def : Pat<(HLOClient_IsNegInfOp NonComplexElementType:$input),
  (HLO_CompareOp
    $input,
    (HLO_ConstantLikeNegInfValue $input),
    HLO_COMPARISON_DIRECTION_EQ,
    (HLO_DEFAULT_COMPARISON_TYPE)
  )>;

// Express `sinh` as
//   sinh(x) = (e^x - e^-x) / 2                     if |x| < 1
//           = e^(x + log(1/2)) - e^(-x + log(1/2)) otherwise.
// TODO(hinsu): Support operands with complex element types by always using the
// second formula. The compare op below is not legal for complex numbers.
def : Pat<(HLOClient_SinhOp NonComplexElementType:$input),
  (HLO_SelectOp
    (HLO_CompareOp
      (HLO_AbsOp $input),
      (HLO_ConstantLike<"1"> $input),
      HLO_COMPARISON_DIRECTION_LT,
      (HLO_DEFAULT_COMPARISON_TYPE)
    ),
    (HLO_DivOp
      (HLO_SubOp
        (HLO_ExpOp $input),
        (HLO_ExpOp
          (HLO_NegOp $input)
        )
      ),
      (HLO_ConstantLike<"2"> $input)
    ),
    (HLO_SubOp
      (HLO_ExpOp
        (HLO_AddOp
          $input,
          (HLO_LogOp
            (HLO_ConstantLike<"0.5"> $input)
          )
        )
      ),
      (HLO_ExpOp
        (HLO_SubOp
          (HLO_LogOp
            (HLO_ConstantLike<"0.5"> $input)
          ),
          $input
        )
      )
    )
  )>;

// Express tan in MHLO dialect as
//   tan(x) = sin(x) / cos(x).
def : Pat<(HLOClient_TanOp NonComplexElementType:$input),
  (HLO_DivOp
    (HLO_SinOp $input),
    (HLO_CosOp $input)
  )>;
