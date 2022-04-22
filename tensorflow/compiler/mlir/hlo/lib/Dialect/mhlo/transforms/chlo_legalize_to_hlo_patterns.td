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

include "mlir/Dialect/Shape/IR/ShapeOps.td"
include "mlir/IR/OpBase.td"
include "mlir-hlo/Dialect/mhlo/IR/hlo_ops.td"
include "mlir-hlo/Dialect/mhlo/IR/chlo_ops.td"

class HLO_ComparisonDirectionValue<string enumStr> :
  ConstantAttr<HLO_ComparisonDirectionAttr, "::mlir::mhlo::ComparisonDirection::" # enumStr>;

//===----------------------------------------------------------------------===//
// Unary op patterns.
//===----------------------------------------------------------------------===//

// Expand acos for non-complex arguments to MHLO dialect as follows:
//   acos(x) = 2 * atan2(sqrt(1 - x^2), (1 + x))  if x != -1
//           = pi                                 if x == -1
//
// TODO(hinsu): Support operands with complex element types separately using
// the following formula.
//   acos(x) = -(i * log(x + i * sqrt((1 + x) * (1 - x))))
def : Pat<(CHLO_AcosOp NonComplexElementType:$input),
  (HLO_SelectOp
    (HLO_CompareOp
      $input,
      (HLO_ConstantLike<"-1"> $input),
      HLO_ComparisonDirectionValue<"NE">,
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
def : Pat<(CHLO_AcoshOp NonComplexElementType:$input),
  (HLO_SelectOp
    (HLO_CompareOp
      $input,
      (HLO_ConstantLike<"-1"> $input),
      HLO_ComparisonDirectionValue<"LT">,
      (HLO_DEFAULT_COMPARISON_TYPE)
    ),
    (HLO_ConstantLike<"NAN"> $input),
    (HLO_SelectOp
      (HLO_CompareOp
        $input,
        (HLO_SqrtOp
          (HLO_ConstantLikeMaxFiniteValue $input)
        ),
        HLO_ComparisonDirectionValue<"GE">,
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

// Expand acosh for complex arguments to MHLO dialect as
//   acosh(x) = log(x + sqrt((x+1)*(x-1)))
//
// Per tensorflow/compiler/xla/client/lib/math.cc at the time of writing:
// "For now, we ignore the question of overflow if x is a
// complex type, because we don't yet have exhaustive tests for complex trig
// functions".
def : Pat<(CHLO_AcoshOp ComplexElementType:$input),
  (HLO_LogOp
    (HLO_AddOp
      $input,
      (HLO_SqrtOp
        (HLO_MulOp
          (HLO_AddOp
            $input,
            (HLO_ConstantLike<"1"> $input)
          ),
          (HLO_SubOp
            $input,
            (HLO_ConstantLike<"1"> $input)
          )
        )
      )
    )
  )>;


// Expand asin to MHLO dialect as follows:
//   asin(x) = 2 * atan(x / (1 + sqrt(1 - x^2)))
def : Pat<(CHLO_AsinOp NonComplexElementType:$input),
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

// Expand asinh for non-complex arguments to MHLO dialect as
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
def : Pat<(CHLO_AsinhOp NonComplexElementType:$input),
  (HLO_MulOp
    (HLO_SignOp $input),
    (HLO_SelectOp
      (HLO_CompareOp
        (HLO_AbsOp $input),
        (HLO_SqrtOp
          (HLO_ConstantLikeMaxFiniteValue $input)
        ),
        HLO_ComparisonDirectionValue<"GE">,
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
          HLO_ComparisonDirectionValue<"LE">,
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

// Expand asinh for complex arguments to MHLO dialect as
//   asinh(x) = log(x + sqrt(x^2 + 1))
//
// Per tensorflow/compiler/xla/client/lib/math.cc at the time of writing:
// "For now, we ignore the question of overflow if x is a
// complex type, because we don't yet have exhaustive tests for complex trig
// functions".
def : Pat<(CHLO_AsinhOp ComplexElementType:$input),
  (HLO_LogOp
    (HLO_AddOp
      $input,
      (HLO_SqrtOp
        (HLO_AddOp
          (HLO_MulOp $input, $input),
          (HLO_ConstantLike<"1"> $input)
        )
      )
    )
  )>;

// Express `atan` as
//   atan(x) = atan2(x, 1)
def : Pat<(CHLO_AtanOp $input),
  (HLO_Atan2Op
    $input,
    (HLO_ConstantLike<"1"> $input)
  )>;

// Express `atanh` for non-complex arguments as follows:
//   atanh(x) = 0.5 * log((1 + x) / (1 - x)) if abs(x) <= 1
//   atanh(x) = nan                          otherwise
def : Pat<(CHLO_AtanhOp NonComplexElementType:$input),
  (HLO_SelectOp
    (HLO_CompareOp
      (HLO_AbsOp $input),
      (HLO_ConstantLike<"1"> $input),
      HLO_ComparisonDirectionValue<"GT">,
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

// Express `atanh` for complex arguments as follows:
//   atanh(x) = (log(1 + x) - log(1 + (-x))) * 0.5
//
// Per tensorflow/compiler/xla/client/lib/math.cc at the time of writing:
// "For now, we ignore the nan edge case for complex inputs,
// because we don't yet have exhaustive tests for complex trig functions".
def : Pat<(CHLO_AtanhOp ComplexElementType:$input),
  (HLO_MulOp
    (HLO_SubOp
      (HLO_Log1pOp $input),
      (HLO_Log1pOp
        (HLO_NegOp $input)
      )
    ),
    (HLO_ConstantLike<"0.5"> $input)
  )>;

// Express `conj` as
//   conj(x) = (re(x), -im(x)).
def : Pat<(CHLO_ConjOp $v),
          (HLO_ComplexOp (HLO_RealOp $v), (HLO_NegOp (HLO_ImagOp $v)))>;

// Express `is_inf` as
//   is_inf(x) = is_pos_inf(|x|)
def : Pat<(CHLO_IsInfOp NonComplexElementType:$input),
  (CHLO_IsPosInfOp
    (HLO_AbsOp $input)
  )>;

// Express `is_pos_inf` as
//   is_pos_inf(x) = (x == +inf)
def : Pat<(CHLO_IsPosInfOp NonComplexElementType:$input),
  (HLO_CompareOp
    $input,
    (HLO_ConstantLikePosInfValue $input),
    HLO_ComparisonDirectionValue<"EQ">,
    (HLO_DEFAULT_COMPARISON_TYPE)
  )>;

// Express `is_neg_inf` as
//   is_neg_inf(x) = (x == -inf)
def : Pat<(CHLO_IsNegInfOp NonComplexElementType:$input),
  (HLO_CompareOp
    $input,
    (HLO_ConstantLikeNegInfValue $input),
    HLO_ComparisonDirectionValue<"EQ">,
    (HLO_DEFAULT_COMPARISON_TYPE)
  )>;
