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

include "mlir/IR/OpBase.td"
include "mlir-hlo/Dialect/mhlo/IR/hlo_ops.td"
include "mlir-hlo/Dialect/mhlo/IR/chlo_ops.td"

//===----------------------------------------------------------------------===//
// Unary op patterns.
//===----------------------------------------------------------------------===//

def NonComplexElementType : Type<
  CPred<"!$_self.cast<ShapedType>().getElementType().isa<ComplexType>()">,
  "Non complex element type">;

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

// Express `atan` as
//   atan(x) = atan2(x, 1)
def : Pat<(HLOClient_AtanOp $input),
  (HLO_Atan2Op
    $input,
    (HLO_ConstantLike<"1"> $input)
  )>;

// Express `conj` as
//   conj(x) = (re(x), -im(x)).
def : Pat<(HLOClient_ConjOp $v),
          (HLO_ComplexOp (HLO_RealOp $v), (HLO_NegOp (HLO_ImagOp $v)))>;

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
def : Pat<(HLOClient_TanOp $input),
  (HLO_DivOp
    (HLO_SinOp $input),
    (HLO_CosOp $input)
  )>;
