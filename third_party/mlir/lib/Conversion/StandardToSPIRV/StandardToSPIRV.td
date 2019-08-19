//==- StandardToSPIRV.td - Standard Ops to SPIR-V Patterns ---*- tablegen -*==//

// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines Patterns to lower standard ops to SPIR-V.
//
//===----------------------------------------------------------------------===//

#ifdef MLIR_CONVERSION_STANDARDTOSPIRV_TD
#else
#define MLIR_CONVERSION_STANDARDTOSPIRV_TD

#ifdef STANDARD_OPS
#else
include "mlir/Dialect/StandardOps/Ops.td"
#endif // STANDARD_OPS

#ifdef SPIRV_OPS
#else
include "mlir/Dialect/SPIRV/SPIRVOps.td"
#endif // SPIRV_OPS

def IsScalar : TypeConstraint<CPred<"!($_self.isa<ShapedType>())">, "scalar">;

class IsVectorLengthPred<int vecLength> :
      CPred<"($_self.cast<VectorType>().getShape().size() == 1 && " #
            "$_self.cast<VectorType>().getShape()[0] == " # vecLength # ")">;

class IsVectorOfLength<int vecLength>:
    TypeConstraint<And<[IsVectorTypePred, IsVectorLengthPred<vecLength>]>,
                   vecLength # "-element vector">;

multiclass BinaryOpPattern<Op src, SPV_Op tgt> {
  def : Pat<(src IsScalar:$l, IsScalar:$r), (tgt $l, $r)>;
  foreach vecLength = [2, 3, 4] in {
    def : Pat<(src IsVectorOfLength<vecLength>:$l,
                   IsVectorOfLength<vecLength>:$r),
              (tgt $l, $r)>;
  }
}

defm : BinaryOpPattern<MulFOp, SPV_FMulOp>;

#endif // MLIR_CONVERSION_STANDARDTOSPIRV_TD
