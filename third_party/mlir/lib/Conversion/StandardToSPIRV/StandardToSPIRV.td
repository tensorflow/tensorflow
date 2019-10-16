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

class BinaryOpPattern<Op src, Op tgt> :
      Pat<(src SPV_ScalarOrVector:$l, SPV_ScalarOrVector:$r),
          (tgt $l, $r)>;

def : BinaryOpPattern<AddFOp, SPV_FAddOp>;
def : BinaryOpPattern<MulFOp, SPV_FMulOp>;

// Constant Op
// TODO(ravishankarm): Handle lowering other constant types.
def : Pat<(ConstantOp:$result $valueAttr),
          (SPV_ConstantOp $valueAttr),
          [(SPV_ScalarOrVector $result)]>;

#endif // MLIR_CONVERSION_STANDARDTOSPIRV_TD
