//==- SPIRVCanonicalization.td - Canonicalization Patterns ---*- tablegen -*==//

// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines SPIR-V canonicalization patterns.
//
//===----------------------------------------------------------------------===//

include "mlir/Dialect/SPIRV/SPIRVOps.td"

//===----------------------------------------------------------------------===//
// spv.Bitcast
//===----------------------------------------------------------------------===//

def ConvertChainedBitcast : Pat<(SPV_BitcastOp (SPV_BitcastOp $operand)),
                                (SPV_BitcastOp $operand)>;

//===----------------------------------------------------------------------===//
// spv.LogicalNot
//===----------------------------------------------------------------------===//

def ConvertLogicalNotOfIEqual : Pat<
    (SPV_LogicalNotOp (SPV_IEqualOp $lhs, $rhs)),
    (SPV_INotEqualOp $lhs, $rhs)>;

def ConvertLogicalNotOfINotEqual : Pat<
    (SPV_LogicalNotOp (SPV_INotEqualOp $lhs, $rhs)),
    (SPV_IEqualOp $lhs, $rhs)>;

def ConvertLogicalNotOfLogicalEqual : Pat<
    (SPV_LogicalNotOp (SPV_LogicalEqualOp $lhs, $rhs)),
    (SPV_LogicalNotEqualOp $lhs, $rhs)>;

def ConvertLogicalNotOfLogicalNotEqual : Pat<
    (SPV_LogicalNotOp (SPV_LogicalNotEqualOp $lhs, $rhs)),
    (SPV_LogicalEqualOp $lhs, $rhs)>;
