/* Copyright 2020 The OpenXLA Authors.

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

#ifndef LHLO_OPS_BASE
#define LHLO_OPS_BASE

include "mlir/Dialect/MemRef/IR/MemRefBase.td"
include "mlir/IR/OpBase.td"
include "mhlo/IR/hlo_ops_common.td"

//===----------------------------------------------------------------------===//
// LMHLO type definitions.
//===----------------------------------------------------------------------===//

// Any integer tensor types
def LHLO_IntBuffer : MemRefOf<[MHLO_Int]>;

// Any floating-point tensor types
def LHLO_FpBuffer : MemRefOf<[AnyFloat]>;

def LHLO_ComplexBuffer : MemRefOf<[AnyComplex]>;

def LHLO_FpOrComplexBuffer : MemRefOf<[AnyFloat, AnyComplex]>;

def LHLO_PredBuffer : MemRefOf<[MHLO_Pred]>;

// Any integer or floating-point tensor types
def LHLO_IntOrFpBuffer : MemRefOf<[MHLO_Int, AnyFloat]>;

def LHLO_PredOrIntBuffer : MemRefOf<[MHLO_Int, MHLO_Pred]>;

def LHLO_Buffer : MemRefOf<[AnyFloat, AnyInteger, AnyComplex]>;

def LHLO_DimensionValue : AnyTypeOf<[Index, MHLO_Pred, MHLO_Int]>;

// Dynamic representation of a shape vector
def LHLO_DimensionBuffer : MemRefRankOf<[LHLO_DimensionValue], [1]>;

#endif // LHLO_OPS_BASE
