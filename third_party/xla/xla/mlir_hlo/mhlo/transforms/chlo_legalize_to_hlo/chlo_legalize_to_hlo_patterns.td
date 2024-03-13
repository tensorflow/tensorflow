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

// This is the legalization pattern definition file for CHLO to MHLO.
// These are included in the PopulateDecomposeChloPatterns factory
// and should only include canonical expansions which are not actually
// ambiguous/different for various backends. Avoid patterns that are actually
// lowering to non-canonical forms.

include "mlir/IR/OpBase.td"
include "mhlo/IR/hlo_ops.td"
include "stablehlo/dialect/ChloOps.td"

//===----------------------------------------------------------------------===//
// Direct CHLO->MHLO conversions
//===----------------------------------------------------------------------===//

def : Pat<(CHLO_TanOp $v),
          (MHLO_TanOp $v),
          [], [], (addBenefit 10)>;

def : Pat<(CHLO_ErfOp $v),
          (MHLO_ErfOp $v),
          [], [], (addBenefit 10)>;

def : Pat<(CHLO_TopKOp AnyRankedTensor:$v, $k),
          (MHLO_TopKOp $v, $k, ConstBoolAttrTrue),
          [], [], (addBenefit 10)>;
