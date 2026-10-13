/* Copyright 2026 The OpenXLA Authors.

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

#ifndef TPU_MATMUL_OPS_
#define TPU_MATMUL_OPS_

// TODO(apaszke): Think hard about precision
def TPU_MatmulOp : TPU_Op<"matmul", [Pure]> {
  let arguments = (ins
    AnyVectorOfNonZeroRank:$lhs,
    AnyVectorOfNonZeroRank:$rhs,
    AnyVectorOfNonZeroRank:$acc,
    // These flags are deprecated - if dimension_numbers are defined,
    // these flags are ignored. They will always be false after canonicalize.
    DefaultValuedAttr<BoolAttr, "false">:$transpose_lhs,
    DefaultValuedAttr<BoolAttr, "false">:$transpose_rhs,
    OptionalAttr<TPU_ContractPrecisionEnum>:$precision,
    // NOTE: User-level optional, once canonicalized, always present.
    OptionalAttr<TPU_DotDimensionNumbersAttr>:$dimension_numbers,
    // Hint to compilers whether to fuse the transposed LHS in the matmul.
    BoolAttr:$transpose_lhs_hint
  );
  let results = (outs AnyVectorOfNonZeroRank:$result);
  let assemblyFormat = [{
    $lhs `,` $rhs `,` $acc prop-dict attr-dict `:` type($lhs) `,` type($rhs) `,` type($acc) `->` type($result)
  }];
  let hasVerifier = 1;
}

#endif  // TPU_MATMUL_OPS_
