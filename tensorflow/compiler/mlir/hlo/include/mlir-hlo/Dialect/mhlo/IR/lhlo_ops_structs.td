/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#ifndef LHLO_OPS_STRUCTS
#define LHLO_OPS_STRUCTS

include "mlir-hlo/Dialect/mhlo/IR/lhlo_dialect.td"

// This structure defines information about how arguments to the LHLO custom
// call operation relate to the arguments of the target function. In most cases
// the mapping will be 1:1, but in certain cases, it may not be. As an example,
// tokens are not represented in the LHLO dialect, but the custom call target
// might still expect to see buffer arguments corresponding to tokens, in which
// case the mapping will not be 1:1.
def CustomCallTargetArgMapping : StructAttr<"CustomCallTargetArgMapping",
                                 LHLO_Dialect, [
   // number of buffer expected by the target for arguments.
   StructFieldAttr<"num_args", I64Attr>,
   // number of buffer expected by the target for results.
   StructFieldAttr<"num_results", I64Attr>,
   // map each custom call op arg to its position in target args.
   StructFieldAttr<"args_to_target_args", I64ArrayAttr>,
   // map each custom call op arg to its position in target results.
   StructFieldAttr<"results_to_target_results", I64ArrayAttr>]> {
   let summary = "Custom call operands to target argument mapping info";
}

#endif // LHLO_OPS_STRUCTS
