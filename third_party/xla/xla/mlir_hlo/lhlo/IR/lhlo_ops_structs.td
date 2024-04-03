/* Copyright 2021 The OpenXLA Authors.

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

include "lhlo/IR/lhlo_dialect.td"
include "mlir/IR/AttrTypeBase.td"

// This attribute defines information about how arguments to the LHLO custom
// call operation relate to the arguments of the target function. In most cases
// the mapping will be 1:1, but in certain cases, it may not be. As an example,
// tokens are not represented in the LHLO dialect, but the custom call target
// might still expect to see buffer arguments corresponding to tokens, in which
// case the mapping will not be 1:1.
def CustomCallTargetArgMappingAttr : AttrDef<
  LHLO_Dialect, "CustomCallTargetArgMapping"> {
  let mnemonic = "custom_call_target_arg_mapping";
  let parameters = (ins
    // number of buffer expected by the target for arguments.
    "int64_t":$num_args,
    // number of buffer expected by the target for results.
    "int64_t":$num_results,
    // map each custom call op arg to its position in target args.
    ArrayRefParameter<"int64_t">:$args_to_target_args,
    // map each custom call op arg to its position in target results.
    ArrayRefParameter<"int64_t">:$results_to_target_results
  );
  // 'struct()' would allow any order, but doesn't support square brackets yet.
  let assemblyFormat = [{`<`
    `num_args` `=` $num_args`,`
    `num_results` `=` $num_results`,`
    `args_to_target_args` `=` `[`$args_to_target_args`]``,`
    `results_to_target_results` `=` `[`$results_to_target_results`]``>`
  }];
  let summary = "Custom call operands to target argument mapping info";
}

#endif // LHLO_OPS_STRUCTS
