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

#include "tensorflow/compiler/mlir/lite/stablehlo/transforms/mhlo_util.h"

#include <algorithm>
#include <string>

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/StringRef.h"

namespace mlir {
namespace TFL {
namespace mhlo {

std::vector<std::string> GetAcceptedDialects() {
  // It returns the default list of accepted dialects.
  std::vector<std::string> accepted_dialects({"mhlo", "builtin", "func"});
  return accepted_dialects;
}

bool IsAcceptedDialect(llvm::StringRef dialect_name,
                       const std::vector<std::string>& accepted_dialects) {
  return std::find(accepted_dialects.begin(), accepted_dialects.end(),
                   dialect_name) != accepted_dialects.end();
}

bool IsMhloOpAllowed(StringRef op_name) {
  // As per go/compute-ir-ops-v01.
  static DenseSet<StringRef>* denylist = new DenseSet<StringRef>{
      // (R2) Part 1: Internal ops.
      "bitcast", "fusion",
      // (R2) Part 2: Modularity ops.
      // NOTE: These ops were proposed to be excluded from Compute IR
      // because we didn't want to necessarily tie the specification to MLIR.
      // In an MLIR-based implementation such as MHLO, these ops are fine.
      // "get_tuple_element", "return", "tuple",
      // (R3) Part 1: Distribution ops.
      "after_all", "all_gather", "all_reduce", "all_to_all",
      "collective_permute", "create_token", "cross-replica-sum", "infeed",
      "outfeed", "print", "recv", "reduce_scatter", "replica_id", "send",
      "trace",
      // (R3) Part 2: Dynamism ops.
      "compute_reshape_shape", "cstr_reshapable", "dynamic_broadcast_in_dim",
      "dynamic_conv", "dynamic_gather", "dynamic_iota", "dynamic_pad",
      "dynamic_reshape", "get_dimension_size", "real_dynamic_slice",
      "set_dimension_size"
      // NOTE: These ops were proposed to be excluded from Compute IR for now
      // because we wanted to unify them with slice and real_dynamic_slice.
      // In the meanwhile, they are very practically important to MHLO,
      // so we should keep them on the allowlist.
      // "dynamic-slice", "dynamic-update-slice"
  };
  return !denylist->contains(op_name);
}

bool IsAcceptedOp(llvm::StringRef dialect_name, llvm::StringRef op_name,
                  const std::vector<std::string>& accepted_dialects) {
  if (!IsAcceptedDialect(dialect_name, accepted_dialects)) return false;
  return dialect_name != "mhlo" || IsMhloOpAllowed(op_name);
}

}  // namespace mhlo
}  // namespace TFL
}  // namespace mlir
