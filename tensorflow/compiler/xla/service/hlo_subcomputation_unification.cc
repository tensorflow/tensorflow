/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/hlo_subcomputation_unification.h"

#include <unordered_map>

namespace xla {

StatusOr<bool> HloSubcomputationUnification::Run(HloModule* module) {
  // For each computation C in the module, find the first computation C0 in the
  // computations_ list that is identical to C, and adds canon[C] = C0.
  absl::flat_hash_map<HloComputation*, HloComputation*> canon;
  const auto& computations = module->computations();
  for (auto i = computations.begin(); i != computations.end(); ++i) {
    for (auto j = computations.begin(); j != i; ++j) {
      // Do not waste time comparing `*i` with `*j` if `*j` is not canonical.
      if (canon.find(*j) == canon.end() && **i == **j) {
        canon[*i] = *j;
        break;
      }
    }
  }

  if (canon.empty()) {
    return false;
  }

  module->ReplaceComputations(canon);
  return true;
}

}  // namespace xla
