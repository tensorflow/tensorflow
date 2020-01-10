/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_JIT_DEADNESS_ANALYSIS_INTERNAL_H_
#define TENSORFLOW_COMPILER_JIT_DEADNESS_ANALYSIS_INTERNAL_H_

#include "absl/container/flat_hash_map.h"
#include "tensorflow/core/graph/tensor_id.h"

namespace tensorflow {
namespace deadness_analysis_internal {

// Returns a map describing the predicate each Tensor was mapped to.  For
// testing purposes only.
using PredicateMapTy = absl::flat_hash_map<TensorId, string, TensorId::Hasher>;
Status ComputePredicates(const Graph& graph, PredicateMapTy* out_predicate_map,
                         bool enable_optimistic = true);

}  // namespace deadness_analysis_internal
}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_JIT_DEADNESS_ANALYSIS_INTERNAL_H_
