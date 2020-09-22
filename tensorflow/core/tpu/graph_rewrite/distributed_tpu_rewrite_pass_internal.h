/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_TPU_GRAPH_REWRITE_DISTRIBUTED_TPU_REWRITE_PASS_INTERNAL_H_
#define TENSORFLOW_CORE_TPU_GRAPH_REWRITE_DISTRIBUTED_TPU_REWRITE_PASS_INTERNAL_H_

#include "tensorflow/core/framework/types.h"

namespace tensorflow {

// Implementation details of distributed_tpu_rewrite_pass.cc, please DO NOT
// depend on these.
namespace internal {

// When set to a value >= 0, overrides the node_id. Used for getting
// deterministic node_ids during testing.
void OverrideNodeIdForTesting(int64 node_id);

// Retrieves the node id, used to make some node names unique in the rewrite
// pass.
uint64 GetNodeId();

}  // namespace internal
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_TPU_GRAPH_REWRITE_DISTRIBUTED_TPU_REWRITE_PASS_INTERNAL_H_
