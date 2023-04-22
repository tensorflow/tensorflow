/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_CORE_COMMON_RUNTIME_PROPAGATOR_DEBUG_UTILS_H_
#define TENSORFLOW_CORE_COMMON_RUNTIME_PROPAGATOR_DEBUG_UTILS_H_

namespace tensorflow {

struct Entry;
struct NodeItem;
class Tensor;

// Returns a pointer to the tensor in `input` if one exists, or `nullptr`.
const Tensor* GetTensorValueForDump(const Entry& input);

// Writes a LOG(WARNING) message describing the state of the given pending node
// in the graph described by `immutable_state`.
void DumpPendingNodeState(const NodeItem& node_item, const Entry* input_vector,
                          const bool show_nodes_with_no_ready_inputs);

// Writes a LOG(WARNING) message describing the state of the given active node
// in the graph described by `immutable_state`.
void DumpActiveNodeState(const NodeItem& node_item, const Entry* input_vector);

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_COMMON_RUNTIME_PROPAGATOR_DEBUG_UTILS_H_
