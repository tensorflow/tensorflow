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
#include "tensorflow/core/common_runtime/simple_propagator_state.h"

#include "tensorflow/core/common_runtime/propagator_debug_utils.h"
#include "tensorflow/core/profiler/lib/traceme.h"

namespace tensorflow {

SimplePropagatorState::SimplePropagatorState(
    const ImmutableExecutorState& immutable_state, int64 step_id)
    : SimplePropagatorState(immutable_state, step_id,
                            immutable_state.get_root_frame_info()) {}

SimplePropagatorState::SimplePropagatorState(
    const ImmutableExecutorState& immutable_state, int64 step_id,
    const ImmutableExecutorState::FrameInfo& finfo)
    : immutable_state_(immutable_state),
      step_id_(step_id),
      vlog_(VLOG_IS_ON(1)),
      input_tensors_(finfo.total_inputs),
      counts_(*finfo.pending_counts),
      nodes_(finfo.nodes.get()) {}

SimplePropagatorState::~SimplePropagatorState() {}

void SimplePropagatorState::ActivateRoots(
    gtl::ArraySlice<const NodeItem*> roots, TaggedNodeSeq* ready) {
  for (const NodeItem* item : roots) {
    DCHECK_EQ(item->num_inputs, 0);
    ready->emplace_back(item);
  }
}

void SimplePropagatorState::PropagateOutputs(const TaggedNode& tagged_node,
                                             EntryVector* outputs,
                                             TaggedNodeSeq* ready) {
  profiler::TraceMe activity(
      [&]() {
        return strings::StrCat(
            "ExecutorPropagateOutputs#", "id=", step_id_,
            ",kernel_name=", tagged_node.node_item->kernel->name_view(),
            ",num_output_edges=", tagged_node.node_item->num_output_edges,
            ",num_output_control_edges=",
            tagged_node.node_item->num_output_control_edges, "#");
      },
      profiler::GetTFTraceMeLevel(/*is_expensive=*/false));

  // Propagates outputs along out edges, and puts newly ready nodes
  // into the ready queue.
  DCHECK(ready->empty());

  const GraphView& gview = immutable_state_.graph_view();
  const NodeItem* item = tagged_node.node_item;

  mutex_lock l(mu_);

  for (const EdgeInfo& e : item->output_edges()) {
    const int dst_id = e.dst_id;
    const PendingCounts::Handle dst_pending_id =
        immutable_state_.pending_ids()[dst_id];
    const int src_slot = e.output_slot;
    int num_pending = counts_.decrement_pending(dst_pending_id, 1);
    const int dst_loc = e.input_slot;
    if (e.is_last) {
      input_tensors_[dst_loc] = std::move((*outputs)[src_slot]);
    } else {
      input_tensors_[dst_loc] = (*outputs)[src_slot];
    }
    if (num_pending == 0) ready->emplace_back(&gview.node_ref(dst_id));
  }

  for (const ControlEdgeInfo& e : item->output_control_edges()) {
    const int dst_id = e.dst_id;
    const PendingCounts::Handle dst_pending_id =
        immutable_state_.pending_ids()[dst_id];
    int num_pending = counts_.decrement_pending(dst_pending_id, 1);
    if (num_pending == 0) ready->emplace_back(&gview.node_ref(dst_id));
  }
}

void SimplePropagatorState::DumpState() {
  mutex_lock l(mu_);
  LOG(WARNING) << "Dumping state";

  // Dump any waiting nodes that are holding on to tensors.
  for (const NodeItem* node : *nodes_) {
    PendingCounts::Handle pending_id =
        immutable_state_.pending_ids()[node->node_id];
    if (counts_.node_state(pending_id) == PendingCounts::PENDING_NOTREADY ||
        counts_.node_state(pending_id) == PendingCounts::PENDING_READY) {
      DumpPendingNodeState(immutable_state_, node->node_id,
                           input_tensors_.data(), false);
    }
  }
  // Then the active nodes.
  for (const NodeItem* node : *nodes_) {
    PendingCounts::Handle pending_id =
        immutable_state_.pending_ids()[node->node_id];
    if (counts_.node_state(pending_id) == PendingCounts::STARTED) {
      DumpActiveNodeState(immutable_state_, node->node_id,
                          input_tensors_.data());
    }
  }
  // Show all input tensors in use.
  size_t total_bytes = 0;
  for (size_t i = 0; i < input_tensors_.size(); ++i) {
    const Entry& input = input_tensors_[i];
    const Tensor* tensor = GetTensorValueForDump(input);
    if (tensor && tensor->IsInitialized()) {
      LOG(WARNING) << "    Input " << i << ": "
                   << strings::StrCat(
                          "Tensor<type: ", DataTypeString(tensor->dtype()),
                          " shape: ", tensor->shape().DebugString(),
                          ", bytes: ", tensor->TotalBytes(), ">");
      total_bytes += tensor->TotalBytes();
    }
  }
  LOG(WARNING) << "    Total bytes " << total_bytes;
}

}  // namespace tensorflow
