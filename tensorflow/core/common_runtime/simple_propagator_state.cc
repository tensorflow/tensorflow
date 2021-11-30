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

#include <atomic>

#include "tensorflow/core/common_runtime/propagator_debug_utils.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/profiler/lib/traceme.h"

namespace tensorflow {

SimplePropagatorState::SimplePropagatorState(
    const ImmutableExecutorState& immutable_state, int64_t step_id, bool vlog)
    : SimplePropagatorState(immutable_state, step_id,
                            immutable_state.get_root_frame_info(), vlog) {}

SimplePropagatorState::SimplePropagatorState(
    const ImmutableExecutorState& immutable_state, int64_t step_id,
    const ImmutableExecutorState::FrameInfo& finfo, bool vlog)
    : immutable_state_(immutable_state),
      step_id_(step_id),
      vlog_(vlog || VLOG_IS_ON(1)),
      input_tensors_(finfo.total_inputs),
      pending_(
          new std::atomic<int32>[immutable_state.graph_view().num_nodes()]),
      active_(vlog_ ? new std::vector<bool>(
                          immutable_state.graph_view().num_nodes())
                    : nullptr),
      nodes_(finfo.nodes.get()) {
  immutable_state_.copy_pending_counts(pending_.get());
}

SimplePropagatorState::~SimplePropagatorState() {}

void SimplePropagatorState::ActivateRoots(
    gtl::ArraySlice<const NodeItem*> roots, TaggedNodeSeq* ready) {
  for (const NodeItem* item : roots) {
    DCHECK_EQ(item->num_inputs, 0);
    ready->push_back(TaggedNode{item});
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

  for (const EdgeInfo& e : item->output_edges()) {
    const int dst_id = e.dst_id;
    const int src_slot = e.output_slot;
    const int dst_loc = e.input_slot;

    // NOTE(mrry): The write to `input_tensors_[dst_loc]` must happen before
    // the pending count update, or else one thread might conclude that the
    // count has dropped to zero before another thread finishes updating the
    // input.
    if (e.is_last) {
      input_tensors_[dst_loc] = std::move((*outputs)[src_slot]);
    } else {
      input_tensors_[dst_loc] = (*outputs)[src_slot];
    }

    int32_t previous_num_pending =
        pending_[dst_id].fetch_sub(1, std::memory_order_release);
    if (previous_num_pending == 1) ready->emplace_back(&gview.node_ref(dst_id));
  }

  for (const ControlEdgeInfo& e : item->output_control_edges()) {
    const int dst_id = e.dst_id;

    int32_t previous_num_pending =
        pending_[dst_id].fetch_sub(1, std::memory_order_release);
    if (previous_num_pending == 1) ready->emplace_back(&gview.node_ref(dst_id));
  }
}

void SimplePropagatorState::DumpState() {
  mutex_lock l(mu_);
  // Dump any waiting nodes that are holding on to tensors.
  for (const NodeItem* node : *nodes_) {
    if (pending_[node->node_id]) {
      DumpPendingNodeState(*node, input_tensors_.data(), false);
    }
  }
  // Then the active nodes.
  for (const NodeItem* node : *nodes_) {
    if ((*active_)[node->node_id]) {
      DumpActiveNodeState(*node, input_tensors_.data());
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
