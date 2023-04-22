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
#ifndef TENSORFLOW_CORE_COMMON_RUNTIME_SIMPLE_PROPAGATOR_STATE_H_
#define TENSORFLOW_CORE_COMMON_RUNTIME_SIMPLE_PROPAGATOR_STATE_H_

#include <vector>

#include "tensorflow/core/common_runtime/entry.h"
#include "tensorflow/core/common_runtime/immutable_executor_state.h"
#include "tensorflow/core/common_runtime/pending_counts.h"
#include "tensorflow/core/framework/control_flow.h"
#include "tensorflow/core/lib/gtl/inlined_vector.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

// Represents the ephemeral "edge state" associated with one invocation of
// `Executor::Run()`.
//
// NOTE: `SimplePropagatorState` does not support "v1-style" control flow,
// including "dead tensors", "Switch" and "Merge" nodes, and cycles in the
// graph. Use `PropagatorState` for graphs with those features.
// `SimplePropagatorState` *does* support "v2-style" or "functional" control
// flow.
//
// `SimplePropagatorState` is responsible for propagating values along dataflow
// edges in a TensorFlow graph and determining which nodes are runnable. The
// executor primarily updates `SimplePropagatorState` by calling
// `PropagateOutputs()` after processing a node, and `SimplePropagatorState`
// dispatches `TaggedNode`s by adding them to a `TaggedNodeSeq`.
class SimplePropagatorState {
 public:
  SimplePropagatorState(const ImmutableExecutorState& immutable_state,
                        int64_t step_id, bool vlog);
  ~SimplePropagatorState();

  // A `TaggedNode` corresponds to a single invocation of a node's kernel,
  // and it is created when the kernel becomes runnable.
  struct TaggedNode {
    const NodeItem* node_item;

    explicit TaggedNode(const NodeItem* node_item) : node_item(node_item) {}

    const NodeItem& get_node_item() const { return *node_item; }

    bool get_is_dead() const { return false; }
    int64 get_iter_num() const { return 0; }
  };

  // A drop-in replacement for std::deque<TaggedNode>.  We typically don't
  // have that many nodes in the ready queue, so we just use a vector and
  // don't free up memory from the queue as we consume nodes.
  // TODO(mrry): Extract this and share it with the version in
  // `PropagatorState`. The correct constants might be different, since
  // sizeof(TaggedNode) is smaller in this version.
  class TaggedNodeReadyQueue {
   public:
    TaggedNodeReadyQueue() : front_index_(0) {}

    void push_back(const TaggedNode& node) { ready_.push_back(node); }
    TaggedNode front() const {
      DCHECK_LT(front_index_, ready_.size());
      return ready_[front_index_];
    }
    void pop_front() {
      DCHECK_LT(front_index_, ready_.size());
      front_index_++;
      if ((front_index_ == ready_.size()) || (front_index_ > kSpillThreshold)) {
        if (front_index_ == ready_.size()) {
          ready_.clear();
        } else {
          // Lots of unused entries at beginning of vector: move everything
          // down to start of vector.
          ready_.erase(ready_.begin(), ready_.begin() + front_index_);
        }
        front_index_ = 0;
      }
    }
    bool empty() const { return ready_.empty(); }

   private:
    // TODO(b/152925936): Re-evaluate these constants with current usage
    // patterns.
    static constexpr int kSpillThreshold = 16384;
    gtl::InlinedVector<TaggedNode, 16> ready_;
    int front_index_;
  };

  // TODO(b/152925936): Re-evaluate this constant with current usage patterns.
  typedef gtl::InlinedVector<TaggedNode, 8> TaggedNodeSeq;

  // Creates and adds a `TaggedNode` for each node in `roots` to `*ready`.
  void ActivateRoots(gtl::ArraySlice<const NodeItem*> roots,
                     TaggedNodeSeq* ready);

  // After processing the outputs, propagates the outputs to their dsts.
  // Contents of *outputs are left in an indeterminate state after
  // returning from this method.
  void PropagateOutputs(const TaggedNode& tagged_node, EntryVector* outputs,
                        TaggedNodeSeq* ready);

  // Returns an array of `Entry` objects corresponding to the inputs of
  // `tagged_node`.
  Entry* GetInputTensors(const TaggedNode& tagged_node) {
#if defined(THREAD_SANITIZER) || defined(DEBUG)
    // NOTE: This read of `pending_[...]` works around a limitation in TSAN.
    // To avoid false positive data race reports, we need to perform an atomic
    // object access that will establish the happens-before relation between
    // the write to input_tensors_ in `PropagateOutputs()` and the read in
    // `PrepareInputs()`.
    CHECK_EQ(pending_[tagged_node.node_item->node_id], 0);
#endif  // defined(THREAD_SANITIZER) || defined(DEBUG)
    return input_tensors_.data() + tagged_node.node_item->input_start;
  }

  FrameAndIter GetFrameAndIter(const TaggedNode& tagged_node) const {
    return {0, 0};
  }

  // Provide debugging output of the state of the executor.
  void DumpState();

  // For debugging/logging only.
  void MaybeMarkStarted(const TaggedNode& tagged_node) {
    // TODO(misard) Replace with a finer-grain enabling flag once we add better
    // optional debugging support.
    if (TF_PREDICT_FALSE(vlog_) && VLOG_IS_ON(1)) {
      mutex_lock l(mu_);
      (*active_)[tagged_node.node_item->node_id] = true;
    }
  }
  void MaybeMarkCompleted(const TaggedNode& tagged_node) {
    // TODO(misard) Replace with a finer-grain enabling flag once we add better
    // optional debugging support.
    if (TF_PREDICT_FALSE(vlog_) && VLOG_IS_ON(1)) {
      mutex_lock l(mu_);
      (*active_)[tagged_node.node_item->node_id] = false;
    }
  }

 private:
  SimplePropagatorState(const ImmutableExecutorState& immutable_state_,
                        int64_t step_id,
                        const ImmutableExecutorState::FrameInfo& finfo,
                        bool vlog);

  const ImmutableExecutorState& immutable_state_;
  const int64 step_id_;
  const bool vlog_;

  // The i-th node's j-th input is stored at
  // `input_tensors[impl_->nodes[i].input_start + j]`.
  //
  // NOTE: No need to protect input_tensors[i] by any locks because it
  // is resized once. Each element of input_tensors is written once by the
  // source node of an edge and is cleared by the destination of the same
  // edge. The destination node always runs after the source node, so there
  // is never concurrent access to the same entry.
  std::vector<Entry> input_tensors_;

  std::unique_ptr<std::atomic<int32>[]> pending_;

  // If `vlog_` is true, this stores a bit vector of active nodes, indexed by
  // node ID.
  mutex mu_;
  std::unique_ptr<std::vector<bool>> active_ TF_GUARDED_BY(mu_);

  const std::vector<const NodeItem*>* const nodes_;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_COMMON_RUNTIME_SIMPLE_PROPAGATOR_STATE_H_
