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
#include <queue>

#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/kernels/boosted_trees/boosted_trees.pb.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/protobuf.h"

namespace tensorflow {

// Try to update a leaf's stats by acquiring its lock.  If it can't be
// acquired, put it in a waiting queue to come back to later and try the next
// one.  Once all leaf_ids have been visited, cycle through the waiting ids
// until they're gone.
void UpdateStats(FertileStatsResource* fertile_stats_resource,
                 const std::unique_ptr<TensorDataSet>& data,
                 const TensorInputTarget& target, int num_targets,
                 const Tensor& leaf_ids_tensor,
                 std::unordered_map<int32, std::unique_ptr<mutex>>* locks,
                 mutex* set_lock, int32 start, int32 end,
                 std::unordered_set<int32>* ready_to_split) {
  const auto leaf_ids = leaf_ids_tensor.unaligned_flat<int32>();

  // Stores leaf_id, leaf_depth, example_id for examples that are waiting
  // on another to finish.
  std::queue<std::tuple<int32, int32>> waiting;

  int32 i = start;
  while (i < end || !waiting.empty()) {
    int32 leaf_id;
    int32 example_id;
    bool was_waiting = false;
    if (i >= end) {
      std::tie(leaf_id, example_id) = waiting.front();
      waiting.pop();
      was_waiting = true;
    } else {
      leaf_id = leaf_ids(i);
      example_id = i;
      ++i;
    }
    const std::unique_ptr<mutex>& leaf_lock = (*locks)[leaf_id];
    if (was_waiting) {
      leaf_lock->lock();
    } else {
      if (!leaf_lock->try_lock()) {
        waiting.emplace(leaf_id, example_id);
        continue;
      }
    }

    bool is_finished;
    fertile_stats_resource->AddExampleToStatsAndInitialize(
        data, &target, {example_id}, leaf_id, &is_finished);
    leaf_lock->unlock();
    if (is_finished) {
      set_lock->lock();
      ready_to_split->insert(leaf_id);
      set_lock->unlock();
    }
  }
}

// Op for traversing the tree with each example, accumulating statistics, and
// outputting node ids that are ready to split.
class TensorForestProcessInputOp : public OpKernel {
 public:
  explicit TensorForestProcessInputOp(OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("random_seed", &random_seed_));
  }

  void Compute(OpKernelContext* context) override {
    using gtl::FindOrNull;

    const Tensor& input_data = context->input(0);
    const Tensor& input_labels = context->input(1);
    const Tensor& leaf_ids_tensor = context->input(2);

    FertileStatsResource* fertile_stats_resource;
    OP_REQUIRES_OK(context, LookupResource(context, HandleFromInput(context, 1),
                                           &fertile_stats_resource));
    DecisionTreeResource* tree_resource;
    OP_REQUIRES_OK(context, LookupResource(context, HandleFromInput(context, 0),
                                           &tree_resource));
    mutex_lock l1(*fertile_stats_resource->get_mutex());
    mutex_lock l2(*tree_resource->get_mutex());

    core::ScopedUnref unref_stats(fertile_stats_resource);
    core::ScopedUnref unref_tree(tree_resource);

    const int32 num_data = data_set->NumItems();

    auto worker_threads = context->device()->tensorflow_cpu_worker_threads();
    int num_threads = worker_threads->num_threads;

    const auto leaf_ids = leaf_ids_tensor.unaligned_flat<int32>();

    // Create one mutex per leaf. We need to protect access to leaf pointers,
    // so instead of grouping examples by leaf, we spread examples out among
    // threads to provide uniform work for each of them and protect access
    // with mutexes.
    std::unordered_map<int, std::unique_ptr<mutex>> locks;
    for (int i = 0; i < num_data; ++i) {
      const int32 id = leaf_ids(i);
      if (FindOrNull(locks, id) == nullptr) {
        // TODO(gilberth): Consider using a memory pool for these.
        locks[id] = std::unique_ptr<mutex>(new mutex);
      }
    }

    const int32 label_dim =
        input_labels.shape().dims() <= 1
            ? 0
            : static_cast<int>(input_labels.shape().dim_size(1));
    const int32 num_targets =
        param_proto_.is_regression() ? (std::max(1, label_dim)) : 1;

    // Ids of leaves that can split.
    std::unordered_set<int32> ready_to_split;
    mutex set_lock;

    // TODO(gilberth): This is a rough approximation based on measurements
    // from a digits run on local desktop.  Heuristics might be necessary
    // if it really matters that much.
    const int64 costPerUpdate = 1000;
    auto update = [this, &target, &leaf_ids_tensor, &num_targets, &data_set,
                   fertile_stats_resource, &locks, &set_lock, &ready_to_split,
                   num_data](int64 start, int64 end) {
      CHECK(start <= end);
      CHECK(end <= num_data);
      UpdateStats(fertile_stats_resource, data_set, target, num_targets,
                  leaf_ids_tensor, &locks, &set_lock, static_cast<int32>(start),
                  static_cast<int32>(end), &ready_to_split);
    };

    Shard(num_threads, worker_threads->workers, num_data, costPerUpdate,
          update);

    Tensor* output_finished_t = nullptr;
    TensorShape output_shape;
    output_shape.AddDim(ready_to_split.size());
    OP_REQUIRES_OK(
        context, context->allocate_output(0, output_shape, &output_finished_t));
    auto output = output_finished_t->unaligned_flat<int32>();
    std::copy(ready_to_split.begin(), ready_to_split.end(), output.data());
  }

 private:
  const int32 random_seed_;
};

// Op for growing finished nodes.
class GrowTreeOp : public OpKernel {
 public:
  explicit GrowTreeOp(OpKernelConstruction* context) : OpKernel(context) {
    string serialized_params;
    OP_REQUIRES_OK(context, context->GetAttr("params", &serialized_params));
    ParseProtoUnlimited(&param_proto_, serialized_params);
  }

  void Compute(OpKernelContext* context) override {
    FertileStatsResource* fertile_stats_resource;
    OP_REQUIRES_OK(context, LookupResource(context, HandleFromInput(context, 1),
                                           &fertile_stats_resource));
    DecisionTreeResource* tree_resource;
    OP_REQUIRES_OK(context, LookupResource(context, HandleFromInput(context, 0),
                                           &tree_resource));
    mutex_lock l1(*fertile_stats_resource->get_mutex());
    mutex_lock l2(*tree_resource->get_mutex());

    core::ScopedUnref unref_stats(fertile_stats_resource);
    core::ScopedUnref unref_tree(tree_resource);

    const Tensor& finished_nodes = context->input(2);

    const auto finished = finished_nodes.unaligned_flat<int32>();

    const int32 num_nodes =
        static_cast<int32>(finished_nodes.shape().dim_size(0));

    for (int i = 0;
         i < num_nodes &&
         tree_resource->decision_tree().decision_tree().nodes_size() <
             param_proto_.max_nodes();
         ++i) {
      const int32 node = finished(i);
      std::unique_ptr<SplitCandidate> best(new SplitCandidate);
      int32 parent_depth;
      bool found =
          fertile_stats_resource->BestSplit(node, best.get(), &parent_depth);
      if (found) {
        std::vector<int32> new_children;
        tree_resource->SplitNode(node, best.get(), &new_children);
        fertile_stats_resource->Allocate(parent_depth, new_children);
        // We are done with best, so it is now safe to clear node.
        fertile_stats_resource->Clear(node);
        CHECK(tree_resource->get_mutable_tree_node(node)->has_leaf() == false);
      } else {  // reset
        fertile_stats_resource->ResetSplitStats(node, parent_depth);
      }
    }
  }

 private:
  tensorforest::TensorForestDataSpec input_spec_;
  TensorForestParams param_proto_;
};

REGISTER_KERNEL_BUILDER(Name("TensorForestProcessInput").Device(DEVICE_CPU),
                        TensorForestProcessInputOp);

}  // namespace tensorflow
