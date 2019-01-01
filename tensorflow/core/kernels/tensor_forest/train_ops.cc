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

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/kernels/tensor_forest/resources.h"
#include "tensorflow/core/lib/random/philox_random.h"
#include "tensorflow/core/lib/random/random.h"
#include "tensorflow/core/lib/random/simple_philox.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow/core/util/work_sharder.h"

namespace tensorflow {

class TensorForestTraverseTreeOp : public OpKernel {
 public:
  explicit TensorForestTraverseTreeOp(OpKernelConstruction* context)
      : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    TensorForestTreeResource* decision_tree_resource;
    OP_REQUIRES_OK(context, LookupResource(context, HandleFromInput(context, 0),
                                           &decision_tree_resource));
    mutex_lock l(*decision_tree_resource->get_mutex());
    core::ScopedUnref unref_me(decision_tree_resource);

    const Tensor* dense_features = nullptr;
    OP_REQUIRES_OK(context, context->input("dense_features", &dense_features));

    auto data_set = dense_features->matrix<float>();
    const int32 batch_size = dense_features->dim_size(0);

    Tensor* output_predictions = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, {batch_size},
                                                     &output_predictions));
    auto out = output_predictions->matrix<int32>();

    if (decision_tree_resource->get_size() <= 1) {
      out.setZero();
      return;
    }

    auto* worker_threads = context->device()->tensorflow_cpu_worker_threads();
    const int32 num_threads = worker_threads->num_threads;
    const int64 cost_per_traverse = 500;
    auto traverse = [&out, &data_set, decision_tree_resource](int64 start,
                                                              int64 end) {
      for (int example_id = start; example_id < end; ++example_id) {
        out(example_id) =
            decision_tree_resource->TraverseTree(example_id, &data_set);
      };
    };
    Shard(num_threads, worker_threads->workers, batch_size, cost_per_traverse,
          traverse);
  }
};

// Op for traversing the tree with each example, accumulating statistics, and
// outputting node ids that are ready to split.
class TensorForestProcessInputOp : public OpKernel {
 public:
  explicit TensorForestProcessInputOp(OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("random_seed", &random_seed_));
    OP_REQUIRES_OK(context, context->GetAttr("is_regression", &is_regression_));
    OP_REQUIRES_OK(context, context->GetAttr("split_node_after_samples",
                                             &split_node_after_samples_));
    OP_REQUIRES_OK(context, context->GetAttr("num_splits_to_consider",
                                             &num_splits_to_consider_));
    // Set up the random number generator.
    if (random_seed_ == 0) {
      single_rand_ = std::unique_ptr<random::PhiloxRandom>(
          new random::PhiloxRandom(random::New64()));
    } else {
      single_rand_ = std::unique_ptr<random::PhiloxRandom>(
          new random::PhiloxRandom(random_seed_));
    }

    rng_ = std::unique_ptr<random::SimplePhilox>(
        new random::SimplePhilox(single_rand_.get()));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor* dense_features_t = nullptr;
    OP_REQUIRES_OK(context,
                   context->input("dense_features", &dense_features_t));
    const Tensor* labels_t = nullptr;
    OP_REQUIRES_OK(context, context->input("labels", &labels_t));
    const Tensor* leaf_ids_t = nullptr;
    OP_REQUIRES_OK(context, context->input("leaf_ids", &leaf_ids_t));

    TensorForestFertileStatsResource* fertile_stats_resource;
    OP_REQUIRES_OK(context, LookupResource(context, HandleFromInput(context, 0),
                                           &fertile_stats_resource));

    mutex_lock l(*fertile_stats_resource->get_mutex());
    core::ScopedUnref unref_me(fertile_stats_resource);

    const auto dense_features = dense_features_t->matrix<float>();
    const auto labels = labels_t->matrix<float>();
    const auto leaf_ids = leaf_ids_t->matrix<float>();
    const int32 batch_size = dense_features_t->dim_size(0);
    number_of_total_feature_ = dense_features_t->dim_size(1);

    // Create one mutex per leaf. We need to protect access to leaf pointers,
    // so instead of grouping examples by leaf, we spread examples out among
    // threads to provide uniform work for each of them and protect access
    // with mutexes.
    std::unordered_map<int, std::unique_ptr<mutex>> locks;
    for (int i = 0; i < batch_size; ++i) {
      const int32 leaf_id = leaf_ids(i);
      if (locks.find(leaf_id) == locks.end()) {
        // TODO(gilberth): Consider using a memory pool for these.
        locks[leaf_id] = std::unique_ptr<mutex>(new mutex);
      }
    }

    const int32 label_dim =
        labels_t->shape().dims() <= 1
            ? 0
            : static_cast<int>(labels_t->shape().dim_size(1));
    const int32 num_targets = is_regression_ ? (std::max(1, label_dim)) : 1;

    // Ids of leaves that can split.
    std::unordered_set<int32> ready_to_split;
    mutex set_lock;

    auto worker_threads = context->device()->tensorflow_cpu_worker_threads();
    int num_threads = worker_threads->num_threads;

    // TODO(gilberth): This is a rough approximation based on measurements
    // from a digits run on local desktop.  Heuristics might be necessary
    // if it really matters that much.
    const int64 costPerUpdate = 1000;
    auto update = [this, &labels, &leaf_ids, &num_targets, &dense_features,
                   fertile_stats_resource, &locks, &set_lock,
                   &ready_to_split](int64 start, int64 end) {
      // Stores leaf_id, example_id for examples that are waiting
      // on another to finish.
      std::queue<std::tuple<int32, int32>> waiting;

      for (int example_id = start; example_id < end; example_id++) {
        const int32 leaf_id = leaf_ids(example_id);
        AddExample(num_targets, leaf_id, example_id, &dense_features, &labels,
                   &locks, &waiting, fertile_stats_resource);
      }

      while (!waiting.empty()) {
        int32 leaf_id, example_id;
        std::tie(leaf_id, example_id) = waiting.front();
        waiting.pop();
        AddExample(num_targets, leaf_id, example_id, &dense_features, &labels,
                   &locks, &waiting, fertile_stats_resource);
      }

      for (int example_id = start; example_id < end; example_id++) {
        const int32 leaf_id = leaf_ids(example_id);
        if (fertile_stats_resource->IsSlotFinished(
                leaf_id, split_node_after_samples_, num_splits_to_consider_)) {
          set_lock.lock();
          ready_to_split.insert(leaf_id);
          set_lock.unlock();
        }
      }
    };

    Shard(num_threads, worker_threads->workers, batch_size, costPerUpdate,
          update);

    Tensor* output_finished_t = nullptr;
    TensorShape output_shape;
    output_shape.AddDim(ready_to_split.size());
    OP_REQUIRES_OK(
        context, context->allocate_output(0, output_shape, &output_finished_t));
    auto output = output_finished_t->unaligned_flat<int32>();
    std::copy(ready_to_split.begin(), ready_to_split.end(), output.data());
  };

  void AddExample(const int32 num_targets, const int32 leaf_id,
                  const int32 example_id,
                  const TTypes<float>::ConstMatrix* dense_features,
                  const TTypes<float>::ConstMatrix* labels,
                  std::unordered_map<int, std::unique_ptr<mutex>>* locks,
                  std::queue<std::tuple<int32, int32>>* waiting,
                  TensorForestFertileStatsResource* fertile_stats_resource) {
    // Try to update a leaf's stats by acquiring its lock.  If it can't be
    // acquired, put it in a waiting queue to come back to later and try the
    // next one.  Once all leaf_ids have been visited, cycle through the waiting
    // ids until they're gone.
    const std::unique_ptr<mutex>& leaf_lock = (*locks)[leaf_id];
    if (leaf_lock->try_lock()) {
      if (fertile_stats_resource->IsSlotInitialized(leaf_id,
                                                    num_splits_to_consider_)) {
        fertile_stats_resource->UpdateSlotStats(is_regression_, leaf_id,
                                                num_targets, example_id,
                                                dense_features, labels);
      } else {
        int32 feature_id;
        {
          mutex_lock lock(mu_);
          feature_id = rng_->Uniform(number_of_total_feature_);
        }
        auto bias = (*dense_features)(example_id, feature_id);

        fertile_stats_resource->AddSplitToSlot(leaf_id, feature_id, bias,
                                               example_id, num_targets,
                                               dense_features, labels);
      }
      leaf_lock->unlock();
    } else {
      waiting->emplace(leaf_id, example_id);
    }
  };

 private:
  int32 random_seed_;
  int32 number_of_total_feature_;
  int32 num_splits_to_consider_;
  int32 split_node_after_samples_;
  bool is_regression_;
  // Mutex for using random number generator.
  mutable mutex mu_;
  std::unique_ptr<random::PhiloxRandom> single_rand_;
  std::unique_ptr<random::SimplePhilox> rng_;
};

// Op for growing finished nodes.
class TensorForestGrowTreeOp : public OpKernel {
 public:
  explicit TensorForestGrowTreeOp(OpKernelConstruction* context)
      : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    TensorForestTreeResource* tree_resource;
    OP_REQUIRES_OK(context, LookupResource(context, HandleFromInput(context, 0),
                                           &tree_resource));

    TensorForestFertileStatsResource* fertile_stats_resource;
    OP_REQUIRES_OK(context, LookupResource(context, HandleFromInput(context, 1),
                                           &fertile_stats_resource));

    mutex_lock l1(*fertile_stats_resource->get_mutex());
    mutex_lock l2(*tree_resource->get_mutex());

    core::ScopedUnref unref_stats(fertile_stats_resource);
    core::ScopedUnref unref_tree(tree_resource);

    const Tensor* finished_nodes_t = nullptr;
    OP_REQUIRES_OK(context,
                   context->input("finished_nodes", &finished_nodes_t));

    auto finished = finished_nodes_t->unaligned_flat<int32>();

    const int32 batch_size = finished_nodes_t->dim_size(0);

    for (int32 i = 0; i < batch_size; i++) {
      auto node = finished(i);
      auto slot = fertile_stats_resource->get_slot(node);
      std::unique_ptr<tensor_forest::SplitCandidate> best_candidate(
          new tensor_forest::SplitCandidate);
      bool found =
          fertile_stats_resource->BestSplitFromSlot(slot, best_candidate.get());
      if (found) {
        std::vector<int32> new_children;
        tree_resource->SplitNode(node, slot, best_candidate.get(),
                                 &new_children);
        for (auto new_node : new_children)
          fertile_stats_resource->Allocate(new_node);
        //
        // We are done with best, so it is now safe to clear node.
        fertile_stats_resource->Clear(node);
        DCHECK(tree_resource->NodeHasLeaf(node) == false)
            << "Node:" << node << " should have being splitted";
      } else {  // reset
        fertile_stats_resource->ResetSplitStats(node);
      }
    }
  }
};

REGISTER_KERNEL_BUILDER(Name("TensorForestProcessInput").Device(DEVICE_CPU),
                        TensorForestProcessInputOp);

}  // namespace tensorflow
