// Copyright 2017 The TensorFlow Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================
#include <queue>

#include "tensorflow/contrib/tensor_forest/kernels/data_spec.h"
#include "tensorflow/contrib/tensor_forest/kernels/v4/decision-tree-resource.h"
#include "tensorflow/contrib/tensor_forest/kernels/v4/fertile-stats-resource.h"
#include "tensorflow/contrib/tensor_forest/kernels/v4/input_data.h"
#include "tensorflow/contrib/tensor_forest/kernels/v4/input_target.h"
#include "tensorflow/contrib/tensor_forest/kernels/v4/params.h"
#include "tensorflow/contrib/tensor_forest/proto/fertile_stats.pb.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/lib/gtl/map_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/work_sharder.h"

namespace tensorflow {
namespace tensorforest {

using gtl::FindOrNull;

// Creates a stats variable.
class CreateFertileStatsVariableOp : public OpKernel {
 public:
  explicit CreateFertileStatsVariableOp(OpKernelConstruction* context)
      : OpKernel(context) {
    string serialized_params;
    OP_REQUIRES_OK(context, context->GetAttr("params", &serialized_params));
    ParseProtoUnlimited(&param_proto_, serialized_params);
  }

  void Compute(OpKernelContext* context) override {
    const Tensor* stats_config_t;
    OP_REQUIRES_OK(context, context->input("stats_config", &stats_config_t));
    OP_REQUIRES(context, TensorShapeUtils::IsScalar(stats_config_t->shape()),
                errors::InvalidArgument("Stats config must be a scalar."));
    auto* result = new FertileStatsResource(param_proto_);
    FertileStats stats;
    if (!ParseProtoUnlimited(&stats, stats_config_t->scalar<string>()())) {
      result->Unref();
      OP_REQUIRES(context, false,
                  errors::InvalidArgument("Unable to parse stats config."));
    }

    result->ExtractFromProto(stats);
    result->MaybeInitialize();

    // Only create one, if one does not exist already. Report status for all
    // other exceptions.
    auto status = CreateResource(context, HandleFromInput(context, 0), result);
    if (!status.ok() && status.code() != tensorflow::error::ALREADY_EXISTS) {
      OP_REQUIRES(context, false, status);
    }
  }

 private:
  TensorForestParams param_proto_;
};

// Op for serializing a model.
class FertileStatsSerializeOp : public OpKernel {
 public:
  explicit FertileStatsSerializeOp(OpKernelConstruction* context)
      : OpKernel(context) {
    string serialized_params;
    OP_REQUIRES_OK(context, context->GetAttr("params", &serialized_params));
    ParseProtoUnlimited(&param_proto_, serialized_params);
  }

  void Compute(OpKernelContext* context) override {
    FertileStatsResource* fertile_stats_resource;
    OP_REQUIRES_OK(context, LookupResource(context, HandleFromInput(context, 0),
                                           &fertile_stats_resource));
    mutex_lock l(*fertile_stats_resource->get_mutex());
    core::ScopedUnref unref_me(fertile_stats_resource);
    Tensor* output_config_t = nullptr;
    OP_REQUIRES_OK(
        context, context->allocate_output(0, TensorShape(), &output_config_t));

    FertileStats stats;
    fertile_stats_resource->PackToProto(&stats);
    output_config_t->scalar<string>()() = stats.SerializeAsString();
  }

 private:
  TensorForestParams param_proto_;
};

// Op for deserializing a stats variable from a checkpoint.
class FertileStatsDeserializeOp : public OpKernel {
 public:
  explicit FertileStatsDeserializeOp(OpKernelConstruction* context)
      : OpKernel(context) {
    string serialized_params;
    OP_REQUIRES_OK(context, context->GetAttr("params", &serialized_params));
    ParseProtoUnlimited(&param_proto_, serialized_params);
  }

  void Compute(OpKernelContext* context) override {
    FertileStatsResource* fertile_stats_resource;
    OP_REQUIRES_OK(context, LookupResource(context, HandleFromInput(context, 0),
                                           &fertile_stats_resource));
    mutex_lock l(*fertile_stats_resource->get_mutex());
    core::ScopedUnref unref_me(fertile_stats_resource);

    const Tensor* stats_config_t;
    OP_REQUIRES_OK(context, context->input("stats_config", &stats_config_t));
    OP_REQUIRES(context, TensorShapeUtils::IsScalar(stats_config_t->shape()),
                errors::InvalidArgument("Stats config must be a scalar."));
    // Deallocate all the previous objects on the resource.
    fertile_stats_resource->Reset();
    FertileStats stats;
    OP_REQUIRES(context,
                ParseProtoUnlimited(&stats, stats_config_t->scalar<string>()()),
                errors::InvalidArgument("Unable to parse stats config."));

    fertile_stats_resource->ExtractFromProto(stats);
    fertile_stats_resource->MaybeInitialize();
  }

 private:
  TensorForestParams param_proto_;
};

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

// Update leaves from start through end in the leaf_examples iterator.
void UpdateStatsCollated(
    FertileStatsResource* fertile_stats_resource,
    DecisionTreeResource* tree_resource,
    const std::unique_ptr<TensorDataSet>& data, const TensorInputTarget& target,
    int num_targets,
    const std::unordered_map<int32, std::vector<int>>& leaf_examples,
    mutex* set_lock, int32 start, int32 end,
    std::unordered_set<int32>* ready_to_split) {
  auto it = leaf_examples.begin();
  std::advance(it, start);
  auto end_it = leaf_examples.begin();
  std::advance(end_it, end);
  while (it != end_it) {
    int32 leaf_id = it->first;
    bool is_finished;
    fertile_stats_resource->AddExampleToStatsAndInitialize(
        data, &target, it->second, leaf_id, &is_finished);
    if (is_finished) {
      set_lock->lock();
      ready_to_split->insert(leaf_id);
      set_lock->unlock();
    }
    ++it;
  }
}

// Op for traversing the tree with each example, accumulating statistics, and
// outputting node ids that are ready to split.
class ProcessInputOp : public OpKernel {
 public:
  explicit ProcessInputOp(OpKernelConstruction* context) : OpKernel(context) {
    string serialized_params;
    OP_REQUIRES_OK(context, context->GetAttr("params", &serialized_params));
    ParseProtoUnlimited(&param_proto_, serialized_params);

    OP_REQUIRES_OK(context, context->GetAttr("random_seed", &random_seed_));

    string serialized_proto;
    OP_REQUIRES_OK(context, context->GetAttr("input_spec", &serialized_proto));
    input_spec_.ParseFromString(serialized_proto);

    data_set_ = std::unique_ptr<TensorDataSet>(
        new TensorDataSet(input_spec_, random_seed_));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& input_data = context->input(2);
    const Tensor& sparse_input_indices = context->input(3);
    const Tensor& sparse_input_values = context->input(4);
    const Tensor& sparse_input_shape = context->input(5);
    const Tensor& input_labels = context->input(6);
    const Tensor& input_weights = context->input(7);
    const Tensor& leaf_ids_tensor = context->input(8);

    data_set_->set_input_tensors(input_data, sparse_input_indices,
                                 sparse_input_values, sparse_input_shape);

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

    const int32 num_data = data_set_->NumItems();
    auto worker_threads = context->device()->tensorflow_cpu_worker_threads();
    int num_threads = worker_threads->num_threads;

    const auto leaf_ids = leaf_ids_tensor.unaligned_flat<int32>();

    // Create one mutex per leaf. We need to protect access to leaf pointers,
    // so instead of grouping examples by leaf, we spread examples out among
    // threads to provide uniform work for each of them and protect access
    // with mutexes.
    std::unordered_map<int, std::unique_ptr<mutex>> locks;
    std::unordered_map<int32, std::vector<int>> leaf_examples;
    if (param_proto_.collate_examples()) {
      for (int i = 0; i < num_data; ++i) {
        leaf_examples[leaf_ids(i)].push_back(i);
      }
    } else {
      for (int i = 0; i < num_data; ++i) {
        const int32 id = leaf_ids(i);
        if (FindOrNull(locks, id) == nullptr) {
          // TODO(gilberth): Consider using a memory pool for these.
          locks[id] = std::unique_ptr<mutex>(new mutex);
        }
      }
    }

    const int32 num_leaves = leaf_examples.size();
    const int32 label_dim =
        input_labels.shape().dims() <= 1
            ? 0
            : static_cast<int>(input_labels.shape().dim_size(1));
    const int32 num_targets =
        param_proto_.is_regression() ? (std::max(1, label_dim)) : 1;

    // Ids of leaves that can split.
    std::unordered_set<int32> ready_to_split;
    mutex set_lock;

    TensorInputTarget target(input_labels, input_weights, num_targets);

    // TODO(gilberth): This is a rough approximation based on measurements
    // from a digits run on local desktop.  Heuristics might be necessary
    // if it really matters that much.
    const int64 costPerUpdate = 1000;
    auto update = [this, &target, &leaf_ids_tensor, &num_targets,
                   fertile_stats_resource, &locks, &set_lock, &ready_to_split,
                   num_data](int64 start, int64 end) {
      CHECK(start <= end);
      CHECK(end <= num_data);
      UpdateStats(fertile_stats_resource, data_set_, target, num_targets,
                  leaf_ids_tensor, &locks, &set_lock, static_cast<int32>(start),
                  static_cast<int32>(end), &ready_to_split);
    };

    auto update_collated = [this, &target, &num_targets, fertile_stats_resource,
                            tree_resource, &leaf_examples, &set_lock,
                            &ready_to_split,
                            num_leaves](int64 start, int64 end) {
      CHECK(start <= end);
      CHECK(end <= num_leaves);
      UpdateStatsCollated(fertile_stats_resource, tree_resource, data_set_,
                          target, num_targets, leaf_examples, &set_lock,
                          static_cast<int32>(start), static_cast<int32>(end),
                          &ready_to_split);
    };

    if (param_proto_.collate_examples()) {
      Shard(num_threads, worker_threads->workers, num_leaves, costPerUpdate,
            update_collated);
    } else {
      Shard(num_threads, worker_threads->workers, num_data, costPerUpdate,
            update);
    }

    Tensor* output_finished_t = nullptr;
    TensorShape output_shape;
    output_shape.AddDim(ready_to_split.size());
    OP_REQUIRES_OK(
        context, context->allocate_output(0, output_shape, &output_finished_t));
    auto output = output_finished_t->unaligned_flat<int32>();
    std::copy(ready_to_split.begin(), ready_to_split.end(), output.data());
  }

 private:
  int32 random_seed_;
  tensorforest::TensorForestDataSpec input_spec_;
  std::unique_ptr<TensorDataSet> data_set_;
  TensorForestParams param_proto_;
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

    // This op takes so little of the time for one batch that it isn't worth
    // threading this.
    for (int i = 0;
         i < num_nodes &&
         tree_resource->decision_tree().decision_tree().nodes_size() <
             param_proto_.max_nodes();
         ++i) {
      const int32 node = finished(i);
      std::unique_ptr<SplitCandidate> best(new SplitCandidate);
      int32 parent_depth;
      // TODO(gilberth): Pushing these to an output would allow the complete
      // decoupling of tree from resource.
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

void FinalizeLeaf(bool is_regression, bool drop_final_class,
                  const std::unique_ptr<LeafModelOperator>& leaf_op,
                  decision_trees::Leaf* leaf) {
  // regression models are already stored in leaf in normalized form.
  if (is_regression) {
    return;
  }

  // TODO(gilberth): Calculate the leaf's sum.
  float sum = 0;
  LOG(FATAL) << "FinalizeTreeOp is disabled for now.";
  if (sum <= 0.0) {
    LOG(WARNING) << "Leaf with sum " << sum << " has stats "
                 << leaf->ShortDebugString();
    return;
  }

  if (leaf->has_vector()) {
    for (int i = 0; i < leaf->vector().value_size(); i++) {
      auto* v = leaf->mutable_vector()->mutable_value(i);
      v->set_float_value(v->float_value() / sum);
    }
    if (drop_final_class) {
      leaf->mutable_vector()->mutable_value()->RemoveLast();
    }
    return;
  }

  if (leaf->has_sparse_vector()) {
    for (auto& it : *leaf->mutable_sparse_vector()->mutable_sparse_value()) {
      it.second.set_float_value(it.second.float_value() / sum);
    }
    return;
  }

  LOG(FATAL) << "Unknown leaf type in " << leaf->DebugString();
}

// Op for finalizing a tree at the end of training.
class FinalizeTreeOp : public OpKernel {
 public:
  explicit FinalizeTreeOp(OpKernelConstruction* context) : OpKernel(context) {
    string serialized_params;
    OP_REQUIRES_OK(context, context->GetAttr("params", &serialized_params));
    ParseProtoUnlimited(&param_proto_, serialized_params);

    model_op_ = LeafModelOperatorFactory::CreateLeafModelOperator(param_proto_);
  }

  void Compute(OpKernelContext* context) override {
    DecisionTreeResource* tree_resource;
    OP_REQUIRES_OK(context, LookupResource(context, HandleFromInput(context, 0),
                                           &tree_resource));
    FertileStatsResource* fertile_stats_resource;
    OP_REQUIRES_OK(context, LookupResource(context, HandleFromInput(context, 1),
                                           &fertile_stats_resource));

    mutex_lock l1(*fertile_stats_resource->get_mutex());
    mutex_lock l2(*tree_resource->get_mutex());

    core::ScopedUnref unref_me(tree_resource);
    core::ScopedUnref unref_stats(fertile_stats_resource);

    // TODO(thomaswc): Add threads
    int num_nodes = tree_resource->decision_tree().decision_tree().nodes_size();
    for (int i = 0; i < num_nodes; i++) {
      auto* node = tree_resource->mutable_decision_tree()
                       ->mutable_decision_tree()
                       ->mutable_nodes(i);
      if (node->has_leaf()) {
        FinalizeLeaf(param_proto_.is_regression(),
                     param_proto_.drop_final_class(), model_op_,
                     node->mutable_leaf());
      }
    }
  }

 private:
  std::unique_ptr<LeafModelOperator> model_op_;
  TensorForestParams param_proto_;
};

REGISTER_RESOURCE_HANDLE_KERNEL(FertileStatsResource);

REGISTER_KERNEL_BUILDER(Name("FertileStatsIsInitializedOp").Device(DEVICE_CPU),
                        IsResourceInitialized<FertileStatsResource>);

REGISTER_KERNEL_BUILDER(Name("CreateFertileStatsVariable").Device(DEVICE_CPU),
                        CreateFertileStatsVariableOp);

REGISTER_KERNEL_BUILDER(Name("FertileStatsSerialize").Device(DEVICE_CPU),
                        FertileStatsSerializeOp);

REGISTER_KERNEL_BUILDER(Name("FertileStatsDeserialize").Device(DEVICE_CPU),
                        FertileStatsDeserializeOp);

REGISTER_KERNEL_BUILDER(Name("ProcessInputV4").Device(DEVICE_CPU),
                        ProcessInputOp);

REGISTER_KERNEL_BUILDER(Name("GrowTreeV4").Device(DEVICE_CPU), GrowTreeOp);

REGISTER_KERNEL_BUILDER(Name("FinalizeTree").Device(DEVICE_CPU),
                        FinalizeTreeOp);

}  // namespace tensorforest
}  // namespace tensorflow
