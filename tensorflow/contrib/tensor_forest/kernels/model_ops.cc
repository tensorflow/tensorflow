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
#include <functional>
#include "tensorflow/contrib/decision_trees/proto/generic_tree_model.pb.h"
#include "tensorflow/contrib/decision_trees/proto/generic_tree_model_extensions.pb.h"
#include "tensorflow/contrib/tensor_forest/kernels/data_spec.h"
#include "tensorflow/contrib/tensor_forest/kernels/v4/decision-tree-resource.h"
#include "tensorflow/contrib/tensor_forest/kernels/v4/input_data.h"
#include "tensorflow/contrib/tensor_forest/proto/tensor_forest_params.pb.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/work_sharder.h"

namespace tensorflow {
namespace tensorforest {

// Creates a tree  variable.
class CreateTreeVariableOp : public OpKernel {
 public:
  explicit CreateTreeVariableOp(OpKernelConstruction* context)
      : OpKernel(context) {
    string serialized_params;
    OP_REQUIRES_OK(context, context->GetAttr("params", &serialized_params));
    ParseProtoUnlimited(&param_proto_, serialized_params);
  }

  void Compute(OpKernelContext* context) override {
    const Tensor* tree_config_t;
    OP_REQUIRES_OK(context, context->input("tree_config", &tree_config_t));
    OP_REQUIRES(context, TensorShapeUtils::IsScalar(tree_config_t->shape()),
                errors::InvalidArgument("Tree config must be a scalar."));

    auto* result = new DecisionTreeResource(param_proto_);
    if (!ParseProtoUnlimited(result->mutable_decision_tree(),
                             tree_config_t->scalar<string>()())) {
      result->Unref();
      OP_REQUIRES(context, false,
                  errors::InvalidArgument("Unable to parse tree  config."));
    }

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
class TreeSerializeOp : public OpKernel {
 public:
  explicit TreeSerializeOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    DecisionTreeResource* decision_tree_resource;
    OP_REQUIRES_OK(context, LookupResource(context, HandleFromInput(context, 0),
                                           &decision_tree_resource));
    mutex_lock l(*decision_tree_resource->get_mutex());
    core::ScopedUnref unref_me(decision_tree_resource);
    Tensor* output_config_t = nullptr;
    OP_REQUIRES_OK(
        context, context->allocate_output(0, TensorShape(), &output_config_t));
    output_config_t->scalar<string>()() =
        decision_tree_resource->decision_tree().SerializeAsString();
  }
};

// Op for deserializing a tree variable from a checkpoint.
class TreeDeserializeOp : public OpKernel {
 public:
  explicit TreeDeserializeOp(OpKernelConstruction* context)
      : OpKernel(context) {
    string serialized_params;
    OP_REQUIRES_OK(context, context->GetAttr("params", &serialized_params));
    ParseProtoUnlimited(&param_proto_, serialized_params);
  }

  void Compute(OpKernelContext* context) override {
    DecisionTreeResource* decision_tree_resource;
    auto handle = HandleFromInput(context, 0);
    OP_REQUIRES_OK(context,
                   LookupResource(context, handle, &decision_tree_resource));
    mutex_lock l(*decision_tree_resource->get_mutex());
    core::ScopedUnref unref_me(decision_tree_resource);

    const Tensor* tree_config_t;
    OP_REQUIRES_OK(context, context->input("tree_config", &tree_config_t));
    OP_REQUIRES(context, TensorShapeUtils::IsScalar(tree_config_t->shape()),
                errors::InvalidArgument("Tree config must be a scalar."));
    // Deallocate all the previous objects on the resource.
    decision_tree_resource->Reset();
    decision_trees::Model* config =
        decision_tree_resource->mutable_decision_tree();
    OP_REQUIRES(context,
                ParseProtoUnlimited(config, tree_config_t->scalar<string>()()),
                errors::InvalidArgument("Unable to parse tree  config."));
    decision_tree_resource->MaybeInitialize();
  }

 private:
  TensorForestParams param_proto_;
};

// Op for getting tree size.
class TreeSizeOp : public OpKernel {
 public:
  explicit TreeSizeOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    DecisionTreeResource* decision_tree_resource;
    OP_REQUIRES_OK(context, LookupResource(context, HandleFromInput(context, 0),
                                           &decision_tree_resource));
    mutex_lock l(*decision_tree_resource->get_mutex());
    core::ScopedUnref unref_me(decision_tree_resource);
    Tensor* output_t = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, TensorShape(), &output_t));
    output_t->scalar<int32>()() =
        decision_tree_resource->decision_tree().decision_tree().nodes_size();
  }
};

void TraverseTree(const DecisionTreeResource* tree_resource,
                  const std::unique_ptr<TensorDataSet>& data, int32 start,
                  int32 end,
                  const std::function<void(int32, int32)>& set_leaf_id,
                  std::vector<TreePath>* tree_paths) {
  for (int i = start; i < end; ++i) {
    const int32 id = tree_resource->TraverseTree(
        data, i, nullptr,
        (tree_paths == nullptr) ? nullptr : &(*tree_paths)[i]);
    set_leaf_id(i, id);
  }
}

// Op for tree inference.
class TreePredictionsV4Op : public OpKernel {
 public:
  explicit TreePredictionsV4Op(OpKernelConstruction* context)
      : OpKernel(context) {
    string serialized_params;
    OP_REQUIRES_OK(context, context->GetAttr("params", &serialized_params));
    ParseProtoUnlimited(&param_proto_, serialized_params);

    string serialized_proto;
    OP_REQUIRES_OK(context, context->GetAttr("input_spec", &serialized_proto));
    input_spec_.ParseFromString(serialized_proto);
    model_op_ = LeafModelOperatorFactory::CreateLeafModelOperator(param_proto_);
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& input_data = context->input(1);
    const Tensor& sparse_input_indices = context->input(2);
    const Tensor& sparse_input_values = context->input(3);
    const Tensor& sparse_input_shape = context->input(4);

    std::unique_ptr<TensorDataSet> data_set(new TensorDataSet(input_spec_, 0));
    data_set->set_input_tensors(input_data, sparse_input_indices,
                                sparse_input_values, sparse_input_shape);

    DecisionTreeResource* decision_tree_resource;
    OP_REQUIRES_OK(context, LookupResource(context, HandleFromInput(context, 0),
                                           &decision_tree_resource));
    mutex_lock l(*decision_tree_resource->get_mutex());
    core::ScopedUnref unref_me(decision_tree_resource);

    const int num_data = data_set->NumItems();
    const int32 num_outputs = param_proto_.num_outputs();

    Tensor* output_predictions = nullptr;
    TensorShape output_shape;
    output_shape.AddDim(num_data);
    output_shape.AddDim(num_outputs);
    OP_REQUIRES_OK(context, context->allocate_output(0, output_shape,
                                                     &output_predictions));
    TTypes<float, 2>::Tensor out = output_predictions->tensor<float, 2>();

    std::vector<TreePath> tree_paths(
        param_proto_.inference_tree_paths() ? num_data : 0);

    auto worker_threads = context->device()->tensorflow_cpu_worker_threads();
    int num_threads = worker_threads->num_threads;
    const int64 costPerTraverse = 500;
    auto traverse = [this, &out, &data_set, decision_tree_resource, num_data,
                     &tree_paths](int64 start, int64 end) {
      CHECK(start <= end);
      CHECK(end <= num_data);
      TraverseTree(decision_tree_resource, data_set, static_cast<int32>(start),
                   static_cast<int32>(end),
                   std::bind(&TreePredictionsV4Op::set_output_value, this,
                             std::placeholders::_1, std::placeholders::_2,
                             decision_tree_resource, &out),
                   param_proto_.inference_tree_paths() ? &tree_paths : nullptr);
    };
    Shard(num_threads, worker_threads->workers, num_data, costPerTraverse,
          traverse);

    Tensor* output_tree_paths = nullptr;
    TensorShape output_paths_shape;
    output_paths_shape.AddDim(param_proto_.inference_tree_paths() ? num_data
                                                                  : 0);
    OP_REQUIRES_OK(context, context->allocate_output(1, output_paths_shape,
                                                     &output_tree_paths));
    auto out_paths = output_tree_paths->unaligned_flat<string>();

    // TODO(gilberth): If this slows down inference too much, consider having
    // a filter that only serializes paths for the predicted label that we're
    // interested in.
    for (int i = 0; i < tree_paths.size(); ++i) {
      out_paths(i) = tree_paths[i].SerializeAsString();
    }
  }

  void set_output_value(int32 i, int32 id,
                        DecisionTreeResource* decision_tree_resource,
                        TTypes<float, 2>::Tensor* out) {
    const decision_trees::Leaf& leaf = decision_tree_resource->get_leaf(id);

    float sum = 0;
    for (int j = 0; j < param_proto_.num_outputs(); ++j) {
      const float count = model_op_->GetOutputValue(leaf, j);
      (*out)(i, j) = count;
      sum += count;
    }

    if (!param_proto_.is_regression() && sum > 0 && sum != 1) {
      for (int j = 0; j < param_proto_.num_outputs(); ++j) {
        (*out)(i, j) /= sum;
      }
    }
  }

 private:
  tensorforest::TensorForestDataSpec input_spec_;
  std::unique_ptr<LeafModelOperator> model_op_;
  TensorForestParams param_proto_;
};

// Outputs leaf ids for the given examples.
class TraverseTreeV4Op : public OpKernel {
 public:
  explicit TraverseTreeV4Op(OpKernelConstruction* context) : OpKernel(context) {
    string serialized_params;
    OP_REQUIRES_OK(context, context->GetAttr("params", &serialized_params));
    ParseProtoUnlimited(&param_proto_, serialized_params);

    string serialized_proto;
    OP_REQUIRES_OK(context, context->GetAttr("input_spec", &serialized_proto));
    input_spec_.ParseFromString(serialized_proto);
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& input_data = context->input(1);
    const Tensor& sparse_input_indices = context->input(2);
    const Tensor& sparse_input_values = context->input(3);
    const Tensor& sparse_input_shape = context->input(4);

    std::unique_ptr<TensorDataSet> data_set(new TensorDataSet(input_spec_, 0));
    data_set->set_input_tensors(input_data, sparse_input_indices,
                                sparse_input_values, sparse_input_shape);

    DecisionTreeResource* decision_tree_resource;
    OP_REQUIRES_OK(context, LookupResource(context, HandleFromInput(context, 0),
                                           &decision_tree_resource));
    mutex_lock l(*decision_tree_resource->get_mutex());
    core::ScopedUnref unref_me(decision_tree_resource);

    const int num_data = data_set->NumItems();

    Tensor* output_predictions = nullptr;
    TensorShape output_shape;
    output_shape.AddDim(num_data);
    OP_REQUIRES_OK(context, context->allocate_output(0, output_shape,
                                                     &output_predictions));

    auto leaf_ids = output_predictions->tensor<int32, 1>();

    auto set_leaf_ids = [&leaf_ids](int32 i, int32 id) { leaf_ids(i) = id; };

    auto worker_threads = context->device()->tensorflow_cpu_worker_threads();
    int num_threads = worker_threads->num_threads;
    const int64 costPerTraverse = 500;
    auto traverse = [&set_leaf_ids, &data_set, decision_tree_resource,
                     num_data](int64 start, int64 end) {
      CHECK(start <= end);
      CHECK(end <= num_data);
      TraverseTree(decision_tree_resource, data_set, static_cast<int32>(start),
                   static_cast<int32>(end), set_leaf_ids, nullptr);
    };
    Shard(num_threads, worker_threads->workers, num_data, costPerTraverse,
          traverse);
  }

 private:
  tensorforest::TensorForestDataSpec input_spec_;
  TensorForestParams param_proto_;
};

// Update the given leaf models using the batch of labels.
class UpdateModelV4Op : public OpKernel {
 public:
  explicit UpdateModelV4Op(OpKernelConstruction* context) : OpKernel(context) {
    string serialized_params;
    OP_REQUIRES_OK(context, context->GetAttr("params", &serialized_params));
    ParseProtoUnlimited(&param_proto_, serialized_params);

    model_op_ = LeafModelOperatorFactory::CreateLeafModelOperator(param_proto_);
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& leaf_ids = context->input(1);
    const Tensor& input_labels = context->input(2);
    const Tensor& input_weights = context->input(3);

    DecisionTreeResource* decision_tree_resource;
    OP_REQUIRES_OK(context, LookupResource(context, HandleFromInput(context, 0),
                                           &decision_tree_resource));
    mutex_lock l(*decision_tree_resource->get_mutex());
    core::ScopedUnref unref_me(decision_tree_resource);

    const int num_data = input_labels.shape().dim_size(0);
    const int32 label_dim =
        input_labels.shape().dims() <= 1
            ? 0
            : static_cast<int>(input_labels.shape().dim_size(1));
    const int32 num_targets =
        param_proto_.is_regression() ? (std::max(1, label_dim)) : 1;

    TensorInputTarget target(input_labels, input_weights, num_targets);

    // TODO(gilberth): Make this thread safe and multi-thread.
    UpdateModel(leaf_ids, target, 0, num_data, decision_tree_resource);
  }

  void UpdateModel(const Tensor& leaf_ids, const TensorInputTarget& target,
                   int32 start, int32 end,
                   DecisionTreeResource* decision_tree_resource) {
    const auto leaves = leaf_ids.unaligned_flat<int32>();
    for (int i = start; i < end; ++i) {
      model_op_->UpdateModel(
          decision_tree_resource->get_mutable_tree_node(leaves(i))
              ->mutable_leaf(),
          &target, i);
    }
  }

 private:
  std::unique_ptr<LeafModelOperator> model_op_;
  TensorForestParams param_proto_;
};

// Op for getting feature usage counts.
class FeatureUsageCountsOp : public OpKernel {
 public:
  explicit FeatureUsageCountsOp(OpKernelConstruction* context)
      : OpKernel(context) {
    string serialized_params;
    OP_REQUIRES_OK(context, context->GetAttr("params", &serialized_params));
    ParseProtoUnlimited(&param_proto_, serialized_params);
  }

  void Compute(OpKernelContext* context) override {
    DecisionTreeResource* decision_tree_resource;
    OP_REQUIRES_OK(context, LookupResource(context, HandleFromInput(context, 0),
                                           &decision_tree_resource));
    mutex_lock l(*decision_tree_resource->get_mutex());
    core::ScopedUnref unref_me(decision_tree_resource);

    const auto& tree = decision_tree_resource->decision_tree();

    Tensor* output_counts = nullptr;
    TensorShape output_shape;
    output_shape.AddDim(param_proto_.num_features());
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, output_shape, &output_counts));

    auto counts = output_counts->unaligned_flat<int32>();
    counts.setZero();

    for (const auto& node : tree.decision_tree().nodes()) {
      if (node.has_custom_node_type()) {
        LOG(WARNING) << "Can't count feature usage for custom nodes.";
      } else if (node.has_binary_node()) {
        const auto& bnode = node.binary_node();
        if (bnode.has_custom_left_child_test()) {
          decision_trees::MatchingValuesTest test;
          if (!bnode.custom_left_child_test().UnpackTo(&test)) {
            LOG(WARNING) << "Unknown custom child test";
            continue;
          }
          int32 feat;
          safe_strto32(test.feature_id().id().value(), &feat);
          ++counts(feat);
        } else {
          const auto& test = bnode.inequality_left_child_test();
          if (test.has_feature_id()) {
            int32 feat;
            safe_strto32(test.feature_id().id().value(), &feat);
            ++counts(feat);
          } else if (test.has_oblique()) {
            for (const auto& featid : test.oblique().features()) {
              int32 feat;
              safe_strto32(featid.id().value(), &feat);
              ++counts(feat);
            }
          }
        }
      }
    }
  }

 private:
  TensorForestParams param_proto_;
};

REGISTER_RESOURCE_HANDLE_KERNEL(DecisionTreeResource);

REGISTER_KERNEL_BUILDER(Name("TreeIsInitializedOp").Device(DEVICE_CPU),
                        IsResourceInitialized<DecisionTreeResource>);

REGISTER_KERNEL_BUILDER(Name("CreateTreeVariable").Device(DEVICE_CPU),
                        CreateTreeVariableOp);

REGISTER_KERNEL_BUILDER(Name("TreeSerialize").Device(DEVICE_CPU),
                        TreeSerializeOp);

REGISTER_KERNEL_BUILDER(Name("TreeDeserialize").Device(DEVICE_CPU),
                        TreeDeserializeOp);

REGISTER_KERNEL_BUILDER(Name("TreeSize").Device(DEVICE_CPU), TreeSizeOp);

REGISTER_KERNEL_BUILDER(Name("TreePredictionsV4").Device(DEVICE_CPU),
                        TreePredictionsV4Op);

REGISTER_KERNEL_BUILDER(Name("TraverseTreeV4").Device(DEVICE_CPU),
                        TraverseTreeV4Op);

REGISTER_KERNEL_BUILDER(Name("FeatureUsageCounts").Device(DEVICE_CPU),
                        FeatureUsageCountsOp);

REGISTER_KERNEL_BUILDER(Name("UpdateModelV4").Device(DEVICE_CPU),
                        UpdateModelV4Op);

}  // namespace tensorforest
}  // namespace tensorflow
