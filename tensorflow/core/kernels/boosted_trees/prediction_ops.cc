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

#include <algorithm>
#include <string>
#include <vector>

#include "tensorflow/core/framework/device_base.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/kernels/boosted_trees/boosted_trees.pb.h"
#include "tensorflow/core/kernels/boosted_trees/resources.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/refcount.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/work_sharder.h"

namespace tensorflow {

static void ConvertVectorsToMatrices(
    OpKernelContext* const context, const OpInputList bucketized_features_list,
    std::vector<tensorflow::TTypes<int32>::ConstMatrix>& bucketized_features) {
  for (const Tensor& tensor : bucketized_features_list) {
    if (tensor.dims() == 1) {
      const auto v = tensor.vec<int32>();
      bucketized_features.emplace_back(
          TTypes<int32>::ConstMatrix(v.data(), v.size(), 1));
    } else {
      OP_REQUIRES(context, TensorShapeUtils::IsMatrix(tensor.shape()),
                  errors::Internal("Cannot use tensor as matrix, expected "
                                   "vector or matrix, received shape ",
                                   tensor.shape().DebugString()));
      bucketized_features.emplace_back(tensor.matrix<int32>());
    }
  }
}

// The Op used during training time to get the predictions so far with the
// current ensemble being built.
// Expect some logits are cached from the previous step and passed through
// to be reused.
class BoostedTreesTrainingPredictOp : public OpKernel {
 public:
  explicit BoostedTreesTrainingPredictOp(OpKernelConstruction* const context)
      : OpKernel(context) {
    VLOG(1) << "Boosted Trees kernels in TF are deprecated. Please use "
            << "TensorFlow Decision Forests instead "
            << "(https://github.com/tensorflow/decision-forests).\n";
    OP_REQUIRES_OK(context, context->GetAttr("num_bucketized_features",
                                             &num_bucketized_features_));
    OP_REQUIRES_OK(context,
                   context->GetAttr("logits_dimension", &logits_dimension_));
  }

  void Compute(OpKernelContext* const context) override {
    core::RefCountPtr<BoostedTreesEnsembleResource> resource;
    // Get the resource.
    OP_REQUIRES_OK(context, LookupResource(context, HandleFromInput(context, 0),
                                           &resource));

    // Get the inputs.
    OpInputList bucketized_features_list;
    OP_REQUIRES_OK(context, context->input_list("bucketized_features",
                                                &bucketized_features_list));
    std::vector<tensorflow::TTypes<int32>::ConstMatrix> bucketized_features;
    bucketized_features.reserve(bucketized_features_list.size());
    ConvertVectorsToMatrices(context, bucketized_features_list,
                             bucketized_features);
    const int batch_size = bucketized_features[0].dimension(0);

    const Tensor* cached_tree_ids_t;
    OP_REQUIRES_OK(context,
                   context->input("cached_tree_ids", &cached_tree_ids_t));
    OP_REQUIRES(context, TensorShapeUtils::IsVector(cached_tree_ids_t->shape()),
                errors::InvalidArgument(
                    "cached_tree_ids must be a vector, received shape ",
                    cached_tree_ids_t->shape().DebugString()));
    const auto cached_tree_ids = cached_tree_ids_t->vec<int32>();

    const Tensor* cached_node_ids_t;
    OP_REQUIRES_OK(context,
                   context->input("cached_node_ids", &cached_node_ids_t));
    OP_REQUIRES(context, TensorShapeUtils::IsVector(cached_node_ids_t->shape()),
                errors::InvalidArgument(
                    "cached_node_ids must be a vector, received shape ",
                    cached_node_ids_t->shape().DebugString()));
    const auto cached_node_ids = cached_node_ids_t->vec<int32>();

    // Allocate outputs.
    Tensor* output_partial_logits_t = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output("partial_logits",
                                            {batch_size, logits_dimension_},
                                            &output_partial_logits_t));
    auto output_partial_logits = output_partial_logits_t->matrix<float>();

    Tensor* output_tree_ids_t = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output("tree_ids", {batch_size},
                                                     &output_tree_ids_t));
    auto output_tree_ids = output_tree_ids_t->vec<int32>();

    Tensor* output_node_ids_t = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output("node_ids", {batch_size},
                                                     &output_node_ids_t));
    auto output_node_ids = output_node_ids_t->vec<int32>();

    // Indicate that the latest tree was used.
    const int32_t latest_tree = resource->num_trees() - 1;

    if (latest_tree < 0) {
      // Ensemble was empty. Output the very first node.
      output_node_ids.setZero();
      output_tree_ids = cached_tree_ids;
      // All the predictions are zeros.
      output_partial_logits.setZero();
    } else {
      output_tree_ids.setConstant(latest_tree);
      auto do_work = [&context, &resource, &bucketized_features,
                      &cached_tree_ids, &cached_node_ids,
                      &output_partial_logits, &output_node_ids, latest_tree,
                      this](int64_t start, int64_t end) {
        for (int32_t i = start; i < end; ++i) {
          int32_t tree_id = cached_tree_ids(i);
          int32_t node_id = cached_node_ids(i);
          std::vector<float> partial_tree_logits(logits_dimension_, 0.0);

          if (node_id >= 0) {
            // If the tree was pruned, returns the node id into which the
            // current_node_id was pruned, as well the correction of the cached
            // logit prediction.
            resource->GetPostPruneCorrection(tree_id, node_id, &node_id,
                                             &partial_tree_logits);
            // Logic in the loop adds the cached node value again if it is a
            // leaf. If it is not a leaf anymore we need to subtract the old
            // node's value. The following logic handles both of these cases.
            const auto& node_logits = resource->node_value(tree_id, node_id);
            if (!node_logits.empty()) {
              OP_REQUIRES(
                  context, node_logits.size() == logits_dimension_,
                  errors::Internal(
                      "Expected node_logits.size() == logits_dimension_, got ",
                      node_logits.size(), " vs ", logits_dimension_));
              for (int32_t j = 0; j < logits_dimension_; ++j) {
                partial_tree_logits[j] -= node_logits[j];
              }
            }
          } else {
            // No cache exists, start from the very first node.
            node_id = 0;
          }
          std::vector<float> partial_all_logits(logits_dimension_, 0.0);
          while (true) {
            if (resource->is_leaf(tree_id, node_id)) {
              const auto& leaf_logits = resource->node_value(tree_id, node_id);
              OP_REQUIRES(
                  context, leaf_logits.size() == logits_dimension_,
                  errors::Internal(
                      "Expected leaf_logits.size() == logits_dimension_, got ",
                      leaf_logits.size(), " vs ", logits_dimension_));
              // Tree is done
              const float tree_weight = resource->GetTreeWeight(tree_id);
              for (int32_t j = 0; j < logits_dimension_; ++j) {
                partial_all_logits[j] +=
                    tree_weight * (partial_tree_logits[j] + leaf_logits[j]);
                partial_tree_logits[j] = 0;
              }
              // Stop if it was the latest tree.
              if (tree_id == latest_tree) {
                break;
              }
              // Move onto other trees.
              ++tree_id;
              node_id = 0;
            } else {
              node_id =
                  resource->next_node(tree_id, node_id, i, bucketized_features);
            }
          }
          output_node_ids(i) = node_id;
          for (int32_t j = 0; j < logits_dimension_; ++j) {
            output_partial_logits(i, j) = partial_all_logits[j];
          }
        }
      };
      // 30 is the magic number. The actual value might be a function of (the
      // number of layers) * (cpu cycles spent on each layer), but this value
      // would work for many cases. May be tuned later.
      const int64_t cost = 30;
      thread::ThreadPool* const worker_threads =
          context->device()->tensorflow_cpu_worker_threads()->workers;
      Shard(worker_threads->NumThreads(), worker_threads, batch_size,
            /*cost_per_unit=*/cost, do_work);
    }
  }

 private:
  int32 logits_dimension_;         // the size of the output prediction vector.
  int32 num_bucketized_features_;  // Indicates the number of features.
};

REGISTER_KERNEL_BUILDER(Name("BoostedTreesTrainingPredict").Device(DEVICE_CPU),
                        BoostedTreesTrainingPredictOp);

// The Op to get the predictions at the evaluation/inference time.
class BoostedTreesPredictOp : public OpKernel {
 public:
  explicit BoostedTreesPredictOp(OpKernelConstruction* const context)
      : OpKernel(context) {
    VLOG(1) << "Boosted Trees kernels in TF are deprecated. Please use "
            << "TensorFlow Decision Forests instead "
            << "(https://github.com/tensorflow/decision-forests).\n";
    OP_REQUIRES_OK(context, context->GetAttr("num_bucketized_features",
                                             &num_bucketized_features_));
    OP_REQUIRES_OK(context,
                   context->GetAttr("logits_dimension", &logits_dimension_));
  }

  void Compute(OpKernelContext* const context) override {
    core::RefCountPtr<BoostedTreesEnsembleResource> resource;
    // Get the resource.
    OP_REQUIRES_OK(context, LookupResource(context, HandleFromInput(context, 0),
                                           &resource));

    // Get the inputs.
    OpInputList bucketized_features_list;
    OP_REQUIRES_OK(context, context->input_list("bucketized_features",
                                                &bucketized_features_list));
    std::vector<tensorflow::TTypes<int32>::ConstMatrix> bucketized_features;
    bucketized_features.reserve(bucketized_features_list.size());
    ConvertVectorsToMatrices(context, bucketized_features_list,
                             bucketized_features);
    const int batch_size = bucketized_features[0].dimension(0);

    // Allocate outputs.
    Tensor* output_logits_t = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(
                                "logits", {batch_size, logits_dimension_},
                                &output_logits_t));
    auto output_logits = output_logits_t->matrix<float>();

    // Return zero logits if it's an empty ensemble.
    if (resource->num_trees() <= 0) {
      output_logits.setZero();
      return;
    }

    const int32_t last_tree = resource->num_trees() - 1;
    auto do_work = [&context, &resource, &bucketized_features, &output_logits,
                    last_tree, this](int64_t start, int64_t end) {
      for (int32_t i = start; i < end; ++i) {
        std::vector<float> tree_logits(logits_dimension_, 0.0);
        int32_t tree_id = 0;
        int32_t node_id = 0;
        while (true) {
          if (resource->is_leaf(tree_id, node_id)) {
            const float tree_weight = resource->GetTreeWeight(tree_id);
            const auto& leaf_logits = resource->node_value(tree_id, node_id);
            OP_REQUIRES(
                context, leaf_logits.size() == logits_dimension_,
                errors::Internal(
                    "Expected leaf_logits.size() == logits_dimension_, got ",
                    leaf_logits.size(), " vs ", logits_dimension_));
            for (int32_t j = 0; j < logits_dimension_; ++j) {
              tree_logits[j] += tree_weight * leaf_logits[j];
            }
            // Stop if it was the last tree.
            if (tree_id == last_tree) {
              break;
            }
            // Move onto other trees.
            ++tree_id;
            node_id = 0;
          } else {
            node_id =
                resource->next_node(tree_id, node_id, i, bucketized_features);
          }
        }
        for (int32_t j = 0; j < logits_dimension_; ++j) {
          output_logits(i, j) = tree_logits[j];
        }
      }
    };
    // 10 is the magic number. The actual number might depend on (the number of
    // layers in the trees) and (cpu cycles spent on each layer), but this
    // value would work for many cases. May be tuned later.
    const int64_t cost = (last_tree + 1) * 10;
    thread::ThreadPool* const worker_threads =
        context->device()->tensorflow_cpu_worker_threads()->workers;
    Shard(worker_threads->NumThreads(), worker_threads, batch_size,
          /*cost_per_unit=*/cost, do_work);
  }

 private:
  int32
      logits_dimension_;  // Indicates the size of the output prediction vector.
  int32 num_bucketized_features_;  // Indicates the number of features.
};

REGISTER_KERNEL_BUILDER(Name("BoostedTreesPredict").Device(DEVICE_CPU),
                        BoostedTreesPredictOp);

// The Op that returns debugging/model interpretability outputs for each
// example. Currently it outputs the split feature ids and logits after each
// split along the decision path for each example. This will be used to compute
// directional feature contributions at predict time for an arbitrary activation
// function.
// TODO(crawles): return in proto 1) Node IDs for ensemble prediction path
// 2) Leaf node IDs.
class BoostedTreesExampleDebugOutputsOp : public OpKernel {
 public:
  explicit BoostedTreesExampleDebugOutputsOp(
      OpKernelConstruction* const context)
      : OpKernel(context) {
    VLOG(1) << "Boosted Trees kernels in TF are deprecated. Please use "
            << "TensorFlow Decision Forests instead "
            << "(https://github.com/tensorflow/decision-forests).\n";
    OP_REQUIRES_OK(context, context->GetAttr("num_bucketized_features",
                                             &num_bucketized_features_));
    OP_REQUIRES_OK(context,
                   context->GetAttr("logits_dimension", &logits_dimension_));
    OP_REQUIRES(context, logits_dimension_ == 1,
                errors::InvalidArgument(
                    "Currently only one dimensional outputs are supported."));
  }

  void Compute(OpKernelContext* const context) override {
    core::RefCountPtr<BoostedTreesEnsembleResource> resource;
    // Get the resource.
    OP_REQUIRES_OK(context, LookupResource(context, HandleFromInput(context, 0),
                                           &resource));

    // Get the inputs.
    OpInputList bucketized_features_list;
    OP_REQUIRES_OK(context, context->input_list("bucketized_features",
                                                &bucketized_features_list));
    std::vector<tensorflow::TTypes<int32>::ConstMatrix> bucketized_features;
    bucketized_features.reserve(bucketized_features_list.size());
    ConvertVectorsToMatrices(context, bucketized_features_list,
                             bucketized_features);
    const int batch_size = bucketized_features[0].dimension(0);

    // We need to get the feature ids used for splitting and the logits after
    // each split. We will use these to calculate the changes in the prediction
    // (contributions) for an arbitrary activation function (done in Python) and
    // attribute them to the associated feature ids. We will store these in
    // a proto below.
    Tensor* output_debug_info_t = nullptr;
    OP_REQUIRES_OK(
        context, context->allocate_output("examples_debug_outputs_serialized",
                                          {batch_size}, &output_debug_info_t));
    // Will contain serialized protos, per example.
    auto output_debug_info = output_debug_info_t->flat<tstring>();
    const int32_t last_tree = resource->num_trees() - 1;

    // For each given example, traverse through all trees keeping track of the
    // features used to split and the associated logits at each point along the
    // path. Note: feature_ids has one less value than logits_path because the
    // first value of each logit path will be the bias.
    auto do_work = [&context, &resource, &bucketized_features,
                    &output_debug_info, last_tree](int64_t start, int64_t end) {
      for (int32_t i = start; i < end; ++i) {
        // Proto to store debug outputs, per example.
        boosted_trees::DebugOutput example_debug_info;
        // Initial bias prediction. E.g., prediction based off training mean.
        const auto& tree_logits = resource->node_value(0, 0);
        OP_REQUIRES(context, tree_logits.size() == 1,
                    errors::Internal("Expected tree_logits.size() == 1, got ",
                                     tree_logits.size()));
        float tree_logit = resource->GetTreeWeight(0) * tree_logits[0];
        example_debug_info.add_logits_path(tree_logit);
        int32_t node_id = 0;
        int32_t tree_id = 0;
        int32_t feature_id;
        float past_trees_logit = 0;  // Sum of leaf logits from prior trees.
        // Go through each tree and populate proto.
        while (tree_id <= last_tree) {
          if (resource->is_leaf(tree_id, node_id)) {  // Move onto other trees.
            // Accumulate tree_logits only if the leaf is non-root, but do so
            // for bias tree.
            if (tree_id == 0 || node_id > 0) {
              past_trees_logit += tree_logit;
            }
            example_debug_info.add_leaf_node_ids(node_id);
            ++tree_id;
            node_id = 0;
          } else {  // Add to proto.
            // Feature id used to split.
            feature_id = resource->feature_id(tree_id, node_id);
            example_debug_info.add_feature_ids(feature_id);
            // Get logit after split.
            node_id =
                resource->next_node(tree_id, node_id, i, bucketized_features);
            const auto& tree_logits = resource->node_value(tree_id, node_id);
            OP_REQUIRES(
                context, tree_logits.size() == 1,
                errors::Internal("Expected tree_logits.size() == 1, got ",
                                 tree_logits.size()));
            tree_logit = resource->GetTreeWeight(tree_id) * tree_logits[0];
            // Output logit incorporates sum of leaf logits from prior trees.
            example_debug_info.add_logits_path(tree_logit + past_trees_logit);
          }
        }
        // Set output as serialized proto containing debug info.
        string serialized = example_debug_info.SerializeAsString();
        output_debug_info(i) = serialized;
      }
    };

    // 10 is the magic number. The actual number might depend on (the number of
    // layers in the trees) and (cpu cycles spent on each layer), but this
    // value would work for many cases. May be tuned later.
    const int64_t cost = (last_tree + 1) * 10;
    thread::ThreadPool* const worker_threads =
        context->device()->tensorflow_cpu_worker_threads()->workers;
    Shard(worker_threads->NumThreads(), worker_threads, batch_size,
          /*cost_per_unit=*/cost, do_work);
  }

 private:
  int32 logits_dimension_;  // Indicates dimension of logits in the tree nodes.
  int32 num_bucketized_features_;  // Indicates the number of features.
};

REGISTER_KERNEL_BUILDER(
    Name("BoostedTreesExampleDebugOutputs").Device(DEVICE_CPU),
    BoostedTreesExampleDebugOutputsOp);

}  // namespace tensorflow
