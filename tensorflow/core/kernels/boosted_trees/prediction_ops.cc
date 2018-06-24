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

// The Op used during training time to get the predictions so far with the
// current ensemble being built.
// Expect some logits are cached from the previous step and passed through
// to be reused.
class BoostedTreesTrainingPredictOp : public OpKernel {
 public:
  explicit BoostedTreesTrainingPredictOp(OpKernelConstruction* const context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("num_bucketized_features",
                                             &num_bucketized_features_));
    OP_REQUIRES_OK(context,
                   context->GetAttr("logits_dimension", &logits_dimension_));
    OP_REQUIRES(context, logits_dimension_ == 1,
                errors::InvalidArgument(
                    "Currently only one dimensional outputs are supported."));
  }

  void Compute(OpKernelContext* const context) override {
    BoostedTreesEnsembleResource* resource;
    // Get the resource.
    OP_REQUIRES_OK(context, LookupResource(context, HandleFromInput(context, 0),
                                           &resource));
    // Release the reference to the resource once we're done using it.
    core::ScopedUnref unref_me(resource);

    // Get the inputs.
    OpInputList bucketized_features_list;
    OP_REQUIRES_OK(context, context->input_list("bucketized_features",
                                                &bucketized_features_list));
    std::vector<tensorflow::TTypes<int32>::ConstVec> batch_bucketized_features;
    batch_bucketized_features.reserve(bucketized_features_list.size());
    for (const Tensor& tensor : bucketized_features_list) {
      batch_bucketized_features.emplace_back(tensor.vec<int32>());
    }
    const int batch_size = batch_bucketized_features[0].size();

    const Tensor* cached_tree_ids_t;
    OP_REQUIRES_OK(context,
                   context->input("cached_tree_ids", &cached_tree_ids_t));
    const auto cached_tree_ids = cached_tree_ids_t->vec<int32>();

    const Tensor* cached_node_ids_t;
    OP_REQUIRES_OK(context,
                   context->input("cached_node_ids", &cached_node_ids_t));
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
    const int32 latest_tree = resource->num_trees() - 1;

    if (latest_tree < 0) {
      // Ensemble was empty. Nothing changes.
      output_node_ids = cached_node_ids;
      output_tree_ids = cached_tree_ids;
      // All the predictions are zeros.
      output_partial_logits.setZero();
    } else {
      output_tree_ids.setConstant(latest_tree);
      auto do_work = [&resource, &batch_bucketized_features, &cached_tree_ids,
                      &cached_node_ids, &output_partial_logits,
                      &output_node_ids, batch_size,
                      latest_tree](int32 start, int32 end) {
        for (int32 i = start; i < end; ++i) {
          int32 tree_id = cached_tree_ids(i);
          int32 node_id = cached_node_ids(i);
          float partial_tree_logit = 0.0;

          // If the tree was pruned, returns the node id into which the
          // current_node_id was pruned, as well the correction of the cached
          // logit prediction.
          resource->GetPostPruneCorrection(tree_id, node_id, &node_id,
                                           &partial_tree_logit);

          // Logic in the loop adds the cached node value again if it is a leaf.
          // If it is not a leaf anymore we need to subtract the old node's
          // value. The following logic handles both of these cases.
          partial_tree_logit -= resource->node_value(tree_id, node_id);
          float partial_all_logit = 0.0;
          while (true) {
            if (resource->is_leaf(tree_id, node_id)) {
              partial_tree_logit += resource->node_value(tree_id, node_id);

              // Tree is done
              partial_all_logit +=
                  resource->GetTreeWeight(tree_id) * partial_tree_logit;
              partial_tree_logit = 0.0;
              // Stop if it was the latest tree.
              if (tree_id == latest_tree) {
                break;
              }
              // Move onto other trees.
              ++tree_id;
              node_id = 0;
            } else {
              node_id = resource->next_node(tree_id, node_id, i,
                                            batch_bucketized_features);
            }
          }
          output_node_ids(i) = node_id;
          output_partial_logits(i, 0) = partial_all_logit;
        }
      };
      // 30 is the magic number. The actual value might be a function of (the
      // number of layers) * (cpu cycles spent on each layer), but this value
      // would work for many cases. May be tuned later.
      const int64 cost = 30;
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
    OP_REQUIRES_OK(context, context->GetAttr("num_bucketized_features",
                                             &num_bucketized_features_));
    OP_REQUIRES_OK(context,
                   context->GetAttr("logits_dimension", &logits_dimension_));
    OP_REQUIRES(context, logits_dimension_ == 1,
                errors::InvalidArgument(
                    "Currently only one dimensional outputs are supported."));
  }

  void Compute(OpKernelContext* const context) override {
    BoostedTreesEnsembleResource* resource;
    // Get the resource.
    OP_REQUIRES_OK(context, LookupResource(context, HandleFromInput(context, 0),
                                           &resource));
    // Release the reference to the resource once we're done using it.
    core::ScopedUnref unref_me(resource);

    // Get the inputs.
    OpInputList bucketized_features_list;
    OP_REQUIRES_OK(context, context->input_list("bucketized_features",
                                                &bucketized_features_list));
    std::vector<tensorflow::TTypes<int32>::ConstVec> batch_bucketized_features;
    batch_bucketized_features.reserve(bucketized_features_list.size());
    for (const Tensor& tensor : bucketized_features_list) {
      batch_bucketized_features.emplace_back(tensor.vec<int32>());
    }
    const int batch_size = batch_bucketized_features[0].size();

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

    const int32 latest_tree = resource->num_trees() - 1;

    auto do_work = [&resource, &batch_bucketized_features, &output_logits,
                    batch_size, latest_tree](int32 start, int32 end) {
      for (int32 i = start; i < end; ++i) {
        float tree_logit = 0.0;
        int32 tree_id = 0;
        int32 node_id = 0;
        while (true) {
          if (resource->is_leaf(tree_id, node_id)) {
            tree_logit += resource->GetTreeWeight(tree_id) *
                          resource->node_value(tree_id, node_id);

            // Stop if it was the latest tree.
            if (tree_id == latest_tree) {
              break;
            }
            // Move onto other trees.
            ++tree_id;
            node_id = 0;
          } else {
            node_id = resource->next_node(tree_id, node_id, i,
                                          batch_bucketized_features);
          }
        }
        output_logits(i, 0) = tree_logit;
      }
    };
    // 10 is the magic number. The actual number might depend on (the number of
    // layers in the trees) and (cpu cycles spent on each layer), but this
    // value would work for many cases. May be tuned later.
    const int64 cost = (latest_tree + 1) * 10;
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

}  // namespace tensorflow
