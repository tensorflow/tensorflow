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
#include <algorithm>
#include <string>
#include <vector>

#include "tensorflow/contrib/boosted_trees/lib/learner/common/partitioners/example_partitioner.h"
#include "tensorflow/contrib/boosted_trees/lib/models/multiple_additive_trees.h"
#include "tensorflow/contrib/boosted_trees/lib/utils/batch_features.h"
#include "tensorflow/contrib/boosted_trees/lib/utils/dropout_utils.h"
#include "tensorflow/contrib/boosted_trees/lib/utils/tensor_utils.h"
#include "tensorflow/contrib/boosted_trees/proto/learner.pb.h"
#include "tensorflow/contrib/boosted_trees/proto/tree_config.pb.h"
#include "tensorflow/contrib/boosted_trees/resources/decision_tree_ensemble_resource.h"
#include "tensorflow/core/framework/device_base.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/refcount.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/platform/types.h"

using tensorflow::boosted_trees::learner::AveragingConfig;
using tensorflow::boosted_trees::trees::DecisionTreeEnsembleConfig;

namespace tensorflow {
namespace boosted_trees {

using boosted_trees::learner::LearnerConfig;
using boosted_trees::learner::LearningRateConfig;
using boosted_trees::learner::LearningRateDropoutDrivenConfig;
using boosted_trees::models::DecisionTreeEnsembleResource;
using boosted_trees::models::MultipleAdditiveTrees;
using boosted_trees::utils::DropoutUtils;
using boosted_trees::utils::TensorUtils;

namespace {
const char* kLearnerConfigAttributeName = "learner_config";
const char* kSeedTensorName = "seed";
const char* kApplyDropoutAttributeName = "apply_dropout";
const char* kApplyAveragingAttributeName = "apply_averaging";
const char* kDropoutInfoOutputTensorName = "drop_out_tree_indices_weights";
const char* kPredictionsTensorName = "predictions";
const char* kLeafIndexTensorName = "leaf_index";

void CalculateTreesToInclude(
    const boosted_trees::trees::DecisionTreeEnsembleConfig& config,
    const std::vector<int32>& trees_to_drop, const int32 num_trees,
    const bool only_finalized, const bool center_bias,
    std::vector<int32>* trees_to_include) {
  trees_to_include->reserve(num_trees - trees_to_drop.size());

  int32 index = 0;
  // This assumes that trees_to_drop is a sorted list of tree ids.
  for (int32 tree = 0; tree < num_trees; ++tree) {
    // Skip the tree if tree is in the list of trees_to_drop.
    if (!trees_to_drop.empty() && index < trees_to_drop.size() &&
        trees_to_drop[index] == tree) {
      ++index;
      continue;
    }
    // Or skip if the tree is not finalized and only_finalized is set,
    // with the exception of centering bias.
    if (only_finalized && !(center_bias && tree == 0) &&
        config.tree_metadata_size() > 0 &&
        !config.tree_metadata(tree).is_finalized()) {
      continue;
    }
    trees_to_include->push_back(tree);
  }
}
}  // namespace

class GradientTreesPredictionOp : public OpKernel {
 public:
  explicit GradientTreesPredictionOp(OpKernelConstruction* const context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("use_locking", &use_locking_));

    OP_REQUIRES_OK(context, context->GetAttr("center_bias", &center_bias_));

    OP_REQUIRES_OK(
        context, context->GetAttr(kApplyDropoutAttributeName, &apply_dropout_));

    LearnerConfig learner_config;
    string learner_config_str;
    OP_REQUIRES_OK(context, context->GetAttr(kLearnerConfigAttributeName,
                                             &learner_config_str));
    OP_REQUIRES(
        context, ParseProtoUnlimited(&learner_config, learner_config_str),
        errors::InvalidArgument("Unable to parse learner config config."));

    num_classes_ = learner_config.num_classes();
    OP_REQUIRES(context, num_classes_ >= 2,
                errors::InvalidArgument("Number of classes must be >=2"));
    OP_REQUIRES(
        context, ParseProtoUnlimited(&learner_config, learner_config_str),
        errors::InvalidArgument("Unable to parse learner config config."));

    bool reduce_dim;
    OP_REQUIRES_OK(context, context->GetAttr("reduce_dim", &reduce_dim));
    prediction_vector_size_ = reduce_dim ? num_classes_ - 1 : num_classes_;

    only_finalized_trees_ =
        learner_config.growing_mode() == learner_config.WHOLE_TREE;
    if (learner_config.has_learning_rate_tuner() &&
        learner_config.learning_rate_tuner().tuner_case() ==
            LearningRateConfig::kDropout) {
      dropout_config_ = learner_config.learning_rate_tuner().dropout();
      has_dropout_ = true;
    } else {
      has_dropout_ = false;
    }

    OP_REQUIRES_OK(context, context->GetAttr(kApplyAveragingAttributeName,
                                             &apply_averaging_));
    apply_averaging_ =
        apply_averaging_ && learner_config.averaging_config().config_case() !=
                                AveragingConfig::CONFIG_NOT_SET;
    if (apply_averaging_) {
      averaging_config_ = learner_config.averaging_config();

      // If there is averaging config, check that the values are correct.
      switch (averaging_config_.config_case()) {
        case AveragingConfig::kAverageLastNTreesFieldNumber: {
          OP_REQUIRES(context, averaging_config_.average_last_n_trees() > 0,
                      errors::InvalidArgument(
                          "Average last n trees must be a positive number"));
          break;
        }
        case AveragingConfig::kAverageLastPercentTreesFieldNumber: {
          OP_REQUIRES(context,
                      averaging_config_.average_last_percent_trees() > 0 &&
                          averaging_config_.average_last_percent_trees() <= 1.0,
                      errors::InvalidArgument(
                          "Average last percent must be in (0,1] interval."));
          break;
        }
        case AveragingConfig::CONFIG_NOT_SET: {
          LOG(QFATAL) << "We should never get here.";
          break;
        }
      }
    }
  }

  void Compute(OpKernelContext* const context) override {
    DecisionTreeEnsembleResource* ensemble_resource;
    // Gets the resource. Grabs the mutex but releases it.
    OP_REQUIRES_OK(context, LookupResource(context, HandleFromInput(context, 0),
                                           &ensemble_resource));
    // Release the reference to the resource once we're done using it.
    core::ScopedUnref unref_me(ensemble_resource);
    if (use_locking_) {
      tf_shared_lock l(*ensemble_resource->get_mutex());
      DoCompute(context, ensemble_resource,
                /*return_output_leaf_index=*/false);
    } else {
      DoCompute(context, ensemble_resource,
                /*return_output_leaf_index=*/false);
    }
  }

 protected:
  // return_output_leaf_index is a boolean variable indicating whether to output
  // leaf index in prediction. Though this class invokes only with this param
  // value as false, the subclass GradientTreesPredictionVerboseOp will invoke
  // with the true value.
  virtual void DoCompute(OpKernelContext* context,
                         DecisionTreeEnsembleResource* ensemble_resource,
                         const bool return_output_leaf_index) {
    // Read dense float features list;
    OpInputList dense_float_features_list;
    OP_REQUIRES_OK(context, TensorUtils::ReadDenseFloatFeatures(
                                context, &dense_float_features_list));

    // Read sparse float features list;
    OpInputList sparse_float_feature_indices_list;
    OpInputList sparse_float_feature_values_list;
    OpInputList sparse_float_feature_shapes_list;
    OP_REQUIRES_OK(context, TensorUtils::ReadSparseFloatFeatures(
                                context, &sparse_float_feature_indices_list,
                                &sparse_float_feature_values_list,
                                &sparse_float_feature_shapes_list));

    // Read sparse int features list;
    OpInputList sparse_int_feature_indices_list;
    OpInputList sparse_int_feature_values_list;
    OpInputList sparse_int_feature_shapes_list;
    OP_REQUIRES_OK(context, TensorUtils::ReadSparseIntFeatures(
                                context, &sparse_int_feature_indices_list,
                                &sparse_int_feature_values_list,
                                &sparse_int_feature_shapes_list));

    // Infer batch size.
    const int64 batch_size = TensorUtils::InferBatchSize(
        dense_float_features_list, sparse_float_feature_shapes_list,
        sparse_int_feature_shapes_list);

    // Read batch features.
    boosted_trees::utils::BatchFeatures batch_features(batch_size);
    OP_REQUIRES_OK(
        context,
        batch_features.Initialize(
            TensorUtils::OpInputListToTensorVec(dense_float_features_list),
            TensorUtils::OpInputListToTensorVec(
                sparse_float_feature_indices_list),
            TensorUtils::OpInputListToTensorVec(
                sparse_float_feature_values_list),
            TensorUtils::OpInputListToTensorVec(
                sparse_float_feature_shapes_list),
            TensorUtils::OpInputListToTensorVec(
                sparse_int_feature_indices_list),
            TensorUtils::OpInputListToTensorVec(sparse_int_feature_values_list),
            TensorUtils::OpInputListToTensorVec(
                sparse_int_feature_shapes_list)));

    std::vector<int32> dropped_trees;
    std::vector<float> original_weights;

    // Do dropout if needed.
    if (apply_dropout_ && has_dropout_) {
      // Read in seed and cast to uint64.
      const Tensor* seed_t;
      OP_REQUIRES_OK(context, context->input(kSeedTensorName, &seed_t));
      OP_REQUIRES(context, TensorShapeUtils::IsScalar(seed_t->shape()),
                  errors::InvalidArgument("Seed must be a scalar."));
      const uint64 seed = seed_t->scalar<int64>()();

      std::unordered_set<int32> trees_not_to_drop;
      if (center_bias_) {
        trees_not_to_drop.insert(0);
      }
      if (ensemble_resource->decision_tree_ensemble().has_growing_metadata()) {
        // We are in batch mode, the last tree is the tree that is being built,
        // we can't drop it during dropout.
        trees_not_to_drop.insert(ensemble_resource->num_trees() - 1);
      }
      const std::vector<float> weights = ensemble_resource->GetTreeWeights();
      OP_REQUIRES_OK(context, DropoutUtils::DropOutTrees(
                                  seed, dropout_config_, trees_not_to_drop,
                                  weights, &dropped_trees, &original_weights));
    }

    // Prepare the list of trees to include in the prediction.
    std::vector<int32> trees_to_include;
    CalculateTreesToInclude(
        ensemble_resource->decision_tree_ensemble(), dropped_trees,
        ensemble_resource->decision_tree_ensemble().trees_size(),
        only_finalized_trees_, center_bias_, &trees_to_include);

    // Allocate output predictions matrix.
    Tensor* output_predictions_t = nullptr;
    OP_REQUIRES_OK(
        context, context->allocate_output(kPredictionsTensorName,
                                          {batch_size, prediction_vector_size_},
                                          &output_predictions_t));
    auto output_predictions = output_predictions_t->matrix<float>();

    // Allocate output leaf index matrix.
    Tensor* output_leaf_index_t = nullptr;
    if (return_output_leaf_index) {
      OP_REQUIRES_OK(context, context->allocate_output(
                                  kLeafIndexTensorName,
                                  {batch_size, ensemble_resource->num_trees()},
                                  &output_leaf_index_t));
    }
    // Run predictor.
    thread::ThreadPool* const worker_threads =
        context->device()->tensorflow_cpu_worker_threads()->workers;

    if (apply_averaging_) {
      DecisionTreeEnsembleConfig adjusted =
          ensemble_resource->decision_tree_ensemble();
      const int start_averaging = std::max(
          0.0,
          averaging_config_.config_case() ==
                  AveragingConfig::kAverageLastNTreesFieldNumber
              ? adjusted.trees_size() - averaging_config_.average_last_n_trees()
              : adjusted.trees_size() *
                    (1.0 - averaging_config_.average_last_percent_trees()));
      const int num_ensembles = adjusted.trees_size() - start_averaging;
      for (int i = start_averaging; i < adjusted.trees_size(); ++i) {
        float weight = adjusted.tree_weights(i);
        adjusted.mutable_tree_weights()->Set(
            i, weight * (num_ensembles - i + start_averaging) / num_ensembles);
      }
      MultipleAdditiveTrees::Predict(adjusted, trees_to_include, batch_features,
                                     worker_threads, output_predictions,
                                     output_leaf_index_t);
    } else {
      MultipleAdditiveTrees::Predict(
          ensemble_resource->decision_tree_ensemble(), trees_to_include,
          batch_features, worker_threads, output_predictions,
          output_leaf_index_t);
    }

    // Output dropped trees and original weights.
    Tensor* output_dropout_info_t = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(
                                kDropoutInfoOutputTensorName,
                                {2, static_cast<int64>(dropped_trees.size())},
                                &output_dropout_info_t));
    auto output_dropout_info = output_dropout_info_t->matrix<float>();
    for (int32 i = 0; i < dropped_trees.size(); ++i) {
      output_dropout_info(0, i) = dropped_trees[i];
      output_dropout_info(1, i) = original_weights[i];
    }
  }

 private:
  LearningRateDropoutDrivenConfig dropout_config_;
  AveragingConfig averaging_config_;
  bool only_finalized_trees_;
  int num_classes_;
  // What is the size of the output vector for predictions?
  int prediction_vector_size_;
  bool apply_dropout_;
  bool center_bias_;
  bool apply_averaging_;
  bool use_locking_;
  bool has_dropout_;
};

REGISTER_KERNEL_BUILDER(Name("GradientTreesPrediction").Device(DEVICE_CPU),
                        GradientTreesPredictionOp);

// GradientTreesPredictionVerboseOp is derived from GradientTreesPredictionOp
// and have an additional output of tensor of rank 2 containing leaf ids for
// each tree where an instance ended up with.
class GradientTreesPredictionVerboseOp : public GradientTreesPredictionOp {
 public:
  explicit GradientTreesPredictionVerboseOp(OpKernelConstruction* const context)
      : GradientTreesPredictionOp(context) {}

 protected:
  void DoCompute(OpKernelContext* context,
                 DecisionTreeEnsembleResource* ensemble_resource,
                 bool return_output_leaf_index) override {
    GradientTreesPredictionOp::DoCompute(context, ensemble_resource,
                                         /*return_output_leaf_index=*/true);
  }
};

REGISTER_KERNEL_BUILDER(
    Name("GradientTreesPredictionVerbose").Device(DEVICE_CPU),
    GradientTreesPredictionVerboseOp);

class GradientTreesPartitionExamplesOp : public OpKernel {
 public:
  explicit GradientTreesPartitionExamplesOp(OpKernelConstruction* const context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("use_locking", &use_locking_));
  }

  void Compute(OpKernelContext* const context) override {
    DecisionTreeEnsembleResource* ensemble_resource;
    // Gets the resource. Grabs the mutex but releases it.
    OP_REQUIRES_OK(context, LookupResource(context, HandleFromInput(context, 0),
                                           &ensemble_resource));
    // Release the reference to the resource once we're done using it.
    core::ScopedUnref unref_me(ensemble_resource);
    if (use_locking_) {
      tf_shared_lock l(*ensemble_resource->get_mutex());
      DoCompute(context, ensemble_resource);
    } else {
      DoCompute(context, ensemble_resource);
    }
  }

 private:
  void DoCompute(OpKernelContext* context,
                 DecisionTreeEnsembleResource* ensemble_resource) {
    // The last non-finalized tree in the ensemble is by convention the
    // one to partition on. If no such tree exists, a nodeless tree is
    // created.
    boosted_trees::trees::DecisionTreeConfig empty_tree_config;
    const boosted_trees::trees::DecisionTreeConfig& tree_config =
        (ensemble_resource->num_trees() <= 0 ||
         ensemble_resource->LastTreeMetadata()->is_finalized())
            ? empty_tree_config
            : *ensemble_resource->LastTree();

    // Read dense float features list;
    OpInputList dense_float_features_list;
    OP_REQUIRES_OK(context, TensorUtils::ReadDenseFloatFeatures(
                                context, &dense_float_features_list));

    // Read sparse float features list;
    OpInputList sparse_float_feature_indices_list;
    OpInputList sparse_float_feature_values_list;
    OpInputList sparse_float_feature_shapes_list;
    OP_REQUIRES_OK(context, TensorUtils::ReadSparseFloatFeatures(
                                context, &sparse_float_feature_indices_list,
                                &sparse_float_feature_values_list,
                                &sparse_float_feature_shapes_list));

    // Read sparse int features list;
    OpInputList sparse_int_feature_indices_list;
    OpInputList sparse_int_feature_values_list;
    OpInputList sparse_int_feature_shapes_list;
    OP_REQUIRES_OK(context, TensorUtils::ReadSparseIntFeatures(
                                context, &sparse_int_feature_indices_list,
                                &sparse_int_feature_values_list,
                                &sparse_int_feature_shapes_list));

    // Infer batch size.
    const int64 batch_size = TensorUtils::InferBatchSize(
        dense_float_features_list, sparse_float_feature_shapes_list,
        sparse_int_feature_shapes_list);

    // Read batch features.
    boosted_trees::utils::BatchFeatures batch_features(batch_size);
    OP_REQUIRES_OK(
        context,
        batch_features.Initialize(
            TensorUtils::OpInputListToTensorVec(dense_float_features_list),
            TensorUtils::OpInputListToTensorVec(
                sparse_float_feature_indices_list),
            TensorUtils::OpInputListToTensorVec(
                sparse_float_feature_values_list),
            TensorUtils::OpInputListToTensorVec(
                sparse_float_feature_shapes_list),
            TensorUtils::OpInputListToTensorVec(
                sparse_int_feature_indices_list),
            TensorUtils::OpInputListToTensorVec(sparse_int_feature_values_list),
            TensorUtils::OpInputListToTensorVec(
                sparse_int_feature_shapes_list)));

    // Allocate output partitions vector.
    Tensor* partition_ids_t = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, {batch_size}, &partition_ids_t));
    thread::ThreadPool* const worker_threads =
        context->device()->tensorflow_cpu_worker_threads()->workers;
    learner::ExamplePartitioner::PartitionExamples(
        tree_config, batch_features, worker_threads->NumThreads(),
        worker_threads, partition_ids_t->vec<int32>().data());
  }

 private:
  bool use_locking_;
};

REGISTER_KERNEL_BUILDER(
    Name("GradientTreesPartitionExamples").Device(DEVICE_CPU),
    GradientTreesPartitionExamplesOp);

}  // namespace boosted_trees
}  // namespace tensorflow
