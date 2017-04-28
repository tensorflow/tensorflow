/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

// See docs in ../ops/sdca_ops.cc.

#define EIGEN_USE_THREADS

#include <stdint.h>
#include <atomic>
#include <limits>
#include <memory>
#include <new>
#include <string>
#include <vector>

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/device_base.h"
#include "tensorflow/core/framework/kernel_def_builder.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_def_builder.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/kernels/hinge-loss.h"
#include "tensorflow/core/kernels/logistic-loss.h"
#include "tensorflow/core/kernels/loss.h"
#include "tensorflow/core/kernels/sdca_internal.h"
#include "tensorflow/core/kernels/smooth-hinge-loss.h"
#include "tensorflow/core/kernels/squared-loss.h"
#include "tensorflow/core/lib/core/coding.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/gtl/inlined_vector.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/fingerprint.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/work_sharder.h"

namespace tensorflow {

namespace {

using sdca::Regularizations;
using sdca::Example;
using sdca::Examples;
using sdca::ExampleStatistics;
using sdca::ModelWeights;

struct ComputeOptions {
  ComputeOptions(OpKernelConstruction* const context) {
    string loss_type;
    OP_REQUIRES_OK(context, context->GetAttr("loss_type", &loss_type));
    if (loss_type == "logistic_loss") {
      loss_updater.reset(new LogisticLossUpdater);
    } else if (loss_type == "squared_loss") {
      loss_updater.reset(new SquaredLossUpdater);
    } else if (loss_type == "hinge_loss") {
      loss_updater.reset(new HingeLossUpdater);
    } else if (loss_type == "smooth_hinge_loss") {
      loss_updater.reset(new SmoothHingeLossUpdater);
    } else {
      OP_REQUIRES(context, false, errors::InvalidArgument(
                                      "Unsupported loss type: ", loss_type));
    }
    OP_REQUIRES_OK(context, context->GetAttr("adaptative", &adaptative));
    OP_REQUIRES_OK(
        context, context->GetAttr("num_sparse_features", &num_sparse_features));
    OP_REQUIRES_OK(context, context->GetAttr("num_sparse_features_with_values",
                                             &num_sparse_features_with_values));
    OP_REQUIRES_OK(context,
                   context->GetAttr("num_dense_features", &num_dense_features));
    OP_REQUIRES(
        context, num_sparse_features + num_dense_features > 0,
        errors::InvalidArgument("Requires at least one feature to train."));

    OP_REQUIRES(context, static_cast<int64>(num_sparse_features) +
                                 static_cast<int64>(num_dense_features) <=
                             std::numeric_limits<int>::max(),
                errors::InvalidArgument(
                    strings::Printf("Too many feature groups: %lld > %d",
                                    static_cast<int64>(num_sparse_features) +
                                        static_cast<int64>(num_dense_features),
                                    std::numeric_limits<int>::max())));
    OP_REQUIRES_OK(
        context, context->GetAttr("num_loss_partitions", &num_loss_partitions));
    OP_REQUIRES_OK(context, context->GetAttr("num_inner_iterations",
                                             &num_inner_iterations));
    OP_REQUIRES_OK(context, regularizations.Initialize(context));
  }

  std::unique_ptr<DualLossUpdater> loss_updater;
  int num_sparse_features = 0;
  int num_sparse_features_with_values = 0;
  int num_dense_features = 0;
  int num_inner_iterations = 0;
  int num_loss_partitions = 0;
  bool adaptative = false;
  Regularizations regularizations;
};

// TODO(shengx): The helper classes/methods are changed to support multiclass
// SDCA, which lead to changes within this function. Need to revisit the
// convergence once the multiclass SDCA is in.
void DoCompute(const ComputeOptions& options, OpKernelContext* const context) {
  ModelWeights model_weights;
  OP_REQUIRES_OK(context, model_weights.Initialize(context));

  Examples examples;
  OP_REQUIRES_OK(
      context,
      examples.Initialize(context, model_weights, options.num_sparse_features,
                          options.num_sparse_features_with_values,
                          options.num_dense_features));

  const Tensor* example_state_data_t;
  OP_REQUIRES_OK(context,
                 context->input("example_state_data", &example_state_data_t));
  TensorShape expected_example_state_shape({examples.num_examples(), 4});
  OP_REQUIRES(context,
              example_state_data_t->shape() == expected_example_state_shape,
              errors::InvalidArgument(
                  "Expected shape ", expected_example_state_shape.DebugString(),
                  " for example_state_data, got ",
                  example_state_data_t->shape().DebugString()));

  Tensor mutable_example_state_data_t(*example_state_data_t);
  auto example_state_data = mutable_example_state_data_t.matrix<float>();
  OP_REQUIRES_OK(context, context->set_output("out_example_state_data",
                                              mutable_example_state_data_t));

  if (options.adaptative) {
    OP_REQUIRES_OK(context,
                   examples.SampleAdaptativeProbabilities(
                       options.num_loss_partitions, options.regularizations,
                       model_weights, example_state_data, options.loss_updater,
                       /*num_weight_vectors =*/1));
  }

  mutex mu;
  Status train_step_status GUARDED_BY(mu);
  std::atomic<std::int64_t> atomic_index(-1);
  auto train_step = [&](const int64 begin, const int64 end) {
    // The static_cast here is safe since begin and end can be at most
    // num_examples which is an int.
    for (int id = static_cast<int>(begin); id < end; ++id) {
      const int64 example_index =
          examples.sampled_index(++atomic_index, options.adaptative);
      const Example& example = examples.example(example_index);
      const float dual = example_state_data(example_index, 0);
      const float example_weight = example.example_weight();
      float example_label = example.example_label();
      const Status conversion_status =
          options.loss_updater->ConvertLabel(&example_label);
      if (!conversion_status.ok()) {
        mutex_lock l(mu);
        train_step_status = conversion_status;
        // Return from this worker thread - the calling thread is
        // responsible for checking context status and returning on error.
        return;
      }

      // Compute wx, example norm weighted by regularization, dual loss,
      // primal loss.
      // For binary SDCA, num_weight_vectors should be one.
      const ExampleStatistics example_statistics =
          example.ComputeWxAndWeightedExampleNorm(
              options.num_loss_partitions, model_weights,
              options.regularizations, 1 /* num_weight_vectors */);

      const double new_dual = options.loss_updater->ComputeUpdatedDual(
          options.num_loss_partitions, example_label, example_weight, dual,
          example_statistics.wx[0], example_statistics.normalized_squared_norm);

      // Compute new weights.
      const double normalized_bounded_dual_delta =
          (new_dual - dual) * example_weight /
          options.regularizations.symmetric_l2();
      model_weights.UpdateDeltaWeights(
          context->eigen_cpu_device(), example,
          std::vector<double>{normalized_bounded_dual_delta});

      // Update example data.
      example_state_data(example_index, 0) = new_dual;
      example_state_data(example_index, 1) =
          options.loss_updater->ComputePrimalLoss(
              example_statistics.prev_wx[0], example_label, example_weight);
      example_state_data(example_index, 2) =
          options.loss_updater->ComputeDualLoss(dual, example_label,
                                                example_weight);
      example_state_data(example_index, 3) = example_weight;
    }
  };
  // TODO(sibyl-Aix6ihai): Tune this properly based on sparsity of the data,
  // number of cpus, and cost per example.
  const int64 kCostPerUnit = examples.num_features();
  const DeviceBase::CpuWorkerThreads& worker_threads =
      *context->device()->tensorflow_cpu_worker_threads();

  Shard(worker_threads.num_threads, worker_threads.workers,
        examples.num_examples(), kCostPerUnit, train_step);
  OP_REQUIRES_OK(context, train_step_status);
}

}  // namespace

class SdcaOptimizer : public OpKernel {
 public:
  explicit SdcaOptimizer(OpKernelConstruction* const context)
      : OpKernel(context), options_(context) {}

  void Compute(OpKernelContext* context) override {
    DoCompute(options_, context);
  }

 private:
  // TODO(sibyl-Aix6ihai): We could use the type-constraint on loss_type, and
  // template the entire class to avoid the virtual table lookup penalty in
  // the inner loop.
  ComputeOptions options_;
};
REGISTER_KERNEL_BUILDER(Name("SdcaOptimizer").Device(DEVICE_CPU),
                        SdcaOptimizer);

class SdcaShrinkL1 : public OpKernel {
 public:
  explicit SdcaShrinkL1(OpKernelConstruction* const context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context, regularizations_.Initialize(context));
  }

  void Compute(OpKernelContext* context) override {
    OpMutableInputList weights_inputs;
    OP_REQUIRES_OK(context,
                   context->mutable_input_list("weights", &weights_inputs));

    auto do_work = [&](const int64 begin, const int64 end) {
      for (int i = begin; i < end; ++i) {
        auto prox_w = weights_inputs.at(i, /*lock_held=*/true).flat<float>();
        prox_w.device(context->eigen_cpu_device()) =
            regularizations_.EigenShrinkVector(prox_w);
      }
    };

    if (weights_inputs.size() > 0) {
      int64 num_weights = 0;
      for (int i = 0; i < weights_inputs.size(); ++i) {
        num_weights += weights_inputs.at(i, /*lock_held=*/true).NumElements();
      }
      // TODO(sibyl-Aix6ihai): Tune this value.
      const int64 kCostPerUnit = (num_weights * 50) / weights_inputs.size();
      const DeviceBase::CpuWorkerThreads& worker_threads =
          *context->device()->tensorflow_cpu_worker_threads();
      Shard(worker_threads.num_threads, worker_threads.workers,
            weights_inputs.size(), kCostPerUnit, do_work);
    }
  }

 private:
  Regularizations regularizations_;
};
REGISTER_KERNEL_BUILDER(Name("SdcaShrinkL1").Device(DEVICE_CPU), SdcaShrinkL1);

// Computes platform independent, compact and unique (with very high
// probability) representation of an example id. It shouldn't be put in
// persistent storage, as its implementation may change in the future.
//
// The current probability of at least one collision for 1B example_ids is
// approximately 10^-21 (ie 2^60 / 2^129).
class SdcaFprint : public OpKernel {
 public:
  explicit SdcaFprint(OpKernelConstruction* const context)
      : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    const Tensor& input = context->input(0);
    OP_REQUIRES(context, TensorShapeUtils::IsVector(input.shape()),
                errors::InvalidArgument("Input must be a vector, got shape ",
                                        input.shape().DebugString()));
    Tensor* out;
    const int64 num_elements = input.NumElements();
    OP_REQUIRES_OK(context, context->allocate_output(
                                0, TensorShape({num_elements, 2}), &out));

    const auto in_values = input.flat<string>();
    auto out_values = out->matrix<int64>();

    for (int64 i = 0; i < num_elements; ++i) {
      const Fprint128 fprint = Fingerprint128(in_values(i));
      // Never return 0 or 1 as the first value of the hash to allow these to
      // safely be used as sentinel values (e.g. dense hash table empty key).
      out_values(i, 0) = TF_PREDICT_TRUE(fprint.low64 >= 2)
                             ? fprint.low64
                             : fprint.low64 + ~static_cast<uint64>(1);
      out_values(i, 1) = fprint.high64;
    }
  }
};
REGISTER_KERNEL_BUILDER(Name("SdcaFprint").Device(DEVICE_CPU), SdcaFprint);

}  // namespace tensorflow
