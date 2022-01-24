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
#include <iterator>
#include <string>
#include <vector>

#include "tensorflow/core/framework/device_base.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/kernels/boosted_trees/quantiles/quantile_stream_resource.h"
#include "tensorflow/core/kernels/boosted_trees/quantiles/weighted_quantiles_stream.h"
#include "tensorflow/core/kernels/boosted_trees/quantiles/weighted_quantiles_summary.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/refcount.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/work_sharder.h"

namespace tensorflow {

const char* const kExampleWeightsName = "example_weights";
const char* const kMaxElementsName = "max_elements";
const char* const kGenerateQuantiles = "generate_quantiles";
const char* const kNumBucketsName = "num_buckets";
const char* const kEpsilonName = "epsilon";
const char* const kBucketBoundariesName = "bucket_boundaries";
const char* const kBucketsName = "buckets";
const char* const kSummariesName = "summaries";
const char* const kNumStreamsName = "num_streams";
const char* const kNumFeaturesName = "num_features";
const char* const kFloatFeaturesName = "float_values";
const char* const kResourceHandleName = "quantile_stream_resource_handle";

using QuantileStreamResource = BoostedTreesQuantileStreamResource;
using QuantileStream =
    boosted_trees::quantiles::WeightedQuantilesStream<float, float>;
using QuantileSummary =
    boosted_trees::quantiles::WeightedQuantilesSummary<float, float>;
using QuantileSummaryEntry =
    boosted_trees::quantiles::WeightedQuantilesSummary<float,
                                                       float>::SummaryEntry;

// Generates quantiles on a finalized QuantileStream.
std::vector<float> GenerateBoundaries(const QuantileStream& stream,
                                      const int64_t num_boundaries) {
  std::vector<float> boundaries = stream.GenerateBoundaries(num_boundaries);

  // Uniquify elements as we may get dupes.
  auto end_it = std::unique(boundaries.begin(), boundaries.end());
  boundaries.resize(std::distance(boundaries.begin(), end_it));
  return boundaries;
}

// Generates quantiles on a finalized QuantileStream.
std::vector<float> GenerateQuantiles(const QuantileStream& stream,
                                     const int64_t num_quantiles) {
  // Do not de-dup boundaries. Exactly num_quantiles+1 boundary values
  // will be returned.
  std::vector<float> boundaries = stream.GenerateQuantiles(num_quantiles - 1);
  CHECK_EQ(boundaries.size(), num_quantiles);
  return boundaries;
}

std::vector<float> GetBuckets(const int32_t feature,
                              const OpInputList& buckets_list) {
  const auto& buckets = buckets_list[feature].flat<float>();
  std::vector<float> buckets_vector(buckets.data(),
                                    buckets.data() + buckets.size());
  return buckets_vector;
}

REGISTER_RESOURCE_HANDLE_KERNEL(BoostedTreesQuantileStreamResource);

REGISTER_KERNEL_BUILDER(
    Name("IsBoostedTreesQuantileStreamResourceInitialized").Device(DEVICE_CPU),
    IsResourceInitialized<BoostedTreesQuantileStreamResource>);

class BoostedTreesCreateQuantileStreamResourceOp : public OpKernel {
 public:
  explicit BoostedTreesCreateQuantileStreamResourceOp(
      OpKernelConstruction* const context)
      : OpKernel(context) {
    VLOG(1) << "Boosted Trees kernels in TF are deprecated. Please use "
            << "TensorFlow Decision Forests instead "
            << "(https://github.com/tensorflow/decision-forests).\n";
    OP_REQUIRES_OK(context, context->GetAttr(kMaxElementsName, &max_elements_));
  }

  void Compute(OpKernelContext* context) override {
    // Only create one, if one does not exist already. Report status for all
    // other exceptions. If one already exists, it unrefs the new one.
    // An epsilon value of zero could cause performance issues and is therefore,
    // disallowed.
    const Tensor* epsilon_t;
    OP_REQUIRES_OK(context, context->input(kEpsilonName, &epsilon_t));
    OP_REQUIRES(context, TensorShapeUtils::IsScalar(epsilon_t->shape()),
                errors::InvalidArgument(
                    "epsilon must be a scalar, got a tensor of shape ",
                    epsilon_t->shape().DebugString()));
    float epsilon = epsilon_t->scalar<float>()();
    OP_REQUIRES(
        context, epsilon > 0,
        errors::InvalidArgument("An epsilon value of zero is not allowed."));

    const Tensor* num_streams_t;
    OP_REQUIRES_OK(context, context->input(kNumStreamsName, &num_streams_t));
    OP_REQUIRES(context, TensorShapeUtils::IsScalar(num_streams_t->shape()),
                errors::InvalidArgument(
                    "num_streams must be a scalar, got a tensor of shape ",
                    num_streams_t->shape().DebugString()));
    int64_t num_streams = num_streams_t->scalar<int64_t>()();
    OP_REQUIRES(context, num_streams >= 0,
                errors::InvalidArgument(
                    "Num_streams input cannot be a negative integer"));

    auto result =
        new QuantileStreamResource(epsilon, max_elements_, num_streams);
    auto status = CreateResource(context, HandleFromInput(context, 0), result);
    if (!status.ok() && status.code() != tensorflow::error::ALREADY_EXISTS) {
      OP_REQUIRES(context, false, status);
    }
  }

 private:
  // An upper bound on the number of entries that the summaries might have
  // for a feature.
  int64_t max_elements_;
};

REGISTER_KERNEL_BUILDER(
    Name("BoostedTreesCreateQuantileStreamResource").Device(DEVICE_CPU),
    BoostedTreesCreateQuantileStreamResourceOp);

class BoostedTreesMakeQuantileSummariesOp : public OpKernel {
 public:
  explicit BoostedTreesMakeQuantileSummariesOp(
      OpKernelConstruction* const context)
      : OpKernel(context) {
    VLOG(1) << "Boosted Trees kernels in TF are deprecated. Please use "
            << "TensorFlow Decision Forests instead "
            << "(https://github.com/tensorflow/decision-forests).\n";
    OP_REQUIRES_OK(context, context->GetAttr(kNumFeaturesName, &num_features_));
  }

  void Compute(OpKernelContext* const context) override {
    // Read float features list;
    OpInputList float_features_list;
    OP_REQUIRES_OK(
        context, context->input_list(kFloatFeaturesName, &float_features_list));

    // Parse example weights and get batch size.
    const Tensor* example_weights_t;
    OP_REQUIRES_OK(context,
                   context->input(kExampleWeightsName, &example_weights_t));
    OP_REQUIRES(context, float_features_list.size() > 0,
                errors::Internal("Got empty feature list"));
    auto example_weights = example_weights_t->flat<float>();
    const int64_t weight_size = example_weights.size();
    const int64_t batch_size = float_features_list[0].flat<float>().size();
    OP_REQUIRES(
        context, weight_size == 1 || weight_size == batch_size,
        errors::InvalidArgument(strings::Printf(
            "Weights should be a single value or same size as features.")));
    const Tensor* epsilon_t;
    OP_REQUIRES_OK(context, context->input(kEpsilonName, &epsilon_t));
    OP_REQUIRES(context, TensorShapeUtils::IsScalar(epsilon_t->shape()),
                errors::InvalidArgument(
                    "epsilon must be a scalar, got a tensor of shape ",
                    epsilon_t->shape().DebugString()));
    float epsilon = epsilon_t->scalar<float>()();

    OpOutputList summaries_output_list;
    OP_REQUIRES_OK(
        context, context->output_list(kSummariesName, &summaries_output_list));

    auto do_quantile_summary_gen = [&](const int64_t begin, const int64_t end) {
      // Iterating features.
      for (int64_t index = begin; index < end; index++) {
        const auto feature_values = float_features_list[index].flat<float>();
        QuantileStream stream(epsilon, batch_size + 1);
        // Run quantile summary generation.
        for (int64_t j = 0; j < batch_size; j++) {
          stream.PushEntry(feature_values(j), (weight_size > 1)
                                                  ? example_weights(j)
                                                  : example_weights(0));
        }
        stream.Finalize();
        const auto summary_entry_list = stream.GetFinalSummary().GetEntryList();
        Tensor* output_t;
        OP_REQUIRES_OK(
            context,
            summaries_output_list.allocate(
                index,
                TensorShape(
                    {static_cast<int64_t>(summary_entry_list.size()), 4}),
                &output_t));
        auto output = output_t->matrix<float>();
        for (auto row = 0; row < summary_entry_list.size(); row++) {
          const auto& entry = summary_entry_list[row];
          output(row, 0) = entry.value;
          output(row, 1) = entry.weight;
          output(row, 2) = entry.min_rank;
          output(row, 3) = entry.max_rank;
        }
      }
    };
    // TODO(tanzheny): comment on the magic number.
    const int64_t kCostPerUnit = 500 * batch_size;
    const DeviceBase::CpuWorkerThreads& worker_threads =
        *context->device()->tensorflow_cpu_worker_threads();
    Shard(worker_threads.num_threads, worker_threads.workers, num_features_,
          kCostPerUnit, do_quantile_summary_gen);
  }

 private:
  int64_t num_features_;
};

REGISTER_KERNEL_BUILDER(
    Name("BoostedTreesMakeQuantileSummaries").Device(DEVICE_CPU),
    BoostedTreesMakeQuantileSummariesOp);

class BoostedTreesFlushQuantileSummariesOp : public OpKernel {
 public:
  explicit BoostedTreesFlushQuantileSummariesOp(
      OpKernelConstruction* const context)
      : OpKernel(context) {
    VLOG(1) << "Boosted Trees kernels in TF are deprecated. Please use "
            << "TensorFlow Decision Forests instead "
            << "(https://github.com/tensorflow/decision-forests).\n";
    OP_REQUIRES_OK(context, context->GetAttr(kNumFeaturesName, &num_features_));
  }

  void Compute(OpKernelContext* const context) override {
    ResourceHandle handle;
    OP_REQUIRES_OK(context,
                   HandleFromInput(context, kResourceHandleName, &handle));
    core::RefCountPtr<QuantileStreamResource> stream_resource;
    OP_REQUIRES_OK(context, LookupResource(context, handle, &stream_resource));
    // Remove the reference at the end of this scope.
    mutex_lock l(*stream_resource->mutex());

    OpOutputList summaries_output_list;
    OP_REQUIRES_OK(
        context, context->output_list(kSummariesName, &summaries_output_list));

    auto do_quantile_summary_gen = [&](const int64_t begin, const int64_t end) {
      // Iterating features.
      for (int64_t index = begin; index < end; index++) {
        QuantileStream* stream = stream_resource->stream(index);
        stream->Finalize();

        const auto summary_list = stream->GetFinalSummary().GetEntryList();
        Tensor* output_t;
        const int64_t summary_list_size =
            static_cast<int64_t>(summary_list.size());
        OP_REQUIRES_OK(context, summaries_output_list.allocate(
                                    index, TensorShape({summary_list_size, 4}),
                                    &output_t));
        auto output = output_t->matrix<float>();
        for (auto row = 0; row < summary_list_size; row++) {
          const auto& entry = summary_list[row];
          output(row, 0) = entry.value;
          output(row, 1) = entry.weight;
          output(row, 2) = entry.min_rank;
          output(row, 3) = entry.max_rank;
        }
      }
    };
    // TODO(tanzheny): comment on the magic number.
    const int64_t kCostPerUnit = 500 * num_features_;
    const DeviceBase::CpuWorkerThreads& worker_threads =
        *context->device()->tensorflow_cpu_worker_threads();
    Shard(worker_threads.num_threads, worker_threads.workers, num_features_,
          kCostPerUnit, do_quantile_summary_gen);
    stream_resource->ResetStreams();
  }

 private:
  int64_t num_features_;
};

REGISTER_KERNEL_BUILDER(
    Name("BoostedTreesFlushQuantileSummaries").Device(DEVICE_CPU),
    BoostedTreesFlushQuantileSummariesOp);

class BoostedTreesQuantileStreamResourceAddSummariesOp : public OpKernel {
 public:
  explicit BoostedTreesQuantileStreamResourceAddSummariesOp(
      OpKernelConstruction* const context)
      : OpKernel(context) {
    VLOG(1) << "Boosted Trees kernels in TF are deprecated. Please use "
            << "TensorFlow Decision Forests instead "
            << "(https://github.com/tensorflow/decision-forests).\n";
  }

  void Compute(OpKernelContext* context) override {
    ResourceHandle handle;
    OP_REQUIRES_OK(context,
                   HandleFromInput(context, kResourceHandleName, &handle));
    core::RefCountPtr<QuantileStreamResource> stream_resource;
    // Create a reference to the underlying resource using the handle.
    OP_REQUIRES_OK(context, LookupResource(context, handle, &stream_resource));
    // Remove the reference at the end of this scope.
    mutex_lock l(*stream_resource->mutex());

    OpInputList summaries_list;
    OP_REQUIRES_OK(context,
                   context->input_list(kSummariesName, &summaries_list));
    auto num_streams = stream_resource->num_streams();
    OP_REQUIRES(
        context, num_streams == summaries_list.size(),
        errors::Internal("Expected num_streams == summaries_list.size(), got ",
                         num_streams, " vs ", summaries_list.size()));

    auto do_quantile_add_summary = [&](const int64_t begin, const int64_t end) {
      // Iterating all features.
      for (int64_t feature_idx = begin; feature_idx < end; ++feature_idx) {
        QuantileStream* stream = stream_resource->stream(feature_idx);
        if (stream->IsFinalized()) {
          VLOG(1) << "QuantileStream has already been finalized for feature"
                  << feature_idx << ".";
          continue;
        }
        const Tensor& summaries = summaries_list[feature_idx];
        const auto summary_values = summaries.matrix<float>();
        const auto& tensor_shape = summaries.shape();
        const int64_t entries_size = tensor_shape.dim_size(0);
        OP_REQUIRES(
            context, tensor_shape.dim_size(1) == 4,
            errors::Internal("Expected tensor_shape.dim_size(1) == 4, got ",
                             tensor_shape.dim_size(1)));
        std::vector<QuantileSummaryEntry> summary_entries;
        summary_entries.reserve(entries_size);
        for (int64_t i = 0; i < entries_size; i++) {
          float value = summary_values(i, 0);
          float weight = summary_values(i, 1);
          float min_rank = summary_values(i, 2);
          float max_rank = summary_values(i, 3);
          QuantileSummaryEntry entry(value, weight, min_rank, max_rank);
          summary_entries.push_back(entry);
        }
        stream_resource->stream(feature_idx)->PushSummary(summary_entries);
      }
    };

    // TODO(tanzheny): comment on the magic number.
    const int64_t kCostPerUnit = 500 * num_streams;
    const DeviceBase::CpuWorkerThreads& worker_threads =
        *context->device()->tensorflow_cpu_worker_threads();
    Shard(worker_threads.num_threads, worker_threads.workers, num_streams,
          kCostPerUnit, do_quantile_add_summary);
  }
};

REGISTER_KERNEL_BUILDER(
    Name("BoostedTreesQuantileStreamResourceAddSummaries").Device(DEVICE_CPU),
    BoostedTreesQuantileStreamResourceAddSummariesOp);

class BoostedTreesQuantileStreamResourceDeserializeOp : public OpKernel {
 public:
  explicit BoostedTreesQuantileStreamResourceDeserializeOp(
      OpKernelConstruction* const context)
      : OpKernel(context) {
    VLOG(1) << "Boosted Trees kernels in TF are deprecated. Please use "
            << "TensorFlow Decision Forests instead "
            << "(https://github.com/tensorflow/decision-forests).\n";
    OP_REQUIRES_OK(context, context->GetAttr(kNumStreamsName, &num_features_));
  }

  void Compute(OpKernelContext* context) override {
    core::RefCountPtr<QuantileStreamResource> streams_resource;
    // Create a reference to the underlying resource using the handle.
    OP_REQUIRES_OK(context, LookupResource(context, HandleFromInput(context, 0),
                                           &streams_resource));
    // Remove the reference at the end of this scope.
    mutex_lock l(*streams_resource->mutex());

    OpInputList bucket_boundaries_list;
    OP_REQUIRES_OK(context, context->input_list(kBucketBoundariesName,
                                                &bucket_boundaries_list));

    auto do_quantile_deserialize = [&](const int64_t begin, const int64_t end) {
      // Iterating over all streams.
      for (int64_t stream_idx = begin; stream_idx < end; stream_idx++) {
        const Tensor& bucket_boundaries_t = bucket_boundaries_list[stream_idx];
        OP_REQUIRES(
            context, TensorShapeUtils::IsVector(bucket_boundaries_t.shape()),
            errors::InvalidArgument("bucket boundaries for each stream must be "
                                    "a vector, received shape ",
                                    bucket_boundaries_t.shape().DebugString(),
                                    " for stream ", stream_idx));
        const auto& bucket_boundaries = bucket_boundaries_t.vec<float>();
        std::vector<float> result;
        result.reserve(bucket_boundaries.size());
        for (size_t i = 0; i < bucket_boundaries.size(); ++i) {
          result.push_back(bucket_boundaries(i));
        }
        streams_resource->set_boundaries(result, stream_idx);
      }
    };

    // TODO(tanzheny): comment on the magic number.
    const int64_t kCostPerUnit = 500 * num_features_;
    const DeviceBase::CpuWorkerThreads& worker_threads =
        *context->device()->tensorflow_cpu_worker_threads();
    Shard(worker_threads.num_threads, worker_threads.workers, num_features_,
          kCostPerUnit, do_quantile_deserialize);
  }

 private:
  int64_t num_features_;
};

REGISTER_KERNEL_BUILDER(
    Name("BoostedTreesQuantileStreamResourceDeserialize").Device(DEVICE_CPU),
    BoostedTreesQuantileStreamResourceDeserializeOp);

class BoostedTreesQuantileStreamResourceFlushOp : public OpKernel {
 public:
  explicit BoostedTreesQuantileStreamResourceFlushOp(
      OpKernelConstruction* const context)
      : OpKernel(context) {
    VLOG(1) << "Boosted Trees kernels in TF are deprecated. Please use "
            << "TensorFlow Decision Forests instead "
            << "(https://github.com/tensorflow/decision-forests).\n";
    OP_REQUIRES_OK(context,
                   context->GetAttr(kGenerateQuantiles, &generate_quantiles_));
  }

  void Compute(OpKernelContext* context) override {
    ResourceHandle handle;
    OP_REQUIRES_OK(context,
                   HandleFromInput(context, kResourceHandleName, &handle));
    core::RefCountPtr<QuantileStreamResource> stream_resource;
    // Create a reference to the underlying resource using the handle.
    OP_REQUIRES_OK(context, LookupResource(context, handle, &stream_resource));
    // Remove the reference at the end of this scope.
    mutex_lock l(*stream_resource->mutex());

    const Tensor* num_buckets_t;
    OP_REQUIRES_OK(context, context->input(kNumBucketsName, &num_buckets_t));
    OP_REQUIRES(context, TensorShapeUtils::IsScalar(num_buckets_t->shape()),
                errors::InvalidArgument(
                    "num_buckets must be a scalar, got a tensor of shape ",
                    num_buckets_t->shape().DebugString()));
    const int64_t num_buckets = num_buckets_t->scalar<int64_t>()();
    const int64_t num_streams = stream_resource->num_streams();

    auto do_quantile_flush = [&](const int64_t begin, const int64_t end) {
      // Iterating over all streams.
      for (int64_t stream_idx = begin; stream_idx < end; ++stream_idx) {
        QuantileStream* stream = stream_resource->stream(stream_idx);
        stream->Finalize();
        stream_resource->set_boundaries(
            generate_quantiles_ ? GenerateQuantiles(*stream, num_buckets)
                                : GenerateBoundaries(*stream, num_buckets),
            stream_idx);
      }
    };

    // TODO(tanzheny): comment on the magic number.
    const int64_t kCostPerUnit = 500 * num_streams;
    const DeviceBase::CpuWorkerThreads& worker_threads =
        *context->device()->tensorflow_cpu_worker_threads();
    Shard(worker_threads.num_threads, worker_threads.workers, num_streams,
          kCostPerUnit, do_quantile_flush);

    stream_resource->ResetStreams();
    stream_resource->set_buckets_ready(true);
  }

 private:
  bool generate_quantiles_;
};

REGISTER_KERNEL_BUILDER(
    Name("BoostedTreesQuantileStreamResourceFlush").Device(DEVICE_CPU),
    BoostedTreesQuantileStreamResourceFlushOp);

class BoostedTreesQuantileStreamResourceGetBucketBoundariesOp
    : public OpKernel {
 public:
  explicit BoostedTreesQuantileStreamResourceGetBucketBoundariesOp(
      OpKernelConstruction* const context)
      : OpKernel(context) {
    VLOG(1) << "Boosted Trees kernels in TF are deprecated. Please use "
            << "TensorFlow Decision Forests instead "
            << "(https://github.com/tensorflow/decision-forests).\n";
    OP_REQUIRES_OK(context, context->GetAttr(kNumFeaturesName, &num_features_));
  }

  void Compute(OpKernelContext* const context) override {
    ResourceHandle handle;
    OP_REQUIRES_OK(context,
                   HandleFromInput(context, kResourceHandleName, &handle));
    core::RefCountPtr<QuantileStreamResource> stream_resource;
    // Create a reference to the underlying resource using the handle.
    OP_REQUIRES_OK(context, LookupResource(context, handle, &stream_resource));
    // Remove the reference at the end of this scope.
    mutex_lock l(*stream_resource->mutex());

    const int64_t num_streams = stream_resource->num_streams();
    OP_REQUIRES(context, num_streams == num_features_,
                errors::Internal("Expected num_streams == num_features_, got ",
                                 num_streams, " vs ", num_features_));
    OpOutputList bucket_boundaries_list;
    OP_REQUIRES_OK(context, context->output_list(kBucketBoundariesName,
                                                 &bucket_boundaries_list));

    auto do_quantile_get_buckets = [&](const int64_t begin, const int64_t end) {
      // Iterating over all streams.
      for (int64_t stream_idx = begin; stream_idx < end; stream_idx++) {
        const auto& boundaries = stream_resource->boundaries(stream_idx);
        Tensor* bucket_boundaries_t = nullptr;
        OP_REQUIRES_OK(
            context, bucket_boundaries_list.allocate(
                         stream_idx, {static_cast<int64_t>(boundaries.size())},
                         &bucket_boundaries_t));
        auto* quantiles_flat = bucket_boundaries_t->flat<float>().data();
        memcpy(quantiles_flat, boundaries.data(),
               sizeof(float) * boundaries.size());
      }
    };

    // TODO(tanzheny): comment on the magic number.
    const int64_t kCostPerUnit = 500 * num_streams;
    const DeviceBase::CpuWorkerThreads& worker_threads =
        *context->device()->tensorflow_cpu_worker_threads();
    Shard(worker_threads.num_threads, worker_threads.workers, num_streams,
          kCostPerUnit, do_quantile_get_buckets);
  }

 private:
  int64_t num_features_;
};

REGISTER_KERNEL_BUILDER(
    Name("BoostedTreesQuantileStreamResourceGetBucketBoundaries")
        .Device(DEVICE_CPU),
    BoostedTreesQuantileStreamResourceGetBucketBoundariesOp);

// Given the calculated quantiles thresholds and input data, this operation
// converts the input features into the buckets (categorical values), depending
// on which quantile they fall into.
class BoostedTreesBucketizeOp : public OpKernel {
 public:
  explicit BoostedTreesBucketizeOp(OpKernelConstruction* const context)
      : OpKernel(context) {
    VLOG(1) << "Boosted Trees kernels in TF are deprecated. Please use "
            << "TensorFlow Decision Forests instead "
            << "(https://github.com/tensorflow/decision-forests).\n";
    OP_REQUIRES_OK(context, context->GetAttr(kNumFeaturesName, &num_features_));
  }

  void Compute(OpKernelContext* const context) override {
    // Read float features list;
    OpInputList float_features_list;
    OP_REQUIRES_OK(
        context, context->input_list(kFloatFeaturesName, &float_features_list));
    OpInputList bucket_boundaries_list;
    OP_REQUIRES_OK(context, context->input_list(kBucketBoundariesName,
                                                &bucket_boundaries_list));
    OP_REQUIRES(context,
                tensorflow::TensorShapeUtils::IsVector(
                    bucket_boundaries_list[0].shape()),
                errors::InvalidArgument(
                    strings::Printf("Buckets should be flat vectors.")));
    OpOutputList buckets_list;
    OP_REQUIRES_OK(context, context->output_list(kBucketsName, &buckets_list));

    auto do_quantile_get_quantiles = [&](const int64_t begin,
                                         const int64_t end) {
      // Iterating over all resources
      for (int64_t feature_idx = begin; feature_idx < end; feature_idx++) {
        const Tensor& values_tensor = float_features_list[feature_idx];
        const int64_t num_values = values_tensor.dim_size(0);

        Tensor* output_t = nullptr;
        OP_REQUIRES_OK(context,
                       buckets_list.allocate(
                           feature_idx, TensorShape({num_values}), &output_t));
        auto output = output_t->flat<int32>();

        const std::vector<float>& bucket_boundaries_vector =
            GetBuckets(feature_idx, bucket_boundaries_list);
        auto flat_values = values_tensor.flat<float>();
        const auto& iter_begin = bucket_boundaries_vector.begin();
        const auto& iter_end = bucket_boundaries_vector.end();
        for (int64_t instance = 0; instance < num_values; instance++) {
          if (iter_begin == iter_end) {
            output(instance) = 0;
            continue;
          }
          const float value = flat_values(instance);
          auto bucket_iter = std::lower_bound(iter_begin, iter_end, value);
          if (bucket_iter == iter_end) {
            --bucket_iter;
          }
          const int32_t bucket = static_cast<int32>(bucket_iter - iter_begin);
          // Bucket id.
          output(instance) = bucket;
        }
      }
    };

    // TODO(tanzheny): comment on the magic number.
    const int64_t kCostPerUnit = 500 * num_features_;
    const DeviceBase::CpuWorkerThreads& worker_threads =
        *context->device()->tensorflow_cpu_worker_threads();
    Shard(worker_threads.num_threads, worker_threads.workers, num_features_,
          kCostPerUnit, do_quantile_get_quantiles);
  }

 private:
  int64_t num_features_;
};

REGISTER_KERNEL_BUILDER(Name("BoostedTreesBucketize").Device(DEVICE_CPU),
                        BoostedTreesBucketizeOp);

}  // namespace tensorflow
