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

#include "tensorflow/contrib/boosted_trees/lib/quantiles/weighted_quantiles_stream.h"
#include "tensorflow/contrib/boosted_trees/lib/utils/parallel_for.h"
#include "tensorflow/contrib/boosted_trees/lib/utils/tensor_utils.h"
#include "tensorflow/contrib/boosted_trees/proto/quantiles.pb.h"
#include "tensorflow/contrib/boosted_trees/resources/quantile_stream_resource.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/work_sharder.h"

namespace tensorflow {

using ::boosted_trees::QuantileConfig;
using boosted_trees::utils::TensorUtils;
using boosted_trees::QuantileStreamResource;

namespace {
const char* const kExampleWeightsName = "example_weights";
const char* const kMaxElementsName = "max_elements";
const char* const kNextStampTokenName = "next_stamp_token";
const char* const kStampTokenName = "stamp_token";
const char* const kAreBucketsReadyName = "are_buckets_ready";
const char* const kGenerateQuantiles = "generate_quantiles";
// Names for sparse arguments.
const char* const kNumSparseFeaturesName = "num_sparse_features";
const char* const kSparseBucketsName = "sparse_buckets";
const char* const kSparseValuesName = "sparse_values";
const char* const kSparseIndicesName = "sparse_indices";
const char* const kSparseSummariesName = "sparse_summaries";
const char* const kSparseConfigName = "sparse_config";
const char* const kSparseOutputTensorName = "sparse_quantiles";
// Names for dense arguments.
const char* const kDenseBucketsName = "dense_buckets";
const char* const kDenseConfigName = "dense_config";
const char* const kDenseOutputTensorName = "dense_quantiles";
const char* const kDenseSummariesName = "dense_summaries";
const char* const kDenseValuesName = "dense_values";
const char* const kNumDenseFeaturesName = "num_dense_features";
const char* const kResourceHandlesName = "quantile_accumulator_handles";
const char* const kNumQuantilesName = "num_quantiles";
const char* const kEpsilonName = "epsilon";
const char* const kBucketsName = "buckets";
const char* const kStreamStateName = "stream_state";
const char* const kSummariesName = "summaries";

using QuantileStream =
    boosted_trees::quantiles::WeightedQuantilesStream<float, float>;
using QuantileSummary =
    boosted_trees::quantiles::WeightedQuantilesSummary<float, float>;
using QuantileSummaryEntry =
    boosted_trees::quantiles::WeightedQuantilesSummary<float,
                                                       float>::SummaryEntry;

std::vector<float> GetBuckets(const int32 feature,
                              const OpInputList& buckets_list) {
  const auto& buckets = buckets_list[feature].flat<float>();
  std::vector<float> buckets_vector(buckets.data(),
                                    buckets.data() + buckets.size());
  return buckets_vector;
}

int32 GetFeatureDimension(const int32 feature_index, const int64 instance,
                          const OpInputList* const indices_list) {
  if (indices_list != nullptr) {
    // Sparse multidimensional.
    return (*indices_list)[feature_index].matrix<int64>()(instance, 1);
  }
  // No indices, assume one-dimensional tensor.
  return 0;
}

// Allows quantization for each of multiple dimensions of a sparse feature.
void QuantizeFeatures(
    const string& output_name, const OpInputList& values_list,
    const OpInputList& buckets_list,
    const OpInputList* const
        indices_list /** Optional, provide for sparse features **/,
    OpKernelContext* const context) {
  if (values_list.size() == 0) {
    return;
  }
  OpOutputList output_list;
  OP_REQUIRES_OK(context, context->output_list(output_name, &output_list));

  for (int32 feature_index = 0; feature_index < values_list.size();
       ++feature_index) {
    const Tensor& values_tensor = values_list[feature_index];
    const int64 num_values = values_tensor.dim_size(0);

    Tensor* output_t = nullptr;
    // Output will have bucket id and dimension of the features for that bucket.
    OP_REQUIRES_OK(
        context, output_list.allocate(feature_index,
                                      TensorShape({num_values, 2}), &output_t));

    auto output = output_t->matrix<int32>();

    const std::vector<float>& buckets_vector =
        GetBuckets(feature_index, buckets_list);
    auto flat_values = values_tensor.flat<float>();
    for (int64 instance = 0; instance < num_values; ++instance) {
      const float value = flat_values(instance);
      auto bucket_iter =
          std::lower_bound(buckets_vector.begin(), buckets_vector.end(), value);
      if (bucket_iter == buckets_vector.end()) {
        --bucket_iter;
      }
      const int32 bucket =
          static_cast<int32>(bucket_iter - buckets_vector.begin());
      // Bucket id.
      output(instance, 0) = bucket;
      // Dimension.
      output(instance, 1) =
          GetFeatureDimension(feature_index, instance, indices_list);
    }
  }
}

// Validates attributes for the quantile ops.
Status ReadAndValidateAttributes(OpKernelConstruction* const context,
                                 int* num_dense_features,
                                 int* num_sparse_features) {
  TF_RETURN_IF_ERROR(
      context->GetAttr(kNumDenseFeaturesName, num_dense_features));
  TF_RETURN_IF_ERROR(
      context->GetAttr(kNumSparseFeaturesName, num_sparse_features));
  if ((*num_dense_features) + (*num_sparse_features) == 0) {
    return errors::InvalidArgument(
        "Please provide at least sparse or dense features.");
  }
  return Status::OK();
}

void ParseConfig(OpKernelConstruction* const context, const string& name,
                 std::vector<QuantileConfig>* output) {
  std::vector<string> serialized_config;
  OP_REQUIRES_OK(context, context->GetAttr(name, &serialized_config));
  output->reserve(serialized_config.size());
  QuantileConfig tmp;
  for (const auto& serialized_string : serialized_config) {
    OP_REQUIRES(context, tmp.ParseFromString(serialized_string),
                errors::InvalidArgument("Malformed QuantileConfig passed in."));
    output->push_back(tmp);
  }
}

// Generates quantiles on a finalized QuantileStream.
std::vector<float> GenerateBoundaries(const QuantileStream& stream,
                                      int num_boundaries) {
  std::vector<float> boundaries = stream.GenerateBoundaries(num_boundaries);

  // Uniquify elements as we may get dupes.
  auto end_it = std::unique(boundaries.begin(), boundaries.end());
  boundaries.resize(std::distance(boundaries.begin(), end_it));
  return boundaries;
}

// Generates quantiles on a finalized QuantileStream.
std::vector<float> GenerateQuantiles(const QuantileStream& stream,
                                     int num_quantiles) {
  // Do not de-dup boundaries. Exactly num_quantiles+1 boundary values
  // will be returned.
  std::vector<float> boundaries = stream.GenerateQuantiles(num_quantiles);
  CHECK_EQ(boundaries.size(), num_quantiles + 1);
  return boundaries;
}

// Copies quantiles to output list.
void CopyBoundaries(OpKernelContext* const context,
                    const std::vector<float>& boundaries, const int64 index,
                    OpOutputList* output_list) {
  // Output to tensor.
  Tensor* output_t = nullptr;
  OP_REQUIRES_OK(
      context, output_list->allocate(
                   index, {static_cast<int64>(boundaries.size())}, &output_t));
  auto* quantiles_flat = output_t->flat<float>().data();
  memcpy(quantiles_flat, boundaries.data(), sizeof(float) * boundaries.size());
}

void CopySummaryToProto(const QuantileSummary& summary,
                        ::boosted_trees::QuantileSummaryState* summary_proto) {
  summary_proto->mutable_entries()->Reserve(summary.Size());
  for (const auto& entry : summary.GetEntryList()) {
    auto* new_entry = summary_proto->add_entries();
    new_entry->set_value(entry.value);
    new_entry->set_weight(entry.weight);
    new_entry->set_min_rank(entry.min_rank);
    new_entry->set_max_rank(entry.max_rank);
  }
}

}  // namespace

// Accumulator for Quantile Summaries.
REGISTER_RESOURCE_HANDLE_KERNEL(QuantileStreamResource);

REGISTER_KERNEL_BUILDER(
    Name("QuantileAccumulatorIsInitialized").Device(DEVICE_CPU),
    IsResourceInitialized<QuantileStreamResource>);

class CreateQuantileAccumulatorOp : public OpKernel {
 public:
  explicit CreateQuantileAccumulatorOp(OpKernelConstruction* const context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr(kEpsilonName, &epsilon_));
    OP_REQUIRES_OK(context,
                   context->GetAttr(kNumQuantilesName, &num_quantiles_));
    OP_REQUIRES_OK(context, context->GetAttr(kMaxElementsName, &max_elements_));
    OP_REQUIRES_OK(context,
                   context->GetAttr(kGenerateQuantiles, &generate_quantiles_));
  }

  void Compute(OpKernelContext* context) override {
    // Only create one, if one does not exist already. Report status for all
    // other exceptions. If one already exists, it unrefs the new one.
    const Tensor* stamp_token_t;
    OP_REQUIRES_OK(context, context->input(kStampTokenName, &stamp_token_t));
    auto result = new QuantileStreamResource(epsilon_, num_quantiles_,
                                             max_elements_, generate_quantiles_,
                                             stamp_token_t->scalar<int64>()());
    auto status = CreateResource(context, HandleFromInput(context, 0), result);
    if (!status.ok() && status.code() != tensorflow::error::ALREADY_EXISTS) {
      OP_REQUIRES(context, false, status);
    }
  }

 private:
  float epsilon_;
  int32 num_quantiles_;
  // An upperbound on the number of enteries that the summaries might have
  // for a feature.
  int64 max_elements_;
  bool generate_quantiles_;
};

REGISTER_KERNEL_BUILDER(Name("CreateQuantileAccumulator").Device(DEVICE_CPU),
                        CreateQuantileAccumulatorOp);

// Adds a summary to the quantile summary stream.
class QuantileAccumulatorAddSummariesOp : public OpKernel {
 public:
  explicit QuantileAccumulatorAddSummariesOp(
      OpKernelConstruction* const context)
      : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    OpInputList resource_handle_list;
    OP_REQUIRES_OK(context, context->input_list(kResourceHandlesName,
                                                &resource_handle_list));
    OpInputList summary_list;
    OP_REQUIRES_OK(context, context->input_list(kSummariesName, &summary_list));

    const Tensor* stamp_token_t;
    OP_REQUIRES_OK(context, context->input(kStampTokenName, &stamp_token_t));
    int64 stamp_token = stamp_token_t->scalar<int64>()();

    thread::ThreadPool* const worker_threads =
        context->device()->tensorflow_cpu_worker_threads()->workers;
    boosted_trees::utils::ParallelFor(
        resource_handle_list.size(), worker_threads->NumThreads(),
        worker_threads,
        [&context, &resource_handle_list, &summary_list, stamp_token](
            int64 start, int64 end) {
          for (int resource_handle_idx = start; resource_handle_idx < end;
               ++resource_handle_idx) {
            ResourceHandle handle = resource_handle_list[resource_handle_idx]
                                        .flat<ResourceHandle>()(0);
            QuantileStreamResource* streams_resource;
            // Create a reference to the underlying resource using the handle.
            OP_REQUIRES_OK(context,
                           LookupResource(context, handle, &streams_resource));
            // Remove the reference at the end of this scope.
            mutex_lock l(*streams_resource->mutex());
            core::ScopedUnref unref_me(streams_resource);

            // If the stamp is invalid we drop the update.
            if (!streams_resource->is_stamp_valid(stamp_token)) {
              VLOG(1)
                  << "Invalid stamp token in QuantileAccumulatorAddSummariesOp."
                  << " Passed stamp token: " << stamp_token << " "
                  << "Current token: " << streams_resource->stamp();
              return;
            }

            protobuf::Arena arena;
            ::boosted_trees::QuantileSummaryState* summary_proto =
                protobuf::Arena::CreateMessage<
                    ::boosted_trees::QuantileSummaryState>(&arena);
            OP_REQUIRES(
                context,
                ParseProtoUnlimited(
                    summary_proto,
                    summary_list[resource_handle_idx].scalar<string>()()),
                errors::InvalidArgument("Unable to parse quantile summary."));
            std::vector<QuantileSummaryEntry> entries;
            entries.reserve(summary_proto->entries_size());
            for (const auto& entry : summary_proto->entries()) {
              entries.emplace_back(entry.value(), entry.weight(),
                                   entry.min_rank(), entry.max_rank());
            }

            // Add the summary to the quantile stream.
            streams_resource->stream(stamp_token)->PushSummary(entries);
          }
        });
  }
};

REGISTER_KERNEL_BUILDER(
    Name("QuantileAccumulatorAddSummaries").Device(DEVICE_CPU),
    QuantileAccumulatorAddSummariesOp);

// Generates summaries for given set of float values, and the given config.
class MakeQuantileSummariesOp : public OpKernel {
 public:
  explicit MakeQuantileSummariesOp(OpKernelConstruction* const context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context,
                   ReadAndValidateAttributes(context, &num_dense_features_,
                                             &num_sparse_features_));
    OP_REQUIRES_OK(context, context->GetAttr(kEpsilonName, &epsilon_));
  }

  void Compute(OpKernelContext* const context) override {
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

    // Parse example weights and get batch size.
    const Tensor* example_weights_t;
    OP_REQUIRES_OK(context,
                   context->input(kExampleWeightsName, &example_weights_t));
    auto example_weights = example_weights_t->flat<float>();
    const int64 batch_size = example_weights.size();

    OpOutputList sparse_summaries_output_list;
    OP_REQUIRES_OK(context,
                   context->output_list(kSparseSummariesName,
                                        &sparse_summaries_output_list));
    OpOutputList dense_summaries_output_list;
    OP_REQUIRES_OK(context, context->output_list(kDenseSummariesName,
                                                 &dense_summaries_output_list));

    auto do_quantile_summary_gen = [&](const int64 begin, const int64 end) {
      auto copy_over_summaries = [&](const QuantileStream& stream,
                                     const int64 index,
                                     OpOutputList* output_list) {
        protobuf::Arena arena;
        ::boosted_trees::QuantileSummaryState* summary_proto =
            protobuf::Arena::CreateMessage<
            ::boosted_trees::QuantileSummaryState>(&arena);
        const auto& summary = stream.GetFinalSummary();
        CopySummaryToProto(summary, summary_proto);
        // Output to tensor.
        Tensor* output_t = nullptr;
        OP_REQUIRES_OK(context, output_list->allocate(index, {}, &output_t));
        summary_proto->SerializeToString(&output_t->scalar<string>()());
      };

      // These are blocks of ranges. We are iterating over both sparse and
      // dense features i.e. [0, sparse_features.size() + dense_features.size()]
      for (int64 i = begin; i < end; ++i) {
        if (i < num_dense_features_) {
          const int64 dense_index = i;
          const auto dense_values =
              dense_float_features_list[dense_index].flat<float>();
          QuantileStream stream(epsilon_, batch_size + 1);
          // Run quantile summary generation.
          for (int64 j = 0; j < batch_size; ++j) {
            stream.PushEntry(dense_values(j), example_weights(j));
          }
          stream.Finalize();
          // Copy summaries to output.
          copy_over_summaries(stream, dense_index,
                              &dense_summaries_output_list);
        } else {
          const int64 sparse_index = i - num_dense_features_;
          const auto sparse_values =
              sparse_float_feature_values_list[sparse_index].flat<float>();
          const auto sparse_indices =
              sparse_float_feature_indices_list[sparse_index].matrix<int64>();
          const auto dense_shape =
              sparse_float_feature_shapes_list[sparse_index].flat<int64>();
          OP_REQUIRES(context, batch_size == dense_shape(0),
                      errors::InvalidArgument(
                          "Sparse column shape doesn't match the batch size."));
          QuantileStream stream(epsilon_, batch_size + 1);
          // Run quantile summary generation.
          const int64 num_sparse_rows =
              sparse_float_feature_indices_list[sparse_index].dim_size(0);
          for (int64 j = 0; j < num_sparse_rows; ++j) {
            const int64 example_id = sparse_indices(j, 0);
            stream.PushEntry(sparse_values(j), example_weights(example_id));
          }
          stream.Finalize();
          // Copy summaries to output.
          copy_over_summaries(stream, sparse_index,
                              &sparse_summaries_output_list);
        }
      }
    };
    const int64 kCostPerUnit = 500 * batch_size;
    const int64 num_features = num_sparse_features_ + num_dense_features_;
    const DeviceBase::CpuWorkerThreads& worker_threads =
        *context->device()->tensorflow_cpu_worker_threads();
    Shard(worker_threads.num_threads, worker_threads.workers, num_features,
          kCostPerUnit, do_quantile_summary_gen);
  }

 private:
  int num_dense_features_;
  int num_sparse_features_;
  float epsilon_;
};

REGISTER_KERNEL_BUILDER(Name("MakeQuantileSummaries").Device(DEVICE_CPU),
                        MakeQuantileSummariesOp);

// Serializes the state of streams.
class QuantileAccumulatorSerializeOp : public OpKernel {
 public:
  explicit QuantileAccumulatorSerializeOp(OpKernelConstruction* const context)
      : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    QuantileStreamResource* streams_resource;
    // Create a reference to the underlying resource using the handle.
    OP_REQUIRES_OK(context, LookupResource(context, HandleFromInput(context, 0),
                                           &streams_resource));
    // Remove the reference at the end of this scope.
    mutex_lock l(*streams_resource->mutex());
    core::ScopedUnref unref_me(streams_resource);

    int64 stamp_token = streams_resource->stamp();
    Tensor* stream_state_t;
    OP_REQUIRES_OK(context,
                   context->allocate_output(kStreamStateName, TensorShape({}),
                                            &stream_state_t));
    bool are_buckets_ready = streams_resource->are_buckets_ready();

    // We are iterating over both dense and sparse features. First we go
    // through the dense features and then the sparse features.
    const QuantileStream& stream = *streams_resource->stream(stamp_token);
    const std::vector<float>& boundaries =
        are_buckets_ready ? streams_resource->boundaries(stamp_token)
                          : std::vector<float>();
    protobuf::Arena arena;
    ::boosted_trees::QuantileStreamState* stream_proto =
        protobuf::Arena::CreateMessage<::boosted_trees::QuantileStreamState>(
            &arena);
    for (const auto& summary : stream.SerializeInternalSummaries()) {
      CopySummaryToProto(summary, stream_proto->add_summaries());
    }
    stream_proto->SerializeToString(&stream_state_t->scalar<string>()());
    Tensor* buckets_t = nullptr;
    OP_REQUIRES_OK(
        context,
        context->allocate_output(
            kBucketsName, {static_cast<int64>(boundaries.size())}, &buckets_t));
    auto* quantiles_flat = buckets_t->flat<float>().data();
    memcpy(quantiles_flat, boundaries.data(),
           sizeof(float) * boundaries.size());
    Tensor* stamp_token_t = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(kStampTokenName, TensorShape({}),
                                            &stamp_token_t));
    stamp_token_t->scalar<int64>()() = stamp_token;
    Tensor* are_buckets_ready_t = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(kAreBucketsReadyName, {},
                                                     &are_buckets_ready_t));
    are_buckets_ready_t->scalar<bool>()() = are_buckets_ready;
  }
};

REGISTER_KERNEL_BUILDER(Name("QuantileAccumulatorSerialize").Device(DEVICE_CPU),
                        QuantileAccumulatorSerializeOp);

// Serializes the state of streams.
class QuantileAccumulatorDeserializeOp : public OpKernel {
 public:
  explicit QuantileAccumulatorDeserializeOp(OpKernelConstruction* const context)
      : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    QuantileStreamResource* streams_resource;
    // Create a reference to the underlying resource using the handle.
    OP_REQUIRES_OK(context, LookupResource(context, HandleFromInput(context, 0),
                                           &streams_resource));
    // Remove the reference at the end of this scope.
    mutex_lock l(*streams_resource->mutex());
    core::ScopedUnref unref_me(streams_resource);

    int64 old_stamp_token = streams_resource->stamp();

    const Tensor* stream_state_t;
    OP_REQUIRES_OK(context, context->input(kStreamStateName, &stream_state_t));
    const Tensor* buckets_t;
    OP_REQUIRES_OK(context, context->input(kBucketsName, &buckets_t));

    QuantileStream* stream = streams_resource->stream(old_stamp_token);
    ::boosted_trees::QuantileStreamState state_proto;
    OP_REQUIRES(
        context,
        ParseProtoUnlimited(&state_proto, stream_state_t->scalar<string>()()),
        errors::InvalidArgument("Unabnle to parse quantile stream state."));
    std::vector<QuantileSummary> summaries;
    summaries.reserve(state_proto.summaries_size());
    std::vector<QuantileSummaryEntry> entries;
    for (const auto& summary : state_proto.summaries()) {
      entries.clear();
      entries.reserve(summary.entries_size());
      for (const auto& entry : summary.entries()) {
        entries.emplace_back(entry.value(), entry.weight(), entry.min_rank(),
                             entry.max_rank());
      }
      summaries.emplace_back();
      summaries[summaries.size() - 1].BuildFromSummaryEntries(entries);
    }
    stream->DeserializeInternalSummaries(summaries);

    const auto& buckets = buckets_t->vec<float>();
    std::vector<float> result;
    result.reserve(buckets.size());

    for (size_t i = 0; i < buckets.size(); ++i) {
      result.push_back(buckets(i));
    }
    streams_resource->set_boundaries(old_stamp_token, result);

    // Reset the stamp token.
    const Tensor* stamp_token_t = nullptr;
    OP_REQUIRES_OK(context, context->input(kStampTokenName, &stamp_token_t));
    int64 stamp_token = stamp_token_t->scalar<int64>()();
    streams_resource->set_stamp(stamp_token);

    const Tensor* are_buckets_ready_t = nullptr;
    OP_REQUIRES_OK(context,
                   context->input(kAreBucketsReadyName, &are_buckets_ready_t));
    streams_resource->set_buckets_ready(are_buckets_ready_t->scalar<bool>()());
  }
};

REGISTER_KERNEL_BUILDER(
    Name("QuantileAccumulatorDeserialize").Device(DEVICE_CPU),
    QuantileAccumulatorDeserializeOp);

// Flushes the quantile summary stream resource.
class QuantileAccumulatorFlushOp : public OpKernel {
 public:
  explicit QuantileAccumulatorFlushOp(OpKernelConstruction* const context)
      : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    QuantileStreamResource* streams_resource;
    // Create a reference to the underlying resource using the handle.
    OP_REQUIRES_OK(context, LookupResource(context, HandleFromInput(context, 0),
                                           &streams_resource));
    // Remove the reference at the end of this scope.
    mutex_lock l(*streams_resource->mutex());
    core::ScopedUnref unref_me(streams_resource);

    const Tensor* next_stamp_token_t;
    OP_REQUIRES_OK(context,
                   context->input(kNextStampTokenName, &next_stamp_token_t));
    int64 next_stamp_token = next_stamp_token_t->scalar<int64>()();

    const Tensor* stamp_token_t;
    OP_REQUIRES_OK(context, context->input(kStampTokenName, &stamp_token_t));
    int64 stamp_token = stamp_token_t->scalar<int64>()();
    CHECK(streams_resource->is_stamp_valid(stamp_token))
        << "Invalid stamp token in QuantileAccumulatorFlushOp. "
        << "Passed stamp token: " << stamp_token << " "
        << "Current token: " << streams_resource->stamp();
    QuantileStream* stream = streams_resource->stream(stamp_token);
    bool generate_quantiles = streams_resource->generate_quantiles();
    stream->Finalize();

    streams_resource->set_boundaries(
        stamp_token,
        generate_quantiles
            ? GenerateQuantiles(*stream, streams_resource->num_quantiles())
            : GenerateBoundaries(*stream, streams_resource->num_quantiles()));

    streams_resource->Reset(next_stamp_token);
  }
};

REGISTER_KERNEL_BUILDER(Name("QuantileAccumulatorFlush").Device(DEVICE_CPU),
                        QuantileAccumulatorFlushOp);

// Flushes the quantile summary stream resource. This version computes the
// summary.
class QuantileAccumulatorFlushSummaryOp : public OpKernel {
 public:
  explicit QuantileAccumulatorFlushSummaryOp(
      OpKernelConstruction* const context)
      : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    QuantileStreamResource* streams_resource;
    // Create a reference to the underlying resource using the handle.
    OP_REQUIRES_OK(context, LookupResource(context, HandleFromInput(context, 0),
                                           &streams_resource));
    // Remove the reference at the end of this scope.
    mutex_lock l(*streams_resource->mutex());
    core::ScopedUnref unref_me(streams_resource);

    const Tensor* next_stamp_token_t;
    OP_REQUIRES_OK(context,
                   context->input(kNextStampTokenName, &next_stamp_token_t));
    int64 next_stamp_token = next_stamp_token_t->scalar<int64>()();

    const Tensor* stamp_token_t;
    OP_REQUIRES_OK(context, context->input(kStampTokenName, &stamp_token_t));
    int64 stamp_token = stamp_token_t->scalar<int64>()();
    CHECK(streams_resource->is_stamp_valid(stamp_token))
        << "Invalid stamp token in QuantileAccumulatorFlushSummaryOp. "
        << "Passed stamp token: " << stamp_token << " "
        << "Current token: " << streams_resource->stamp();
    QuantileStream* stream = streams_resource->stream(stamp_token);
    stream->Finalize();
    protobuf::Arena arena;
    ::boosted_trees::QuantileSummaryState* summary_proto =
        protobuf::Arena::CreateMessage<::boosted_trees::QuantileSummaryState>(
            &arena);
    const auto& summary = stream->GetFinalSummary();
    CopySummaryToProto(summary, summary_proto);
    // Output to tensor.
    Tensor* output_t = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, TensorShape({}), &output_t));
    summary_proto->SerializeToString(&output_t->scalar<string>()());
    streams_resource->Reset(next_stamp_token);
  }
};

REGISTER_KERNEL_BUILDER(
    Name("QuantileAccumulatorFlushSummary").Device(DEVICE_CPU),
    QuantileAccumulatorFlushSummaryOp);

// Get bucket boundaries from summaries.
class QuantileAccumulatorGetBucketsOp : public OpKernel {
 public:
  explicit QuantileAccumulatorGetBucketsOp(OpKernelConstruction* const context)
      : OpKernel(context) {}

  void Compute(OpKernelContext* const context) override {
    OpInputList resource_handle_list;
    OP_REQUIRES_OK(context, context->input_list(kResourceHandlesName,
                                                &resource_handle_list));
    OpOutputList are_buckets_ready_list;
    OP_REQUIRES_OK(context, context->output_list(kAreBucketsReadyName,
                                                 &are_buckets_ready_list));
    OpOutputList buckets_list;
    OP_REQUIRES_OK(context, context->output_list(kBucketsName, &buckets_list));
    const Tensor* stamp_token_t;
    OP_REQUIRES_OK(context, context->input(kStampTokenName, &stamp_token_t));
    int64 stamp_token = stamp_token_t->scalar<int64>()();

    thread::ThreadPool* const worker_threads =
        context->device()->tensorflow_cpu_worker_threads()->workers;
    boosted_trees::utils::ParallelFor(
        resource_handle_list.size(), worker_threads->NumThreads(),
        worker_threads,
        [&context, &resource_handle_list, &are_buckets_ready_list,
         &buckets_list, stamp_token](int64 start, int64 end) {
          for (int resource_handle_idx = start; resource_handle_idx < end;
               ++resource_handle_idx) {
            ResourceHandle handle = resource_handle_list[resource_handle_idx]
                                        .flat<ResourceHandle>()(0);
            QuantileStreamResource* streams_resource;
            OP_REQUIRES_OK(context,
                           LookupResource(context, handle, &streams_resource));
            // Remove the reference at the end of this scope.
            mutex_lock l(*streams_resource->mutex());
            core::ScopedUnref unref_me(streams_resource);

            bool are_buckets_ready =
                streams_resource->is_stamp_valid(stamp_token) &&
                streams_resource->are_buckets_ready();

            Tensor* are_buckets_ready_t = nullptr;
            OP_REQUIRES_OK(context,
                           are_buckets_ready_list.allocate(
                               resource_handle_idx, {}, &are_buckets_ready_t));
            are_buckets_ready_t->scalar<bool>()() = are_buckets_ready;

            const std::vector<float>& boundaries =
                are_buckets_ready ? streams_resource->boundaries(stamp_token)
                                  : std::vector<float>();
            Tensor* output_t = nullptr;
            OP_REQUIRES_OK(context, buckets_list.allocate(
                                        resource_handle_idx,
                                        {static_cast<int64>(boundaries.size())},
                                        &output_t));
            auto* quantiles_flat = output_t->flat<float>().data();
            memcpy(quantiles_flat, boundaries.data(),
                   sizeof(float) * boundaries.size());
          }
        });
  }
};

REGISTER_KERNEL_BUILDER(
    Name("QuantileAccumulatorGetBuckets").Device(DEVICE_CPU),
    QuantileAccumulatorGetBucketsOp);

// Generates buckets for given set of float values, and the given config.
class QuantileBucketsOp : public OpKernel {
 public:
  explicit QuantileBucketsOp(OpKernelConstruction* const context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context,
                   ReadAndValidateAttributes(context, &num_dense_features_,
                                             &num_sparse_features_));

    ParseConfig(context, kDenseConfigName, &dense_configs_);
    OP_REQUIRES(context, dense_configs_.size() == num_dense_features_,
                errors::InvalidArgument(
                    "Mismatch in number of dense quantile configs."));
    ParseConfig(context, kSparseConfigName, &sparse_configs_);
    OP_REQUIRES(context, sparse_configs_.size() == num_sparse_features_,
                errors::InvalidArgument(
                    "Mismatch in number of sparse quantile configs."));
  }

  void Compute(OpKernelContext* const context) override {
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

    // Parse example weights and get batch size.
    const Tensor* example_weights_t;
    OP_REQUIRES_OK(context,
                   context->input(kExampleWeightsName, &example_weights_t));
    auto example_weights = example_weights_t->flat<float>();
    const int64 batch_size = example_weights.size();

    OpOutputList sparse_buckets_output_list;
    OP_REQUIRES_OK(context, context->output_list(kSparseBucketsName,
                                                 &sparse_buckets_output_list));
    OpOutputList dense_buckets_output_list;
    OP_REQUIRES_OK(context, context->output_list(kDenseBucketsName,
                                                 &dense_buckets_output_list));

    auto do_quantile_bucket_gen = [&](const int64 begin, const int64 end) {
      // These are blocks of ranges. We are iterating over both sparse and
      // dense features i.e. [0, sparse_features.size() + dense_features.size()]
      for (int64 i = begin; i < end; ++i) {
        if (i < sparse_configs_.size()) {
          const int64 sparse_index = i;
          const auto sparse_values =
              sparse_float_feature_values_list[sparse_index].flat<float>();
          const auto sparse_indices =
              sparse_float_feature_indices_list[sparse_index].matrix<int64>();
          QuantileStream stream(sparse_configs_[sparse_index].eps(),
                                batch_size);
          // Run quantile summary generation.
          const int64 num_sparse_rows =
              sparse_float_feature_indices_list[sparse_index].dim_size(0);
          for (int64 j = 0; j < num_sparse_rows; ++j) {
            const int64 example_id = sparse_indices(j, 0);
            stream.PushEntry(sparse_values(j), example_weights(example_id));
          }
          stream.Finalize();
          // Create buckets.
          const auto boundaries = GenerateBoundaries(
              stream, sparse_configs_[sparse_index].num_quantiles());
          CopyBoundaries(context, boundaries, sparse_index,
                         &sparse_buckets_output_list);

        } else {
          const int64 dense_index = i - sparse_configs_.size();
          const auto dense_values =
              dense_float_features_list[dense_index].flat<float>();
          QuantileStream stream(dense_configs_[dense_index].eps(), batch_size);
          // Run quantile summary generation.
          for (int64 j = 0; j < batch_size; ++j) {
            stream.PushEntry(dense_values(j), example_weights(j));
          }
          stream.Finalize();
          // Create buckets.
          const auto boundaries = GenerateBoundaries(
              stream, dense_configs_[dense_index].num_quantiles());
          CopyBoundaries(context, boundaries, dense_index,
                         &dense_buckets_output_list);
        }
      }
    };

    const int64 kCostPerUnit = 500 * batch_size;
    const int64 num_features = sparse_configs_.size() + dense_configs_.size();
    const DeviceBase::CpuWorkerThreads& worker_threads =
        *context->device()->tensorflow_cpu_worker_threads();
    Shard(worker_threads.num_threads, worker_threads.workers, num_features,
          kCostPerUnit, do_quantile_bucket_gen);
  }

 private:
  int num_dense_features_;
  int num_sparse_features_;
  std::vector<QuantileConfig> dense_configs_;
  std::vector<QuantileConfig> sparse_configs_;
};

REGISTER_KERNEL_BUILDER(Name("QuantileBuckets").Device(DEVICE_CPU),
                        QuantileBucketsOp);

// Given the calculated quantiles thresholds and input data, this operation
// converts the input features into the buckets (categorical values), depending
// on which quantile they fall into.
class QuantilesOp : public OpKernel {
 public:
  explicit QuantilesOp(OpKernelConstruction* const context)
      : OpKernel(context) {
    int num_dense_features;
    int num_sparse_features;
    OP_REQUIRES_OK(context,
                   ReadAndValidateAttributes(context, &num_dense_features,
                                             &num_sparse_features));
  }

  void Compute(OpKernelContext* const context) override {
    // Dense features inputs
    OpInputList dense_float_features_list;
    OP_REQUIRES_OK(context, context->input_list(kDenseValuesName,
                                                &dense_float_features_list));
    OpInputList dense_buckets_list;
    OP_REQUIRES_OK(context,
                   context->input_list(kDenseBucketsName, &dense_buckets_list));

    if (dense_buckets_list.size() > 0) {
      // Check the first tensor to make sure it is the right shape
      OP_REQUIRES(
          context,
          tensorflow::TensorShapeUtils::IsVector(dense_buckets_list[0].shape()),
          errors::InvalidArgument(
              strings::Printf("Dense buckets should be flat vectors")));
    }

    // Sparse features inputs
    OpInputList sparse_float_feature_values_list;
    OP_REQUIRES_OK(context,
                   context->input_list(kSparseValuesName,
                                       &sparse_float_feature_values_list));

    OpInputList sparse_float_indices_list;
    OP_REQUIRES_OK(context, context->input_list(kSparseIndicesName,
                                                &sparse_float_indices_list));

    OpInputList sparse_buckets_list;
    OP_REQUIRES_OK(
        context, context->input_list(kSparseBucketsName, &sparse_buckets_list));

    if (sparse_buckets_list.size() > 0) {
      OP_REQUIRES(
          context,
          tensorflow::TensorShapeUtils::IsVector(
              sparse_buckets_list[0].shape()),
          errors::InvalidArgument("Sparse buckets should be flat vectors"));
    }

    // Quantize the feature values
    QuantizeFeatures(kDenseOutputTensorName, dense_float_features_list,
                     dense_buckets_list, nullptr, context);

    QuantizeFeatures(kSparseOutputTensorName, sparse_float_feature_values_list,
                     sparse_buckets_list, &sparse_float_indices_list, context);
  }
};

REGISTER_KERNEL_BUILDER(Name("Quantiles").Device(DEVICE_CPU), QuantilesOp);

template <typename T>
class BucketizeWithInputBoundariesOp : public OpKernel {
 public:
  explicit BucketizeWithInputBoundariesOp(OpKernelConstruction* context)
      : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    const Tensor& boundaries_tensor = context->input(1);
    VLOG(1) << "boundaries has shape: "
            << boundaries_tensor.shape().DebugString();
    auto boundaries = boundaries_tensor.flat<float>();
    std::vector<T> boundaries_vector;
    boundaries_vector.reserve(boundaries.size());
    for (size_t i = 0; i < boundaries.size(); i++) {
      boundaries_vector.push_back(boundaries(i));
      VLOG(1) << "boundaries(" << i << ") : " << boundaries(i);
    }
    OP_REQUIRES(
        context,
        std::is_sorted(boundaries_vector.begin(), boundaries_vector.end()),
        errors::InvalidArgument("Expected sorted boundaries"));

    const Tensor& input_tensor = context->input(0);
    VLOG(1) << "Inputs has shape: " << input_tensor.shape().DebugString()
            << " Dtype: " << tensorflow::DataTypeString(input_tensor.dtype());
    auto input = input_tensor.flat<T>();

    Tensor* output_tensor = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor.shape(),
                                                     &output_tensor));
    auto output = output_tensor->template flat<int32>();

    for (size_t i = 0; i < input.size(); i++) {
      output(i) = CalculateBucketIndex(input(i), boundaries_vector);
    }
  }

 private:
  int32 CalculateBucketIndex(const T value, std::vector<T>& boundaries_vector) {
    auto first_bigger_it = std::upper_bound(boundaries_vector.begin(),
                                            boundaries_vector.end(), value);
    int32 index = first_bigger_it - boundaries_vector.begin();
    CHECK(index >= 0 && index <= boundaries_vector.size())
        << "Invalid bucket index: " << index
        << " boundaries_vector.size(): " << boundaries_vector.size();
    return index;
  }
};

#define REGISTER_KERNEL(T)                                     \
  REGISTER_KERNEL_BUILDER(Name("BucketizeWithInputBoundaries") \
                              .Device(DEVICE_CPU)              \
                              .TypeConstraint<T>("T"),         \
                          BucketizeWithInputBoundariesOp<T>);

REGISTER_KERNEL(int32);
REGISTER_KERNEL(int64);
REGISTER_KERNEL(float);
REGISTER_KERNEL(double);
#undef REGISTER_KERNEL

}  // namespace tensorflow
