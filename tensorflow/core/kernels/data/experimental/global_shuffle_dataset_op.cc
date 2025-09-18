/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/synchronization/mutex.h"
#include "tensorflow/core/data/name_utils.h"
#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/framework/dataset_options.pb.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/op_requires.h"
#include "tensorflow/core/framework/resource_handle.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/kernels/data/random_seed_ops.h"
#include "tensorflow/core/kernels/random_index_shuffle.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/statusor.h"

namespace tensorflow {
namespace data {
namespace {

constexpr int32_t kIndexShuffleRounds = 8;

constexpr const char kDatasetType[] = "GlobalShuffle";
constexpr const char kElementCount[] = "element_count";
constexpr const char kGlobalShuffleDataset[] = "GlobalShuffleDataset";
constexpr const char kReshuffleEachIteration[] = "reshuffle_each_iteration";
constexpr const char kSeed[] = "seed";
constexpr const char kSeed2[] = "seed2";
constexpr const char kSeed3[] = "seed3";
constexpr const char kSeedGenerator[] = "SeedGenerator";
constexpr const char kEpochNumRandomSamples[] = "epoch_num_random_samples";

class GlobalShuffleDatasetOp : public UnaryDatasetOpKernel {
 public:
  explicit GlobalShuffleDatasetOp(OpKernelConstruction* ctx);

 protected:
  void MakeDataset(OpKernelContext* ctx, DatasetBase* input,
                   DatasetBase** output) override;

 private:
  class Dataset;

  bool reshuffle_each_iteration_ = true;
};

class GlobalShuffleDatasetOp::Dataset : public DatasetBase {
 public:
  Dataset(OpKernelContext* ctx, const DatasetBase* input,
          SeedGeneratorManager* seed_generator, RandomSeeds&& input_seeds,
          bool owns_resource, ResourceHandle&& resource_handle)
      : DatasetBase(DatasetContext(ctx)),
        input_(input),
        seed_generator_(seed_generator),
        input_seeds_(std::move(input_seeds)),
        owns_resource_(owns_resource),
        resource_handle_(std::move(resource_handle)),
        resource_mgr_(ctx->resource_manager()) {
    input_->Ref();
  }

  ~Dataset() override {
    seed_generator_->Unref();
    if (owns_resource_) {
      absl::Status s = resource_mgr_->Delete<SeedGeneratorManager>(
          resource_handle_.container(), resource_handle_.name());
      if (!s.ok()) {
        LOG(WARNING) << "Failed to delete random seed generator resource for "
                     << "tf.data global shuffle dataset: " << s;
      }
    }
    input_->Unref();
  }

  const DataTypeVector& output_dtypes() const override {
    return input_->output_dtypes();
  }

  const std::vector<PartialTensorShape>& output_shapes() const override {
    return input_->output_shapes();
  }

  std::string DebugString() const override {
    return name_utils::DatasetDebugString(kDatasetType);
  }

  int64_t CardinalityInternal(CardinalityOptions options) const override {
    return input_->Cardinality(options);
  }

  absl::Status InputDatasets(
      std::vector<const DatasetBase*>* inputs) const override {
    inputs->push_back(input_);
    return absl::OkStatus();
  }

  absl::Status CheckExternalState() const override {
    return input_->CheckExternalState();
  }

 protected:
  std::unique_ptr<IteratorBase> MakeIteratorInternal(
      const std::string& prefix) const override;

  absl::Status AsGraphDefInternal(SerializationContext* ctx,
                                  DatasetGraphDefBuilder* b,
                                  Node** output) const override {
    // Inputs
    Node* input_graph_node = nullptr;
    TF_RETURN_IF_ERROR(b->AddInputDataset(ctx, input_, &input_graph_node));
    Node* seed_node = nullptr;
    Node* seed2_node = nullptr;
    TF_RETURN_IF_ERROR(b->AddScalar(input_seeds_.input_seed(), &seed_node));
    TF_RETURN_IF_ERROR(b->AddScalar(input_seeds_.input_seed2(), &seed2_node));

    Node* resource_handle_node = nullptr;
    Tensor handle(DT_RESOURCE, TensorShape({}));
    handle.scalar<ResourceHandle>()() = resource_handle_;
    TF_RETURN_IF_ERROR(b->AddTensor(handle, &resource_handle_node));

    // Attrs
    AttrValue reshuffle_each_iteration;
    b->BuildAttrValue(seed_generator_->get()->reshuffle_each_iteration(),
                      &reshuffle_each_iteration);
    return b->AddDataset(
        this, /*inputs=*/
        {input_graph_node, seed_node, seed2_node, resource_handle_node},
        /*attrs=*/
        {std::make_pair(kReshuffleEachIteration, reshuffle_each_iteration)},
        output);
  }

  absl::Status RandomIndexingCompatible() const override {
    return absl::OkStatus();
  }

 private:
  class Iterator;

  const DatasetBase* const input_;
  SeedGeneratorManager* const seed_generator_;  // Owned
  const RandomSeeds input_seeds_;
  const bool owns_resource_;
  const ResourceHandle resource_handle_;
  ResourceMgr* const resource_mgr_;  // Not owned.
};

class GlobalShuffleDatasetOp::Dataset::Iterator
    : public DatasetIterator<Dataset> {
 public:
  explicit Iterator(const Params& params,
                    std::shared_ptr<SeedGenerator> seed_generator)
      : DatasetIterator<Dataset>(params),
        cardinality_(dataset()->Cardinality()),
        seed_generator_(seed_generator) {}

  bool SymbolicCheckpointCompatible() const override { return true; }

  absl::Status Initialize(IteratorContext* ctx) override
      ABSL_LOCKS_EXCLUDED(mu_) {
    absl::MutexLock l(mu_);
    if (ctx->cancellation_manager()->IsCancelled()) {
      return absl::CancelledError(
          "ctx->cancellation_manager()->IsCancelled() is true. Would not "
          "execute `seed_generator_` to prevent incorrect results when "
          "restoring.");
    }
    int64_t seed4;
    seed_generator_->GenerateSeeds(&seed_, &seed2_);
    seed_generator_->GenerateSeeds(&seed3_, &seed4);

    // Snapshots `num_random_samples()` so that
    // we know how to recover the seed later.
    num_random_samples_ = seed_generator_->num_random_samples();
    TF_RETURN_IF_ERROR(
        dataset()->input_->MakeIterator(ctx, this, prefix(), &input_impl_));
    return absl::OkStatus();
  }

  absl::Status GetNextInternal(IteratorContext* ctx,
                               std::vector<Tensor>* out_tensors,
                               bool* end_of_sequence) override
      ABSL_LOCKS_EXCLUDED(mu_) {
    absl::MutexLock l(mu_);
    IteratorContext::Params params(ctx);
    params.index_mapper = GetIndexMapper(ctx->index_mapper());
    IteratorContext global_shuffle_ctx(params);
    TF_RETURN_IF_ERROR(input_impl_->GetNext(&global_shuffle_ctx, out_tensors,
                                            end_of_sequence));
    ctx->MergeCheckpoint(global_shuffle_ctx.checkpoint());
    ++element_count_;
    return absl::OkStatus();
  }

  IndexMapperFn GetIndexMapper(IndexMapperFn parent_index_mapper) const override
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
    uint32_t seed = static_cast<uint32_t>(seed_);
    uint32_t seed2 = static_cast<uint32_t>(seed2_);
    uint32_t seed3 = static_cast<uint32_t>(seed3_);
    uint64_t max_index =
        cardinality_ > 0 ? static_cast<uint64_t>(cardinality_ - 1) : 0;
    return [parent_index_mapper, seed, seed2, seed3,
            max_index](size_t element_position) -> absl::StatusOr<size_t> {
      if (parent_index_mapper != nullptr) {
        TF_ASSIGN_OR_RETURN(element_position,
                            parent_index_mapper(element_position));
      }
      if (element_position > max_index) {
        return absl::OutOfRangeError("Out of range");
      }
      if (max_index == 0) {
        return 0;
      }
      return static_cast<int64_t>(tensorflow::random::index_shuffle(
          static_cast<uint64_t>(element_position), {seed, seed2, seed3},
          max_index, kIndexShuffleRounds));
    };
  }

  absl::Status SaveInternal(SerializationContext* ctx,
                            IteratorStateWriter* writer) override {
    absl::MutexLock l(mu_);
    TF_RETURN_IF_ERROR(
        writer->WriteScalar(prefix(), kElementCount, element_count_));
    TF_RETURN_IF_ERROR(writer->WriteScalar(prefix(), kEpochNumRandomSamples,
                                           num_random_samples_));

    TF_RETURN_IF_ERROR(writer->WriteScalar(prefix(), kSeed, seed_));
    TF_RETURN_IF_ERROR(writer->WriteScalar(prefix(), kSeed2, seed2_));
    TF_RETURN_IF_ERROR(writer->WriteScalar(prefix(), kSeed3, seed3_));
    TF_RETURN_IF_ERROR(SaveInput(ctx, writer, input_impl_));
    return absl::OkStatus();
  }

  absl::Status RestoreInternal(IteratorContext* ctx,
                               IteratorStateReader* reader) override {
    absl::MutexLock l(mu_);
    if (ctx->restored_element_count().has_value()) {
      element_count_ = *ctx->restored_element_count();
    } else {
      TF_RETURN_IF_ERROR(
          reader->ReadScalar(prefix(), kElementCount, &element_count_));
    }

    // Restoring the seed_generator is necessary when
    // combine this op with `.repeat()`.
    // This is similar to how shuffle dataset recovers the seed generator -
    // `tensorflow::data::ShuffleDatasetOpBase::ShuffleDatasetBase::Iterator::RestoreInternal`.
    TF_RETURN_IF_ERROR(reader->ReadScalar(prefix(), kEpochNumRandomSamples,
                                          &num_random_samples_));
    seed_generator_->set_num_random_samples(num_random_samples_);
    seed_generator_->Reset();

    // Required to recover seeds because `Initialize` is always called
    // before `RestoreInternal.
    TF_RETURN_IF_ERROR(reader->ReadScalar(prefix(), kSeed, &seed_));
    TF_RETURN_IF_ERROR(reader->ReadScalar(prefix(), kSeed2, &seed2_));
    TF_RETURN_IF_ERROR(reader->ReadScalar(prefix(), kSeed3, &seed3_));

    IteratorContext::Params params(ctx);
    params.restored_element_count = element_count_;
    params.index_mapper = GetIndexMapper(ctx->index_mapper());
    IteratorContext ctx_copy(params);
    TF_RETURN_IF_ERROR(RestoreInput(&ctx_copy, reader, input_impl_));
    ctx->MergeCheckpoint(ctx_copy.checkpoint());
    return absl::OkStatus();
  }

 private:
  const int64_t cardinality_;

  mutable absl::Mutex mu_;
  std::shared_ptr<SeedGenerator> seed_generator_ ABSL_GUARDED_BY(mu_);
  int64_t seed_ ABSL_GUARDED_BY(mu_) = 0;
  int64_t seed2_ ABSL_GUARDED_BY(mu_) = 0;
  int64_t seed3_ ABSL_GUARDED_BY(mu_) = 0;

  std::unique_ptr<IteratorBase> input_impl_ ABSL_GUARDED_BY(mu_);
  int64_t element_count_ ABSL_GUARDED_BY(mu_) = 0;
  int64_t num_random_samples_ ABSL_GUARDED_BY(mu_) = 0;
};

GlobalShuffleDatasetOp::GlobalShuffleDatasetOp(OpKernelConstruction* ctx)
    : UnaryDatasetOpKernel(ctx) {
  if (ctx->HasAttr(kReshuffleEachIteration)) {
    OP_REQUIRES_OK(
        ctx, ctx->GetAttr(kReshuffleEachIteration, &reshuffle_each_iteration_));
  }
}

void GlobalShuffleDatasetOp::MakeDataset(OpKernelContext* ctx,
                                         DatasetBase* input,
                                         DatasetBase** output) {
  OP_REQUIRES(ctx, input->RandomIndexingCompatible().ok(),
              absl::FailedPreconditionError(absl::StrCat(
                  "`global_shuffle` requires all upstream transformations be "
                  "compatible with random access. Got: ",
                  input->RandomIndexingCompatible().ToString())));

  CardinalityOptions options;
  options.set_compute_level(CardinalityOptions::CARDINALITY_COMPUTE_MODERATE);
  int64_t cardinality = input->Cardinality(std::move(options));
  OP_REQUIRES(ctx, cardinality > 0,
              absl::InvalidArgumentError(absl::StrCat(
                  "`global_shuffle` requires the input dataset to have a "
                  "non-empty finite cardinality. Got cardinality ",
                  cardinality, " for dataset ", input->DebugString())));

  int64_t seed, seed2;
  OP_REQUIRES_OK(ctx, ParseScalarArgument<int64_t>(ctx, kSeed, &seed));
  OP_REQUIRES_OK(ctx, ParseScalarArgument<int64_t>(ctx, kSeed2, &seed2));
  RandomSeeds input_seeds(seed, seed2);

  static std::atomic<int64_t> resource_id_counter(0);
  const std::string& container = ctx->resource_manager()->default_container();
  std::string name = absl::StrCat(ctx->op_kernel().name(), "/", kSeedGenerator,
                                  "_", resource_id_counter.fetch_add(1));

  auto handle = HandleFromInput(ctx, 3);
  SeedGeneratorManager* seed_generator = nullptr;
  absl::Status s = ctx->resource_manager()->Lookup<SeedGeneratorManager>(
      handle.container(), handle.name(), &seed_generator);

  bool owns_resource = false;
  if (absl::IsNotFound(s)) {
    owns_resource = true;
    OP_REQUIRES_OK(
        ctx, ctx->resource_manager()->LookupOrCreate<SeedGeneratorManager>(
                 container, name, &seed_generator,
                 [reshuffle = reshuffle_each_iteration_,
                  &input_seeds](SeedGeneratorManager** seed_generator) {
                   if (reshuffle) {
                     *seed_generator = new SeedGeneratorManager(
                         new RandomSeedGenerator(input_seeds));
                   } else {
                     *seed_generator = new SeedGeneratorManager(
                         new FixedSeedGenerator(input_seeds));
                   }
                   return absl::OkStatus();
                 }));
    handle = MakeResourceHandle<SeedGenerator>(ctx, container, name);
  } else {
    OP_REQUIRES_OK(ctx, s);
  }

  *output = new Dataset(ctx, input, seed_generator, std::move(input_seeds),
                        owns_resource, std::move(handle));
}

std::unique_ptr<IteratorBase>
GlobalShuffleDatasetOp::Dataset::MakeIteratorInternal(
    const std::string& prefix) const {
  return std::make_unique<GlobalShuffleDatasetOp::Dataset::Iterator>(
      Iterator::Params{this, name_utils::IteratorPrefix(kDatasetType, prefix)},
      seed_generator_->get());
}

REGISTER_KERNEL_BUILDER(Name(kGlobalShuffleDataset).Device(DEVICE_CPU),
                        GlobalShuffleDatasetOp);

}  // namespace
}  // namespace data
}  // namespace tensorflow
