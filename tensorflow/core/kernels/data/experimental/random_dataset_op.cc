/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/core/kernels/data/experimental/random_dataset_op.h"

#include <string>
#include <utility>

#include "tensorflow/core/data/dataset_utils.h"
#include "tensorflow/core/data/name_utils.h"
#include "tensorflow/core/data/split_utils.h"
#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/partial_tensor_shape.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/kernels/data/random_seed_ops.h"
#include "tensorflow/core/lib/random/philox_random.h"
#include "tensorflow/core/lib/random/random.h"
#include "tensorflow/core/lib/random/random_distributions.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace data {
namespace experimental {

// Constants declared in random_dataset_op.h and used both here and in test
// cases.
/* static */ constexpr const char* const RandomDatasetOp::kDatasetType;
/* static */ constexpr const char* const RandomDatasetOp::kSeed;
/* static */ constexpr const char* const RandomDatasetOp::kSeed2;
/* static */ constexpr const char* const RandomDatasetOp::kOutputTypes;
/* static */ constexpr const char* const RandomDatasetOp::kOutputShapes;
/* static */ constexpr const char* const
    RandomDatasetOp::kRerandomizeEachIteration;

namespace {

constexpr char kRandomDatasetV1[] = "RandomDataset";
constexpr char kRandomDatasetV2[] = "RandomDatasetV2";
constexpr char kSeedGenerator[] = "SeedGenerator";
constexpr char kEpochNumRandomSamples[] = "epoch_num_random_samples";
constexpr char kNumRandomSamples[] = "num_random_samples";

}  // namespace

class RandomDatasetOp::Dataset : public DatasetBase {
 public:
  Dataset(OpKernelContext* ctx, RandomSeeds&& seeds,
          SeedGeneratorManager* manager, ResourceHandle&& resource_handle,
          bool owns_resource, int op_version)
      : DatasetBase(DatasetContext(ctx)),
        seeds_(std::move(seeds)),
        op_version_(op_version),
        manager_(manager),
        resource_handle_(resource_handle),
        resource_mgr_(ctx->resource_manager()),
        owns_resource_(owns_resource) {}

  ~Dataset() override {
    manager_->Unref();
    if (owns_resource_) {
      absl::Status s = resource_mgr_->Delete<SeedGeneratorManager>(
          resource_handle_.container(), resource_handle_.name());
      if (!s.ok()) {
        LOG(WARNING) << "Failed to delete RNG resource: " << s;
      }
    }
  }

  absl::Status MakeSplitProviders(std::vector<std::unique_ptr<SplitProvider>>*
                                      split_providers) const override {
    // We use kint64 to generate an effectively infinite number of "splits".
    // These splits aren't actually used during iteration.
    // TODO(aaudibert): Avoid sending dummy splits over RPC when using tf.data
    // service with RandomDataset.
    split_providers->push_back(std::make_unique<IndexSplitProvider>(kint64max));
    return absl::OkStatus();
  }

  std::unique_ptr<IteratorBase> MakeIteratorInternal(
      const string& prefix) const override {
    return std::make_unique<Iterator>(
        Iterator::Params{this, strings::StrCat(prefix, "::Random")},
        manager_->get().get());
  }

  const DataTypeVector& output_dtypes() const override {
    static DataTypeVector* dtypes = new DataTypeVector({DT_INT64});
    return *dtypes;
  }

  const std::vector<PartialTensorShape>& output_shapes() const override {
    static std::vector<PartialTensorShape>* shapes =
        new std::vector<PartialTensorShape>({{}});
    return *shapes;
  }

  string DebugString() const override {
    name_utils::DatasetDebugStringParams params;
    params.op_version = op_version_;
    params.set_args(seeds_.input_seed(), seeds_.input_seed2());
    return name_utils::DatasetDebugString(RandomDatasetOp::kDatasetType,
                                          params);
  }

  int64_t CardinalityInternal(CardinalityOptions options) const override {
    return kInfiniteCardinality;
  }

  absl::Status InputDatasets(
      std::vector<const DatasetBase*>* inputs) const override {
    return absl::OkStatus();
  }

  absl::Status CheckExternalState() const override { return absl::OkStatus(); }

 protected:
  absl::Status AsGraphDefInternal(SerializationContext* ctx,
                                  DatasetGraphDefBuilder* b,
                                  Node** output) const override {
    Node* seed_node = nullptr;
    Node* seed2_node = nullptr;
    TF_RETURN_IF_ERROR(b->AddScalar(seeds_.input_seed(), &seed_node));
    TF_RETURN_IF_ERROR(b->AddScalar(seeds_.input_seed2(), &seed2_node));
    if (op_version_ == 1) {
      return b->AddDataset(this, {seed_node, seed2_node}, output);
    }
    Node* resource_handle_node = nullptr;
    Tensor handle(DT_RESOURCE, TensorShape({}));
    handle.scalar<ResourceHandle>()() = resource_handle_;
    TF_RETURN_IF_ERROR(b->AddTensor(handle, &resource_handle_node));
    AttrValue rerandomize_each_iteration;
    b->BuildAttrValue(manager_->get()->reshuffle_each_iteration(),
                      &rerandomize_each_iteration);
    return b->AddDataset(
        this, {seed_node, seed2_node, resource_handle_node},
        {std::make_pair(kRerandomizeEachIteration, rerandomize_each_iteration)},
        output);
  }

 private:
  class Iterator : public DatasetIterator<Dataset> {
   public:
    Iterator(const Params& params, SeedGenerator* seed_generator)
        : DatasetIterator<Dataset>(params),
          seed_generator_(seed_generator),
          parent_generator_(seed_generator_->seed(), seed_generator_->seed2()),
          generator_(&parent_generator_) {}

    bool SymbolicCheckpointCompatible() const override { return true; }

    absl::Status Initialize(IteratorContext* ctx) override {
      mutex_lock l(mu_);
      seed_generator_->GenerateSeeds(&seed_, &seed2_);
      ResetRngs();
      return absl::OkStatus();
    }

    absl::Status GetNextInternal(IteratorContext* ctx,
                                 std::vector<Tensor>* out_tensors,
                                 bool* end_of_sequence) override {
      out_tensors->reserve(1);
      mutex_lock l(mu_);
      out_tensors->emplace_back(ctx->allocator({}), DT_INT64, TensorShape({}));
      out_tensors->back().scalar<int64_t>()() = Random();
      *end_of_sequence = false;
      return absl::OkStatus();
    }

    std::shared_ptr<model::Node> CreateNode(
        IteratorContext* ctx, model::Node::Args args) const override {
      return model::MakeSourceNode(std::move(args));
    }

    absl::Status SaveInternal(SerializationContext* ctx,
                              IteratorStateWriter* writer) override {
      mutex_lock l(mu_);
      // Save state needed to restore the random number generators.
      TF_RETURN_IF_ERROR(
          writer->WriteScalar(full_name(kEpochNumRandomSamples),
                              seed_generator_->num_random_samples()));
      TF_RETURN_IF_ERROR(writer->WriteScalar(this->full_name(kNumRandomSamples),
                                             num_random_samples_));
      TF_RETURN_IF_ERROR(writer->WriteScalar(this->full_name(kSeed), seed_));
      TF_RETURN_IF_ERROR(writer->WriteScalar(this->full_name(kSeed2), seed2_));
      return absl::OkStatus();
    }

    absl::Status RestoreInternal(IteratorContext* ctx,
                                 IteratorStateReader* reader) override {
      mutex_lock l(mu_);
      // Restore the random number generators.
      int64_t num_random_samples;
      TF_RETURN_IF_ERROR(reader->ReadScalar(full_name(kEpochNumRandomSamples),
                                            &num_random_samples));
      seed_generator_->set_num_random_samples(num_random_samples);
      seed_generator_->Reset();
      TF_RETURN_IF_ERROR(reader->ReadScalar(this->full_name(kNumRandomSamples),
                                            &num_random_samples_));
      TF_RETURN_IF_ERROR(reader->ReadScalar(this->full_name(kSeed), &seed_));
      TF_RETURN_IF_ERROR(reader->ReadScalar(this->full_name(kSeed2), &seed2_));
      ResetRngs();
      return absl::OkStatus();
    }

   protected:
    void ResetRngs() TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
      // Reset the generators based on the current iterator seeds.
      parent_generator_ = random::PhiloxRandom(seed_, seed2_);
      generator_ =
          random::SingleSampleAdapter<random::PhiloxRandom>(&parent_generator_);
      generator_.Skip(num_random_samples_);
    }

   private:
    random::SingleSampleAdapter<random::PhiloxRandom>::ResultType Random()
        TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
      num_random_samples_++;
      auto out = generator_();
      return out;
    }

    mutex mu_;
    SeedGenerator* const seed_generator_ TF_GUARDED_BY(mu_);  // Not owned.
    random::PhiloxRandom parent_generator_ TF_GUARDED_BY(mu_);
    random::SingleSampleAdapter<random::PhiloxRandom> generator_
        TF_GUARDED_BY(mu_);
    int64_t num_random_samples_ TF_GUARDED_BY(mu_) = 0;
    int64_t seed_ TF_GUARDED_BY(mu_) = 0;
    int64_t seed2_ TF_GUARDED_BY(mu_) = 0;
  };

 private:
  const RandomSeeds seeds_;
  const int op_version_;
  SeedGeneratorManager* const manager_;  // Owned
  const ResourceHandle resource_handle_;
  ResourceMgr* const resource_mgr_;  // Not owned.
  const bool owns_resource_;
};

RandomDatasetOp::RandomDatasetOp(OpKernelConstruction* ctx)
    : DatasetOpKernel(ctx) {
  auto& op_name = ctx->def().op();
  if (op_name == kRandomDatasetV2) {
    op_version_ = 2;
  } else if (op_name == kRandomDatasetV1) {
    op_version_ = 1;
  }
  if (ctx->HasAttr(kRerandomizeEachIteration)) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr(kRerandomizeEachIteration,
                                     &rerandomize_each_iteration_));
  }
}

void RandomDatasetOp::MakeDataset(OpKernelContext* ctx, DatasetBase** output) {
  int64_t seed;
  OP_REQUIRES_OK(ctx, ParseScalarArgument<int64_t>(ctx, "seed", &seed));
  int64_t seed2;
  OP_REQUIRES_OK(ctx, ParseScalarArgument<int64_t>(ctx, "seed2", &seed2));
  RandomSeeds seeds(seed, seed2);
  static std::atomic<int64_t> resource_id_counter(0);
  const string& container = ctx->resource_manager()->default_container();
  auto name = strings::StrCat(ctx->op_kernel().name(), "/", kSeedGenerator, "_",
                              resource_id_counter.fetch_add(1));
  SeedGeneratorManager* manager = nullptr;
  ResourceHandle handle;
  bool owns_resource = true;
  if (op_version_ == 2) {
    OP_REQUIRES_OK(ctx, HandleFromInput(ctx, 2, &handle));
    absl::Status s = ctx->resource_manager()->Lookup<SeedGeneratorManager>(
        handle.container(), handle.name(), &manager);
    owns_resource = false;
    if (absl::IsNotFound(s)) {
      owns_resource = true;
    } else {
      OP_REQUIRES_OK(ctx, s);
    }
  }

  // TODO(b/259308104): Rather than managing resources directly, use ref
  // counting resource handles: go/tf-ref-counting-resource-handles.
  if (owns_resource) {
    OP_REQUIRES_OK(
        ctx,
        ctx->resource_manager()->LookupOrCreate<SeedGeneratorManager>(
            container, name, &manager,
            [rerandomize = rerandomize_each_iteration_,
             &seeds](SeedGeneratorManager** manager) {
              if (rerandomize) {
                *manager =
                    new SeedGeneratorManager(new RandomSeedGenerator(seeds));
              } else {
                *manager =
                    new SeedGeneratorManager(new FixedSeedGenerator(seeds));
              }
              return absl::OkStatus();
            }));
    handle = MakeResourceHandle<SeedGenerator>(ctx, container, name);
  }

  *output = new RandomDatasetOp::Dataset(ctx, std::move(seeds), manager,
                                         std::move(handle), owns_resource,
                                         op_version_);
}

namespace {

REGISTER_KERNEL_BUILDER(Name(kRandomDatasetV1).Device(DEVICE_CPU),
                        RandomDatasetOp);
REGISTER_KERNEL_BUILDER(Name(kRandomDatasetV2).Device(DEVICE_CPU),
                        RandomDatasetOp);
REGISTER_KERNEL_BUILDER(Name("ExperimentalRandomDataset").Device(DEVICE_CPU),
                        RandomDatasetOp);

}  // namespace
}  // namespace experimental
}  // namespace data
}  // namespace tensorflow
