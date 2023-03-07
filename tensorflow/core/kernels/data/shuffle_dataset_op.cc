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
#include "tensorflow/core/kernels/data/shuffle_dataset_op.h"

#include <cstdint>
#include <deque>
#include <memory>
#include <numeric>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "tensorflow/core/data/dataset_utils.h"
#include "tensorflow/core/data/name_utils.h"
#include "tensorflow/core/data/serialization_utils.h"
#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/framework/partial_tensor_shape.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/kernels/data/random_seed_ops.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/random/philox_random.h"
#include "tensorflow/core/lib/random/random.h"
#include "tensorflow/core/lib/random/random_distributions.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/stringprintf.h"

namespace tensorflow {
namespace data {

// See documentation in ../../ops/dataset_ops.cc for a high-level
// description of the following op.

/* static */ constexpr const char* const ShuffleDatasetOpBase::kInputDataset;
/* static */ constexpr const char* const ShuffleDatasetOpBase::kBufferSize;
/* static */ constexpr const char* const ShuffleDatasetOpBase::kSeed;
/* static */ constexpr const char* const ShuffleDatasetOpBase::kSeed2;
/* static */ constexpr const char* const ShuffleDatasetOpBase::kOutputTypes;
/* static */ constexpr const char* const ShuffleDatasetOpBase::kOutputShapes;
/* static */ constexpr const char* const
    ShuffleDatasetOpBase::kReshuffleEachIteration;

/* static */ constexpr const char* const ShuffleDatasetOp::kDatasetType;

/* static */ constexpr const char* const
    ShuffleAndRepeatDatasetOp::kDatasetType;
/* static */ constexpr const char* const ShuffleAndRepeatDatasetOp::kCount;

const int64_t kLogIntervalMicros = 10 * 1000000;  // 10 seconds.
const int64_t kMaxEpochsInBuffer = 3;

constexpr char kNumRandomSamples[] = "num_random_samples";
constexpr char kDataProduced[] = "data_produced";
constexpr char kEndOfInputSequence[] = "end_of_input_sequence";
constexpr char kEpoch[] = "epoch";
constexpr char kNumElements[] = "num_elements";
constexpr char kSlicesSize[] = "slices_size";
constexpr char kSlicesStart[] = "slices_start";
constexpr char kSlicesEnd[] = "slices_end";
constexpr char kSeedGenerator[] = "SeedGenerator";
constexpr char kEpochNumRandomSamples[] = "epoch_num_random_samples";
constexpr char kShuffleDatasetV1[] = "ShuffleDataset";
constexpr char kShuffleDatasetV2[] = "ShuffleDatasetV2";
constexpr char kShuffleDatasetV3[] = "ShuffleDatasetV3";
constexpr char kShuffleAndRepeatDatasetV1[] = "ShuffleAndRepeatDataset";
constexpr char kShuffleAndRepeatDatasetV2[] = "ShuffleAndRepeatDatasetV2";

ShuffleDatasetOpBase::ShuffleDatasetOpBase(OpKernelConstruction* ctx)
    : UnaryDatasetOpKernel(ctx) {}

// Abstract base dataset that implements a shuffling iterator.
class ShuffleDatasetOpBase::ShuffleDatasetBase : public DatasetBase {
 public:
  ShuffleDatasetBase(OpKernelContext* ctx, const DatasetBase* input,
                     int64_t buffer_size,
                     std::shared_ptr<SeedGenerator> seed_generator,
                     int64_t count)
      : DatasetBase(DatasetContext(ctx)),
        input_(input),
        buffer_size_(buffer_size),
        seed_generator_(std::move(seed_generator)),
        count_(count),
        traceme_metadata_(
            {{"buffer_size",
              strings::Printf("%lld", static_cast<long long>(buffer_size))}}) {
    input_->Ref();
  }

  ~ShuffleDatasetBase() override { input_->Unref(); }

  virtual string op_type() const = 0;

  const DataTypeVector& output_dtypes() const override {
    return input_->output_dtypes();
  }

  const std::vector<PartialTensorShape>& output_shapes() const override {
    return input_->output_shapes();
  }

  int64_t CardinalityInternal(CardinalityOptions options) const override {
    if (count_ == -1 || input_->Cardinality(options) == kInfiniteCardinality) {
      return kInfiniteCardinality;
    } else if (input_->Cardinality(options) == kUnknownCardinality) {
      return kUnknownCardinality;
    } else {
      return input_->Cardinality(options) * count_;
    }
  }

  Status InputDatasets(std::vector<const DatasetBase*>* inputs) const override {
    inputs->push_back(input_);
    return OkStatus();
  }

  Status CheckExternalState() const override {
    return input_->CheckExternalState();
  }

  Status Get(OpKernelContext* ctx, int64 index,
             std::vector<Tensor>* out_tensors) const override {
    TF_RETURN_IF_ERROR(CheckRandomAccessCompatible(index));
    {
      mutex_lock l(mu_);
      if (shuffled_indices_.empty()) {
        InitializeRandomAccessIndices();
      }
    }
    int64 shuffled_index;
    {
      tf_shared_lock l(mu_);
      shuffled_index = shuffled_indices_[index];
    }
    TF_RETURN_IF_ERROR(input_->Get(ctx, shuffled_index, out_tensors));
    return OkStatus();
  }

  string DebugString() const override {
    name_utils::DatasetDebugStringParams params;
    params.set_args(buffer_size_, seed_generator_->seed(),
                    seed_generator_->seed2(), count_);
    return name_utils::DatasetDebugString(op_type(), params);
  }

  std::unique_ptr<IteratorBase> MakeIteratorInternal(
      const string& prefix) const override {
    return std::make_unique<Iterator>(
        Iterator::Params{this, name_utils::IteratorPrefix(op_type(), prefix)},
        seed_generator_.get());
  }

  void InitializeRandomAccessIndices() const TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
    const int64 cardinality = Cardinality();
    shuffled_indices_ = std::vector<std::int64_t>(cardinality);
    std::iota(shuffled_indices_.begin(), shuffled_indices_.end(), 0);
    int64_t shuffled_index = 0;
    random::PhiloxRandom parent_generator =
        random::PhiloxRandom(seed_generator_->seed(), seed_generator_->seed2());
    random::SingleSampleAdapter<random::PhiloxRandom> generator =
        random::SingleSampleAdapter<random::PhiloxRandom>(&parent_generator);

    while (shuffled_index < cardinality) {
      int64_t offset = generator() % (cardinality - shuffled_index);
      std::swap(shuffled_indices_[shuffled_index + offset],
                shuffled_indices_[shuffled_index]);
      shuffled_index += 1;
    }
  }

 protected:
  class Iterator : public DatasetIterator<ShuffleDatasetBase> {
   public:
    explicit Iterator(const Params& params, SeedGenerator* seed_generator)
        : DatasetIterator<ShuffleDatasetBase>(params),
          seed_generator_(seed_generator),
          parent_generator_(seed_generator->seed(), seed_generator->seed2()),
          generator_(&parent_generator_) {
      buffer_ = std::make_unique<std::vector<std::vector<Tensor>>>(
          params.dataset->buffer_size_);
    }

    Status Initialize(IteratorContext* ctx) override {
      mutex_lock l(mu_);
      seed_generator_->GenerateSeeds(&seed_, &seed2_);
      ResetRngs();
      return OkStatus();
    }

    Status GetNextInternal(IteratorContext* ctx,
                           std::vector<Tensor>* out_tensors,
                           bool* end_of_sequence) override {
      mutex_lock l(mu_);
      TF_RETURN_IF_ERROR(FillBuffer(ctx));
      if (num_elements_ == 0) {
        DCHECK(input_impl_ == nullptr);
        *end_of_sequence = true;
        return OkStatus();
      }

      *end_of_sequence = false;
      ClearEmptySlices();
      DCHECK(!slices_.empty());
      // Choose an element to produce uniformly at random from the first
      // slice, and then remove the element from the slice.
      int64_t offset =
          Random() % (slices_.front()->end - slices_.front()->start);
      int64_t index = (slices_.front()->start + offset) % buffer_->size();
      *out_tensors = std::move(buffer_->at(index));
      this->RecordBufferDequeue(ctx, *out_tensors);
      std::swap(buffer_->at(index),
                buffer_->at(slices_.front()->start % buffer_->size()));
      slices_.front()->start++;
      num_elements_--;
      return OkStatus();
    }

   protected:
    std::shared_ptr<model::Node> CreateNode(
        IteratorContext* ctx, model::Node::Args args) const override {
      return model::MakeKnownRatioNode(std::move(args),
                                       /*ratio=*/1);
    }

    void ResetRngs() TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
      // Reset the generators based on the current iterator seeds.
      parent_generator_ = random::PhiloxRandom(seed_, seed2_);
      generator_ =
          random::SingleSampleAdapter<random::PhiloxRandom>(&parent_generator_);
      generator_.Skip(num_random_samples_);
    }

    Status SaveInternal(SerializationContext* ctx,
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

      // Save input iterator if it hasn't been exhausted else write
      // "end_of_input_sequence".
      if (!input_impl_) {
        TF_RETURN_IF_ERROR(
            writer->WriteScalar(this->full_name(kEndOfInputSequence), ""));
      } else {
        TF_RETURN_IF_ERROR(this->SaveInput(ctx, writer, input_impl_));
      }

      // Save the epoch counter, buffer, and buffer slices.
      TF_RETURN_IF_ERROR(writer->WriteScalar(this->full_name(kEpoch), epoch_));
      TF_RETURN_IF_ERROR(
          writer->WriteScalar(this->full_name(kNumElements), num_elements_));
      TF_RETURN_IF_ERROR(WriteElementsToCheckpoint(writer, prefix(), *buffer_));
      TF_RETURN_IF_ERROR(
          writer->WriteScalar(this->full_name(kSlicesSize), slices_.size()));
      for (size_t i = 0; i < slices_.size(); ++i) {
        TF_RETURN_IF_ERROR(
            writer->WriteScalar(this->full_name(absl::StrJoin(
                                    std::make_tuple(kSlicesStart, i), "_")),
                                slices_[i]->start));
        TF_RETURN_IF_ERROR(writer->WriteScalar(
            this->full_name(absl::StrJoin(std::make_tuple(kSlicesEnd, i), "_")),
            slices_[i]->end));
      }
      if (data_produced_) {
        TF_RETURN_IF_ERROR(
            writer->WriteScalar(this->full_name(kDataProduced), ""));
      }

      return OkStatus();
    }

    Status RestoreInternal(IteratorContext* ctx,
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

      // Restore the input iterator if it wasn't already exhausted.
      if (!reader->Contains(this->full_name(kEndOfInputSequence))) {
        TF_RETURN_IF_ERROR(this->dataset()->input_->MakeIterator(
            ctx, this, this->prefix(), &input_impl_));
        TF_RETURN_IF_ERROR(this->RestoreInput(ctx, reader, input_impl_));
      } else {
        input_impl_.reset();
      }

      // Restore the epoch counter, buffer, and buffer slices.
      TF_RETURN_IF_ERROR(reader->ReadScalar(this->full_name(kEpoch), &epoch_));
      TF_RETURN_IF_ERROR(
          reader->ReadScalar(this->full_name(kNumElements), &num_elements_));
      size_t slices_size;
      {
        int64_t temp;
        TF_RETURN_IF_ERROR(
            reader->ReadScalar(this->full_name(kSlicesSize), &temp));
        slices_size = static_cast<size_t>(temp);
      }
      buffer_ = std::make_unique<std::vector<std::vector<Tensor>>>();
      TF_RETURN_IF_ERROR(
          ReadElementsFromCheckpoint(ctx, reader, prefix(), buffer_.get()));
      for (const auto& element : *buffer_) {
        RecordBufferEnqueue(ctx, element);
      }
      buffer_->resize(dataset()->buffer_size_);
      slices_.clear();
      for (size_t i = 0; i < slices_size; ++i) {
        int64_t start;
        TF_RETURN_IF_ERROR(
            reader->ReadScalar(this->full_name(absl::StrJoin(
                                   std::make_tuple(kSlicesStart, i), "_")),
                               &start));
        int64_t end;
        TF_RETURN_IF_ERROR(reader->ReadScalar(
            this->full_name(absl::StrJoin(std::make_tuple(kSlicesEnd, i), "_")),
            &end));
        slices_.push_back(std::make_unique<Slice>(start, end));
      }
      data_produced_ = reader->Contains(this->full_name(kDataProduced));

      return OkStatus();
    }

    TraceMeMetadata GetTraceMeMetadata() const override {
      return this->dataset()->traceme_metadata_;
    }

   private:
    // Used to represent slices of `buffer_` that belong to different epochs.
    // The invariant maintained by the implementation is: `start` <= `end`.
    // When using `start` and `end` to index into `buffer_`, their values
    // should be taken modulo the size of `buffer_` as their absolute value
    // can be greater than the range of `buffer_`.
    struct Slice {
      Slice(int64_t start, int64_t end) : start(start), end(end) {}

      int64_t start;
      int64_t end;
    };

    random::SingleSampleAdapter<random::PhiloxRandom>::ResultType Random()
        TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
      num_random_samples_++;
      auto out = generator_();
      return out;
    }

    // Fills the shuffle buffer, preparing the buffer for sampling.
    Status FillBuffer(IteratorContext* ctx) TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
      int64_t start_micros = EnvTime::NowMicros();
      int64_t num_log_entries = 0;
      while (ShouldFillBuffer()) {
        if (EnvTime::NowMicros() >
            ((num_log_entries + 1) * kLogIntervalMicros) + start_micros) {
          num_log_entries++;
          LOG(INFO) << "Filling up shuffle buffer (this may take a while): "
                    << num_elements_ << " of " << BufferSizeString();
        }
        if (!input_impl_) {
          TF_RETURN_IF_ERROR(PrepareNextEpoch(ctx));
        }
        std::vector<Tensor> input_element;
        bool end_of_input_sequence = false;
        TF_RETURN_IF_ERROR(
            input_impl_->GetNext(ctx, &input_element, &end_of_input_sequence));
        if (!end_of_input_sequence) {
          AddToShuffleBuffer(ctx, std::move(input_element));
          continue;
        }
        input_impl_.reset();
        // Reached end of input_impl_.
        if (ctx->split_providers().empty() && !data_produced_ &&
            this->dataset()->count_ == -1) {
          // If we encounter the end of sequence without producing data, we
          // terminate the iteration immediately. (Otherwise, this iterator
          // would loop infinitely and never produce a value.)
          return OkStatus();
        }
      }
      if (num_log_entries > 0) {
        LOG(INFO) << "Shuffle buffer filled.";
      }
      return OkStatus();
    }

    bool ShouldFillBuffer() TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
      if (!input_impl_ && dataset()->count_ != -1 &&
          epoch_ >= dataset()->count_) {
        return false;
      }
      if (slices_.size() > kMaxEpochsInBuffer && num_elements_ > 0) {
        // When the elements stored in `buffer_` span more than
        // `kMaxEpochsInBuffer` epochs, we do not fill the buffer further to
        // conserve memory. This means that the upper bound on the size of
        // `buffer_` is `kMaxEpochsInBuffer * cardinality(input_dataset) +
        // 1`.
        return false;
      }
      return num_elements_ < buffer_->size();
    }

    Status PrepareNextEpoch(IteratorContext* ctx)
        TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
      if (epoch_ == 0) {
        slices_.push_back(std::make_unique<Slice>(0, 0));
      } else {
        int64_t n = slices_.back()->end;
        slices_.push_back(std::make_unique<Slice>(n, n));
        for (const auto& provider : ctx->split_providers()) {
          TF_RETURN_IF_ERROR(provider->Reset());
        }
      }
      TF_RETURN_IF_ERROR(this->dataset()->input_->MakeIterator(
          ctx, this, this->prefix(), &input_impl_));
      epoch_++;
      return OkStatus();
    }

    void AddToShuffleBuffer(IteratorContext* ctx, std::vector<Tensor>&& element)
        TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
      data_produced_ = true;
      if (num_elements_ == 0) {
        VLOG(1) << "Starting to fill up shuffle buffer of size: "
                << BufferSizeString();
      }
      this->RecordBufferEnqueue(ctx, element);
      size_t index = slices_.back()->end % buffer_->size();
      buffer_->at(index) = std::move(element);
      num_elements_++;
      slices_.back()->end++;
    }

    void ClearEmptySlices() TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
      // Garbage collect all empty slices.
      while (slices_.front()->start == slices_.front()->end) {
        slices_.pop_front();
        // Reinitialize the RNG state for the next epoch.
        num_random_samples_ = 0;
        seed_generator_->GenerateSeeds(&seed_, &seed2_);
        ResetRngs();
      }
    }

    std::string BufferSizeString() {
      return absl::StrCat(dataset()->buffer_size_);
    }

    mutex mu_;
    SeedGenerator* const seed_generator_ TF_GUARDED_BY(mu_);  // Not owned.
    std::unique_ptr<std::vector<std::vector<Tensor>>> buffer_
        TF_GUARDED_BY(mu_);
    std::unique_ptr<IteratorBase> input_impl_ TF_GUARDED_BY(mu_) = nullptr;
    int64_t epoch_ TF_GUARDED_BY(mu_) = 0;
    int64_t num_elements_ TF_GUARDED_BY(mu_) = 0;
    int64_t seed_ TF_GUARDED_BY(mu_) = 0;
    int64_t seed2_ TF_GUARDED_BY(mu_) = 0;
    // Indices into `buffer_` indicating which data belongs to which epoch.
    // The slice at the front of the deque references data from the earliest
    // buffered epoch. It is an invariant that all slices reference
    // non-overlapping sections of `buffer_`.
    std::deque<std::unique_ptr<Slice>> slices_ TF_GUARDED_BY(mu_);
    random::PhiloxRandom parent_generator_ TF_GUARDED_BY(mu_);
    random::SingleSampleAdapter<random::PhiloxRandom> generator_
        TF_GUARDED_BY(mu_);
    int64_t num_random_samples_ TF_GUARDED_BY(mu_) = 0;
    bool data_produced_ TF_GUARDED_BY(mu_) = false;
  };

  const DatasetBase* const input_;
  const int64_t buffer_size_;
  const std::shared_ptr<SeedGenerator> seed_generator_;
  // The number of epochs to run for. Normally this is just 1, but sometimes we
  // fuse shuffle and repeat together, and make the shuffle dataset op
  // responsible for repeating as well.
  const int64_t count_;
  const TraceMeMetadata traceme_metadata_;
  mutable mutex mu_;
  mutable std::vector<std::int64_t> shuffled_indices_ TF_GUARDED_BY(mu_);
};  // ShuffleDatasetBase

// This version of memory dataset has an exclusive ownership of the seed
// generator resource. It supports sharing of the seed generator across
// different iterations of the `repeat` transformation but not across different
// iterators.
class ShuffleDatasetOp::Dataset : public ShuffleDatasetBase {
 public:
  Dataset(OpKernelContext* ctx, const DatasetBase* input, int64_t buffer_size,
          int64_t count, RandomSeeds&& seeds, SeedGeneratorManager* manager,
          ResourceHandle&& resource_handle)
      : ShuffleDatasetBase(ctx, input, buffer_size, manager->get(), count),
        manager_(manager),
        resource_handle_(std::move(resource_handle)),
        resource_mgr_(ctx->resource_manager()),
        seeds_(std::move(seeds)) {}

  ~Dataset() override {
    manager_->Unref();
    Status s = resource_mgr_->Delete<SeedGeneratorManager>(
        resource_handle_.container(), resource_handle_.name());
    if (!s.ok()) {
      LOG(WARNING) << "Failed to delete RNG resource: " << s.ToString();
    }
  }

  string op_type() const override { return kDatasetType; }

 protected:
  Status AsGraphDefInternal(SerializationContext* ctx,
                            DatasetGraphDefBuilder* b,
                            Node** output) const override {
    Node* input_graph_node = nullptr;
    TF_RETURN_IF_ERROR(b->AddInputDataset(ctx, input_, &input_graph_node));
    Node* buffer_size_node = nullptr;
    Node* seed_node = nullptr;
    Node* seed2_node = nullptr;
    AttrValue reshuffle_each_iteration;

    TF_RETURN_IF_ERROR(b->AddScalar(buffer_size_, &buffer_size_node));
    TF_RETURN_IF_ERROR(b->AddScalar(seeds_.input_seed(), &seed_node));
    TF_RETURN_IF_ERROR(b->AddScalar(seeds_.input_seed2(), &seed2_node));
    b->BuildAttrValue(seed_generator_->reshuffle_each_iteration(),
                      &reshuffle_each_iteration);
    TF_RETURN_IF_ERROR(b->AddDataset(
        this,
        {input_graph_node, buffer_size_node, seed_node, seed2_node},  // Inputs
        {std::make_pair(kReshuffleEachIteration,
                        reshuffle_each_iteration)},  // Attrs
        output));
    return OkStatus();
  }

 private:
  SeedGeneratorManager* const manager_;  // Owned.
  const ResourceHandle resource_handle_;
  ResourceMgr* const resource_mgr_;  // Not owned.
  const RandomSeeds seeds_;
};

// This version of shuffle dataset has a shared ownership of the seed generator
// resource. It supports sharing of the generator state across different
// iterations of the `repeat` transformation and also across different
// iterators.
class ShuffleDatasetOp::DatasetV2 : public ShuffleDatasetBase {
 public:
  DatasetV2(OpKernelContext* ctx, const DatasetBase* input, int64_t buffer_size,
            int64_t count, SeedGeneratorManager* manager,
            ResourceHandle&& resource_handle, bool owns_resource)
      : ShuffleDatasetBase(ctx, input, buffer_size, manager->get(), count),
        manager_(manager),
        owns_resource_(owns_resource),
        resource_handle_(std::move(resource_handle)),
        resource_mgr_(ctx->resource_manager()) {}

  ~DatasetV2() override {
    manager_->Unref();
    if (owns_resource_) {
      Status s = resource_mgr_->Delete<SeedGeneratorManager>(
          resource_handle_.container(), resource_handle_.name());
      if (!s.ok()) {
        LOG(WARNING) << "Failed to delete RNG resource: " << s.ToString();
      }
    }
  }

  string op_type() const override { return kDatasetType; }

 protected:
  Status AsGraphDefInternal(SerializationContext* ctx,
                            DatasetGraphDefBuilder* b,
                            Node** output) const override {
    Node* input_graph_node = nullptr;
    TF_RETURN_IF_ERROR(b->AddInputDataset(ctx, input_, &input_graph_node));
    Node* buffer_size_node = nullptr;
    TF_RETURN_IF_ERROR(b->AddScalar(buffer_size_, &buffer_size_node));
    Node* resource_handle_node = nullptr;
    Tensor handle(DT_RESOURCE, TensorShape({}));
    handle.scalar<ResourceHandle>()() = resource_handle_;
    TF_RETURN_IF_ERROR(b->AddTensor(handle, &resource_handle_node));
    TF_RETURN_IF_ERROR(b->AddDataset(
        this,
        {input_graph_node, buffer_size_node, resource_handle_node},  // Inputs
        {},                                                          // Attrs
        output));
    return OkStatus();
  }

 private:
  SeedGeneratorManager* const manager_;  // Owned.
  const bool owns_resource_;
  const ResourceHandle resource_handle_;
  ResourceMgr* const resource_mgr_;  // Not owned.
};

// This version of shuffle dataset extends the functionality of DatasetV2 with
// the ability to preserve seed generator configuration (i.e. initial seeds and
// whether to reshuffle each iteration) across serialization of the dataset.
class ShuffleDatasetOp::DatasetV3 : public ShuffleDatasetBase {
 public:
  DatasetV3(OpKernelContext* ctx, const DatasetBase* input, int64_t buffer_size,
            int64_t count, RandomSeeds&& seeds, SeedGeneratorManager* manager,
            ResourceHandle&& resource_handle, bool owns_resource)
      : ShuffleDatasetBase(ctx, input, buffer_size, manager->get(), count),
        manager_(manager),
        owns_resource_(owns_resource),
        resource_handle_(std::move(resource_handle)),
        resource_mgr_(ctx->resource_manager()),
        seeds_(std::move(seeds)) {}

  ~DatasetV3() override {
    manager_->Unref();
    if (owns_resource_) {
      Status s = resource_mgr_->Delete<SeedGeneratorManager>(
          resource_handle_.container(), resource_handle_.name());
      if (!s.ok()) {
        LOG(WARNING) << "Failed to delete RNG resource: " << s.ToString();
      }
    }
  }

  string op_type() const override { return kDatasetType; }

 protected:
  Status AsGraphDefInternal(SerializationContext* ctx,
                            DatasetGraphDefBuilder* b,
                            Node** output) const override {
    Node* input_graph_node = nullptr;
    TF_RETURN_IF_ERROR(b->AddInputDataset(ctx, input_, &input_graph_node));
    Node* buffer_size_node = nullptr;
    Node* seed_node = nullptr;
    Node* seed2_node = nullptr;
    TF_RETURN_IF_ERROR(b->AddScalar(buffer_size_, &buffer_size_node));
    TF_RETURN_IF_ERROR(b->AddScalar(seeds_.input_seed(), &seed_node));
    TF_RETURN_IF_ERROR(b->AddScalar(seeds_.input_seed2(), &seed2_node));
    Node* resource_handle_node = nullptr;
    Tensor handle(DT_RESOURCE, TensorShape({}));
    handle.scalar<ResourceHandle>()() = resource_handle_;
    TF_RETURN_IF_ERROR(b->AddTensor(handle, &resource_handle_node));
    AttrValue reshuffle_each_iteration;
    b->BuildAttrValue(seed_generator_->reshuffle_each_iteration(),
                      &reshuffle_each_iteration);
    TF_RETURN_IF_ERROR(
        b->AddDataset(this,
                      {input_graph_node, buffer_size_node, seed_node,
                       seed2_node, resource_handle_node},  // Inputs
                      {std::make_pair(kReshuffleEachIteration,
                                      reshuffle_each_iteration)},  // Attrs
                      output));
    return OkStatus();
  }

 private:
  SeedGeneratorManager* const manager_;  // Owned
  const bool owns_resource_;
  const ResourceHandle resource_handle_;
  ResourceMgr* const resource_mgr_;  // Not owned.
  const RandomSeeds seeds_;
};

ShuffleDatasetOp::ShuffleDatasetOp(OpKernelConstruction* ctx)
    : ShuffleDatasetOpBase(ctx) {
  auto& op_name = ctx->def().op();
  if (op_name == kShuffleDatasetV3) {
    op_version_ = 3;
  } else if (op_name == kShuffleDatasetV2) {
    op_version_ = 2;
  } else if (op_name == kShuffleDatasetV1) {
    op_version_ = 1;
  }
  if (ctx->HasAttr(kReshuffleEachIteration)) {
    OP_REQUIRES_OK(
        ctx, ctx->GetAttr(kReshuffleEachIteration, &reshuffle_each_iteration_));
  }
}

void ShuffleDatasetOp::MakeDataset(OpKernelContext* ctx, DatasetBase* input,
                                   DatasetBase** output) {
  int64_t buffer_size = 0;
  OP_REQUIRES_OK(ctx,
                 ParseScalarArgument<int64_t>(ctx, kBufferSize, &buffer_size));
  OP_REQUIRES(
      ctx, buffer_size > 0,
      errors::InvalidArgument("buffer_size must be greater than zero."));

  int64_t count = 1;
  static std::atomic<int64_t> resource_id_counter(0);
  const string& container = ctx->resource_manager()->default_container();
  auto name = strings::StrCat(ctx->op_kernel().name(), "/", kSeedGenerator, "_",
                              resource_id_counter.fetch_add(1));
  if (op_version_ == 3) {
    auto handle = HandleFromInput(ctx, 4);
    SeedGeneratorManager* manager = nullptr;
    Status s = ctx->resource_manager()->Lookup<SeedGeneratorManager>(
        handle.container(), handle.name(), &manager);
    int64_t seed;
    OP_REQUIRES_OK(ctx, ParseScalarArgument<int64_t>(ctx, kSeed, &seed));
    int64_t seed2;
    OP_REQUIRES_OK(ctx, ParseScalarArgument<int64_t>(ctx, kSeed2, &seed2));
    RandomSeeds seeds(seed, seed2);
    bool owns_resource = false;
    if (errors::IsNotFound(s)) {
      owns_resource = true;
      OP_REQUIRES_OK(
          ctx,
          ctx->resource_manager()->LookupOrCreate<SeedGeneratorManager>(
              container, name, &manager,
              [reshuffle = reshuffle_each_iteration_,
               &seeds](SeedGeneratorManager** manager) {
                if (reshuffle) {
                  *manager =
                      new SeedGeneratorManager(new RandomSeedGenerator(seeds));
                } else {
                  *manager =
                      new SeedGeneratorManager(new FixedSeedGenerator(seeds));
                }
                return OkStatus();
              }));
      handle = MakeResourceHandle<SeedGenerator>(ctx, container, name);
    } else {
      OP_REQUIRES_OK(ctx, s);
    }

    // Ownership of manager is transferred onto `DatasetV3`.
    *output = new ShuffleDatasetOp::DatasetV3(ctx, input, buffer_size, count,
                                              std::move(seeds), manager,
                                              std::move(handle), owns_resource);
  } else if (op_version_ == 2) {
    auto handle = HandleFromInput(ctx, 2);
    SeedGeneratorManager* manager = nullptr;
    Status s = ctx->resource_manager()->Lookup<SeedGeneratorManager>(
        handle.container(), handle.name(), &manager);
    bool owns_resource = false;
    if (errors::IsNotFound(s)) {
      owns_resource = true;
      LOG(WARNING) << "Failed to find seed generator resource. Falling back to "
                      "using a non-deterministically seeded generator and "
                      "reshuffling each iteration.";
      RandomSeeds seeds(0, 0);
      OP_REQUIRES_OK(
          ctx, ctx->resource_manager()->LookupOrCreate<SeedGeneratorManager>(
                   container, name, &manager,
                   [&seeds](SeedGeneratorManager** manager) {
                     *manager = new SeedGeneratorManager(
                         new RandomSeedGenerator(seeds));
                     return OkStatus();
                   }));
      handle = MakeResourceHandle<SeedGeneratorManager>(ctx, container, name);
    } else {
      OP_REQUIRES_OK(ctx, s);
    }

    // Ownership of manager is transferred onto `DatasetV2`.
    *output =
        new ShuffleDatasetOp::DatasetV2(ctx, input, buffer_size, count, manager,
                                        std::move(handle), owns_resource);
  } else {
    if (op_version_ != 1) {
      LOG(WARNING) << "Unsupported version of shuffle dataset op: "
                   << op_version_ << ". Defaulting to version 1.";
    }
    int64_t seed;
    OP_REQUIRES_OK(ctx, ParseScalarArgument<int64_t>(ctx, kSeed, &seed));
    int64_t seed2;
    OP_REQUIRES_OK(ctx, ParseScalarArgument<int64_t>(ctx, kSeed2, &seed2));
    RandomSeeds seeds(seed, seed2);
    SeedGeneratorManager* manager;
    OP_REQUIRES_OK(
        ctx,
        ctx->resource_manager()->LookupOrCreate<SeedGeneratorManager>(
            container, name, &manager,
            [reshuffle = reshuffle_each_iteration_,
             &seeds](SeedGeneratorManager** manager) {
              if (reshuffle) {
                *manager =
                    new SeedGeneratorManager(new RandomSeedGenerator(seeds));
              } else {
                *manager =
                    new SeedGeneratorManager(new FixedSeedGenerator(seeds));
              }
              return OkStatus();
            }));
    auto handle =
        MakeResourceHandle<SeedGeneratorManager>(ctx, container, name);

    // Ownership of manager is transferred onto `Dataset`.
    *output = new ShuffleDatasetOp::Dataset(ctx, input, buffer_size, count,
                                            std::move(seeds), manager,
                                            std::move(handle));
  }
}

class ShuffleAndRepeatDatasetOp::Dataset : public ShuffleDatasetBase {
 public:
  Dataset(OpKernelContext* ctx, const DatasetBase* input, int64_t buffer_size,
          RandomSeeds&& seeds, SeedGeneratorManager* manager, int64_t count,
          ResourceHandle&& resource_handle)
      : ShuffleDatasetBase(ctx, input, buffer_size, manager->get(), count),
        manager_(manager),
        resource_handle_(std::move(resource_handle)),
        resource_mgr_(ctx->resource_manager()),
        seeds_(std::move(seeds)) {}

  ~Dataset() override {
    manager_->Unref();
    Status s = resource_mgr_->Delete<SeedGeneratorManager>(
        resource_handle_.container(), resource_handle_.name());
    if (!s.ok()) {
      LOG(WARNING) << "Failed to delete RNG resource: " << s.ToString();
    }
  }

  string op_type() const override { return kDatasetType; }

 protected:
  Status AsGraphDefInternal(SerializationContext* ctx,
                            DatasetGraphDefBuilder* b,
                            Node** output) const override {
    Node* input_graph_node = nullptr;
    TF_RETURN_IF_ERROR(b->AddInputDataset(ctx, input_, &input_graph_node));
    Node* buffer_size = nullptr;
    Node* seed = nullptr;
    Node* seed2 = nullptr;
    Node* count = nullptr;

    TF_RETURN_IF_ERROR(b->AddScalar(buffer_size_, &buffer_size));
    TF_RETURN_IF_ERROR(b->AddScalar(seeds_.input_seed(), &seed));
    TF_RETURN_IF_ERROR(b->AddScalar(seeds_.input_seed2(), &seed2));
    TF_RETURN_IF_ERROR(b->AddScalar(count_, &count));
    AttrValue reshuffle_each_iteration;
    b->BuildAttrValue(seed_generator_->reshuffle_each_iteration(),
                      &reshuffle_each_iteration);
    TF_RETURN_IF_ERROR(b->AddDataset(
        this, {input_graph_node, buffer_size, seed, seed2, count},  // Inputs
        {std::make_pair(kReshuffleEachIteration,
                        reshuffle_each_iteration)},  // Attrs
        output));
    return OkStatus();
  }

 private:
  SeedGeneratorManager* const manager_;  // Owned.
  const ResourceHandle resource_handle_;
  ResourceMgr* const resource_mgr_;  // Not owned.
  const RandomSeeds seeds_;
};

class ShuffleAndRepeatDatasetOp::DatasetV2 : public ShuffleDatasetBase {
 public:
  DatasetV2(OpKernelContext* ctx, const DatasetBase* input, int64_t buffer_size,
            int64_t count, RandomSeeds&& seeds, SeedGeneratorManager* manager,
            ResourceHandle&& resource_handle, bool owns_resource)
      : ShuffleDatasetBase(ctx, input, buffer_size, manager->get(), count),
        manager_(manager),
        owns_resource_(owns_resource),
        resource_handle_(std::move(resource_handle)),
        resource_mgr_(ctx->resource_manager()),
        seeds_(std::move(seeds)) {}

  ~DatasetV2() override {
    manager_->Unref();
    if (owns_resource_) {
      Status s = resource_mgr_->Delete<SeedGeneratorManager>(
          resource_handle_.container(), resource_handle_.name());
      if (!s.ok()) {
        LOG(WARNING) << "Failed to delete RNG resource: " << s.ToString();
      }
    }
  }

  string op_type() const override { return kDatasetType; }

 protected:
  Status AsGraphDefInternal(SerializationContext* ctx,
                            DatasetGraphDefBuilder* b,
                            Node** output) const override {
    Node* input_graph_node = nullptr;
    TF_RETURN_IF_ERROR(b->AddInputDataset(ctx, input_, &input_graph_node));
    Node* buffer_size_node = nullptr;
    Node* seed_node = nullptr;
    Node* seed2_node = nullptr;
    Node* count_node = nullptr;
    TF_RETURN_IF_ERROR(b->AddScalar(buffer_size_, &buffer_size_node));
    TF_RETURN_IF_ERROR(b->AddScalar(seeds_.input_seed(), &seed_node));
    TF_RETURN_IF_ERROR(b->AddScalar(seeds_.input_seed2(), &seed2_node));
    TF_RETURN_IF_ERROR(b->AddScalar(count_, &count_node));
    Node* resource_handle_node = nullptr;
    Tensor handle(DT_RESOURCE, TensorShape({}));
    handle.scalar<ResourceHandle>()() = resource_handle_;
    TF_RETURN_IF_ERROR(b->AddTensor(handle, &resource_handle_node));
    AttrValue reshuffle_each_iteration;
    b->BuildAttrValue(seed_generator_->reshuffle_each_iteration(),
                      &reshuffle_each_iteration);
    TF_RETURN_IF_ERROR(
        b->AddDataset(this,
                      {input_graph_node, buffer_size_node, seed_node,
                       seed2_node, count_node, resource_handle_node},  // Inputs
                      {std::make_pair(kReshuffleEachIteration,
                                      reshuffle_each_iteration)},  // Attrs
                      output));
    return OkStatus();
  }

 private:
  SeedGeneratorManager* const manager_;  // Owned
  const bool owns_resource_;
  const ResourceHandle resource_handle_;
  ResourceMgr* const resource_mgr_;  // Not owned.
  const RandomSeeds seeds_;
};

ShuffleAndRepeatDatasetOp::ShuffleAndRepeatDatasetOp(OpKernelConstruction* ctx)
    : ShuffleDatasetOpBase(ctx) {
  auto& op_name = ctx->def().op();
  if (op_name == kShuffleAndRepeatDatasetV2) {
    op_version_ = 2;
  } else if (op_name == kShuffleAndRepeatDatasetV1) {
    op_version_ = 1;
  }
  if (ctx->HasAttr(kReshuffleEachIteration)) {
    OP_REQUIRES_OK(
        ctx, ctx->GetAttr(kReshuffleEachIteration, &reshuffle_each_iteration_));
  }
}

void ShuffleAndRepeatDatasetOp::MakeDataset(OpKernelContext* ctx,
                                            DatasetBase* input,
                                            DatasetBase** output) {
  int64_t buffer_size = 0;
  OP_REQUIRES_OK(ctx,
                 ParseScalarArgument<int64_t>(ctx, kBufferSize, &buffer_size));
  OP_REQUIRES(
      ctx, buffer_size > 0,
      errors::InvalidArgument("buffer_size must be greater than zero."));

  int64_t seed;
  OP_REQUIRES_OK(ctx, ParseScalarArgument<int64_t>(ctx, kSeed, &seed));

  int64_t seed2;
  OP_REQUIRES_OK(ctx, ParseScalarArgument<int64_t>(ctx, kSeed2, &seed2));

  int64_t count;
  OP_REQUIRES_OK(ctx, ParseScalarArgument<int64_t>(ctx, kCount, &count));

  OP_REQUIRES(ctx, count > 0 || count == -1,
              errors::InvalidArgument(
                  "count must be greater than zero or equal to -1."));

  RandomSeeds seeds(seed, seed2);

  static std::atomic<int64_t> resource_id_counter(0);
  const string& container = ctx->resource_manager()->default_container();
  auto name = strings::StrCat(ctx->op_kernel().name(), "/", kSeedGenerator, "_",
                              resource_id_counter.fetch_add(1));
  if (op_version_ == 2) {
    auto handle = HandleFromInput(ctx, 5);
    SeedGeneratorManager* manager = nullptr;
    Status s = ctx->resource_manager()->Lookup<SeedGeneratorManager>(
        handle.container(), handle.name(), &manager);
    bool owns_resource = false;
    if (errors::IsNotFound(s)) {
      owns_resource = true;
      OP_REQUIRES_OK(
          ctx,
          ctx->resource_manager()->LookupOrCreate<SeedGeneratorManager>(
              container, name, &manager,
              [reshuffle = reshuffle_each_iteration_,
               &seeds](SeedGeneratorManager** manager) {
                if (reshuffle) {
                  *manager =
                      new SeedGeneratorManager(new RandomSeedGenerator(seeds));
                } else {
                  *manager =
                      new SeedGeneratorManager(new FixedSeedGenerator(seeds));
                }
                return OkStatus();
              }));
      handle = MakeResourceHandle<SeedGenerator>(ctx, container, name);
    } else {
      OP_REQUIRES_OK(ctx, s);
    }

    // Ownership of manager is transferred onto `DatasetV2`.
    *output = new ShuffleAndRepeatDatasetOp::DatasetV2(
        ctx, input, buffer_size, count, std::move(seeds), manager,
        std::move(handle), owns_resource);
  } else {
    if (op_version_ != 1) {
      LOG(WARNING) << "Unsupported version of shuffle dataset op: "
                   << op_version_ << ". Defaulting to version 1.";
    }
    SeedGeneratorManager* manager;
    OP_REQUIRES_OK(
        ctx,
        ctx->resource_manager()->LookupOrCreate<SeedGeneratorManager>(
            container, name, &manager,
            [reshuffle = reshuffle_each_iteration_,
             &seeds](SeedGeneratorManager** manager) {
              if (reshuffle) {
                *manager =
                    new SeedGeneratorManager(new RandomSeedGenerator(seeds));
              } else {
                *manager =
                    new SeedGeneratorManager(new FixedSeedGenerator(seeds));
              }
              return OkStatus();
            }));
    auto handle =
        MakeResourceHandle<SeedGeneratorManager>(ctx, container, name);

    // Ownership of manager is transferred onto `Dataset`.
    *output = new Dataset(ctx, input, buffer_size, std::move(seeds), manager,
                          count, std::move(handle));
  }
}

namespace {
REGISTER_KERNEL_BUILDER(Name("ShuffleDataset").Device(DEVICE_CPU),
                        ShuffleDatasetOp);

REGISTER_KERNEL_BUILDER(Name("ShuffleDatasetV2").Device(DEVICE_CPU),
                        ShuffleDatasetOp);

REGISTER_KERNEL_BUILDER(Name("ShuffleDatasetV3").Device(DEVICE_CPU),
                        ShuffleDatasetOp);

REGISTER_KERNEL_BUILDER(Name("ShuffleAndRepeatDataset").Device(DEVICE_CPU),
                        ShuffleAndRepeatDatasetOp);

REGISTER_KERNEL_BUILDER(Name("ShuffleAndRepeatDatasetV2").Device(DEVICE_CPU),
                        ShuffleAndRepeatDatasetOp);
}  // namespace
}  // namespace data
}  // namespace tensorflow
