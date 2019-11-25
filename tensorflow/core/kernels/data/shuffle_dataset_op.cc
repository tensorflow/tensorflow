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

#include <deque>
#include <tuple>
#include <vector>

#include "tensorflow/core/framework/partial_tensor_shape.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/kernels/data/dataset_utils.h"
#include "tensorflow/core/kernels/data/name_utils.h"
#include "tensorflow/core/kernels/data/random_seed_ops.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/random/philox_random.h"
#include "tensorflow/core/lib/random/random.h"
#include "tensorflow/core/lib/random/random_distributions.h"

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

/* static */ constexpr const char* const ShuffleDatasetOp::kDatasetType;
/* static */ constexpr const char* const
    ShuffleDatasetOp::kReshuffleEachIteration;

/* static */ constexpr const char* const
    ShuffleAndRepeatDatasetOp::kDatasetType;
/* static */ constexpr const char* const ShuffleAndRepeatDatasetOp::kCount;

const int64 kLogIntervalMicros = 10 * 1000000;  // 10 seconds.
const int64 kMaxEpochsInBuffer = 3;

constexpr char kNumRandomSamples[] = "num_random_samples";
constexpr char kDataProduced[] = "data_produced";
constexpr char kEndOfInputSequence[] = "end_of_input_sequence";
constexpr char kEpoch[] = "epoch";
constexpr char kNumElements[] = "num_elements";
constexpr char kSlicesSize[] = "slices_size";
constexpr char kSlicesStart[] = "slices_start";
constexpr char kSlicesEnd[] = "slices_end";
constexpr char kBuffer[] = "buffer";
constexpr char kSize[] = "size";
constexpr char kRandomSeedGenerator[] = "RandomSeedGenerator";
constexpr char kTFData[] = "tf_data";
constexpr char kDSNumRandomSamples[] = "ds_num_random_samples";
constexpr char kFixedSeedDatasetPrefix[] = "FixedSeed";
constexpr char kReshufflingDatasetPrefix[] = "Reshuffling";
constexpr char kShuffleDataset[] = "ShuffleDataset";

ShuffleDatasetOpBase::ShuffleDatasetOpBase(OpKernelConstruction* ctx)
    : UnaryDatasetOpKernel(ctx) {}

// Abstract base dataset that implements a shuffling iterator.
class ShuffleDatasetOpBase::ShuffleDatasetBase : public DatasetBase {
 public:
  ShuffleDatasetBase(OpKernelContext* ctx, const DatasetBase* input,
                     int64 buffer_size, int64 count)
      : DatasetBase(DatasetContext(ctx)),
        input_(input),
        buffer_size_(buffer_size),
        count_(count) {
    input_->Ref();
  }

  ~ShuffleDatasetBase() override { input_->Unref(); }

  const DataTypeVector& output_dtypes() const override {
    return input_->output_dtypes();
  }

  const std::vector<PartialTensorShape>& output_shapes() const override {
    return input_->output_shapes();
  }

  int64 Cardinality() const override {
    if (count_ == -1 || input_->Cardinality() == kInfiniteCardinality) {
      return kInfiniteCardinality;
    } else if (input_->Cardinality() == kUnknownCardinality) {
      return kUnknownCardinality;
    } else {
      return input_->Cardinality() * count_;
    }
  }

  Status CheckExternalState() const override {
    return input_->CheckExternalState();
  }

 protected:
  template <class T>
  class Iterator : public DatasetIterator<T> {
   public:
    explicit Iterator(const typename DatasetIterator<T>::Params& params,
                      int64 seed, int64 seed2)
        : DatasetIterator<T>(params),
          seed_(seed),
          seed2_(seed2),
          input_impl_(nullptr),
          epoch_(0),
          num_elements_(0),
          parent_generator_(seed, seed2),
          generator_(&parent_generator_) {
      buffer_ = absl::make_unique<std::vector<Tensor>[]>(
          params.dataset->buffer_size_);
      slices_.push_back(absl::make_unique<Slice>(0, 0));
    }

    string BuildTraceMeName() override {
      return strings::StrCat(
          this->prefix(), "#buffer_size=", this->dataset()->buffer_size_, "#");
    }

    Status GetNextInternal(IteratorContext* ctx,
                           std::vector<Tensor>* out_tensors,
                           bool* end_of_sequence) override {
      mutex_lock l(mu_);
      int64 start_micros = ctx->env()->NowMicros();
      int64 num_log_entries = 0;
      if (!input_impl_ && epoch_ == 0) {
        TF_RETURN_IF_ERROR(this->dataset()->input_->MakeIterator(
            ctx, this->prefix(), &input_impl_));
      }
      while (input_impl_ && num_elements_ < this->dataset()->buffer_size_) {
        if (ctx->env()->NowMicros() >
            ((num_log_entries + 1) * kLogIntervalMicros) + start_micros) {
          num_log_entries++;
          LOG(INFO) << "Filling up shuffle buffer (this may take a while): "
                    << num_elements_ << " of " << this->dataset()->buffer_size_;
        }
        std::vector<Tensor> input_element;
        bool end_of_input_sequence = false;
        while (this->dataset()->count_ == -1 ||
               epoch_ < this->dataset()->count_) {
          TF_RETURN_IF_ERROR(input_impl_->GetNext(ctx, &input_element,
                                                  &end_of_input_sequence));
          if (!end_of_input_sequence) {
            data_produced_ = true;
            break;
          }
          if (!data_produced_ && this->dataset()->count_ == -1) {
            // If we encounter the end of sequence without producing data, we
            // terminate the iteration immediately. (Otherwise, this iterator
            // would loop infinitely and never produce a value.)
            *end_of_sequence = true;
            return Status::OK();
          }
          epoch_++;
          int64 n = slices_.back()->end;
          slices_.push_back(absl::make_unique<Slice>(n, n));
          TF_RETURN_IF_ERROR(this->dataset()->input_->MakeIterator(
              ctx, this->prefix(), &input_impl_));
        }
        if (!end_of_input_sequence) {
          if (num_elements_ == 0) {
            VLOG(1) << "Starting to fill up shuffle buffer of size: "
                    << this->dataset()->buffer_size_;
          }
          this->RecordBufferEnqueue(ctx, input_element);
          buffer_[slices_.back()->end % this->dataset()->buffer_size_] =
              std::move(input_element);
          num_elements_++;
          slices_.back()->end++;
        } else {
          input_impl_.reset();
        }
        if (slices_.size() > kMaxEpochsInBuffer) {
          // When the elements stored in `buffer_` span more than
          // `kMaxEpochsInBuffer` epochs, we do not fill the buffer further to
          // conserve memory. This means that the upper bound on the size of
          // `buffer_` is `kMaxEpochsInBuffer * cardinality(input_dataset) +
          // 1`.
          break;
        }
      }
      if (num_log_entries > 0) {
        LOG(INFO) << "Shuffle buffer filled.";
      }

      if (num_elements_ > 0) {
        *end_of_sequence = false;
        // Garbage collect all empty slices.
        while (!slices_.empty() &&
               slices_.front()->start == slices_.front()->end) {
          slices_.pop_front();
        }
        DCHECK(!slices_.empty());
        // Choose an element to produce uniformly at random from the first
        // slice, and then remove the element from the slice.
        int64 offset =
            Random() % (slices_.front()->end - slices_.front()->start);
        int64 index =
            (slices_.front()->start + offset) % this->dataset()->buffer_size_;
        *out_tensors = std::move(buffer_[index]);
        this->RecordBufferDequeue(ctx, *out_tensors);
        std::swap(
            buffer_[index],
            buffer_[slices_.front()->start % this->dataset()->buffer_size_]);
        slices_.front()->start++;
        num_elements_--;
      } else {
        DCHECK(input_impl_ == nullptr);
        *end_of_sequence = true;
      }
      return Status::OK();
    }

   protected:
    std::shared_ptr<model::Node> CreateNode(
        IteratorContext* ctx, model::Node::Args args) const override {
      return model::MakeKnownRatioNode(std::move(args),
                                       /*ratio=*/1);
    }

    void ResetRngs() EXCLUSIVE_LOCKS_REQUIRED(mu_) {
      // Reset the generators based on the current iterator seeds.
      parent_generator_ = random::PhiloxRandom(seed_, seed2_);
      generator_ =
          random::SingleSampleAdapter<random::PhiloxRandom>(&parent_generator_);
      generator_.Skip(num_random_samples_);
    }

    Status SaveInternal(IteratorStateWriter* writer) override {
      mutex_lock l(mu_);
      // Save state needed to restore the random number generators.
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
        TF_RETURN_IF_ERROR(this->SaveInput(writer, input_impl_));
      }

      // Save the epoch counter, buffer, and buffer slices.
      TF_RETURN_IF_ERROR(writer->WriteScalar(this->full_name(kEpoch), epoch_));
      TF_RETURN_IF_ERROR(
          writer->WriteScalar(this->full_name(kNumElements), num_elements_));
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
        for (size_t j = slices_[i]->start; j < slices_[i]->end; ++j) {
          size_t index = j % this->dataset()->buffer_size_;
          TF_RETURN_IF_ERROR(writer->WriteScalar(
              this->full_name(
                  absl::StrJoin(std::make_tuple(kBuffer, index, kSize), "_")),
              buffer_[index].size()));
          for (size_t k = 0; k < buffer_[index].size(); ++k) {
            TF_RETURN_IF_ERROR(writer->WriteTensor(
                this->full_name(
                    absl::StrJoin(std::make_tuple(kBuffer, index, k), "_")),
                buffer_[index][k]));
          }
        }
      }
      if (data_produced_) {
        TF_RETURN_IF_ERROR(
            writer->WriteScalar(this->full_name(kDataProduced), ""));
      }

      return Status::OK();
    }

    Status RestoreInternal(IteratorContext* ctx,
                           IteratorStateReader* reader) override {
      mutex_lock l(mu_);
      // Restore the random number generators.
      TF_RETURN_IF_ERROR(reader->ReadScalar(this->full_name(kNumRandomSamples),
                                            &num_random_samples_));
      TF_RETURN_IF_ERROR(reader->ReadScalar(this->full_name(kSeed), &seed_));
      TF_RETURN_IF_ERROR(reader->ReadScalar(this->full_name(kSeed2), &seed2_));
      ResetRngs();

      // Restore the input iterator if it wasn't already exhausted.
      if (!reader->Contains(this->full_name(kEndOfInputSequence))) {
        TF_RETURN_IF_ERROR(this->dataset()->input_->MakeIterator(
            ctx, this->prefix(), &input_impl_));
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
        int64 temp;
        TF_RETURN_IF_ERROR(
            reader->ReadScalar(this->full_name(kSlicesSize), &temp));
        slices_size = static_cast<size_t>(temp);
      }
      buffer_ = absl::make_unique<std::vector<Tensor>[]>(
          this->dataset()->buffer_size_);
      slices_.clear();
      for (size_t i = 0; i < slices_size; ++i) {
        int64 start;
        TF_RETURN_IF_ERROR(
            reader->ReadScalar(this->full_name(absl::StrJoin(
                                   std::make_tuple(kSlicesStart, i), "_")),
                               &start));
        int64 end;
        TF_RETURN_IF_ERROR(reader->ReadScalar(
            this->full_name(absl::StrJoin(std::make_tuple(kSlicesEnd, i), "_")),
            &end));
        slices_.push_back(absl::make_unique<Slice>(start, end));
        for (size_t j = start; j < end; ++j) {
          size_t index = j % this->dataset()->buffer_size_;
          int64 list_size;
          TF_RETURN_IF_ERROR(reader->ReadScalar(
              this->full_name(
                  absl::StrJoin(std::make_tuple(kBuffer, index, kSize), "_")),
              &list_size));
          buffer_[index] = std::vector<Tensor>(list_size);
          for (int k = 0; k < list_size; ++k) {
            TF_RETURN_IF_ERROR(reader->ReadTensor(
                this->full_name(
                    absl::StrJoin(std::make_tuple(kBuffer, index, k), "_")),
                &buffer_[index][k]));
          }
        }
      }
      data_produced_ = reader->Contains(this->full_name(kDataProduced));

      return Status::OK();
    }

    mutex mu_;
    int64 seed_ GUARDED_BY(mu_);
    int64 seed2_ GUARDED_BY(mu_);

   private:
    // Used to represent slices of `buffer_` that belong to different epochs.
    // The invariant maintained by the implementation is: `start` <= `end`.
    // When using `start` and `end` to index into `buffer_`, their values
    // should be taken modulo the size of `buffer_` as their absolute value
    // can be greater than the range of `buffer_`.
    struct Slice {
      Slice(int64 start, int64 end) : start(start), end(end) {}

      int64 start;
      int64 end;
    };

    random::SingleSampleAdapter<random::PhiloxRandom>::ResultType Random()
        EXCLUSIVE_LOCKS_REQUIRED(mu_) {
      num_random_samples_++;
      auto out = generator_();
      return out;
    }

    std::unique_ptr<std::vector<Tensor>[]> buffer_ GUARDED_BY(mu_);
    std::unique_ptr<IteratorBase> input_impl_ GUARDED_BY(mu_);
    int64 epoch_ GUARDED_BY(mu_);
    int64 num_elements_ GUARDED_BY(mu_);
    // Indices into `buffer_` indicating which data belongs to which epoch.
    // The slice at the front of the deque references data from the earliest
    // buffered epoch. It is an invariant that all slices reference
    // non-overlapping sections of `buffer_`.
    std::deque<std::unique_ptr<Slice>> slices_ GUARDED_BY(mu_);
    random::PhiloxRandom parent_generator_ GUARDED_BY(mu_);
    random::SingleSampleAdapter<random::PhiloxRandom> generator_
        GUARDED_BY(mu_);
    int64 num_random_samples_ GUARDED_BY(mu_) = 0;
    bool data_produced_ GUARDED_BY(mu_) = false;
  };

  const DatasetBase* const input_;
  const int64 buffer_size_;
  // The number of epochs to run for. Normally this is just 1, but sometimes we
  // fuse shuffle and repeat together, and make the shuffle dataset op
  // responsible for repeating as well.
  const int64 count_;
};

// A dataset that uses a pseudorandom sequence of seeds for the iterators
// created from it. Used when `reshuffle_each_iteration` is true.
class ShuffleDatasetOp::ReshufflingDataset : public ShuffleDatasetBase {
 public:
  ReshufflingDataset(OpKernelContext* ctx, const DatasetBase* input,
                     int64 buffer_size, int64 seed, int64 seed2, int64 count)
      : ShuffleDatasetBase(ctx, input, buffer_size, count),
        seed_(seed),
        seed2_(seed2) {}

  string DebugString() const override {
    name_utils::DatasetDebugStringParams params;
    params.dataset_prefix = kReshufflingDatasetPrefix;
    params.set_args(buffer_size_, seed_, seed2_);
    return name_utils::DatasetDebugString(kDatasetType, params);
  }

  std::unique_ptr<IteratorBase> MakeIteratorInternal(
      const string& prefix) const override {
    return absl::make_unique<Iterator>(
        Iterator::Params{this,
                         name_utils::IteratorPrefix(kDatasetType, prefix)},
        seed_, seed2_);
  }

 protected:
  class Iterator : public ShuffleDatasetBase::Iterator<ReshufflingDataset> {
   public:
    Iterator(const Params& params, int64 seed, int64 seed2)
        : ShuffleDatasetBase::Iterator<ReshufflingDataset>(params, seed,
                                                           seed2) {}

    ~Iterator() override { seed_generator_->Unref(); }

    Status Initialize(IteratorContext* ctx) override {
      // Firstly, lookup or create a seed generator from the IteratorResource
      // resource_mgr.
      ResourceMgr* mgr = ctx->resource_mgr();
      RandomSeedGenerator* seed_generator;
      const string name = strings::StrCat(
          prefix(), name_utils::kDelimiter, dataset()->type_string(),
          name_utils::kDelimiter, kRandomSeedGenerator);

      int64 dataset_seed, dataset_seed2;
      {
        tf_shared_lock l(mu_);
        // Ideally we'd like to hold this lock in the LookupOrCreate method,
        // but that trips up our Deadlock detection code.
        dataset_seed = seed_;
        dataset_seed2 = seed2_;
      }
      TF_RETURN_IF_ERROR(mgr->LookupOrCreate<RandomSeedGenerator>(
          kTFData, name, &seed_generator,
          [dataset_seed, dataset_seed2](RandomSeedGenerator** seed_generator) {
            // On the first iterator creation, use the original seeds from the
            // dataset to seed a `RandomSeedGenerator` that will provide seeds
            // for subsequent repetitions of the same dataset.
            *seed_generator =
                new RandomSeedGenerator(dataset_seed, dataset_seed2);
            return Status::OK();
          }));
      seed_generator_ = seed_generator;
      seed_generator_->GenerateRandomSeeds(&seed_, &seed2_);
      mutex_lock l(mu_);
      ResetRngs();
      return Status::OK();
    }

   protected:
    std::shared_ptr<model::Node> CreateNode(
        IteratorContext* ctx, model::Node::Args args) const override {
      return model::MakeKnownRatioNode(std::move(args),
                                       /*ratio=*/1);
    }

    Status SaveInternal(IteratorStateWriter* writer) override {
      // Save RNG state of Dataset.
      TF_RETURN_IF_ERROR(
          writer->WriteScalar(full_name(kDSNumRandomSamples),
                              seed_generator_->num_random_samples()));

      // Save the Iterator.
      return ShuffleDatasetBase::Iterator<ReshufflingDataset>::SaveInternal(
          writer);
    }

    Status RestoreInternal(IteratorContext* ctx,
                           IteratorStateReader* reader) override {
      // Restore RNG state of Dataset.
      int64 num_random_samples;
      TF_RETURN_IF_ERROR(reader->ReadScalar(full_name(kDSNumRandomSamples),
                                            &num_random_samples));
      seed_generator_->set_num_random_samples(num_random_samples);
      seed_generator_->Reset();

      // Restore the Iterator.
      return ShuffleDatasetBase::Iterator<ReshufflingDataset>::RestoreInternal(
          ctx, reader);
    }

   private:
    RandomSeedGenerator* seed_generator_;
  };

  Status AsGraphDefInternal(SerializationContext* ctx,
                            DatasetGraphDefBuilder* b,
                            Node** output) const override {
    Node* input_graph_node = nullptr;
    TF_RETURN_IF_ERROR(b->AddInputDataset(ctx, input_, &input_graph_node));
    Node* buffer_size = nullptr;
    Node* seed = nullptr;
    Node* seed2 = nullptr;
    AttrValue reshuffle_each_iteration;

    TF_RETURN_IF_ERROR(b->AddScalar(buffer_size_, &buffer_size));
    TF_RETURN_IF_ERROR(b->AddScalar(seed_, &seed));
    TF_RETURN_IF_ERROR(b->AddScalar(seed2_, &seed2));
    b->BuildAttrValue(true, &reshuffle_each_iteration);
    TF_RETURN_IF_ERROR(b->AddDataset(
        this, {input_graph_node, buffer_size, seed, seed2},  // Inputs
        {std::make_pair(kReshuffleEachIteration,
                        reshuffle_each_iteration)},  // Attrs
        output));
    return Status::OK();
  }

 private:
  const int64 seed_;
  const int64 seed2_;
};

// A dataset that uses a pseudorandom sequence of seeds for the iterators
// created from it. Used in TF 2.0 when `reshuffle_each_iteration` is true.
class ShuffleDatasetOp::ReshufflingDatasetV2 : public ShuffleDatasetBase {
 public:
  ReshufflingDatasetV2(OpKernelContext* ctx, const DatasetBase* input,
                       int64 buffer_size, int64 count,
                       RandomSeedGenerator* seed_generator,
                       std::unique_ptr<OwnedResourceHandle> handle)
      : ShuffleDatasetBase(ctx, input, buffer_size, count),
        seed_generator_(seed_generator),
        handle_(std::move(handle)) {}

  ~ReshufflingDatasetV2() override { seed_generator_->Unref(); }

  string DebugString() const override {
    name_utils::DatasetDebugStringParams params;
    params.dataset_prefix = kReshufflingDatasetPrefix;
    params.set_args(buffer_size_);
    return name_utils::DatasetDebugString(kDatasetType, params);
  }

  Status CheckExternalState() const override {
    return errors::FailedPrecondition(
        DebugString(), " depends on random seed generator resource.");
  }

  std::unique_ptr<IteratorBase> MakeIteratorInternal(
      const string& prefix) const override {
    return absl::make_unique<Iterator>(
        Iterator::Params{this,
                         name_utils::IteratorPrefix(kDatasetType, prefix)},
        seed_generator_);
  }

 protected:
  class Iterator : public ShuffleDatasetBase::Iterator<ReshufflingDatasetV2> {
   public:
    Iterator(const Params& params, RandomSeedGenerator* seed_generator)
        : ShuffleDatasetBase::Iterator<ReshufflingDatasetV2>(params, 0, 0),
          seed_generator_(seed_generator) {}

    Status Initialize(IteratorContext* ctx) override {
      mutex_lock l(mu_);
      seed_generator_->GenerateRandomSeeds(&seed_, &seed2_);
      ResetRngs();
      return Status::OK();
    }

   protected:
    std::shared_ptr<model::Node> CreateNode(
        IteratorContext* ctx, model::Node::Args args) const override {
      return model::MakeKnownRatioNode(std::move(args), /*ratio=*/1);
    }

    Status SaveInternal(IteratorStateWriter* writer) override {
      // Save state of the seed generator.
      TF_RETURN_IF_ERROR(
          writer->WriteScalar(full_name(kDSNumRandomSamples),
                              seed_generator_->num_random_samples()));

      // Save the tterator state.
      return ShuffleDatasetBase::Iterator<ReshufflingDatasetV2>::SaveInternal(
          writer);
    }

    Status RestoreInternal(IteratorContext* ctx,
                           IteratorStateReader* reader) override {
      // Restore state of the seed generator.
      int64 num_random_samples;
      TF_RETURN_IF_ERROR(reader->ReadScalar(full_name(kDSNumRandomSamples),
                                            &num_random_samples));
      seed_generator_->set_num_random_samples(num_random_samples);
      seed_generator_->Reset();

      // Restore the iterator state.
      return ShuffleDatasetBase::Iterator<
          ReshufflingDatasetV2>::RestoreInternal(ctx, reader);
    }

   private:
    RandomSeedGenerator* seed_generator_;
  };

  Status AsGraphDefInternal(SerializationContext* ctx,
                            DatasetGraphDefBuilder* b,
                            Node** output) const override {
    Node* input_graph_node = nullptr;
    TF_RETURN_IF_ERROR(b->AddInputDataset(ctx, input_, &input_graph_node));
    Node* buffer_size_node = nullptr;
    TF_RETURN_IF_ERROR(b->AddScalar(buffer_size_, &buffer_size_node));
    Node* resource_handle_node = nullptr;
    Tensor handle(DT_RESOURCE, TensorShape({}));
    handle.scalar<ResourceHandle>()() = handle_->handle();
    TF_RETURN_IF_ERROR(b->AddTensor(handle, &resource_handle_node));
    TF_RETURN_IF_ERROR(b->AddDataset(
        this,
        {input_graph_node, buffer_size_node, resource_handle_node},  // Inputs
        {},                                                          // Attrs
        output));
    return Status::OK();
  }

 private:
  RandomSeedGenerator* seed_generator_ = nullptr;
  std::unique_ptr<OwnedResourceHandle> handle_;
};

// A dataset that uses the same fixed seed for all iterators created from it.
// Used when `reshuffle_each_iteration` is false.
class ShuffleDatasetOp::FixedSeedDataset : public ShuffleDatasetBase {
 public:
  FixedSeedDataset(OpKernelContext* ctx, const DatasetBase* input,
                   int64 buffer_size, int64 seed, int64 seed2, int64 count)
      : ShuffleDatasetBase(ctx, input, buffer_size, count),
        seed_(seed),
        seed2_(seed2) {}

  string DebugString() const override {
    name_utils::DatasetDebugStringParams params;
    params.dataset_prefix = kFixedSeedDatasetPrefix;
    params.set_args(buffer_size_, seed_, seed2_);
    return name_utils::DatasetDebugString(kDatasetType, params);
  }

  std::unique_ptr<IteratorBase> MakeIteratorInternal(
      const string& prefix) const override {
    return absl::make_unique<ShuffleDatasetBase::Iterator<ShuffleDatasetBase>>(
        ShuffleDatasetBase::Iterator<ShuffleDatasetBase>::Params{
            this, name_utils::IteratorPrefix(kDatasetType, prefix)},
        seed_, seed2_);
  }

 protected:
  Status AsGraphDefInternal(SerializationContext* ctx,
                            DatasetGraphDefBuilder* b,
                            Node** output) const override {
    Node* input_graph_node = nullptr;
    TF_RETURN_IF_ERROR(b->AddInputDataset(ctx, input_, &input_graph_node));
    Node* buffer_size = nullptr;
    Node* seed = nullptr;
    Node* seed2 = nullptr;
    AttrValue reshuffle_each_iteration;

    TF_RETURN_IF_ERROR(b->AddScalar(buffer_size_, &buffer_size));
    TF_RETURN_IF_ERROR(b->AddScalar(seed_, &seed));
    TF_RETURN_IF_ERROR(b->AddScalar(seed2_, &seed2));
    b->BuildAttrValue(false, &reshuffle_each_iteration);
    TF_RETURN_IF_ERROR(b->AddDataset(
        this, {input_graph_node, buffer_size, seed, seed2},  // Inputs
        {std::make_pair(kReshuffleEachIteration,
                        reshuffle_each_iteration)},  // Attrs
        output));
    return Status::OK();
  }

 private:
  const int64 seed_;
  const int64 seed2_;
};

ShuffleDatasetOp::ShuffleDatasetOp(OpKernelConstruction* ctx)
    : ShuffleDatasetOpBase(ctx),
      op_version_(ctx->def().op() == kShuffleDataset ? 1 : 2) {
  if (ctx->HasAttr(kReshuffleEachIteration)) {
    OP_REQUIRES_OK(
        ctx, ctx->GetAttr(kReshuffleEachIteration, &reshuffle_each_iteration_));
  }
}

void ShuffleDatasetOp::MakeDataset(OpKernelContext* ctx, DatasetBase* input,
                                   DatasetBase** output) {
  int64 buffer_size = 0;
  OP_REQUIRES_OK(ctx,
                 ParseScalarArgument<int64>(ctx, kBufferSize, &buffer_size));
  OP_REQUIRES(
      ctx, buffer_size > 0,
      errors::InvalidArgument("buffer_size must be greater than zero."));

  int64 count = 1;
  if (op_version_ == 2) {
    RandomSeedGenerator* seed_generator = nullptr;
    OP_REQUIRES_OK(
        ctx, LookupResource(ctx, HandleFromInput(ctx, 2), &seed_generator));

    // Create a fresh handle for the resource because the input handle can
    // become invalid after this op executes.
    std::unique_ptr<OwnedResourceHandle> handle;
    OP_REQUIRES_OK(ctx,
                   OwnedResourceHandle::Create(ctx, seed_generator,
                                               kRandomSeedGenerator, &handle));

    // Ownership of seed generator is transferred onto `ReshufflingDatasetV2`.
    *output = new ReshufflingDatasetV2(ctx, input, buffer_size, count,
                                       seed_generator, std::move(handle));
    return;
  }

  int64 seed;
  OP_REQUIRES_OK(ctx, ParseScalarArgument<int64>(ctx, kSeed, &seed));

  int64 seed2;
  OP_REQUIRES_OK(ctx, ParseScalarArgument<int64>(ctx, kSeed2, &seed2));

  // By TensorFlow convention, passing 0 for both seeds indicates
  // that the shuffling should be seeded non-deterministically.
  if (seed == 0 && seed2 == 0) {
    seed = random::New64();
    seed2 = random::New64();
  }

  if (reshuffle_each_iteration_) {
    *output =
        new ReshufflingDataset(ctx, input, buffer_size, seed, seed2, count);
  } else {
    *output = new FixedSeedDataset(ctx, input, buffer_size, seed, seed2, count);
  }
}

class ShuffleAndRepeatDatasetOp::Dataset : public ShuffleDatasetBase {
 public:
  Dataset(OpKernelContext* ctx, const DatasetBase* input, int64 buffer_size,
          int64 seed, int64 seed2, int64 count)
      : ShuffleDatasetBase(ctx, input, buffer_size, count),
        seed_(seed),
        seed2_(seed2) {}

  string DebugString() const override {
    name_utils::DatasetDebugStringParams params;
    params.set_args(buffer_size_, seed_, seed2_);
    return name_utils::DatasetDebugString(kDatasetType, params);
  }

  std::unique_ptr<IteratorBase> MakeIteratorInternal(
      const string& prefix) const override {
    return absl::make_unique<ShuffleDatasetBase::Iterator<ShuffleDatasetBase>>(
        ShuffleDatasetBase::Iterator<ShuffleDatasetBase>::Params{
            this, name_utils::IteratorPrefix(kDatasetType, prefix)},
        seed_, seed2_);
  }

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
    TF_RETURN_IF_ERROR(b->AddScalar(seed_, &seed));
    TF_RETURN_IF_ERROR(b->AddScalar(seed2_, &seed2));
    TF_RETURN_IF_ERROR(b->AddScalar(count_, &count));
    TF_RETURN_IF_ERROR(b->AddDataset(
        this, {input_graph_node, buffer_size, seed, seed2, count},  // Inputs
        {},                                                         // Attrs
        output));
    return Status::OK();
  }

 private:
  const int64 seed_;
  const int64 seed2_;
};

ShuffleAndRepeatDatasetOp::ShuffleAndRepeatDatasetOp(OpKernelConstruction* ctx)
    : ShuffleDatasetOpBase(ctx) {}

void ShuffleAndRepeatDatasetOp::MakeDataset(OpKernelContext* ctx,
                                            DatasetBase* input,
                                            DatasetBase** output) {
  int64 buffer_size = 0;
  OP_REQUIRES_OK(ctx,
                 ParseScalarArgument<int64>(ctx, kBufferSize, &buffer_size));
  OP_REQUIRES(
      ctx, buffer_size > 0,
      errors::InvalidArgument("buffer_size must be greater than zero."));

  int64 seed;
  OP_REQUIRES_OK(ctx, ParseScalarArgument<int64>(ctx, kSeed, &seed));

  int64 seed2;
  OP_REQUIRES_OK(ctx, ParseScalarArgument<int64>(ctx, kSeed2, &seed2));

  int64 count;
  OP_REQUIRES_OK(ctx, ParseScalarArgument<int64>(ctx, kCount, &count));

  OP_REQUIRES(ctx, count > 0 || count == -1,
              errors::InvalidArgument(
                  "count must be greater than zero or equal to -1."));

  // By TensorFlow convention, if both seeds are 0, then shuffling should be
  // seeded non-deterministically.
  if (seed == 0 && seed2 == 0) {
    seed = random::New64();
    seed2 = random::New64();
  }

  *output = new Dataset(ctx, input, buffer_size, seed, seed2, count);
}

namespace {
REGISTER_KERNEL_BUILDER(Name("ShuffleDataset").Device(DEVICE_CPU),
                        ShuffleDatasetOp);

REGISTER_KERNEL_BUILDER(Name("ShuffleDatasetV2").Device(DEVICE_CPU),
                        ShuffleDatasetOp);

REGISTER_KERNEL_BUILDER(Name("ShuffleAndRepeatDataset").Device(DEVICE_CPU),
                        ShuffleAndRepeatDatasetOp);
}  // namespace
}  // namespace data
}  // namespace tensorflow
