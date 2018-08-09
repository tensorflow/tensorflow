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

#include <deque>
#include <vector>

#include "tensorflow/core/framework/partial_tensor_shape.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/kernels/data/dataset.h"
#include "tensorflow/core/lib/random/philox_random.h"
#include "tensorflow/core/lib/random/random.h"
#include "tensorflow/core/lib/random/random_distributions.h"

namespace tensorflow {

namespace {

const int64 kLogIntervalMicros = 10 * 1000000;  // 10 seconds.

// See documentation in ../ops/dataset_ops.cc for a high-level
// description of the following op.

class ShuffleDatasetOpBase : public UnaryDatasetOpKernel {
 public:
  explicit ShuffleDatasetOpBase(OpKernelConstruction* ctx)
      : UnaryDatasetOpKernel(ctx) {}

 protected:
  // Abstract base dataset that implements a shuffling iterator.
  class ShuffleDatasetBase : public GraphDatasetBase {
   public:
    ShuffleDatasetBase(OpKernelContext* ctx, const DatasetBase* input,
                       int64 buffer_size, int64 count)
        : GraphDatasetBase(ctx),
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

   protected:
    template <class T>
    class Iterator : public DatasetIterator<T> {
     public:
      explicit Iterator(const typename DatasetIterator<T>::Params& params,
                        int64 seed, int64 seed2)
          : DatasetIterator<T>(params),
            input_impl_(nullptr),
            seed_(seed),
            seed2_(seed2),
            epoch_(0),
            num_elements_(0),
            parent_generator_(seed, seed2),
            generator_(&parent_generator_) {
        buffer_.reset(new std::vector<Tensor>[params.dataset->buffer_size_]);
        slices_.emplace_back(new Slice{0, 0});
      }

      Status GetNextInternal(IteratorContext* ctx,
                             std::vector<Tensor>* out_tensors,
                             bool* end_of_sequence) override {
        mutex_lock l(mu_);
        int64 start_micros = ctx->env()->NowMicros();
        int64 num_log_entries = 0;
        bool first_call = false;
        if (!input_impl_ && epoch_ == 0) {
          first_call = true;
          TF_RETURN_IF_ERROR(this->dataset()->input_->MakeIterator(
              ctx, this->prefix(), &input_impl_));
        }
        while (input_impl_ && num_elements_ < this->dataset()->buffer_size_) {
          if (ctx->env()->NowMicros() >
              ((num_log_entries + 1) * kLogIntervalMicros) + start_micros) {
            num_log_entries++;
            LOG(INFO) << "Filling up shuffle buffer (this may take a while): "
                      << num_elements_ << " of "
                      << this->dataset()->buffer_size_;
          }
          std::vector<Tensor> input_element;
          bool end_of_input_sequence = false;
          while (this->dataset()->count_ == -1 ||
                 epoch_ < this->dataset()->count_) {
            TF_RETURN_IF_ERROR(input_impl_->GetNext(ctx, &input_element,
                                                    &end_of_input_sequence));
            if (!end_of_input_sequence) {
              first_call = false;
              break;
            }
            if (first_call && this->dataset()->count_ == -1) {
              // If the first call to GetNext() fails because the end
              // of sequence has been reached, we terminate the
              // iteration immediately. (Otherwise, this iterator
              // would loop infinitely and never produce a value.)
              *end_of_sequence = true;
              return Status::OK();
            }
            epoch_++;
            int64 n = slices_.back()->end;
            slices_.emplace_back(new Slice{n, n});
            TF_RETURN_IF_ERROR(this->dataset()->input_->MakeIterator(
                ctx, this->prefix(), &input_impl_));
          }
          if (!end_of_input_sequence) {
            buffer_[slices_.back()->end % this->dataset()->buffer_size_] =
                std::move(input_element);
            num_elements_++;
            slices_.back()->end++;
          } else {
            input_impl_.reset();
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
      Status SaveInternal(IteratorStateWriter* writer) override {
        mutex_lock l(mu_);
        // Save state needed to restore the random number generators.
        TF_RETURN_IF_ERROR(writer->WriteScalar(
            this->full_name("num_random_samples"), num_random_samples_));
        TF_RETURN_IF_ERROR(writer->WriteScalar(this->full_name("seed"), seed_));
        TF_RETURN_IF_ERROR(
            writer->WriteScalar(this->full_name("seed2"), seed2_));

        // Save input iterator if it hasn't been exhausted else write
        // "end_of_input_sequence".
        if (!input_impl_) {
          TF_RETURN_IF_ERROR(writer->WriteScalar(
              this->full_name("end_of_input_sequence"), ""));
        } else {
          TF_RETURN_IF_ERROR(this->SaveParent(writer, input_impl_));
        }

        // Save the epoch counter, buffer, and buffer slices.
        TF_RETURN_IF_ERROR(
            writer->WriteScalar(this->full_name("epoch"), epoch_));
        TF_RETURN_IF_ERROR(writer->WriteScalar(this->full_name("num_elements"),
                                               num_elements_));
        TF_RETURN_IF_ERROR(writer->WriteScalar(this->full_name("slices_size"),
                                               slices_.size()));
        for (size_t i = 0; i < slices_.size(); ++i) {
          TF_RETURN_IF_ERROR(writer->WriteScalar(
              this->full_name(strings::StrCat("slices_start_", i)),
              slices_[i]->start));
          TF_RETURN_IF_ERROR(writer->WriteScalar(
              this->full_name(strings::StrCat("slices_end_", i)),
              slices_[i]->end));
          for (size_t j = slices_[i]->start; j < slices_[i]->end; ++j) {
            size_t index = j % this->dataset()->buffer_size_;
            TF_RETURN_IF_ERROR(writer->WriteScalar(
                this->full_name(strings::StrCat("buffer_", index, "_size")),
                buffer_[index].size()));
            for (size_t k = 0; k < buffer_[index].size(); ++k) {
              TF_RETURN_IF_ERROR(writer->WriteTensor(
                  this->full_name(strings::StrCat("buffer_", index, "_", k)),
                  buffer_[index][k]));
            }
          }
        }

        return Status::OK();
      }

      Status RestoreInternal(IteratorContext* ctx,
                             IteratorStateReader* reader) override {
        mutex_lock l(mu_);
        // Restore the random number generators.
        TF_RETURN_IF_ERROR(reader->ReadScalar(
            this->full_name("num_random_samples"), &num_random_samples_));
        TF_RETURN_IF_ERROR(reader->ReadScalar(this->full_name("seed"), &seed_));
        TF_RETURN_IF_ERROR(
            reader->ReadScalar(this->full_name("seed2"), &seed2_));
        ResetRngs();

        // Restore the input iterator if it wasn't already exhausted.
        if (!reader->Contains(this->full_name("end_of_input_sequence"))) {
          TF_RETURN_IF_ERROR(this->dataset()->input_->MakeIterator(
              ctx, this->prefix(), &input_impl_));
          TF_RETURN_IF_ERROR(this->RestoreParent(ctx, reader, input_impl_));
        } else {
          input_impl_.reset();
        }

        // Restore the epoch counter, buffer, and buffer slices.
        TF_RETURN_IF_ERROR(
            reader->ReadScalar(this->full_name("epoch"), &epoch_));
        TF_RETURN_IF_ERROR(reader->ReadScalar(this->full_name("num_elements"),
                                              &num_elements_));
        size_t slices_size;
        {
          int64 temp;
          TF_RETURN_IF_ERROR(
              reader->ReadScalar(this->full_name("slices_size"), &temp));
          slices_size = static_cast<size_t>(temp);
        }
        buffer_.reset(new std::vector<Tensor>[this->dataset()->buffer_size_]);
        for (size_t i = 0; i < slices_size; ++i) {
          int64 start;
          TF_RETURN_IF_ERROR(reader->ReadScalar(
              this->full_name(strings::StrCat("slices_start_", i)), &start));
          int64 end;
          TF_RETURN_IF_ERROR(reader->ReadScalar(
              this->full_name(strings::StrCat("slices_end_", i)), &end));
          slices_.emplace_back(new Slice{start, end});
          for (size_t j = start; j < end; ++j) {
            size_t index = j % this->dataset()->buffer_size_;
            int64 list_size;
            TF_RETURN_IF_ERROR(reader->ReadScalar(
                this->full_name(strings::StrCat("buffer_", index, "_size")),
                &list_size));
            buffer_[index] = std::vector<Tensor>(list_size);
            for (int k = 0; k < list_size; ++k) {
              TF_RETURN_IF_ERROR(reader->ReadTensor(
                  this->full_name(strings::StrCat("buffer_", index, "_", k)),
                  &buffer_[index][k]));
            }
          }
        }

        return Status::OK();
      }

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

      void ResetRngs() EXCLUSIVE_LOCKS_REQUIRED(mu_) {
        // Reset the generators based on the current iterator seeds.
        parent_generator_ = random::PhiloxRandom(seed_, seed2_);
        generator_ = random::SingleSampleAdapter<random::PhiloxRandom>(
            &parent_generator_);
        generator_.Skip(num_random_samples_);
      }

      mutex mu_;
      std::unique_ptr<std::vector<Tensor>[]> buffer_ GUARDED_BY(mu_);
      std::unique_ptr<IteratorBase> input_impl_ GUARDED_BY(mu_);
      int64 seed_ GUARDED_BY(mu_);
      int64 seed2_ GUARDED_BY(mu_);
      int64 epoch_ GUARDED_BY(mu_);
      int64 num_elements_ GUARDED_BY(mu_);
      std::deque<std::unique_ptr<Slice>> slices_ GUARDED_BY(mu_);
      random::PhiloxRandom parent_generator_ GUARDED_BY(mu_);
      random::SingleSampleAdapter<random::PhiloxRandom> generator_
          GUARDED_BY(mu_);
      int64 num_random_samples_ GUARDED_BY(mu_) = 0;
    };

    const DatasetBase* const input_;
    const int64 buffer_size_;
    const int64 count_;
  };
};

class ShuffleDatasetOp : public ShuffleDatasetOpBase {
 public:
  explicit ShuffleDatasetOp(OpKernelConstruction* ctx)
      : ShuffleDatasetOpBase(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("reshuffle_each_iteration",
                                     &reshuffle_each_iteration_));
  }

  void MakeDataset(OpKernelContext* ctx, DatasetBase* input,
                   DatasetBase** output) override {
    int64 buffer_size;
    OP_REQUIRES_OK(
        ctx, ParseScalarArgument<int64>(ctx, "buffer_size", &buffer_size));
    OP_REQUIRES(
        ctx, buffer_size > 0,
        errors::InvalidArgument("buffer_size must be greater than zero."));

    int64 seed;
    OP_REQUIRES_OK(ctx, ParseScalarArgument<int64>(ctx, "seed", &seed));

    int64 seed2;
    OP_REQUIRES_OK(ctx, ParseScalarArgument<int64>(ctx, "seed2", &seed2));

    // By TensorFlow convention, passing 0 for both seeds indicates
    // that the shuffling should be seeded non-deterministically.
    if (seed == 0 && seed2 == 0) {
      seed = random::New64();
      seed2 = random::New64();
    }

    int64 count = 1;
    if (reshuffle_each_iteration_) {
      *output =
          new ReshufflingDataset(ctx, input, buffer_size, seed, seed2, count);
    } else {
      *output =
          new FixedSeedDataset(ctx, input, buffer_size, seed, seed2, count);
    }
  }

 private:
  // A dataset that uses a pseduorandom sequence of seeds for the iterators
  // created from it. Used when `reshuffle_each_iteration` is true.
  class ReshufflingDataset : public ShuffleDatasetBase {
   public:
    ReshufflingDataset(OpKernelContext* ctx, const DatasetBase* input,
                       int64 buffer_size, int64 seed, int64 seed2, int64 count)
        : ShuffleDatasetBase(ctx, input, buffer_size, count),
          seed_(seed),
          seed2_(seed2),
          parent_generator_(seed, seed2),
          generator_(&parent_generator_) {}

    string DebugString() const override {
      mutex_lock l(mu_);
      return strings::StrCat("ShuffleDatasetOp(", buffer_size_, ", ", seed_,
                             ", ", seed2_, ")::ReshufflingDataset");
    }

    std::unique_ptr<IteratorBase> MakeIteratorInternal(
        const string& prefix) const override {
      int64 iterator_seed;
      int64 iterator_seed2;
      {
        mutex_lock l(mu_);
        iterator_seed = Random();
        iterator_seed2 = Random();
      }
      return std::unique_ptr<IteratorBase>(
          new Iterator({this, strings::StrCat(prefix, "::Shuffle")},
                       iterator_seed, iterator_seed2));
    }

   protected:
    class Iterator : public ShuffleDatasetBase::Iterator<ReshufflingDataset> {
     public:
      explicit Iterator(const Params& params, int64 seed, int64 seed2)
          : ShuffleDatasetBase::Iterator<ReshufflingDataset>(params, seed,
                                                             seed2) {}

     protected:
      Status SaveInternal(IteratorStateWriter* writer) override {
        mutex_lock l(dataset()->mu_);

        // Save RNG state of Dataset.
        TF_RETURN_IF_ERROR(
            writer->WriteScalar(full_name("ds_num_random_samples"),
                                dataset()->num_random_samples_));

        // Save the Iterator.
        return ShuffleDatasetBase::Iterator<ReshufflingDataset>::SaveInternal(
            writer);
      }

      Status RestoreInternal(IteratorContext* ctx,
                             IteratorStateReader* reader) override {
        mutex_lock l(dataset()->mu_);

        // Restore RNG state of Dataset.
        TF_RETURN_IF_ERROR(
            reader->ReadScalar(full_name("ds_num_random_samples"),
                               &dataset()->num_random_samples_));
        dataset()->ResetRngs();

        // Restore the Iterator.
        return ShuffleDatasetBase::Iterator<
            ReshufflingDataset>::RestoreInternal(ctx, reader);
      }
    };

    Status AsGraphDefInternal(OpKernelContext* ctx, DatasetGraphDefBuilder* b,
                              Node** output) const override {
      mutex_lock l(mu_);
      Node* input_graph_node = nullptr;
      TF_RETURN_IF_ERROR(b->AddParentDataset(ctx, input_, &input_graph_node));
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
          {std::make_pair("reshuffle_each_iteration",
                          reshuffle_each_iteration)},  // Attrs
          output));
      return Status::OK();
    }

   private:
    random::SingleSampleAdapter<random::PhiloxRandom>::ResultType Random() const
        EXCLUSIVE_LOCKS_REQUIRED(mu_) {
      num_random_samples_++;
      auto out = generator_();
      return out;
    }

    void ResetRngs() const EXCLUSIVE_LOCKS_REQUIRED(mu_) {
      // Reset the generators based on the current seeds.
      parent_generator_ = random::PhiloxRandom(seed_, seed2_);
      generator_ =
          random::SingleSampleAdapter<random::PhiloxRandom>(&parent_generator_);
      generator_.Skip(num_random_samples_);
    }

    mutable int64 seed_ GUARDED_BY(mu_);
    mutable int64 seed2_ GUARDED_BY(mu_);
    mutable mutex mu_;
    mutable random::PhiloxRandom parent_generator_ GUARDED_BY(mu_);
    mutable random::SingleSampleAdapter<random::PhiloxRandom> generator_
        GUARDED_BY(mu_);
    mutable int64 num_random_samples_ GUARDED_BY(mu_) = 0;
  };

  // A dataset that uses the same fixed seed for all iterators created from it.
  // Used when `reshuffle_each_iteration` is false.
  class FixedSeedDataset : public ShuffleDatasetBase {
   public:
    FixedSeedDataset(OpKernelContext* ctx, const DatasetBase* input,
                     int64 buffer_size, int64 seed, int64 seed2, int64 count)
        : ShuffleDatasetBase(ctx, input, buffer_size, count),
          seed_(seed),
          seed2_(seed) {}

    string DebugString() const override {
      return strings::StrCat("ShuffleDatasetOp(", buffer_size_, ", ", seed_,
                             ", ", seed2_, ")::FixedSeedDataset");
    }

    std::unique_ptr<IteratorBase> MakeIteratorInternal(
        const string& prefix) const override {
      return std::unique_ptr<IteratorBase>(
          new ShuffleDatasetBase::Iterator<ShuffleDatasetBase>(
              {this, strings::StrCat(prefix, "::Shuffle")}, seed_, seed2_));
    }

   protected:
    Status AsGraphDefInternal(OpKernelContext* ctx, DatasetGraphDefBuilder* b,
                              Node** output) const override {
      Node* input_graph_node = nullptr;
      TF_RETURN_IF_ERROR(b->AddParentDataset(ctx, input_, &input_graph_node));
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
          {std::make_pair("reshuffle_each_iteration",
                          reshuffle_each_iteration)},  // Attrs
          output));
      return Status::OK();
    }

   private:
    const int64 seed_;
    const int64 seed2_;
  };

  bool reshuffle_each_iteration_;
};

class ShuffleAndRepeatDatasetOp : public ShuffleDatasetOpBase {
 public:
  explicit ShuffleAndRepeatDatasetOp(OpKernelConstruction* ctx)
      : ShuffleDatasetOpBase(ctx) {}

  void MakeDataset(OpKernelContext* ctx, DatasetBase* input,
                   DatasetBase** output) override {
    int64 buffer_size;
    OP_REQUIRES_OK(
        ctx, ParseScalarArgument<int64>(ctx, "buffer_size", &buffer_size));
    OP_REQUIRES(
        ctx, buffer_size > 0,
        errors::InvalidArgument("buffer_size must be greater than zero."));

    int64 seed;
    OP_REQUIRES_OK(ctx, ParseScalarArgument<int64>(ctx, "seed", &seed));

    int64 seed2;
    OP_REQUIRES_OK(ctx, ParseScalarArgument<int64>(ctx, "seed2", &seed2));

    int64 count;
    OP_REQUIRES_OK(ctx, ParseScalarArgument<int64>(ctx, "count", &count));

    // By TensorFlow convention, if both seeds are 0, then shuffling should be
    // seeded non-deterministically.
    if (seed == 0 && seed2 == 0) {
      seed = random::New64();
      seed2 = random::New64();
    }

    *output = new Dataset(ctx, input, buffer_size, seed, seed2, count);
  }

 private:
  class Dataset : public ShuffleDatasetBase {
   public:
    Dataset(OpKernelContext* ctx, const DatasetBase* input, int64 buffer_size,
            int64 seed, int64 seed2, int64 count)
        : ShuffleDatasetBase(ctx, input, buffer_size, count),
          seed_(seed),
          seed2_(seed2) {}

    string DebugString() const override {
      return strings::StrCat("ShuffleAndRepeatDatasetOp(", buffer_size_, ", ",
                             seed_, ", ", seed2_, ", ", count_, ")::Dataset");
    }

    std::unique_ptr<IteratorBase> MakeIteratorInternal(
        const string& prefix) const override {
      return std::unique_ptr<IteratorBase>(
          new ShuffleDatasetBase::Iterator<ShuffleDatasetBase>(
              {this, strings::StrCat(prefix, "::ShuffleAndRepeat")}, seed_,
              seed2_));
    }

   protected:
    Status AsGraphDefInternal(OpKernelContext* ctx, DatasetGraphDefBuilder* b,
                              Node** output) const override {
      Node* input_graph_node = nullptr;
      TF_RETURN_IF_ERROR(b->AddParentDataset(ctx, input_, &input_graph_node));
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
};

REGISTER_KERNEL_BUILDER(Name("ShuffleDataset").Device(DEVICE_CPU),
                        ShuffleDatasetOp);

REGISTER_KERNEL_BUILDER(Name("ShuffleAndRepeatDataset").Device(DEVICE_CPU),
                        ShuffleAndRepeatDatasetOp);

}  // namespace

}  // namespace tensorflow
