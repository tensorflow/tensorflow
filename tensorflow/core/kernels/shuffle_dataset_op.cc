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
#include "tensorflow/core/kernels/dataset.h"

#include "tensorflow/core/framework/partial_tensor_shape.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/random/philox_random.h"
#include "tensorflow/core/lib/random/random.h"
#include "tensorflow/core/lib/random/random_distributions.h"

namespace tensorflow {

namespace {

const int64 kLogIntervalMicros = 10 * 1000000;  // 10 seconds.

// See documentation in ../ops/dataset_ops.cc for a high-level
// description of the following op.

class ShuffleDatasetOp : public UnaryDatasetOpKernel {
 public:
  explicit ShuffleDatasetOp(OpKernelConstruction* ctx)
      : UnaryDatasetOpKernel(ctx) {
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

    if (reshuffle_each_iteration_) {
      *output = new ReshufflingDataset(ctx, input, buffer_size, seed, seed2);
    } else {
      *output = new FixedSeedDataset(ctx, input, buffer_size, seed, seed2);
    }
  }

 private:
  // Abstract base dataset that implements a shuffling iterator.
  class ShuffleDatasetBase : public GraphDatasetBase {
   public:
    ShuffleDatasetBase(OpKernelContext* ctx, const DatasetBase* input,
                       int64 buffer_size)
        : GraphDatasetBase(ctx), input_(input), buffer_size_(buffer_size) {
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
    class Iterator : public DatasetIterator<ShuffleDatasetBase> {
     public:
      explicit Iterator(const Params& params, int64 seed, int64 seed2)
          : DatasetIterator<ShuffleDatasetBase>(params),
            input_impl_(params.dataset->input_->MakeIterator(params.prefix)),
            seed_(seed),
            seed2_(seed2),
            parent_generator_(seed, seed2),
            generator_(&parent_generator_) {
        buffer_.reserve(params.dataset->buffer_size_);
      }

      Status GetNextInternal(IteratorContext* ctx,
                             std::vector<Tensor>* out_tensors,
                             bool* end_of_sequence) override {
        mutex_lock l(mu_);
        int64 start_micros = ctx->env()->NowMicros();
        int64 num_log_entries = 0;
        while (input_impl_ && buffer_.size() < dataset()->buffer_size_) {
          if (ctx->env()->NowMicros() >
              ((num_log_entries + 1) * kLogIntervalMicros) + start_micros) {
            num_log_entries++;
            LOG(INFO) << "Filling up shuffle buffer (this may take a while): "
                      << buffer_.size() << " of " << dataset()->buffer_size_;
          }
          std::vector<Tensor> input_element;
          bool end_of_input_sequence;
          TF_RETURN_IF_ERROR(input_impl_->GetNext(ctx, &input_element,
                                                  &end_of_input_sequence));
          if (!end_of_input_sequence) {
            buffer_.emplace_back(std::move(input_element));
          } else {
            input_impl_.reset();
          }
        }
        if (num_log_entries > 0) {
          LOG(INFO) << "Shuffle buffer filled.";
        }

        if (!buffer_.empty()) {
          *end_of_sequence = false;
          // Choose an element to produce uniformly at random, and
          // swap the last element into its place in the buffer.
          int64 index = Random() % buffer_.size();
          *out_tensors = std::move(buffer_[index]);
          std::swap(buffer_[index], buffer_.back());
          buffer_.pop_back();
        } else {
          DCHECK(input_impl_ == nullptr);
          *end_of_sequence = true;
        }
        return Status::OK();
      }

     protected:
      Status SaveInternal(IteratorStateWriter* writer) override {
        mutex_lock l(mu_);

        // Save the tensors in the buffer.
        TF_RETURN_IF_ERROR(
            writer->WriteScalar(full_name("buffer_size"), buffer_.size()));
        for (size_t i = 0; i < buffer_.size(); i++) {
          TF_RETURN_IF_ERROR(writer->WriteScalar(
              full_name(strings::StrCat("buffer_", i, "_size")),
              buffer_[i].size()));
          for (size_t j = 0; j < buffer_[i].size(); j++) {
            TF_RETURN_IF_ERROR(writer->WriteTensor(
                full_name(strings::StrCat("buffer_", i, "_", j)),
                buffer_[i][j]));
          }
        }

        // Save state needed to restore the random number generators.
        TF_RETURN_IF_ERROR(writer->WriteScalar(full_name("num_random_samples"),
                                               num_random_samples_));

        // Save input iterator if it hasn't been exhausted else write
        // "end_of_input_sequence".
        if (!input_impl_) {
          TF_RETURN_IF_ERROR(
              writer->WriteScalar(full_name("end_of_input_sequence"), ""));
        } else {
          TF_RETURN_IF_ERROR(SaveParent(writer, input_impl_));
        }
        return Status::OK();
      }

      Status RestoreInternal(OpKernelContext* ctx,
                             IteratorStateReader* reader) override {
        mutex_lock l(mu_);
        buffer_.clear();

        // Restore the buffer.
        size_t buffer_size;
        {
          int64 temp;
          TF_RETURN_IF_ERROR(
              reader->ReadScalar(full_name("buffer_size"), &temp));
          buffer_size = static_cast<size_t>(temp);
        }
        buffer_.reserve(buffer_size);
        for (size_t i = 0; i < buffer_size; i++) {
          int64 list_size;
          TF_RETURN_IF_ERROR(reader->ReadScalar(
              full_name(strings::StrCat("buffer_", i, "_size")), &list_size));
          buffer_.emplace_back(std::vector<Tensor>(list_size));
          for (int j = 0; j < list_size; j++) {
            TF_RETURN_IF_ERROR(reader->ReadTensor(
                full_name(strings::StrCat("buffer_", i, "_", j)),
                &buffer_[i][j]));
          }
        }

        // Restore the random number generators.
        TF_RETURN_IF_ERROR(reader->ReadScalar(full_name("num_random_samples"),
                                              &num_random_samples_));
        ResetRngs();

        // Restore the input iterator if it wasn't already exhausted.
        if (!reader->Contains(full_name("end_of_input_sequence"))) {
          input_impl_ = dataset()->input_->MakeIterator(prefix());
          TF_RETURN_IF_ERROR(RestoreParent(ctx, reader, input_impl_));
        } else {
          input_impl_.reset();
        }
        return Status::OK();
      }

     private:
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
      std::vector<std::vector<Tensor>> buffer_ GUARDED_BY(mu_);
      std::unique_ptr<IteratorBase> input_impl_ GUARDED_BY(mu_);
      const int64 seed_ GUARDED_BY(mu_);
      const int64 seed2_ GUARDED_BY(mu_);
      random::PhiloxRandom parent_generator_ GUARDED_BY(mu_);
      random::SingleSampleAdapter<random::PhiloxRandom> generator_
          GUARDED_BY(mu_);
      int64 num_random_samples_ GUARDED_BY(mu_) = 0;
    };

    const DatasetBase* const input_;
    const int64 buffer_size_;
  };

  // A dataset that uses a pseduorandom sequence of seeds for the iterators
  // created from it. Used when `reshuffle_each_iteration` is true.
  class ReshufflingDataset : public ShuffleDatasetBase {
   public:
    ReshufflingDataset(OpKernelContext* ctx, const DatasetBase* input,
                       int64 buffer_size, int64 seed, int64 seed2)
        : ShuffleDatasetBase(ctx, input, buffer_size),
          seed_(seed),
          seed2_(seed2),
          parent_generator_(seed, seed2),
          generator_(&parent_generator_) {}

    string DebugString() override {
      return strings::StrCat("ShuffleDatasetOp(", buffer_size_, ", ", seed_,
                             ", ", seed2_, ")::ReshufflingDataset");
    }

    std::unique_ptr<IteratorBase> MakeIterator(
        const string& prefix) const override {
      int64 iterator_seed;
      int64 iterator_seed2;
      {
        mutex_lock l(mu_);
        iterator_seed = generator_();
        iterator_seed2 = generator_();
      }
      return std::unique_ptr<IteratorBase>(new ShuffleDatasetBase::Iterator(
          {this, strings::StrCat(prefix, "::Shuffle")}, iterator_seed,
          iterator_seed2));
    }

   private:
    const int64 seed_;
    const int64 seed2_;
    mutable mutex mu_;
    mutable random::PhiloxRandom parent_generator_ GUARDED_BY(mu_);
    mutable random::SingleSampleAdapter<random::PhiloxRandom> generator_
        GUARDED_BY(mu_);
  };

  // A dataset that uses the same fixed seed for all iterators created from it.
  // Used when `reshuffle_each_iteration` is false.
  class FixedSeedDataset : public ShuffleDatasetBase {
   public:
    FixedSeedDataset(OpKernelContext* ctx, const DatasetBase* input,
                     int64 buffer_size, int64 seed, int64 seed2)
        : ShuffleDatasetBase(ctx, input, buffer_size),
          seed_(seed),
          seed2_(seed) {}

    string DebugString() override {
      return strings::StrCat("ShuffleDatasetOp(", buffer_size_, ", ", seed_,
                             ", ", seed2_, ")::FixedSeedDataset");
    }

    std::unique_ptr<IteratorBase> MakeIterator(
        const string& prefix) const override {
      return std::unique_ptr<IteratorBase>(new ShuffleDatasetBase::Iterator(
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

REGISTER_KERNEL_BUILDER(Name("ShuffleDataset").Device(DEVICE_CPU),
                        ShuffleDatasetOp);

}  // namespace

}  // namespace tensorflow
