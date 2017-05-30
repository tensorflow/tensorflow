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

// See documentation in ../ops/iterator_ops.cc for a high-level
// description of the following op.

class ShuffleDatasetOp : public OpKernel {
 public:
  explicit ShuffleDatasetOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    // Create a new ShuffleDatasetOp::Dataset, insert it in the step-local
    // container, and return it as the output.
    DatasetBase* input;
    OP_REQUIRES_OK(ctx, LookupResource(ctx, HandleFromInput(ctx, 0), &input));
    core::ScopedUnref unref_input(input);

    const Tensor* buffer_size_t;
    OP_REQUIRES_OK(ctx, ctx->input("buffer_size", &buffer_size_t));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(buffer_size_t->shape()),
                errors::InvalidArgument("buffer_size must be a scalar"));
    const int64 buffer_size = buffer_size_t->flat<int64>()(0);
    OP_REQUIRES(
        ctx, buffer_size > 0,
        errors::InvalidArgument("buffer_size must be greater than zero."));

    const Tensor* seed_t;
    OP_REQUIRES_OK(ctx, ctx->input("seed", &seed_t));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(seed_t->shape()),
                errors::InvalidArgument("seed must be a scalar"));
    const int64 seed = seed_t->flat<int64>()(0);

    const Tensor* seed2_t;
    OP_REQUIRES_OK(ctx, ctx->input("seed2", &seed2_t));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(seed2_t->shape()),
                errors::InvalidArgument("seed2 must be a scalar"));
    const int64 seed2 = seed2_t->flat<int64>()(0);

    DatasetBase* dataset = new Dataset(input, buffer_size, seed, seed2);
    Tensor* output = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape({}), &output));
    ResourceHandle handle = MakeResourceHandle<DatasetBase>(
        ctx, ctx->step_container()->name(), name());
    OP_REQUIRES_OK(ctx, CreateResource(ctx, handle, dataset));
    output->flat<ResourceHandle>()(0) = handle;
  }

 private:
  class Dataset : public DatasetBase {
   public:
    Dataset(const DatasetBase* input, int64 buffer_size, int64 seed,
            int64 seed2)
        : input_(input), buffer_size_(buffer_size), seed_(seed), seed2_(seed2) {
      input_->Ref();
    }

    ~Dataset() override { input_->Unref(); }

    std::unique_ptr<IteratorBase> MakeIterator() const override {
      return std::unique_ptr<IteratorBase>(new Iterator(this));
    }

    const DataTypeVector& output_dtypes() const override {
      return input_->output_dtypes();
    }

    const std::vector<PartialTensorShape>& output_shapes() const override {
      return input_->output_shapes();
    }

    string DebugString() override {
      return strings::StrCat("ShuffleDatasetOp(", buffer_size_, ", ", seed_,
                             ", ", seed2_, ")::Dataset");
    }

   private:
    class Iterator : public DatasetIterator<Dataset> {
     public:
      explicit Iterator(const Dataset* dataset)
          : DatasetIterator<Dataset>(dataset),
            input_impl_(dataset->input_->MakeIterator()),
            generator_(&parent_generator_) {
        buffer_.reserve(dataset->buffer_size_);
        int64 seed = dataset->seed_;
        int64 seed2 = dataset->seed2_;
        if (seed == 0 && seed2 == 0) {
          // If both seeds are unspecified, use completely random seeds.
          seed = random::New64();
          seed2 = random::New64();
        }
        parent_generator_ = random::PhiloxRandom(seed, seed2);
      }

      Status GetNext(IteratorContext* ctx, std::vector<Tensor>* out_tensors,
                     bool* end_of_sequence) override {
        mutex_lock l(mu_);
        while (!end_of_input_sequence_ &&
               buffer_.size() < dataset()->buffer_size_) {
          std::vector<Tensor> input_element;
          TF_RETURN_IF_ERROR(input_impl_->GetNext(ctx, &input_element,
                                                  &end_of_input_sequence_));
          if (!end_of_input_sequence_) {
            buffer_.emplace_back(std::move(input_element));
          }
        }

        if (!buffer_.empty()) {
          *end_of_sequence = false;
          // Choose an element to produce uniformly at random, and
          // swap the last element into its place in the buffer.
          int64 index = generator_() % buffer_.size();
          *out_tensors = std::move(buffer_[index]);
          std::swap(buffer_[index], buffer_.back());
          buffer_.pop_back();
        } else {
          DCHECK(end_of_input_sequence_);
          *end_of_sequence = true;
        }
        return Status::OK();
      }

     private:
      mutex mu_;
      std::vector<std::vector<Tensor>> buffer_ GUARDED_BY(mu_);
      std::unique_ptr<IteratorBase> input_impl_ GUARDED_BY(mu_);
      bool end_of_input_sequence_ GUARDED_BY(mu_) = false;
      random::PhiloxRandom parent_generator_ GUARDED_BY(mu_);
      random::SingleSampleAdapter<random::PhiloxRandom> generator_
          GUARDED_BY(mu_);
    };

    const DatasetBase* const input_;
    const int64 buffer_size_;
    const int64 seed_;
    const int64 seed2_;
  };
};

REGISTER_KERNEL_BUILDER(Name("ShuffleDataset").Device(DEVICE_CPU),
                        ShuffleDatasetOp);

}  // namespace

}  // namespace tensorflow
