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

namespace tensorflow {

namespace {

// See documentation in ../ops/iterator_ops.cc for a high-level
// description of the following op.

class RepeatDatasetOp : public OpKernel {
 public:
  explicit RepeatDatasetOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    // Create a new RepeatDatasetOp::Dataset, insert it in the step-local
    // container, and return it as the output.
    DatasetBase* input;
    OP_REQUIRES_OK(ctx, LookupResource(ctx, HandleFromInput(ctx, 0), &input));
    core::ScopedUnref unref_input(input);

    const Tensor* count_t;
    OP_REQUIRES_OK(ctx, ctx->input("count", &count_t));
    const int64 count = count_t->flat<int64>()(0);

    DatasetBase* dataset = new Dataset(count, input);
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
    Dataset(int64 count, const DatasetBase* input)
        : count_(count), input_(input) {
      input_->Ref();
    }

    ~Dataset() override { input_->Unref(); }

    std::unique_ptr<IteratorBase> MakeIterator() const override {
      if (count_ < 0) {
        return std::unique_ptr<IteratorBase>(new ForeverIterator(this));
      } else if (count_ == 0) {
        return std::unique_ptr<IteratorBase>(new EmptyIterator(this));
      } else {
        return std::unique_ptr<IteratorBase>(new FiniteIterator(this));
      }
    }

    const DataTypeVector& output_dtypes() const override {
      return input_->output_dtypes();
    }
    const std::vector<PartialTensorShape>& output_shapes() const override {
      return input_->output_shapes();
    }

    string DebugString() override { return "RepeatDatasetOp::Dataset"; }

   private:
    class EmptyIterator : public DatasetIterator<Dataset> {
     public:
      explicit EmptyIterator(const Dataset* dataset)
          : DatasetIterator<Dataset>(dataset) {}
      Status GetNext(IteratorContext* ctx, std::vector<Tensor>* out_tensors,
                     bool* end_of_sequence) override {
        *end_of_sequence = true;
        return Status::OK();
      }
    };

    class FiniteIterator : public DatasetIterator<Dataset> {
     public:
      explicit FiniteIterator(const Dataset* dataset)
          : DatasetIterator<Dataset>(dataset),
            i_(0),
            input_impl_(dataset->input_->MakeIterator()) {}

      Status GetNext(IteratorContext* ctx, std::vector<Tensor>* out_tensors,
                     bool* end_of_sequence) override {
        mutex_lock l(mu_);  // TODO(mrry): Make locking less conservative.
        while (i_ < dataset()->count_) {
          TF_RETURN_IF_ERROR(
              input_impl_->GetNext(ctx, out_tensors, end_of_sequence));
          if (!*end_of_sequence) {
            return Status::OK();
          }
          ++i_;
          input_impl_ = dataset()->input_->MakeIterator();
        }
        *end_of_sequence = true;
        input_impl_.reset();
        return Status::OK();
      }

     private:
      mutex mu_;
      int64 i_ GUARDED_BY(mu_);
      std::unique_ptr<IteratorBase> input_impl_ GUARDED_BY(mu_);
    };

    class ForeverIterator : public DatasetIterator<Dataset> {
     public:
      explicit ForeverIterator(const Dataset* dataset)
          : DatasetIterator<Dataset>(dataset), input_impl_(nullptr) {}

      Status GetNext(IteratorContext* ctx, std::vector<Tensor>* out_tensors,
                     bool* end_of_sequence) override {
        mutex_lock l(mu_);  // TODO(mrry): Make locking less conservative.
        do {
          if (!input_impl_) {
            input_impl_ = dataset()->input_->MakeIterator();
            TF_RETURN_IF_ERROR(
                input_impl_->GetNext(ctx, out_tensors, end_of_sequence));
            // If the first call to GetNext() fails because the end of
            // sequence has been reached, we return an OutOfRange
            // error to terminate the iteration. (Otherwise, this
            // iterator would loop infinitely and never produce a
            // value.)
            if (!*end_of_sequence) {
              return Status::OK();
            } else {
              input_impl_.reset();
              return errors::OutOfRange(
                  "Attempted to repeat an empty dataset infinitely.");
            }
          } else {
            TF_RETURN_IF_ERROR(
                input_impl_->GetNext(ctx, out_tensors, end_of_sequence));
            if (!*end_of_sequence) {
              return Status::OK();
            } else {
              input_impl_.reset();
            }
          }
        } while (true);
      }

     private:
      mutex mu_;
      std::unique_ptr<IteratorBase> input_impl_ GUARDED_BY(mu_);
    };

    const int64 count_;
    const DatasetBase* const input_;
  };
};

REGISTER_KERNEL_BUILDER(Name("RepeatDataset").Device(DEVICE_CPU),
                        RepeatDatasetOp);

}  // namespace

}  // namespace tensorflow
