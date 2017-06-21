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

// See documentation in ../ops/dataset_ops.cc for a high-level
// description of the following op.

class SkipDatasetOp : public UnaryDatasetOpKernel {
 public:
  explicit SkipDatasetOp(OpKernelConstruction* ctx)
      : UnaryDatasetOpKernel(ctx) {}

  void MakeDataset(OpKernelContext* ctx, DatasetBase* input,
                   DatasetBase** output) override {
    // Create a new RepeatDatasetOp::Dataset, and return it as the output.
    int64 count;
    OP_REQUIRES_OK(ctx, ParseScalarArgument<int64>(ctx, "count", &count));

    *output = new Dataset(count, input);
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
        return std::unique_ptr<IteratorBase>(new EmptyIterator(this));
      }  else if (count_ == 0) {
        // Pass through.
        return input_->MakeIterator();
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

    string DebugString() override { return "SkipDatasetOp::Dataset"; }

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

        // Keep calling GetNext().  TODO(vrv): Figure out a way to
        // skip records without reading, perhaps by adding an
        // interface to iterator.
        while (i_ < dataset()->count_) {
          // Fetch and throw away Tensors.
          std::vector<Tensor> dummy_out_tensors;
          TF_RETURN_IF_ERROR(input_impl_->GetNext(ctx, &dummy_out_tensors,
                                                  end_of_sequence));
          if (*end_of_sequence) {
            // We reached the end before the count was reached.
            input_impl_.reset();
            return Status::OK();
          }

          ++i_;
        }

        // Return GetNext() on the underlying iterator.
        TF_RETURN_IF_ERROR(input_impl_->GetNext(ctx, out_tensors,
                                                end_of_sequence));
        return Status::OK();
      }

     private:
      mutex mu_;
      int64 i_ GUARDED_BY(mu_);
      std::unique_ptr<IteratorBase> input_impl_ GUARDED_BY(mu_);
    };

    const int64 count_;
    const DatasetBase* const input_;
  };
};

REGISTER_KERNEL_BUILDER(Name("SkipDataset").Device(DEVICE_CPU),
                        SkipDatasetOp);

}  // namespace

}  // namespace tensorflow
