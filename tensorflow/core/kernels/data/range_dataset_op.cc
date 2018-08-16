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
#include "tensorflow/core/framework/partial_tensor_shape.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/kernels/data/dataset.h"

namespace tensorflow {

namespace {

// See documentation in ../ops/dataset_ops.cc for a high-level
// description of the following op.

class RangeDatasetOp : public DatasetOpKernel {
 public:
  explicit RangeDatasetOp(OpKernelConstruction* ctx) : DatasetOpKernel(ctx) {}

  void MakeDataset(OpKernelContext* ctx, DatasetBase** output) override {
    int64 start;
    OP_REQUIRES_OK(ctx, ParseScalarArgument<int64>(ctx, "start", &start));

    int64 stop;
    OP_REQUIRES_OK(ctx, ParseScalarArgument<int64>(ctx, "stop", &stop));

    int64 step;
    OP_REQUIRES_OK(ctx, ParseScalarArgument<int64>(ctx, "step", &step));
    OP_REQUIRES(ctx, step != 0,
                errors::InvalidArgument("step must be a non-zero integer."));

    *output = new Dataset(ctx, start, stop, step);
  }

 private:
  class Dataset : public DatasetBase {
   public:
    Dataset(OpKernelContext* ctx, int64 start, int64 stop, int64 step)
        : DatasetBase(DatasetContext(ctx)),
          start_(start),
          stop_(stop),
          step_(step) {}

    std::unique_ptr<IteratorBase> MakeIteratorInternal(
        const string& prefix) const override {
      return std::unique_ptr<IteratorBase>(
          new Iterator({this, strings::StrCat(prefix, "::Range")}));
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
      return strings::StrCat("RangeDatasetOp(", start_, ", ", stop_, ", ",
                             step_, ")::Dataset");
    }

   protected:
    Status AsGraphDefInternal(SerializationContext* ctx,
                              DatasetGraphDefBuilder* b,
                              Node** output) const override {
      Node* start = nullptr;
      Node* stop = nullptr;
      Node* step = nullptr;
      TF_RETURN_IF_ERROR(b->AddScalar(start_, &start));
      TF_RETURN_IF_ERROR(b->AddScalar(stop_, &stop));
      TF_RETURN_IF_ERROR(b->AddScalar(step_, &step));
      TF_RETURN_IF_ERROR(b->AddDataset(this, {start, stop, step}, output));
      return Status::OK();
    }

   private:
    class Iterator : public DatasetIterator<Dataset> {
     public:
      explicit Iterator(const Params& params)
          : DatasetIterator<Dataset>(params) {
        next_ = params.dataset->start_;
      }

      Status GetNextInternal(IteratorContext* ctx,
                             std::vector<Tensor>* out_tensors,
                             bool* end_of_sequence) override {
        mutex_lock l(mu_);
        if ((dataset()->step_ > 0 && next_ >= dataset()->stop_) ||
            (dataset()->step_ < 0 && next_ <= dataset()->stop_)) {
          *end_of_sequence = true;
          return Status::OK();
        }
        Tensor value_tensor(ctx->allocator({}), DT_INT64, {});
        value_tensor.scalar<int64>()() = next_;
        out_tensors->emplace_back(std::move(value_tensor));
        *end_of_sequence = false;
        next_ += dataset()->step_;

        return Status::OK();
      }

     protected:
      Status SaveInternal(IteratorStateWriter* writer) override {
        mutex_lock l(mu_);
        TF_RETURN_IF_ERROR(writer->WriteScalar(full_name("next"), next_));
        return Status::OK();
      }

      Status RestoreInternal(IteratorContext* ctx,
                             IteratorStateReader* reader) override {
        mutex_lock l(mu_);
        TF_RETURN_IF_ERROR(reader->ReadScalar(full_name("next"), &next_));
        return Status::OK();
      }

     private:
      mutex mu_;
      int64 next_ GUARDED_BY(mu_);
    };

    const int64 start_;
    const int64 stop_;
    const int64 step_;
  };
};

REGISTER_KERNEL_BUILDER(Name("RangeDataset").Device(DEVICE_CPU),
                        RangeDatasetOp);

}  // namespace

}  // namespace tensorflow
