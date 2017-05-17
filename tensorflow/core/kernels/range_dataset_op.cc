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

class RangeDatasetOp : public OpKernel {
 public:
  explicit RangeDatasetOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    const Tensor* start_t;
    OP_REQUIRES_OK(ctx, ctx->input("start", &start_t));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(start_t->shape()),
                errors::InvalidArgument("start must be a scalar"));
    const int64 start = start_t->flat<int64>()(0);

    const Tensor* stop_t;
    OP_REQUIRES_OK(ctx, ctx->input("stop", &stop_t));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(stop_t->shape()),
                errors::InvalidArgument("stop must be a scalar"));
    const int64 stop = stop_t->flat<int64>()(0);

    const Tensor* step_t;
    OP_REQUIRES_OK(ctx, ctx->input("step", &step_t));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(step_t->shape()),
                errors::InvalidArgument("step must be a scalar"));
    const int64 step = step_t->flat<int64>()(0);
    OP_REQUIRES(ctx, step != 0,
                errors::InvalidArgument("step must be a non-zero integer."));

    DatasetBase* dataset = new Dataset(start, stop, step);
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
    Dataset(int64 start, int64 stop, int64 step)
        : start_(start), stop_(stop), step_(step) {}

    std::unique_ptr<IteratorBase> MakeIterator() const override {
      return std::unique_ptr<IteratorBase>(new Iterator(this));
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

    string DebugString() override {
      return strings::StrCat("RangeDatasetOp(", start_, ", ", stop_, ", ",
                             step_, ")::Dataset");
    }

   private:
    class Iterator : public DatasetIterator<Dataset> {
     public:
      explicit Iterator(const Dataset* dataset)
          : DatasetIterator<Dataset>(dataset) {
        next_ = dataset->start_;
      }

      Status GetNext(IteratorContext* ctx, std::vector<Tensor>* out_tensors,
                     bool* end_of_sequence) override {
        mutex_lock l(mu_);
        if ((dataset()->step_ > 0 && next_ >= dataset()->stop_) ||
            (dataset()->step_ < 0 && next_ <= dataset()->stop_)) {
          *end_of_sequence = true;
          return Status::OK();
        }
        Tensor value_tensor(cpu_allocator(), DT_INT64, {});
        value_tensor.scalar<int64>()() = next_;
        out_tensors->emplace_back(std::move(value_tensor));
        *end_of_sequence = false;
        next_ += dataset()->step_;

        return Status::OK();
      }

     private:
      mutex mu_;
      int64 next_;
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
