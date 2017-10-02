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

class TensorDatasetOp : public DatasetOpKernel {
 public:
  explicit TensorDatasetOp(OpKernelConstruction* ctx) : DatasetOpKernel(ctx) {}

  void MakeDataset(OpKernelContext* ctx, DatasetBase** output) override {
    // Create a new TensorDatasetOp::Dataset, insert it in the step
    // container, and return it as the output.
    OpInputList inputs;
    OP_REQUIRES_OK(ctx, ctx->input_list("components", &inputs));
    // TODO(mrry): Validate that the shapes of the "components" tensors match
    // the "shapes" attr.;
    std::vector<Tensor> components;
    components.reserve(inputs.size());
    for (const Tensor& t : inputs) {
      components.push_back(t);
    }
    *output = new Dataset(std::move(components));
  }

 private:
  class Dataset : public DatasetBase {
   public:
    explicit Dataset(std::vector<Tensor> tensors)
        : tensors_(std::move(tensors)) {
      for (const Tensor& t : tensors_) {
        dtypes_.push_back(t.dtype());
        shapes_.emplace_back(t.shape().dim_sizes());
      }
    }

    std::unique_ptr<IteratorBase> MakeIterator(
        const string& prefix) const override {
      return std::unique_ptr<IteratorBase>(
          new Iterator({this, strings::StrCat(prefix, "::FromTensor")}));
    }

    const DataTypeVector& output_dtypes() const override { return dtypes_; }
    const std::vector<PartialTensorShape>& output_shapes() const override {
      return shapes_;
    }

    string DebugString() override { return "TensorDatasetOp::Dataset"; }

   private:
    class Iterator : public DatasetIterator<Dataset> {
     public:
      explicit Iterator(const Params& params)
          : DatasetIterator<Dataset>(params), produced_(false) {}

      Status GetNextInternal(IteratorContext* ctx,
                             std::vector<Tensor>* out_tensors,
                             bool* end_of_sequence) override {
        mutex_lock l(mu_);
        if (!produced_) {
          *out_tensors = dataset()->tensors_;
          produced_ = true;
          *end_of_sequence = false;
          return Status::OK();
        } else {
          *end_of_sequence = true;
          return Status::OK();
        }
      }

     private:
      mutex mu_;
      bool produced_ GUARDED_BY(mu_);
    };

    const std::vector<Tensor> tensors_;
    DataTypeVector dtypes_;
    std::vector<PartialTensorShape> shapes_;
  };
};

REGISTER_KERNEL_BUILDER(Name("TensorDataset").Device(DEVICE_CPU),
                        TensorDatasetOp);

}  // namespace

}  // namespace tensorflow
