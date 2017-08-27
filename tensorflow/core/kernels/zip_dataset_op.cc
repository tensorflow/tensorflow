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

class ZipDatasetOp : public DatasetOpKernel {
 public:
  explicit ZipDatasetOp(OpKernelConstruction* ctx) : DatasetOpKernel(ctx) {}

  void MakeDataset(OpKernelContext* ctx, DatasetBase** output) override {
    std::vector<DatasetBase*> inputs;
    Status s;
    for (size_t i = 0; i < ctx->num_inputs(); ++i) {
      // Create a new ZipDatasetOp::Dataset, insert it in the step-local
      // container, and return it as the output.
      DatasetBase* input;
      s.Update(LookupResource(ctx, HandleFromInput(ctx, i), &input));
      if (!s.ok()) {
        break;
      }
      inputs.push_back(input);
    }

    if (s.ok()) {
      *output = new Dataset(inputs);
    }

    // TODO(mrry): Implement a container that acts as a
    // `std::vector<core::ScopedUnref>`, to avoid having to unref the
    // inputs manually, and re-enable the use of `OP_REQUIRES_OK()`.
    for (DatasetBase* input : inputs) {
      input->Unref();
    }
    ctx->SetStatus(s);
  }

 private:
  class Dataset : public DatasetBase {
   public:
    explicit Dataset(const std::vector<DatasetBase*>& inputs)
        : inputs_(inputs) {
      for (const auto& input : inputs_) {
        input->Ref();
        for (DataType dt : input->output_dtypes()) {
          output_dtypes_.push_back(dt);
        }
        output_shapes_.insert(output_shapes_.end(),
                              input->output_shapes().begin(),
                              input->output_shapes().end());
      }
    }

    ~Dataset() override {
      for (const auto& input : inputs_) {
        input->Unref();
      }
    }

    std::unique_ptr<IteratorBase> MakeIterator(
        const string& prefix) const override {
      return std::unique_ptr<IteratorBase>(
          new Iterator({this, strings::StrCat(prefix, "::Zip")}));
    }

    const DataTypeVector& output_dtypes() const override {
      return output_dtypes_;
    }

    const std::vector<PartialTensorShape>& output_shapes() const override {
      return output_shapes_;
    }

    string DebugString() override { return "ZipDatasetOp::Dataset"; }

   private:
    class Iterator : public DatasetIterator<Dataset> {
     public:
      explicit Iterator(const Params& params)
          : DatasetIterator<Dataset>(params) {
        input_impls_.reserve(params.dataset->inputs_.size());
        size_t idx = 0;
        for (const auto& input : params.dataset->inputs_) {
          input_impls_.emplace_back(input->MakeIterator(
              strings::StrCat(params.prefix, "[", idx++, "]")));
        }
      }

      Status GetNextInternal(IteratorContext* ctx,
                             std::vector<Tensor>* out_tensors,
                             bool* end_of_sequence) override {
        mutex_lock l(mu_);
        out_tensors->clear();
        out_tensors->reserve(dataset()->output_dtypes().size());
        for (const auto& input_impl : input_impls_) {
          std::vector<Tensor> input_tensors;
          TF_RETURN_IF_ERROR(
              input_impl->GetNext(ctx, &input_tensors, end_of_sequence));
          if (*end_of_sequence) {
            return Status::OK();
          }
          out_tensors->insert(out_tensors->end(), input_tensors.begin(),
                              input_tensors.end());
        }
        *end_of_sequence = false;
        return Status::OK();
      }

     private:
      mutex mu_;
      std::vector<std::unique_ptr<IteratorBase>> input_impls_ GUARDED_BY(mu_);
    };

    const std::vector<DatasetBase*> inputs_;
    DataTypeVector output_dtypes_;
    std::vector<PartialTensorShape> output_shapes_;
  };
};

REGISTER_KERNEL_BUILDER(Name("ZipDataset").Device(DEVICE_CPU), ZipDatasetOp);

}  // namespace

}  // namespace tensorflow
