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

class ConcatenateDatasetOp : public BinaryDatasetOpKernel {
 public:
  explicit ConcatenateDatasetOp(OpKernelConstruction* ctx)
      : BinaryDatasetOpKernel(ctx) {}
  void MakeDataset(OpKernelContext* ctx, DatasetBase* input,
                   DatasetBase* to_concatenate, DatasetBase** output) override {
    OP_REQUIRES(ctx, input->output_dtypes() == to_concatenate->output_dtypes(),
                errors::InvalidArgument(
                    "input dataset and dataset to concatenate"
                    " have different output_types %s and %s",
                    (DataTypeVectorString(input->output_dtypes()),
                     DataTypeVectorString(to_concatenate->output_dtypes()))));
    *output = new Dataset(ctx, input, to_concatenate);
  }

 private:
  class Dataset : public DatasetBase {
   public:
    explicit Dataset(OpKernelContext* ctx, const DatasetBase* input,
                     const DatasetBase* to_concatenate)
        : DatasetBase(DatasetContext(ctx)),
          input_(input),
          to_concatenate_(to_concatenate) {
      input_->Ref();
      to_concatenate_->Ref();

      auto os_input = input->output_shapes();
      auto os_concatenate = to_concatenate->output_shapes();
      for (int i = 0; i < os_input.size(); i++) {
        output_shapes_.push_back(
            MostSpecificCompatibleShape(os_input[i], os_concatenate[i]));
      }
    }
    ~Dataset() override {
      input_->Unref();
      to_concatenate_->Unref();
    }

    std::unique_ptr<IteratorBase> MakeIteratorInternal(
        const string& prefix) const override {
      return std::unique_ptr<IteratorBase>(
          new Iterator({this, strings::StrCat(prefix, "::Concatenate")}));
    }

    const DataTypeVector& output_dtypes() const override {
      return input_->output_dtypes();
    }

    const std::vector<PartialTensorShape>& output_shapes() const override {
      return output_shapes_;
    }

    string DebugString() const override {
      return "ConcatenateDatasetOp::Dataset";
    }

   protected:
    Status AsGraphDefInternal(SerializationContext* ctx,
                              DatasetGraphDefBuilder* b,
                              Node** output) const override {
      Node* input_graph = nullptr;
      TF_RETURN_IF_ERROR(b->AddInputDataset(ctx, input_, &input_graph));
      Node* to_concatenate_graph = nullptr;
      TF_RETURN_IF_ERROR(
          b->AddInputDataset(ctx, to_concatenate_, &to_concatenate_graph));
      TF_RETURN_IF_ERROR(
          b->AddDataset(this, {input_graph, to_concatenate_graph}, output));
      return Status::OK();
    }

   private:
    class Iterator : public DatasetIterator<Dataset> {
     public:
      explicit Iterator(const Params& params)
          : DatasetIterator<Dataset>(params), i_(0) {}

      Status Initialize(IteratorContext* ctx) override {
        return dataset()->input_->MakeIterator(
            ctx, strings::StrCat(prefix(), "[0]"), &input_impl_);
      }

      Status GetNextInternal(IteratorContext* ctx,
                             std::vector<Tensor>* out_tensors,
                             bool* end_of_sequence) override {
        mutex_lock l(mu_);
        if (!input_impl_) {
          *end_of_sequence = true;
          return Status::OK();
        }
        while (i_ < 2) {
          TF_RETURN_IF_ERROR(
              input_impl_->GetNext(ctx, out_tensors, end_of_sequence));
          if (!*end_of_sequence) {
            return Status::OK();
          }
          if (++i_ < 2) {
            TF_RETURN_IF_ERROR(dataset()->to_concatenate_->MakeIterator(
                ctx, strings::StrCat(prefix(), "[1]"), &input_impl_));
          }
        }
        *end_of_sequence = true;
        input_impl_.reset();
        return Status::OK();
      }

     protected:
      Status SaveInternal(IteratorStateWriter* writer) override {
        mutex_lock l(mu_);
        TF_RETURN_IF_ERROR(writer->WriteScalar(full_name("i"), i_));
        if (input_impl_) {
          TF_RETURN_IF_ERROR(SaveInput(writer, input_impl_));
        } else {
          TF_RETURN_IF_ERROR(
              writer->WriteScalar(full_name("input_impl_uninitialized"), ""));
        }
        return Status::OK();
      }

      Status RestoreInternal(IteratorContext* ctx,
                             IteratorStateReader* reader) override {
        mutex_lock l(mu_);
        TF_RETURN_IF_ERROR(reader->ReadScalar(full_name("i"), &i_));
        if (reader->Contains(full_name("input_impl_uninitialized"))) {
          input_impl_.reset();
          return Status::OK();
        }
        if (!TF_PREDICT_TRUE(i_ >= 0 && i_ <= 2))
          return errors::InvalidArgument("i_ must be in range [0, 2].");
        if (i_ == 1) {
          TF_RETURN_IF_ERROR(dataset()->to_concatenate_->MakeIterator(
              ctx, strings::StrCat(prefix(), "[1]"), &input_impl_));
        } else if (i_ == 2) {
          input_impl_.reset();
        }
        if (input_impl_) {
          TF_RETURN_IF_ERROR(RestoreInput(ctx, reader, input_impl_));
        }
        return Status::OK();
      }

     private:
      mutex mu_;
      int64 i_ GUARDED_BY(mu_);
      std::unique_ptr<IteratorBase> input_impl_ GUARDED_BY(mu_);
    };

    static PartialTensorShape MostSpecificCompatibleShape(
        const PartialTensorShape& ts1, const PartialTensorShape& ts2) {
      PartialTensorShape output_tensorshape;
      if (ts1.dims() != ts2.dims() || ts1.unknown_rank() || ts2.unknown_rank())
        return output_tensorshape;
      auto dims1 = ts1.dim_sizes();
      auto dims2 = ts2.dim_sizes();
      for (int d = 0; d < ts1.dims(); d++) {
        if (dims1[d] == dims2[d])
          output_tensorshape.Concatenate(dims1[d]);
        else
          output_tensorshape.Concatenate(-1);
      }
      return output_tensorshape;
    }

    const DatasetBase* input_;
    const DatasetBase* to_concatenate_;
    std::vector<PartialTensorShape> output_shapes_;
  };
};

REGISTER_KERNEL_BUILDER(Name("ConcatenateDataset").Device(DEVICE_CPU),
                        ConcatenateDatasetOp);

}  // namespace

}  // namespace tensorflow
