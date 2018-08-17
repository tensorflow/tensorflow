/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/framework/partial_tensor_shape.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/kernels/data/dataset.h"
#include "tensorflow/core/lib/gtl/cleanup.h"
#include "tensorflow/core/lib/random/random.h"

namespace tensorflow {

namespace {

// See documentation in ../ops/dataset_ops.cc for a high-level
// description of the following op.
// TODO(prazek): Filter already has a logic of filtering by the given tensor,
// but it must return both components.  We could introduce kernel like
// DropComponentDatasetOp and use FilterDataset for filtering.
class FilterByLastComponentDatasetOp : public UnaryDatasetOpKernel {
 public:
  explicit FilterByLastComponentDatasetOp(OpKernelConstruction* ctx)
      : UnaryDatasetOpKernel(ctx),
        graph_def_version_(ctx->graph_def_version()) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("output_types", &output_types_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("output_shapes", &output_shapes_));
  }

  void MakeDataset(OpKernelContext* ctx, DatasetBase* input,
                   DatasetBase** output) override {
    *output = new Dataset(ctx, input, output_types_, output_shapes_);
  }

 private:
  const int graph_def_version_;
  DataTypeVector output_types_;
  std::vector<PartialTensorShape> output_shapes_;

  class Dataset : public DatasetBase {
   public:
    Dataset(OpKernelContext* ctx, const DatasetBase* input,
            const DataTypeVector& output_types,
            std::vector<PartialTensorShape> output_shapes)
        : DatasetBase(DatasetContext(ctx)),
          input_(input),
          output_types_(output_types),
          output_shapes_(std::move(output_shapes)) {
      input_->Ref();
    }

    ~Dataset() override { input_->Unref(); }

    std::unique_ptr<IteratorBase> MakeIteratorInternal(
        const string& prefix) const override {
      return std::unique_ptr<Iterator>(new Iterator(
          {this, strings::StrCat(prefix, "::FilterByLastComponent")}));
    }

    const DataTypeVector& output_dtypes() const override {
      return output_types_;
    }
    const std::vector<PartialTensorShape>& output_shapes() const override {
      return output_shapes_;
    }

    string DebugString() const override {
      return "FilterByLastComponentDatasetOp::Dataset";
    }

   protected:
    Status AsGraphDefInternal(SerializationContext* ctx,
                              DatasetGraphDefBuilder* b,
                              Node** output) const override {
      Node* input_graph_node = nullptr;
      TF_RETURN_IF_ERROR(b->AddInputDataset(ctx, input_, &input_graph_node));

      TF_RETURN_IF_ERROR(b->AddDataset(
          this, {std::make_pair(0, input_graph_node)},  // Single tensor inputs.
          {}, {}, output));
      return Status::OK();
    }

   private:
    const DatasetBase* const input_;
    const DataTypeVector output_types_;
    const std::vector<PartialTensorShape> output_shapes_;

   private:
    class Iterator : public DatasetIterator<Dataset> {
     public:
      explicit Iterator(const Params& params)
          : DatasetIterator<Dataset>(params) {}

      Status Initialize(IteratorContext* ctx) override {
        return dataset()->input_->MakeIterator(ctx, prefix(), &input_impl_);
      }

      Status GetNextInternal(IteratorContext* ctx,
                             std::vector<Tensor>* out_tensors,
                             bool* end_of_sequence) override {
        // NOTE(mrry): This method is thread-safe as long as `input_impl_` is
        // thread-safe. However, if multiple threads enter this method, outputs
        // may be observed in a non-deterministic order.
        bool matched;
        do {
          {
            tf_shared_lock l(mu_);
            if (!input_impl_) {
              *end_of_sequence = true;
              return Status::OK();
            }
            TF_RETURN_IF_ERROR(
                input_impl_->GetNext(ctx, out_tensors, end_of_sequence));
          }
          if (*end_of_sequence) {
            mutex_lock l(mu_);
            input_impl_.reset();
            return Status::OK();
          }

          matched = out_tensors->back().scalar<bool>()();
          out_tensors->pop_back();
          if (!matched) {
            // Clear the output tensor list since it didn't match.
            out_tensors->clear();
          }
        } while (!matched);
        *end_of_sequence = false;
        return Status::OK();
      }

     protected:
      Status SaveInternal(IteratorStateWriter* writer) override {
        mutex_lock l(mu_);
        TF_RETURN_IF_ERROR(SaveInput(writer, input_impl_));
        return Status::OK();
      }

      Status RestoreInternal(IteratorContext* ctx,
                             IteratorStateReader* reader) override {
        mutex_lock l(mu_);
        TF_RETURN_IF_ERROR(RestoreInput(ctx, reader, input_impl_));
        return Status::OK();
      }

     private:
      mutex mu_;
      std::unique_ptr<IteratorBase> input_impl_ GUARDED_BY(mu_);
    };
  };
};

REGISTER_KERNEL_BUILDER(Name("FilterByLastComponentDataset").Device(DEVICE_CPU),
                        FilterByLastComponentDatasetOp);

}  // namespace

}  // namespace tensorflow
