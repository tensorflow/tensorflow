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
#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/framework/partial_tensor_shape.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/logging.h"

namespace tensorflow {
namespace data {
namespace experimental {
namespace {

class IgnoreErrorsDatasetOp : public UnaryDatasetOpKernel {
 public:
  explicit IgnoreErrorsDatasetOp(OpKernelConstruction* ctx)
      : UnaryDatasetOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("log_warning", &log_warning_));
  }

  void MakeDataset(OpKernelContext* ctx, DatasetBase* input,
                   DatasetBase** output) override {
    *output = new Dataset(ctx, input, log_warning_);
  }

 private:
  class Dataset : public DatasetBase {
   public:
    explicit Dataset(OpKernelContext* ctx, const DatasetBase* input,
                     const bool log_warning)
        : DatasetBase(DatasetContext(ctx)),
          input_(input),
          log_warning_(log_warning) {
      input_->Ref();
    }

    ~Dataset() override { input_->Unref(); }

    std::unique_ptr<IteratorBase> MakeIteratorInternal(
        const string& prefix) const override {
      return absl::make_unique<Iterator>(
          Iterator::Params{this, strings::StrCat(prefix, "::IgnoreErrors")});
    }

    const DataTypeVector& output_dtypes() const override {
      return input_->output_dtypes();
    }
    const std::vector<PartialTensorShape>& output_shapes() const override {
      return input_->output_shapes();
    }

    string DebugString() const override {
      return "IgnoreErrorsDatasetOp::Dataset";
    }

    int64 Cardinality() const override { return kUnknownCardinality; }

    Status InputDatasets(
        std::vector<const DatasetBase*>* inputs) const override {
      inputs->push_back(input_);
      return Status::OK();
    }

    Status CheckExternalState() const override {
      return input_->CheckExternalState();
    }

   protected:
    Status AsGraphDefInternal(SerializationContext* ctx,
                              DatasetGraphDefBuilder* b,
                              Node** output) const override {
      Node* input_graph_node = nullptr;
      TF_RETURN_IF_ERROR(b->AddInputDataset(ctx, input_, &input_graph_node));
      AttrValue log_warning_attr;
      b->BuildAttrValue<bool>(log_warning_, &log_warning_attr);
      TF_RETURN_IF_ERROR(
          b->AddDataset(this, {std::make_pair(0, input_graph_node)}, {},
                        {{"log_warning", log_warning_attr}}, output));
      return Status::OK();
    }

   private:
    class Iterator : public DatasetIterator<Dataset> {
     public:
      explicit Iterator(const Params& params)
          : DatasetIterator<Dataset>(params) {}

      Status Initialize(IteratorContext* ctx) override {
        return dataset()->input_->MakeIterator(ctx, this, prefix(),
                                               &input_impl_);
      }

      Status GetNextInternal(IteratorContext* ctx,
                             std::vector<Tensor>* out_tensors,
                             bool* end_of_sequence) override {
        Status s;
        {
          tf_shared_lock l(mu_);
          if (!input_impl_) {
            *end_of_sequence = true;
            return Status::OK();
          }
          s = input_impl_->GetNext(ctx, out_tensors, end_of_sequence);
          while (!s.ok() && !errors::IsCancelled(s)) {
            if (dataset()->log_warning_) {
              LOG(WARNING) << "Error raised with error message "
                           << s.error_message();
            }
            out_tensors->clear();
            s = input_impl_->GetNext(ctx, out_tensors, end_of_sequence);
          }
        }
        if (*end_of_sequence) {
          mutex_lock l(mu_);
          input_impl_.reset();
        }
        return s;
      }

     protected:
      std::shared_ptr<model::Node> CreateNode(
          IteratorContext* ctx, model::Node::Args args) const override {
        return model::MakeKnownRatioNode(std::move(args),
                                         /*ratio=*/1);
      }

      Status SaveInternal(SerializationContext* ctx,
                          IteratorStateWriter* writer) override {
        mutex_lock l(mu_);
        if (input_impl_)
          TF_RETURN_IF_ERROR(SaveInput(ctx, writer, input_impl_));
        else
          TF_RETURN_IF_ERROR(
              writer->WriteScalar(full_name("input_impls_empty"), ""));
        return Status::OK();
      }

      Status RestoreInternal(IteratorContext* ctx,
                             IteratorStateReader* reader) override {
        mutex_lock l(mu_);
        if (reader->Contains(full_name("input_impls_empty")))
          input_impl_.reset();
        else
          TF_RETURN_IF_ERROR(RestoreInput(ctx, reader, input_impl_));
        return Status::OK();
      }

     private:
      mutex mu_;
      std::unique_ptr<IteratorBase> input_impl_ TF_GUARDED_BY(mu_);
    };

    const DatasetBase* const input_;
    const bool log_warning_;
  };
  bool log_warning_;
};

REGISTER_KERNEL_BUILDER(Name("IgnoreErrorsDataset").Device(DEVICE_CPU),
                        IgnoreErrorsDatasetOp);
REGISTER_KERNEL_BUILDER(
    Name("ExperimentalIgnoreErrorsDataset").Device(DEVICE_CPU),
    IgnoreErrorsDatasetOp);

}  // namespace
}  // namespace experimental
}  // namespace data
}  // namespace tensorflow
