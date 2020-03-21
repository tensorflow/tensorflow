/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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
#include <iterator>
#include <vector>

#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/common_runtime/input_colocation_exemption_registry.h"
#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/framework/partial_tensor_shape.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/kernels/data/captured_function.h"
#include "tensorflow/core/kernels/data/dataset_utils.h"
#include "tensorflow/core/util/ptr_util.h"

namespace tensorflow {
namespace data {
namespace experimental {
namespace {

class TakeWhileDatasetOp : public UnaryDatasetOpKernel {
 public:
  explicit TakeWhileDatasetOp(OpKernelConstruction* ctx)
      : UnaryDatasetOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, FunctionMetadata::Create(
                            ctx, "predicate", /*params=*/{}, &func_metadata_));
    OP_REQUIRES(ctx, func_metadata_->short_circuit_info().indices.size() <= 1,
                errors::InvalidArgument(
                    "predicate function has more than one return value."));
  }

  void MakeDataset(OpKernelContext* ctx, DatasetBase* input,
                   DatasetBase** output) override {
    std::unique_ptr<CapturedFunction> captured_func;
    OP_REQUIRES_OK(
        ctx, CapturedFunction::Create(ctx, func_metadata_, "other_arguments",
                                      &captured_func));
    *output = new Dataset(ctx, input, std::move(captured_func));
  }

 private:
  class Dataset : public DatasetBase {
   public:
    Dataset(OpKernelContext* ctx, const DatasetBase* input,
            std::unique_ptr<CapturedFunction> captured_func)
        : DatasetBase(DatasetContext(ctx)),
          input_(input),
          captured_func_(std::move(captured_func)) {
      input_->Ref();
    }

    ~Dataset() override { input_->Unref(); }

    std::unique_ptr<IteratorBase> MakeIteratorInternal(
        const string& prefix) const override {
      return MakeUnique<Iterator>(
          Iterator::Params{this, strings::StrCat(prefix, "::TakeWhile")});
    }

    const DataTypeVector& output_dtypes() const override {
      return input_->output_dtypes();
    }

    const std::vector<PartialTensorShape>& output_shapes() const override {
      return input_->output_shapes();
    }

    string DebugString() const override {
      return "TakeWhileDatasetOp::Dataset";
    }

    int64 Cardinality() const override { return kUnknownCardinality; }

    Status CheckExternalState() const override {
      TF_RETURN_IF_ERROR(captured_func_->CheckExternalState());
      return input_->CheckExternalState();
    }

   protected:
    Status AsGraphDefInternal(SerializationContext* ctx,
                              DatasetGraphDefBuilder* b,
                              Node** output) const override {
      Node* input_node;
      TF_RETURN_IF_ERROR(b->AddInputDataset(ctx, input_, &input_node));

      std::vector<Node*> other_arguments;
      DataTypeVector other_arguments_types;
      TF_RETURN_IF_ERROR(captured_func_->AddToGraph(ctx, b, &other_arguments,
                                                    &other_arguments_types));
      AttrValue f_attr;
      b->BuildAttrValue(captured_func_->func(), &f_attr);

      AttrValue other_arguments_types_attr;
      b->BuildAttrValue(other_arguments_types, &other_arguments_types_attr);

      TF_RETURN_IF_ERROR(b->AddDataset(
          this, {std::make_pair(0, input_node)},
          {std::make_pair(1, other_arguments)},
          {std::make_pair("predicate", f_attr),
           std::make_pair("Targuments", other_arguments_types_attr)},
          output));
      return Status::OK();
    }

   private:
    class Iterator : public DatasetIterator<Dataset> {
     public:
      explicit Iterator(const Params& params)
          : DatasetIterator<Dataset>(params) {}

      Status Initialize(IteratorContext* ctx) override {
        TF_RETURN_IF_ERROR(
            dataset()->input_->MakeIterator(ctx, this, prefix(), &input_impl_));
        return dataset()->captured_func_->Instantiate(
            ctx, &instantiated_captured_func_);
      }

      Status GetNextInternal(IteratorContext* ctx,
                             std::vector<Tensor>* out_tensors,
                             bool* end_of_sequence) override {
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
        std::vector<Tensor> result;
        TF_RETURN_IF_ERROR(instantiated_captured_func_->RunWithBorrowedArgs(
            ctx, *out_tensors, &result));

        if (result.size() != 1 || result[0].dtype() != DT_BOOL ||
            result[0].NumElements() != 1) {
          return errors::InvalidArgument(
              "`predicate` must returns a scalar bool tensor.");
        }
        *end_of_sequence = !result[0].scalar<bool>()();
        if (*end_of_sequence) {
          out_tensors->clear();
        }
        return Status::OK();
      }

     protected:
      std::shared_ptr<model::Node> CreateNode(
          IteratorContext* ctx, model::Node::Args args) const override {
        return model::MakeKnownRatioNode(std::move(args),
                                         /*ratio=*/1);
      }

      Status SaveInternal(SerializationContext* ctx,
                          IteratorStateWriter* writer) override {
        TF_RETURN_IF_ERROR(ctx->HandleCheckExternalStateStatus(
            dataset()->captured_func_->CheckExternalState()));
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
      std::unique_ptr<InstantiatedCapturedFunction> instantiated_captured_func_;
    };

    const DatasetBase* const input_;
    const std::unique_ptr<CapturedFunction> captured_func_;
  };

  std::shared_ptr<FunctionMetadata> func_metadata_ = nullptr;
};

REGISTER_KERNEL_BUILDER(Name("TakeWhileDataset").Device(DEVICE_CPU),
                        TakeWhileDatasetOp);
REGISTER_KERNEL_BUILDER(Name("ExperimentalTakeWhileDataset").Device(DEVICE_CPU),
                        TakeWhileDatasetOp);

REGISTER_INPUT_COLOCATION_EXEMPTION("TakeWhileDataset");
REGISTER_INPUT_COLOCATION_EXEMPTION("ExperimentalTakeWhileDataset");

}  // namespace
}  // namespace experimental
}  // namespace data
}  // namespace tensorflow
