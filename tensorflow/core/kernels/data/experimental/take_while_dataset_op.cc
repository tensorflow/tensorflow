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
#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/framework/partial_tensor_shape.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/kernels/data/captured_function.h"

namespace tensorflow {
namespace data {
namespace {

// See documentation in ../../ops/dataset_ops.cc for a high-level
// description of the following op.

class TakeWhileDatasetOp : public UnaryDatasetOpKernel {
 public:
  explicit TakeWhileDatasetOp(OpKernelConstruction* ctx)
      : UnaryDatasetOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("predicate", &func_));
  }

  void MakeDataset(OpKernelContext* ctx, DatasetBase* input,
                   DatasetBase** output) override {
    std::unique_ptr<CapturedFunction> captured_func;
    OP_REQUIRES_OK(ctx, CapturedFunction::Create(func_, ctx, "other_arguments",
                                                 &captured_func));
    
    // TODO (squadrick): check short-circuit
    *output = new Dataset(ctx, input, func_, std::move(captured_func));
  }

 private:
  class Dataset : public DatasetBase {
   public:
    Dataset(OpKernelContext* ctx, const DatasetBase* input,
            const NameAttrList& func, 
            std::unique_ptr<CapturedFunction> captured_func)
        : DatasetBase(DatasetContext(ctx)),
          input_(input),
          func_(func),
          captured_func_(std::move(captured_func)) {
      input_->Ref();
    }

    ~Dataset() override { input_->Unref(); }

    std::unique_ptr<IteratorBase> MakeIteratorInternal(
        const string& prefix) const override {
      return std::unique_ptr<IteratorBase>(
          new Iterator({this, strings::StrCat(prefix, "::TakeWhile")}));
    }

    const DataTypeVector& output_dtypes() const override {
      return input_->output_dtypes();
    }

    const std::vector<PartialTensorShape>& output_shapes() const override {
      return input_->output_shapes();
    }

    string DebugString() const override { return "TakeWhileDatasetOp::Dataset"; }

    int64 Cardinality() const override { return kUnknownCardinality; }

   protected:
    Status AsGraphDefInternal(SerializationContext* ctx,
                              DatasetGraphDefBuilder* b,
                              Node** output) const override {
      TF_RETURN_IF_ERROR(b->AddFunction(ctx, func_.name()));
      Node* input_node;
      TF_RETURN_IF_ERROR(b->AddInputDataset(ctx, input_, &input_node));

      std::vector<Node*> other_arguments;
      other_arguments.reserve(captured_func_->captured_inputs().size());
      DataTypeVector other_arguments_types;
      other_arguments_types.reserve(captured_func_->captured_inputs().size());
      for (const Tensor& t : captured_func_->captured_inputs()) {
        Node* node;
        DatasetBase* input;
        Status s = GetDatasetFromVariantTensor(t, &input);
        if (s.ok()) {
          TF_RETURN_IF_ERROR(b->AddInputDataset(ctx, input, &node));
        } else {
          TF_RETURN_IF_ERROR(b->AddTensor(t, &node));
        }
        other_arguments.emplace_back(node);
        other_arguments_types.emplace_back(t.dtype());
      }
      AttrValue f_attr;
      b->BuildAttrValue(func_, &f_attr);

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
            dataset()->input_->MakeIterator(ctx, prefix(), &input_impl_));
        return dataset()->captured_func_->Instantiate(
            ctx, &instantiated_captured_func_);
      }

      Status GetNextInternal(IteratorContext* ctx,
                             std::vector<Tensor>* out_tensors,
                             bool* end_of_sequence) override {
        {
          tf_shared_lock l(mu_);
          if(!input_impl_) {
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

        std::vector<Tensor> bool_output;

        Status s = instantiated_captured_func_->RunWithBorrowedArgs(
            ctx, *out_tensors, &bool_output);

        if (s.ok()) {
          if(bool_output.size() != 1 || bool_output[0].dtype() != DT_BOOL ||
             bool_output[0].NumElements() != 1) {
            return errors::InvalidArgument(
                "`predicate` must returns a scalar bool tensor.");
          }
          if (!bool_output[0].scalar<bool>()()) {
            *end_of_sequence = true;
            return Status::OK();
          }
        }
        return s; // propagate error to caller
      }

     protected:
      std::shared_ptr<model::Node> CreateNode(
          IteratorContext* ctx, model::Node::Args args) const override {
        return model::MakeKnownRatioNode(std::move(args),
                                         /*ratio=*/1);
      }

      Status SaveInternal(IteratorStateWriter* writer) override {
        mutex_lock l(mu_);
        if (input_impl_)
          TF_RETURN_IF_ERROR(SaveInput(writer, input_impl_));
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
      std::unique_ptr<IteratorBase> input_impl_ GUARDED_BY(mu_);
      std::unique_ptr<InstantiatedCapturedFunction> instantiated_captured_func_;
    };

    const DatasetBase* const input_;
    const NameAttrList func_;
    const std::unique_ptr<CapturedFunction> captured_func_;
  };

  NameAttrList func_;
};

REGISTER_KERNEL_BUILDER(Name("ExperimentalTakeWhileDataset").Device(DEVICE_CPU),
                        TakeWhileDatasetOp);

}  // namespace
}  // namespace data
}  // namespace tensorflow
