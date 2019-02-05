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
#include "tensorflow/core/kernels/data/dataset_utils.h"
#include "tensorflow/core/util/ptr_util.h"

namespace tensorflow {
namespace data {
namespace {

// See documentation in ../../ops/dataset_ops.cc for a high-level
// description of the following op.

class TakeWhileDatasetOp : public UnaryDatasetOpKernel {
 public:
  using LoopIteratorPredicate =
      std::function<Status(IteratorContext*, InstantiatedCapturedFunction*,
                           std::vector<Tensor>&, bool*)>;

  explicit TakeWhileDatasetOp(OpKernelConstruction* ctx)
      : UnaryDatasetOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("predicate", &func_));
  }

  void MakeDataset(OpKernelContext* ctx, DatasetBase* input,
                   DatasetBase** output) override {
    std::unique_ptr<CapturedFunction> captured_func;
    OP_REQUIRES_OK(ctx, CapturedFunction::Create(func_, ctx, "other_arguments",
                                                 &captured_func));

    std::vector<int> indices;
    OP_REQUIRES_OK(ctx, ComputeShortCircuitIndices(ctx, func_, &indices));
    OP_REQUIRES(
        ctx, indices.size() <= 1,
        errors::InvalidArgument("`predicate` has more than one return value."));

    LoopIteratorPredicate loop_pred;
    if (indices.empty()) {
      loop_pred = [](IteratorContext* ctx,
                     InstantiatedCapturedFunction* inst_captured_func,
                     const std::vector<Tensor>& args, bool* end_of_sequence) {
        std::vector<Tensor> result;
        TF_RETURN_IF_ERROR(
            inst_captured_func->RunWithBorrowedArgs(ctx, args, &result));

        if (result.size() != 1 || result[0].dtype() != DT_BOOL ||
            result[0].NumElements() != 1) {
          return errors::InvalidArgument(
              "`predicate` must returns a scalar bool tensor.");
        }
        *end_of_sequence = !result[0].scalar<bool>()();
        return Status::OK();
      };
    } else {
      loop_pred = [indices](IteratorContext* ctx,
                            InstantiatedCapturedFunction* inst_captured_func,
                            const std::vector<Tensor>& args,
                            bool* end_of_sequence) {
        const Tensor& predicate = args[indices[0]];
        if (predicate.dtype() != DT_BOOL || predicate.NumElements() != 1) {
          return errors::InvalidArgument(
              "`predicate` must returns a scalar bool tensor.");
        }
        *end_of_sequence = !predicate.scalar<bool>()();
        return Status::OK();
      };
    }
    *output = new Dataset(ctx, input, func_, std::move(captured_func),
                          std::move(loop_pred));
  }

 private:
  class Dataset : public DatasetBase {
   public:
    Dataset(OpKernelContext* ctx, const DatasetBase* input,
            const NameAttrList& func,
            std::unique_ptr<CapturedFunction> captured_func,
            LoopIteratorPredicate loop_pred)
        : DatasetBase(DatasetContext(ctx)),
          input_(input),
          func_(func),
          captured_func_(std::move(captured_func)),
          loop_pred_(std::move(loop_pred)) {
      input_->Ref();
    }

    ~Dataset() override { input_->Unref(); }

    std::unique_ptr<IteratorBase> MakeIteratorInternal(
        const string& prefix) const override {
      return MakeUnique<Iterator>(
          Iterator::Params{this, strings::StrCat(prefix, "::TakeWhile")},
          loop_pred_);
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
      explicit Iterator(const Params& params, LoopIteratorPredicate loop_pred)
          : DatasetIterator<Dataset>(params), loop_pred_(loop_pred) {}

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
        return loop_pred_(ctx, instantiated_captured_func_.get(), *out_tensors,
                          end_of_sequence);
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
      const LoopIteratorPredicate loop_pred_;
    };

    const DatasetBase* const input_;
    const NameAttrList func_;
    const std::unique_ptr<CapturedFunction> captured_func_;
    const LoopIteratorPredicate loop_pred_;
  };

  NameAttrList func_;
};

REGISTER_KERNEL_BUILDER(Name("ExperimentalTakeWhileDataset").Device(DEVICE_CPU),
                        TakeWhileDatasetOp);

}  // namespace
}  // namespace data
}  // namespace tensorflow
