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
#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/framework/partial_tensor_shape.h"
#include "tensorflow/core/framework/stats_aggregator.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/kernels/data/captured_function.h"
#include "tensorflow/core/kernels/data/dataset.h"
#include "tensorflow/core/lib/gtl/cleanup.h"
#include "tensorflow/core/lib/random/random.h"
#include "tensorflow/core/lib/strings/str_util.h"

namespace tensorflow {
namespace data {
namespace {

// See documentation in ../ops/dataset_ops.cc for a high-level
// description of the following op.

class FilterDatasetOp : public UnaryDatasetOpKernel {
 public:
  explicit FilterDatasetOp(OpKernelConstruction* ctx)
      : UnaryDatasetOpKernel(ctx),
        graph_def_version_(ctx->graph_def_version()) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("predicate", &func_));
  }

  void MakeDataset(OpKernelContext* ctx, DatasetBase* input,
                   DatasetBase** output) override {
    FunctionLibraryRuntime::Handle pred_handle;
    OP_REQUIRES_OK(ctx,
                   ctx->function_library()->Instantiate(
                       func_.name(), AttrSlice(&func_.attr()), &pred_handle));
    auto cleanup = gtl::MakeCleanup([ctx, pred_handle]() {
      OP_REQUIRES_OK(ctx, ctx->function_library()->ReleaseHandle(pred_handle));
    });

    const FunctionBody* pred_body =
        ctx->function_library()->GetFunctionBody(pred_handle);
    OP_REQUIRES(ctx, pred_body->ret_nodes.size() == 1,
                errors::InvalidArgument(
                    "predicate function must have a single return value."));
    Node* ret_node = pred_body->ret_nodes[0];
    Node* ret_input_node;
    OP_REQUIRES_OK(ctx, ret_node->input_node(0, &ret_input_node));

    std::unique_ptr<CapturedFunction> captured_func;
    OP_REQUIRES_OK(ctx, CapturedFunction::Create(func_, ctx, "other_arguments",
                                                 &captured_func));

    if (ret_input_node->def().op() == "_Arg") {
      int32 index = -1;
      OP_REQUIRES_OK(ctx, GetNodeAttr(ret_input_node->def(), "index", &index));
      *output = new FilterTensorDataset(ctx, input, func_,
                                        std::move(captured_func), index);
    } else {
      *output = new FilterFunctionDataset(ctx, input, func_,
                                          std::move(captured_func));
    }
  }

 private:
  const int graph_def_version_;

  class FilterDatasetBase : public DatasetBase {
   public:
    FilterDatasetBase(OpKernelContext* ctx, const DatasetBase* input,
                      const NameAttrList& func,
                      std::unique_ptr<CapturedFunction> captured_func)
        : DatasetBase(DatasetContext(ctx)),
          input_(input),
          func_(func),
          captured_func_(std::move(captured_func)) {
      input_->Ref();
    }

    ~FilterDatasetBase() override { input_->Unref(); }

    std::unique_ptr<IteratorBase> MakeIteratorInternal(
        const string& prefix) const override {
      return std::unique_ptr<IteratorBase>(
          new Iterator({this, strings::StrCat(prefix, "::Filter")}));
    }

    const DataTypeVector& output_dtypes() const override {
      return input_->output_dtypes();
    }
    const std::vector<PartialTensorShape>& output_shapes() const override {
      return input_->output_shapes();
    }

    string DebugString() const override { return "FilterDatasetOp::Dataset"; }

   protected:
    Status AsGraphDefInternal(SerializationContext* ctx,
                              DatasetGraphDefBuilder* b,
                              Node** output) const override {
      TF_RETURN_IF_ERROR(b->AddFunction(ctx, func_.name()));
      Node* input_graph_node;
      TF_RETURN_IF_ERROR(b->AddInputDataset(ctx, input_, &input_graph_node));

      DataTypeVector other_arguments_types;
      other_arguments_types.reserve(captured_func_->captured_inputs().size());
      std::vector<Node*> other_arguments;
      other_arguments.reserve(captured_func_->captured_inputs().size());
      for (const Tensor& t : captured_func_->captured_inputs()) {
        Node* node;
        TF_RETURN_IF_ERROR(b->AddTensor(t, &node));
        other_arguments.emplace_back(node);
        other_arguments_types.emplace_back(t.dtype());
      }
      AttrValue f;
      b->BuildAttrValue(func_, &f);
      AttrValue other_arguments_types_attr;
      b->BuildAttrValue(other_arguments_types, &other_arguments_types_attr);

      TF_RETURN_IF_ERROR(b->AddDataset(
          this, {{0, input_graph_node}}, {{1, other_arguments}},
          {{"predicate", f}, {"Targuments", other_arguments_types_attr}},
          output));
      return Status::OK();
    }

    virtual Status EvaluatePredicate(IteratorContext* ctx,
                                     const std::vector<Tensor>& element,
                                     bool* out_matched) const = 0;

   private:
    class Iterator : public DatasetIterator<FilterDatasetBase> {
     public:
      explicit Iterator(const Params& params)
          : DatasetIterator<FilterDatasetBase>(params),
            filtered_elements_(0),
            dropped_elements_(0) {
        std::vector<string> components =
            str_util::Split(params.prefix, "::", str_util::SkipEmpty());
        prefix_end_ = components.back();
      }

      Status Initialize(IteratorContext* ctx) override {
        TF_RETURN_IF_ERROR(
            dataset()->input_->MakeIterator(ctx, prefix(), &input_impl_));
        return dataset()->captured_func_->Instantiate(ctx);
      }

      Status GetNextInternal(IteratorContext* ctx,
                             std::vector<Tensor>* out_tensors,
                             bool* end_of_sequence) override {
        // NOTE(mrry): This method is thread-safe as long as
        // `input_impl_` and `f` are thread-safe. However, if multiple
        // threads enter this method, outputs may be observed in a
        // non-deterministic order.
        auto stats_aggregator = ctx->stats_aggregator();
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

          TF_RETURN_IF_ERROR(
              dataset()->EvaluatePredicate(ctx, *out_tensors, &matched));
          if (!matched) {
            // Clear the output tensor list since it didn't match.
            out_tensors->clear();
            if (stats_aggregator) {
              mutex_lock l(mu_);
              dropped_elements_++;
              stats_aggregator->AddScalar(
                  strings::StrCat(prefix_end_, "::dropped_elements"),
                  static_cast<float>((dropped_elements_)));
              // TODO(shivaniagrawal): multiple pipelines would collect
              // aggregated number of dropped elements for all the pipelines,
              // exploit tagged_context here.
              stats_aggregator->IncrementCounter(
                  prefix_end_, "dropped_elements", static_cast<float>(1));
            }
          }
        } while (!matched);
        // TODO(shivaniagrawal): add ratio of dropped_elements and
        // filtered_elements as a histogram.
        if (stats_aggregator) {
          mutex_lock l(mu_);
          filtered_elements_++;
          stats_aggregator->AddScalar(
              strings::StrCat(prefix_end_, "::filtered_elements"),
              static_cast<float>((filtered_elements_)));
          // TODO(shivaniagrawal): multiple pipelines would collect aggregated
          // number of filtered elements for all the pipelines, exploit
          // tagged_context here.
          stats_aggregator->IncrementCounter(prefix_end_, "filtered_elements",
                                             static_cast<float>(1));
        }
        *end_of_sequence = false;
        return Status::OK();
      }

     protected:
      Status SaveInternal(IteratorStateWriter* writer) override {
        mutex_lock l(mu_);
        if (input_impl_)
          TF_RETURN_IF_ERROR(SaveInput(writer, input_impl_));
        else
          TF_RETURN_IF_ERROR(
              writer->WriteScalar(full_name("input_impls_empty"), ""));
        TF_RETURN_IF_ERROR(writer->WriteScalar(full_name("filtered_elements"),
                                               filtered_elements_));
        TF_RETURN_IF_ERROR(writer->WriteScalar(full_name("dropped_elements"),
                                               dropped_elements_));
        return Status::OK();
      }

      Status RestoreInternal(IteratorContext* ctx,
                             IteratorStateReader* reader) override {
        mutex_lock l(mu_);
        if (reader->Contains(full_name("input_impls_empty")))
          input_impl_.reset();
        else
          TF_RETURN_IF_ERROR(RestoreInput(ctx, reader, input_impl_));
        TF_RETURN_IF_ERROR(reader->ReadScalar(full_name("filtered_elements"),
                                              &filtered_elements_));
        TF_RETURN_IF_ERROR(reader->ReadScalar(full_name("dropped_elements"),
                                              &dropped_elements_));
        return Status::OK();
      }

     private:
      mutex mu_;
      std::unique_ptr<IteratorBase> input_impl_ GUARDED_BY(mu_);
      int64 filtered_elements_ GUARDED_BY(mu_);
      int64 dropped_elements_ GUARDED_BY(mu_);
      string prefix_end_;
    };

    const DatasetBase* const input_;
    const NameAttrList func_;

   protected:
    const std::unique_ptr<CapturedFunction> captured_func_;
  };

  class FilterFunctionDataset : public FilterDatasetBase {
   public:
    using FilterDatasetBase::FilterDatasetBase;

   protected:
    Status EvaluatePredicate(IteratorContext* ctx,
                             const std::vector<Tensor>& element,
                             bool* out_matched) const override {
      // TODO(mrry): Avoid blocking a threadpool thread. We will need to
      // stack-rip the iterators and use async kernels.
      std::vector<Tensor> result;
      TF_RETURN_IF_ERROR(
          captured_func_->RunWithBorrowedArgs(ctx, element, &result));

      if (result.size() != 1 || result[0].dtype() != DT_BOOL ||
          result[0].NumElements() != 1) {
        return errors::InvalidArgument(
            "Filter predicate `f` must return a scalar bool.");
      }
      *out_matched = result[0].scalar<bool>()();
      return Status::OK();
    }
  };

  class FilterTensorDataset : public FilterDatasetBase {
   public:
    FilterTensorDataset(OpKernelContext* ctx, const DatasetBase* input,
                        const NameAttrList& func,
                        std::unique_ptr<CapturedFunction> captured_func,
                        int32 index)
        : FilterDatasetBase(ctx, input, func, std::move(captured_func)),
          index_(index) {}

   protected:
    Status EvaluatePredicate(IteratorContext* ctx,
                             const std::vector<Tensor>& element,
                             bool* out_matched) const override {
      const Tensor& predicate = element[index_];
      if (predicate.dtype() != DT_BOOL || predicate.NumElements() != 1) {
        return errors::InvalidArgument(
            "Filter predicate `f` must return a scalar bool.");
      }
      *out_matched = predicate.scalar<bool>()();
      return Status::OK();
    }

   private:
    const int32 index_;
  };

 private:
  NameAttrList func_;
};

REGISTER_KERNEL_BUILDER(Name("FilterDataset").Device(DEVICE_CPU),
                        FilterDatasetOp);

}  // namespace
}  // namespace data
}  // namespace tensorflow
