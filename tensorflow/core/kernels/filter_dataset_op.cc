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

#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/framework/partial_tensor_shape.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/random/random.h"

#include "tensorflow/core/kernels/captured_function.h"

namespace tensorflow {

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
    OpInputList inputs;
    OP_REQUIRES_OK(ctx, ctx->input_list("other_arguments", &inputs));
    std::vector<Tensor> other_arguments;
    other_arguments.reserve(inputs.size());
    for (const Tensor& t : inputs) {
      other_arguments.push_back(t);
    }

    std::unique_ptr<CapturedFunction> captured_func;
    OP_REQUIRES_OK(ctx, CapturedFunction::Create(ctx, func_, graph_def_version_,
                                                 std::move(other_arguments),
                                                 &captured_func));

    *output = new Dataset(input, std::move(captured_func));
  }

 private:
  const int graph_def_version_;

  class Dataset : public DatasetBase {
   public:
    Dataset(const DatasetBase* input,
            std::unique_ptr<CapturedFunction> captured_func)
        : input_(input), captured_func_(std::move(captured_func)) {
      input_->Ref();
    }

    ~Dataset() override { input_->Unref(); }

    std::unique_ptr<IteratorBase> MakeIterator(
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

    string DebugString() override { return "FilterDatasetOp::Dataset"; }

   private:
    class Iterator : public DatasetIterator<Dataset> {
     public:
      explicit Iterator(const Params& params)
          : DatasetIterator<Dataset>(params),
            input_impl_(params.dataset->input_->MakeIterator(params.prefix)) {}

      Status GetNextInternal(IteratorContext* ctx,
                             std::vector<Tensor>* out_tensors,
                             bool* end_of_sequence) override {
        // NOTE(mrry): This method is thread-safe as long as
        // `input_impl_` and `f` are thread-safe. However, if multiple
        // threads enter this method, outputs may be observed in a
        // non-deterministic order.
        bool matched;
        do {
          TF_RETURN_IF_ERROR(
              input_impl_->GetNext(ctx, out_tensors, end_of_sequence));
          if (*end_of_sequence) {
            return Status::OK();
          }

          FunctionLibraryRuntime::Options opts;
          opts.step_id = CapturedFunction::generate_step_id();
          ScopedStepContainer step_container(
              opts.step_id, [this, ctx](const string& name) {
                dataset()
                    ->captured_func_->resource_manager()
                    ->Cleanup(name)
                    .IgnoreError();
              });
          opts.step_container = &step_container;
          opts.runner = ctx->runner();
          // TODO(mrry): Avoid blocking a threadpool thread. We will need to
          // stack-rip the iterators and use async kernels.
          Notification n;
          Status ret;
          std::vector<Tensor> result;
          ret = dataset()->captured_func_->Run(opts, *out_tensors, &result,
                                               prefix());

          if (!ret.ok()) {
            return ret;
          } else if (result.size() != 1 || result[0].dtype() != DT_BOOL ||
                     result[0].NumElements() != 1) {
            return errors::InvalidArgument(
                "Filter predicate `f` must return a scalar bool.");
          }
          matched = result[0].scalar<bool>()();
          if (!matched) {
            // Clear the output tensor list since it didn't match.
            out_tensors->clear();
          }
        } while (!matched);
        *end_of_sequence = false;
        return Status::OK();
      }

     private:
      const std::unique_ptr<IteratorBase> input_impl_;
    };

    const DatasetBase* const input_;
    const std::unique_ptr<CapturedFunction> captured_func_;
  };

 private:
  const NameAttrList* func_;
};

REGISTER_KERNEL_BUILDER(Name("FilterDataset").Device(DEVICE_CPU),
                        FilterDatasetOp);

}  // namespace

}  // namespace tensorflow
