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
#include <deque>

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

class ParallelMapDatasetOp : public UnaryDatasetOpKernel {
 public:
  explicit ParallelMapDatasetOp(OpKernelConstruction* ctx)
      : UnaryDatasetOpKernel(ctx),
        graph_def_version_(ctx->graph_def_version()) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("f", &func_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("output_types", &output_types_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("output_shapes", &output_shapes_));
  }

 protected:
  void MakeDataset(OpKernelContext* ctx, DatasetBase* input,
                   DatasetBase** output) override {
    OpInputList inputs;
    OP_REQUIRES_OK(ctx, ctx->input_list("other_arguments", &inputs));
    std::vector<Tensor> other_arguments;
    other_arguments.reserve(inputs.size());
    for (const Tensor& t : inputs) {
      other_arguments.push_back(t);
    }

    int32 num_parallel_calls;
    OP_REQUIRES_OK(ctx, ParseScalarArgument(ctx, "num_parallel_calls",
                                            &num_parallel_calls));
    OP_REQUIRES(ctx, num_parallel_calls > 0,
                errors::InvalidArgument(
                    "num_parallel_calls must be greater than zero."));

    std::unique_ptr<CapturedFunction> captured_func;
    OP_REQUIRES_OK(ctx, CapturedFunction::Create(ctx, func_, graph_def_version_,
                                                 std::move(other_arguments),
                                                 &captured_func));

    *output = new Dataset(input, num_parallel_calls, output_types_,
                          output_shapes_, std::move(captured_func));
  }

 private:
  class Dataset : public DatasetBase {
   public:
    Dataset(const DatasetBase* input, int32 num_parallel_calls,
            const DataTypeVector& output_types,
            const std::vector<PartialTensorShape>& output_shapes,
            std::unique_ptr<CapturedFunction> captured_func)
        : input_(input),
          num_parallel_calls_(num_parallel_calls),
          output_types_(output_types),
          output_shapes_(output_shapes),
          captured_func_(std::move(captured_func)) {
      input_->Ref();
    }

    ~Dataset() override { input_->Unref(); }

    std::unique_ptr<IteratorBase> MakeIterator(
        const string& prefix) const override {
      return std::unique_ptr<IteratorBase>(
          new Iterator({this, strings::StrCat(prefix, "::ParallelMap")}));
    }

    const DataTypeVector& output_dtypes() const override {
      return output_types_;
    }

    const std::vector<PartialTensorShape>& output_shapes() const override {
      return output_shapes_;
    }

    string DebugString() override { return "ParallelMapDatasetOp::Dataset"; }

   private:
    class Iterator : public DatasetIterator<Dataset> {
     public:
      explicit Iterator(const Params& params)
          : DatasetIterator<Dataset>(params),
            input_impl_(params.dataset->input_->MakeIterator(params.prefix)),
            invocation_results_(params.dataset->num_parallel_calls_) {}

      ~Iterator() override {
        // TODO(mrry): Replace this cancellation logic with a
        // CancellationManager. The syntax would be more heavyweight,
        // but it would be possible to thread a cancellation manager
        // through the IteratorContext to upstream,
        // potentially-blocking iterators, when we add these.
        {
          mutex_lock l(mu_);
          for (size_t i = 0; i < dataset()->num_parallel_calls_; ++i) {
            if (invocation_results_[i].notification) {
              invocation_results_[i].notification->WaitForNotification();
            }
          }
        }
      }

      Status GetNextInternal(IteratorContext* ctx,
                             std::vector<Tensor>* out_tensors,
                             bool* end_of_sequence) override {
        mutex_lock l(mu_);

        // Ensure that there are `dataset()->num_parallel_calls_`
        // invocations of `func_` outstanding at once.
        while (!end_of_input_ && (num_inputs_consumed_ - num_outputs_consumed_ <
                                  dataset()->num_parallel_calls_)) {
          InvokeFunctionLocked(ctx);
        }

        if (end_of_input_ && num_inputs_consumed_ == num_outputs_consumed_) {
          *end_of_sequence = true;
          return Status::OK();
        }

        // Read the next result out of `invocation_results_`, which
        // acts as a circular buffer.
        const size_t result_index =
            num_outputs_consumed_ % dataset()->num_parallel_calls_;
        InvocationResult* result = &invocation_results_[result_index];
        *end_of_sequence = false;
        if (result->notification) {
          result->notification->WaitForNotification();
          if (result->status.ok()) {
            std::swap(*out_tensors, result->return_values);
          }
        }
        ++num_outputs_consumed_;
        return result->status;
      }

     private:
      struct InvocationResult {
        Status status;
        std::unique_ptr<Notification> notification;
        std::vector<Tensor> return_values;
      };

      void InvokeFunctionLocked(IteratorContext* ctx)
          EXCLUSIVE_LOCKS_REQUIRED(mu_) {
        DCHECK(!end_of_input_);
        DCHECK(num_inputs_consumed_ - num_outputs_consumed_ <
               dataset()->num_parallel_calls_);

        // The result of invoking the function will be written into the next
        // slot in `invocation_results_`, which acts as a circular buffer.
        const size_t result_index =
            num_inputs_consumed_ % dataset()->num_parallel_calls_;
        InvocationResult* result = &invocation_results_[result_index];
        *result = InvocationResult();

        // Get the next input element.
        std::vector<Tensor> input_element;
        result->status =
            input_impl_->GetNext(ctx, &input_element, &end_of_input_);
        if (end_of_input_) {
          result->status = errors::OutOfRange("");
        } else {
          ++num_inputs_consumed_;
        }

        if (result->status.ok()) {
          // Call `func_(input_element)`, store the result in
          // `result->return_values`, and notify `result->notification`
          // to unblock a consumer.
          result->notification.reset(new Notification);

          FunctionLibraryRuntime::Options opts;
          opts.step_id = CapturedFunction::generate_step_id();
          ScopedStepContainer* step_container = new ScopedStepContainer(
              opts.step_id, [this, ctx](const string& name) {
                dataset()
                    ->captured_func_->resource_manager()
                    ->Cleanup(name)
                    .IgnoreError();
              });
          opts.step_container = step_container;
          opts.runner = ctx->runner();
          dataset()->captured_func_->RunAsync(
              opts, input_element, &result->return_values,
              [result, step_container, result_index](Status ret_status) {
                delete step_container;
                result->status.Update(ret_status);
                result->notification->Notify();
              });
        }
      }

      mutex mu_;
      const std::unique_ptr<IteratorBase> input_impl_ GUARDED_BY(mu_);
      std::vector<InvocationResult> invocation_results_ GUARDED_BY(mu_);
      bool end_of_input_ GUARDED_BY(mu_) = false;
      int64 num_inputs_consumed_ GUARDED_BY(mu_) = 0;
      int64 num_outputs_consumed_ GUARDED_BY(mu_) = 0;
    };

    const DatasetBase* const input_;
    const NameAttrList func_;
    const int32 num_parallel_calls_;
    const DataTypeVector output_types_;
    const std::vector<PartialTensorShape> output_shapes_;
    const std::unique_ptr<CapturedFunction> captured_func_;
  };

  const int graph_def_version_;
  DataTypeVector output_types_;
  std::vector<PartialTensorShape> output_shapes_;
  NameAttrList func_;
};

REGISTER_KERNEL_BUILDER(Name("ParallelMapDataset").Device(DEVICE_CPU),
                        ParallelMapDatasetOp);

}  // namespace

}  // namespace tensorflow
