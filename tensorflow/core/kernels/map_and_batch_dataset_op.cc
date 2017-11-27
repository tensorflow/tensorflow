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
#define EIGEN_USE_THREADS

#include "tensorflow/core/lib/core/blocking_counter.h"
#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/framework/partial_tensor_shape.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/kernels/captured_function.h"
#include "tensorflow/core/kernels/dataset.h"
#include "tensorflow/core/kernels/inplace_ops_functor.h"
#include "tensorflow/core/lib/random/random.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/tracing.h"

namespace tensorflow {

namespace {

// See documentation in ../ops/dataset_ops.cc for a high-level
// description of the following op.

class MapAndBatchDatasetOp : public UnaryDatasetOpKernel {
 public:
  explicit MapAndBatchDatasetOp(OpKernelConstruction* ctx)
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

    int64 batch_size;
    OP_REQUIRES_OK(ctx, ParseScalarArgument(ctx, "batch_size", &batch_size));
    OP_REQUIRES(
        ctx, batch_size > 0,
        errors::InvalidArgument("batch_size must be greater than zero."));

    int64 num_parallel_batches;
    OP_REQUIRES_OK(ctx, ParseScalarArgument(ctx, "num_parallel_batches",
                                            &num_parallel_batches));
    OP_REQUIRES(ctx, num_parallel_batches > 0,
                errors::InvalidArgument(
                    "num_parallel_batches must be greater than zero."));

    std::unique_ptr<CapturedFunction> captured_func;
    OP_REQUIRES_OK(ctx, CapturedFunction::Create(ctx, func_, graph_def_version_,
                                                 std::move(other_arguments),
                                                 &captured_func));

    *output = new Dataset(input, batch_size, num_parallel_batches,
                          output_types_, output_shapes_,
                          std::move(captured_func), &ctx->eigen_cpu_device());
  }

 private:
  class Dataset : public DatasetBase {
   public:
    Dataset(const DatasetBase* input, int64 batch_size,
            int64 num_parallel_batches, const DataTypeVector& output_types,
            const std::vector<PartialTensorShape>& output_shapes,
            std::unique_ptr<CapturedFunction> captured_func,
            const Eigen::ThreadPoolDevice* device)
        : input_(input),
          batch_size_(batch_size),
          num_parallel_batches_(num_parallel_batches),
          output_types_(output_types),
          output_shapes_(output_shapes),
          captured_func_(std::move(captured_func)),
          device_(device) {
      input_->Ref();
    }

    ~Dataset() override { input_->Unref(); }

    std::unique_ptr<IteratorBase> MakeIterator(
        const string& prefix) const override {
      return std::unique_ptr<IteratorBase>(
          new Iterator({this, strings::StrCat(prefix, "::MapAndBatch")}));
    }

    const DataTypeVector& output_dtypes() const override {
      return output_types_;
    }

    const std::vector<PartialTensorShape>& output_shapes() const override {
      return output_shapes_;
    }

    string DebugString() override { return "MapAndBatchDatasetOp::Dataset"; }

   private:
    class Iterator : public DatasetIterator<Dataset> {
     public:
      explicit Iterator(const Params& params)
          : DatasetIterator<Dataset>(params),
            input_impl_(params.dataset->input_->MakeIterator(params.prefix)),
            invocation_results_(params.dataset->batch_size_ *
                                params.dataset->num_parallel_batches_),
            batch_results_(params.dataset->num_parallel_batches_) {}

      ~Iterator() override {
        // TODO(mrry): Replace this cancellation logic with a
        // CancellationManager. The syntax would be more heavyweight,
        // but it would be possible to thread a cancellation manager
        // through the IteratorContext to upstream,
        // potentially-blocking iterators, when we add these.
        mutex_lock l(mu_);
        if (current_batch_index_ != -1) {
          for (size_t batch_index = 0;
               batch_index < dataset()->num_parallel_batches_; ++batch_index) {
            WaitForBatch(batch_index).IgnoreError();
            // Deallocate tensors allocated for the output.
            batch_results_[batch_index].output.clear();
          }
        }
      }

      // TODO(jsimsa): Implement and profile the following alternative design:
      //
      // 0. Set the number of in-flight batches and invocations independently
      // (though obviously the max number of in-flight invocations must be <
      // batch_size * num_parallel_batches). Maintain a current producing batch
      // index and offset.
      // 1. Issue invocations in order of batch and offset, as you do currently.
      // 2. When an invocation finishes, increment the current producing batch
      // and offset. If that invocation would start a new batch and give more
      // than num_parallel_batches in-flight, block; else start the new
      // invocation into that location.
      // 3. When a GetNext() call arrives, block until there's a full batch.
      // Before returning the batch, if the number of pending invocations is
      // less than the max, issue that number of invocations.
      Status GetNextInternal(IteratorContext* ctx,
                             std::vector<Tensor>* out_tensors,
                             bool* end_of_sequence) override {
        mutex_lock l(mu_);

        // One-time initialization.
        if (current_batch_index_ == -1) {
          current_batch_index_ = 0;
          for (size_t i = 0; i < dataset()->num_parallel_batches_; ++i) {
            StartInvocationBatch(ctx, i);
          }
        }

        if (end_of_input_) {
          *end_of_sequence = true;
          return Status::OK();
        }

        Status status = WaitForBatch(current_batch_index_);
        if (!status.ok()) {
          // Deallocate tensors allocated for the output.
          batch_results_[current_batch_index_].output.clear();
        } else {
          *out_tensors = std::move(batch_results_[current_batch_index_].output);
          *end_of_sequence = false;
        }
        StartInvocationBatch(ctx, current_batch_index_);
        current_batch_index_ =
            (current_batch_index_ + 1) % dataset()->num_parallel_batches_;
        return status;
      }

     private:
      struct BatchResult {
        mutex mu;
        bool output_allocated GUARDED_BY(mu);
        std::vector<Tensor> output;
        std::unique_ptr<BlockingCounter> counter;
      };

      struct InvocationResult {
        Status status;
        std::vector<Tensor> return_values;
      };

      int64 ComputeInvocationIndex(int64 batch_index, int64 offset) {
        return batch_index * dataset()->batch_size_ + offset;
      }

      void EnsureOutputAllocated(BatchResult* batch_result,
                                 const std::vector<Tensor>& return_values) {
        mutex_lock l(batch_result->mu);
        if (batch_result->output_allocated) {
          return;
        }
        const size_t num_components = return_values.size();
        for (size_t i = 0; i < num_components; ++i) {
          TensorShape component_shape({dataset()->batch_size_});
          component_shape.AppendShape(return_values[i].shape());
          Tensor component(cpu_allocator(), return_values[i].dtype(),
                           component_shape);
          batch_result->output.emplace_back(std::move(component));
        }
        batch_result->output_allocated = true;
      }

      void InvokeFunctionLocked(IteratorContext* ctx, int64 batch_index,
                                int64 offset) EXCLUSIVE_LOCKS_REQUIRED(mu_) {
        size_t index = ComputeInvocationIndex(batch_index, offset);
        InvocationResult* result = &invocation_results_[index];
        BatchResult* batch_result = &batch_results_[batch_index];

        // Get the next input element.
        std::vector<Tensor> input_element;
        result->status =
            input_impl_->GetNext(ctx, &input_element, &end_of_input_);
        if (end_of_input_ || !result->status.ok()) {
          batch_result->counter->DecrementCount();
          return;
        }

        // Call `captured_func_(input_element)`, store the result in
        // `result->return_values`, and notify `batch_result->counter`
        // to unblock a consumer.
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
            [this, result, step_container, batch_result,
             offset](Status ret_status) {
              delete step_container;
              result->status.Update(ret_status);
              if (ret_status.ok()) {
                EnsureOutputAllocated(batch_result, result->return_values);
                const size_t num_components = result->return_values.size();
                for (size_t i = 0; i < num_components; ++i) {
                  const Tensor& tensor = result->return_values[i];
                  Tensor* batch = &(batch_result->output)[i];
                  if (tensor.NumElements() !=
                      (batch->NumElements() / batch->dim_size(0))) {
                    TensorShape batch_shape = batch->shape();
                    batch_shape.RemoveDim(0);
                    result->status.Update(errors::InvalidArgument(
                        "Cannot add tensor to the batch: number of "
                        "elements does not match. Shapes are: [tensor]: ",
                        tensor.shape().DebugString(),
                        ", [batch]: ", batch_shape.DebugString()));
                    break;
                  }
                  // TODO(mrry): Add a version of DoParallelConcat that allows
                  // us to move `tensor` where possible, to speed up string
                  // tensor batching.
                  Status copy_status = ::tensorflow::functor::DoParallelConcat(
                      *dataset()->device_, tensor, offset, batch);
                  if (!copy_status.ok()) {
                    result->status.Update(copy_status);
                    break;
                  }
                }
              }
              // NOTE(mrry): We clear the return values here to release any
              // memory associated with them and to paralellize the destruction
              // of the tensors (which can be surprisingly expensive for
              // map functions with large numbers of return values).
              result->return_values.clear();
              batch_result->counter->DecrementCount();
            });
      }

      void StartInvocationBatch(IteratorContext* ctx, int64 batch_index)
          EXCLUSIVE_LOCKS_REQUIRED(mu_) {
        port::Tracing::TraceMe activity(strings::StrCat(prefix(), "::Start"));
        // Initialize batch result.
        {
          mutex_lock l(batch_results_[batch_index].mu);
          batch_results_[batch_index].output_allocated = false;
          batch_results_[batch_index].counter.reset(
              new BlockingCounter(dataset()->batch_size_));
        }
        // Initialize invocation results.
        for (size_t i = 0; i < dataset()->batch_size_; ++i) {
          size_t index = ComputeInvocationIndex(batch_index, i);
          InvocationResult* result = &invocation_results_[index];
          // Reset the state of `result`.
          // NOTE(mrry): `result->return_values` were cleared when the previous
          // invocation completed.
          result->status = Status::OK();
        }
        // Start individual invocations.
        for (size_t i = 0; i < dataset()->batch_size_; ++i) {
          InvokeFunctionLocked(ctx, batch_index, i);
        }
      }

      Status WaitForBatch(int64 batch_index) EXCLUSIVE_LOCKS_REQUIRED(mu_) {
        port::Tracing::TraceMe activity(strings::StrCat(prefix(), "::Wait"));
        batch_results_[batch_index].counter->Wait();
        Status status = Status::OK();
        for (size_t i = 0; i < dataset()->batch_size_; ++i) {
          size_t index = ComputeInvocationIndex(batch_index, i);
          InvocationResult* result = &invocation_results_[index];
          if (!result->status.ok()) {
            VLOG(3) << "failed to process element[" << i
                    << "]: " << result->status;
            status.Update(result->status);
          }
        }
        return status;
      }

      mutex mu_;
      int32 current_batch_index_ GUARDED_BY(mu_) = -1;
      const std::unique_ptr<IteratorBase> input_impl_ GUARDED_BY(mu_);
      std::vector<InvocationResult> invocation_results_ GUARDED_BY(mu_);
      std::vector<BatchResult> batch_results_ GUARDED_BY(mu_);
      bool end_of_input_ GUARDED_BY(mu_) = false;
    };

    const DatasetBase* const input_;
    const NameAttrList func_;
    const int64 batch_size_;
    const int64 num_parallel_batches_;
    const DataTypeVector output_types_;
    const std::vector<PartialTensorShape> output_shapes_;
    const std::unique_ptr<CapturedFunction> captured_func_;
    const Eigen::ThreadPoolDevice* device_;  // not owned
  };

  const int graph_def_version_;
  DataTypeVector output_types_;
  std::vector<PartialTensorShape> output_shapes_;
  NameAttrList func_;
};

REGISTER_KERNEL_BUILDER(Name("MapAndBatchDataset").Device(DEVICE_CPU),
                        MapAndBatchDatasetOp);

}  // namespace

}  // namespace tensorflow
