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

class ParallelMapDatasetOp : public OpKernel {
 public:
  explicit ParallelMapDatasetOp(OpKernelConstruction* ctx)
      : OpKernel(ctx), graph_def_version_(ctx->graph_def_version()) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("f", &func_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("output_types", &output_types_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("output_shapes", &output_shapes_));
  }

  void Compute(OpKernelContext* ctx) override {
    DatasetBase* input;
    OP_REQUIRES_OK(ctx, LookupResource(ctx, HandleFromInput(ctx, 0), &input));
    core::ScopedUnref unref_input(input);

    OpInputList inputs;
    OP_REQUIRES_OK(ctx, ctx->input_list("other_arguments", &inputs));
    std::vector<Tensor> other_arguments;
    other_arguments.reserve(inputs.size());
    for (const Tensor& t : inputs) {
      other_arguments.push_back(t);
    }

    const Tensor* num_threads_t;
    OP_REQUIRES_OK(ctx, ctx->input("num_threads", &num_threads_t));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(num_threads_t->shape()),
                errors::InvalidArgument("num_threads must be a scalar"));
    const int32 num_threads = num_threads_t->flat<int32>()(0);
    OP_REQUIRES(
        ctx, num_threads > 0,
        errors::InvalidArgument("num_threads must be greater than zero."));

    const Tensor* output_buffer_size_t;
    OP_REQUIRES_OK(ctx,
                   ctx->input("output_buffer_size", &output_buffer_size_t));
    OP_REQUIRES(
        ctx, TensorShapeUtils::IsScalar(output_buffer_size_t->shape()),
        errors::InvalidArgument("output_buffer_size must be a scalar."));
    const int64 output_buffer_size = output_buffer_size_t->flat<int64>()(0);

    // TODO(mrry): Relax this requirement? If the output buffer owns
    // the (tuples of) tensors into which `f` writes its output, it
    // seems like this constraint would make it easier to (i)
    // constrain the memory usage of the iterator, and (ii) enforce a
    // consistent ordering between input and output.
    OP_REQUIRES(ctx, output_buffer_size >= num_threads,
                errors::InvalidArgument(
                    "output_buffer_size (", output_buffer_size,
                    ") must be greater than or equal to num_threads (",
                    num_threads, ")."));

    std::unique_ptr<CapturedFunction> captured_func;
    OP_REQUIRES_OK(ctx, CapturedFunction::Create(ctx, func_, graph_def_version_,
                                                 std::move(other_arguments),
                                                 &captured_func));

    // TODO(mrry): It seems unnatural to capture the params from *this
    // kernel's* OpKernelContext, although the captured values should
    // be the same for any kernel in the same session. Consider adding
    // an IteratorContext* argument to Dataset::MakeIterator(), and
    // threading the context information through that
    // way. Alternatively, provide a session-scoped context that will
    // provide this information to all users in the same session (and
    // that will have the appropriate lifetime).
    IteratorContext::Params params;
    params.env = ctx->env();
    params.resource_manager = ctx->resource_manager();
    params.runner = *(ctx->runner());

    DatasetBase* dataset =
        new Dataset(input, num_threads, output_buffer_size, std::move(params),
                    output_types_, output_shapes_, std::move(captured_func));

    Tensor* output = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape({}), &output));
    ResourceHandle handle = MakeResourceHandle<DatasetBase>(
        ctx, ctx->step_container()->name(), name());
    OP_REQUIRES_OK(ctx, CreateResource(ctx, handle, dataset));
    output->flat<ResourceHandle>()(0) = handle;
  }

 private:
  class Dataset : public DatasetBase {
   public:
    Dataset(const DatasetBase* input, int32 num_threads,
            int64 output_buffer_size, IteratorContext::Params ctx_params,
            const DataTypeVector& output_types,
            const std::vector<PartialTensorShape>& output_shapes,
            std::unique_ptr<CapturedFunction> captured_func)
        : input_(input),
          num_threads_(num_threads),
          output_buffer_size_(output_buffer_size),
          ctx_params_(std::move(ctx_params)),
          output_types_(output_types),
          output_shapes_(output_shapes),
          captured_func_(std::move(captured_func)) {
      input_->Ref();
    }

    ~Dataset() override { input_->Unref(); }

    std::unique_ptr<IteratorBase> MakeIterator() const override {
      return std::unique_ptr<IteratorBase>(new Iterator(this));
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
      explicit Iterator(const Dataset* dataset)
          : DatasetIterator<Dataset>(dataset),
            iter_ctx_(dataset->ctx_params_),
            input_impl_(dataset->input_->MakeIterator()) {}

      ~Iterator() override {
        // Signal the mapper threads, if any, so that they terminate.
        // We will then join those threads when we delete
        // `this->mapper_threads_`.
        //
        // TODO(mrry): Replace this cancellation logic with a
        // CancellationManager. The syntax would be more heavyweight,
        // but it would be possible to thread a cancellation manager
        // through the IteratorContext to upstream,
        // potentially-blocking iterators, when we add these.
        {
          mutex_lock l(output_mu_);
          cancelled_ = true;
          cond_var_.notify_all();
        }
      }

      Status GetNext(IteratorContext* ctx, std::vector<Tensor>* out_tensors,
                     bool* end_of_sequence) override {
        mutex_lock l(output_mu_);
        TF_RETURN_IF_ERROR(EnsureMapperThreadsStarted(ctx));

        while (true) {
          // 1. Wait until the next element in the output queue has
          // been produced, or we are shutting down.
          while (
              !cancelled_ && active_threads_ > 0 &&
              (output_buffer_.empty() || !output_buffer_.front().is_produced)) {
            cond_var_.wait(l);
          }

          if (cancelled_) {
            return errors::Cancelled(
                "ParallelMapDatasetOp::Dataset::Iterator::GetNext");
          }

          if (!output_buffer_.empty() && output_buffer_.front().is_produced) {
            // A new output element is available. Forward the status
            // from computing it, and (if we successfully got an
            // element) the output values.
            Status s = output_buffer_.front().output_status;
            if (s.ok()) {
              *out_tensors = std::move(output_buffer_.front().output_value);
            }
            output_buffer_.pop_front();
            *end_of_sequence = false;

            // Wake one of the producing threads, in case they have been
            // waiting for space in the queue.
            cond_var_.notify_one();
            return s;
          } else if (active_threads_ == 0) {
            *end_of_sequence = true;
            return Status::OK();
          }
        }
      }

     private:
      // An output queue element comprises a bool (which indicates
      // whether the element has been produced yet) and a vector of
      // tensors (which contains the tuple of tensors if the bool is
      // true).
      struct OutputQueueElement {
        // The producer must set `is_produced` to `true` after
        // `output_status` or `output_value` has been written.
        bool is_produced = false;
        // The producer sets `output_status` if either getting the
        // input element or applying the mapper function to it fails.
        Status output_status;
        // The mapped data element.
        std::vector<Tensor> output_value;
      };

      Status EnsureMapperThreadsStarted(IteratorContext* ctx)
          EXCLUSIVE_LOCKS_REQUIRED(output_mu_) {
        if (mapper_threads_.empty()) {
          // Choose a step ID that is guaranteed not to clash with any
          // Session-generated step ID. DirectSession only generates
          // non-negative step IDs (contiguous, starting from 0), and
          // MasterSession generates 56-bit random step IDs whose MSB
          // is always 0, so a negative random step ID should suffice.
          f_opts_.step_id = -std::abs(static_cast<int64>(random::New64()));
          f_opts_.runner = iter_ctx_.runner();

          active_threads_ = dataset()->num_threads_;
          for (int i = 0; i < dataset()->num_threads_; ++i) {
            mapper_threads_.emplace_back(
                std::unique_ptr<Thread>(ctx->env()->StartThread(
                    {}, "mapper_thread", [this]() { MapperThread(); })));
          }
        }
        return Status::OK();
      }

      void MapperThread() {
        while (true) {
          OutputQueueElement* output_queue_element_;

          std::vector<Tensor> input_args;
          std::vector<Tensor> output_value;

          Status s;

          // 1. Acquire a slot in the output queue and a corresponding input
          // element.
          {
            // First acquire the input lock. Only one MapperThread may
            // call GetNext() on the input iterator at a time, to
            // preserve the ordering of elements.
            mutex_lock input_lock(input_mu_);
            {
              // This MapperThread is now responsible for producing
              // the next element in the output queue. We acquire a
              // slot in the output queue atomically, which may block,
              // but we deliberately do not release input_mu_ to
              // prevent another MapperThread from overtaking us.
              mutex_lock output_lock(output_mu_);
              while (!cancelled_ &&
                     output_buffer_.size() == dataset()->output_buffer_size_) {
                cond_var_.wait(output_lock);
              }

              if (cancelled_) {
                --active_threads_;
                return;
              }

              output_buffer_.push_back(OutputQueueElement());
              output_queue_element_ = &output_buffer_.back();
            }

            bool end_of_sequence;
            s = input_impl_->GetNext(&iter_ctx_, &input_args, &end_of_sequence);
            if (s.ok() && end_of_sequence) {
              mutex_lock output_lock(output_mu_);
              --active_threads_;
              if (active_threads_ == 0) {
                cond_var_.notify_all();
              }
              return;
            }
          }

          if (s.ok()) {
            s = dataset()->captured_func_->Run(f_opts_, input_args,
                                               &output_value);
          }

          // 3. Signal that the element has been produced.
          {
            mutex_lock output_lock(output_mu_);
            output_queue_element_->output_status.Update(s);
            output_queue_element_->is_produced = true;
            std::swap(output_queue_element_->output_value, output_value);
            cond_var_.notify_all();
          }
        }
      }

      IteratorContext iter_ctx_;
      mutex input_mu_;
      const std::unique_ptr<IteratorBase> input_impl_ GUARDED_BY(input_mu_);
      FunctionLibraryRuntime::Options f_opts_;
      mutex output_mu_;
      condition_variable cond_var_;
      std::deque<OutputQueueElement> output_buffer_ GUARDED_BY(output_mu_);
      std::vector<std::unique_ptr<Thread>> mapper_threads_
          GUARDED_BY(output_mu_);
      bool cancelled_ GUARDED_BY(output_mu_) = false;
      int32 active_threads_ GUARDED_BY(output_mu_);
    };

    const DatasetBase* const input_;
    const NameAttrList func_;
    const int32 num_threads_;
    const int64 output_buffer_size_;
    const IteratorContext::Params ctx_params_;
    const DataTypeVector output_types_;
    const std::vector<PartialTensorShape> output_shapes_;
    const std::unique_ptr<CapturedFunction> captured_func_;
  };

  const int graph_def_version_;
  DataTypeVector output_types_;
  std::vector<PartialTensorShape> output_shapes_;
  const NameAttrList* func_;
};

REGISTER_KERNEL_BUILDER(Name("ParallelMapDataset").Device(DEVICE_CPU),
                        ParallelMapDatasetOp);

}  // namespace

}  // namespace tensorflow
