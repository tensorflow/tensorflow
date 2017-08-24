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
#include "tensorflow/core/framework/partial_tensor_shape.h"
#include "tensorflow/core/framework/tensor.h"

namespace tensorflow {

namespace {

// See documentation in ../ops/dataset_ops.cc for a high-level
// description of the following op.

class PrefetchDatasetOp : public UnaryDatasetOpKernel {
 public:
  explicit PrefetchDatasetOp(OpKernelConstruction* ctx)
      : UnaryDatasetOpKernel(ctx) {}

 protected:
  void MakeDataset(OpKernelContext* ctx, DatasetBase* input,
                   DatasetBase** output) override {
    int64 buffer_size;
    OP_REQUIRES_OK(
        ctx, ParseScalarArgument<int64>(ctx, "buffer_size", &buffer_size));

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

    *output = new Dataset(input, buffer_size, std::move(params));
  }

 private:
  class Dataset : public DatasetBase {
   public:
    Dataset(const DatasetBase* input, int64 buffer_size,
            IteratorContext::Params ctx_params)
        : input_(input),

          buffer_size_(buffer_size),
          ctx_params_(std::move(ctx_params)) {
      input_->Ref();
    }

    ~Dataset() override { input_->Unref(); }

    std::unique_ptr<IteratorBase> MakeIterator(
        const string& prefix) const override {
      return std::unique_ptr<IteratorBase>(
          new Iterator({this, strings::StrCat(prefix, "::Prefetch")}));
    }

    const DataTypeVector& output_dtypes() const override {
      return input_->output_dtypes();
    }
    const std::vector<PartialTensorShape>& output_shapes() const override {
      return input_->output_shapes();
    }

    string DebugString() override { return "PrefetchDatasetOp::Dataset"; }

   private:
    class Iterator : public DatasetIterator<Dataset> {
     public:
      explicit Iterator(const Params& params)
          : DatasetIterator<Dataset>(params),
            input_impl_(params.dataset->input_->MakeIterator(params.prefix)) {}

      ~Iterator() override {
        // Signal the prefetch thread to terminate it. We will then
        // join that thread when we delete `this->prefetch_thread_`.
        //
        // TODO(mrry): Replace this cancellation logic with a
        // CancellationManager. The syntax would be more heavyweight,
        // but it would be possible to thread a cancellation manager
        // through the IteratorContext to upstream,
        // potentially-blocking iterators, when we add these.
        {
          mutex_lock l(mu_);
          cancelled_ = true;
          cond_var_.notify_all();
        }
      }

      Status GetNextInternal(IteratorContext* ctx,
                             std::vector<Tensor>* out_tensors,
                             bool* end_of_sequence) override {
        mutex_lock l(mu_);
        TF_RETURN_IF_ERROR(EnsurePrefetchThreadStarted(ctx));

        while (true) {
          // Wait until the next element in the buffer has been
          // produced, or we are shutting down.
          while (!cancelled_ && !prefetch_thread_finished_ && buffer_.empty()) {
            cond_var_.wait(l);
          }

          if (cancelled_) {
            return errors::Cancelled(
                "PrefetchDatasetOp::Dataset::Iterator::GetNext");
          }

          if (!buffer_.empty()) {
            // A new element is available. Forward the status from
            // computing it, and (if we successfully got an element)
            // the output values.
            Status s = buffer_.front().status;
            if (s.ok()) {
              *out_tensors = std::move(buffer_.front().value);
            }
            buffer_.pop_front();
            *end_of_sequence = false;

            // Wake the prefetch thread, in case it has been waiting
            // for space in the buffer.
            cond_var_.notify_one();
            return s;
          } else if (prefetch_thread_finished_) {
            *end_of_sequence = true;
            return Status::OK();
          }
        }
      }

     private:
      // A buffer element comprises a status and (if that status is
      // OK) a vector of tensors, representing an element of the input dataset.
      struct BufferElement {
        // The producer sets `status` if getting the input element fails.
        Status status;
        // The buffered data element.
        std::vector<Tensor> value;
      };

      Status EnsurePrefetchThreadStarted(IteratorContext* ctx)
          EXCLUSIVE_LOCKS_REQUIRED(mu_) {
        if (!prefetch_thread_) {
          prefetch_thread_.reset(
              ctx->env()->StartThread({}, "prefetch_thread",
                                      std::bind(&Iterator::PrefetchThread, this,
                                                new IteratorContext(*ctx))));
        }
        return Status::OK();
      }

      // Prefetches elements of the input, storing results in an internal
      // buffer.
      //
      // It owns the iterator context passed to it.
      void PrefetchThread(IteratorContext* ctx) {
        std::unique_ptr<IteratorContext> cleanup(ctx);
        while (true) {
          std::vector<Tensor> value;

          // 1. Wait for a slot in the buffer.
          {
            mutex_lock l(mu_);
            while (!cancelled_ && buffer_.size() == dataset()->buffer_size_) {
              cond_var_.wait(l);
            }

            if (cancelled_) {
              return;
            }
          }

          // 2. Read the next element.
          bool end_of_sequence;
          BufferElement buffer_element;
          buffer_element.status = input_impl_->GetNext(
              ctx, &buffer_element.value, &end_of_sequence);
          if (buffer_element.status.ok() && end_of_sequence) {
            mutex_lock l(mu_);
            prefetch_thread_finished_ = true;
            cond_var_.notify_all();
            return;
          }

          // 3. Signal that the element has been produced.
          {
            mutex_lock l(mu_);
            buffer_.push_back(std::move(buffer_element));
            cond_var_.notify_all();
          }
        }
      }

      mutex mu_;
      const std::unique_ptr<IteratorBase> input_impl_;
      condition_variable cond_var_;
      std::deque<BufferElement> buffer_ GUARDED_BY(mu_);
      std::unique_ptr<Thread> prefetch_thread_ GUARDED_BY(mu_);
      bool cancelled_ GUARDED_BY(mu_) = false;
      bool prefetch_thread_finished_ GUARDED_BY(mu_) = false;
    };

    const DatasetBase* const input_;
    const int64 buffer_size_;
    const IteratorContext::Params ctx_params_;
  };
};

REGISTER_KERNEL_BUILDER(Name("PrefetchDataset").Device(DEVICE_CPU),
                        PrefetchDatasetOp);

}  // namespace

}  // namespace tensorflow
