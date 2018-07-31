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

#include "tensorflow/core/framework/partial_tensor_shape.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/kernels/data/dataset.h"
#include "tensorflow/core/kernels/data/prefetch_autotuner.h"
#include "tensorflow/core/lib/core/error_codes.pb.h"

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
    OP_REQUIRES(ctx,
                buffer_size >= 0 || buffer_size == PrefetchAutotuner::kAutoTune,
                errors::InvalidArgument("buffer_size must be >= 0"));

    *output = new Dataset(ctx, input, buffer_size);
  }

 private:
  class Dataset : public GraphDatasetBase {
   public:
    Dataset(OpKernelContext* ctx, const DatasetBase* input, int64 buffer_size)
        : GraphDatasetBase(ctx), input_(input), buffer_size_(buffer_size) {
      input_->Ref();
    }

    ~Dataset() override { input_->Unref(); }

    std::unique_ptr<IteratorBase> MakeIteratorInternal(
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

    string DebugString() const override { return "PrefetchDatasetOp::Dataset"; }

   protected:
    Status AsGraphDefInternal(OpKernelContext* ctx, DatasetGraphDefBuilder* b,
                              Node** output) const override {
      Node* input_graph_node = nullptr;
      TF_RETURN_IF_ERROR(b->AddParentDataset(ctx, input_, &input_graph_node));
      Node* buffer_size = nullptr;
      TF_RETURN_IF_ERROR(b->AddScalar(buffer_size_, &buffer_size));
      TF_RETURN_IF_ERROR(
          b->AddDataset(this, {input_graph_node, buffer_size}, output));
      return Status::OK();
    }

   private:
    class Iterator : public DatasetIterator<Dataset> {
     public:
      explicit Iterator(const Params& params)
          : DatasetIterator<Dataset>(params),
            auto_tuner_(params.dataset->buffer_size_) {}

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

      Status Initialize(IteratorContext* ctx) override {
        return dataset()->input_->MakeIterator(ctx, prefix(), &input_impl_);
      }

      Status GetNextInternal(IteratorContext* ctx,
                             std::vector<Tensor>* out_tensors,
                             bool* end_of_sequence) override {
        {
          mutex_lock l(mu_);
          TF_RETURN_IF_ERROR(EnsurePrefetchThreadStarted(ctx));
          // Wait until the next element in the buffer has been
          // produced, or we are shutting down.
          while (!cancelled_ && buffer_.empty() && !prefetch_thread_finished_ &&
                 auto_tuner_.buffer_limit() != 0) {
            auto_tuner_.RecordEmpty();
            cond_var_.wait(l);
          }

          if (cancelled_) {
            return errors::Cancelled(
                "PrefetchDatasetOp::Dataset::Iterator::GetNext");
          }

          if (!buffer_.empty()) {
            return Consume(out_tensors, end_of_sequence);
          }

          if (prefetch_thread_finished_) {
            *end_of_sequence = true;
            return Status::OK();
          }

          DCHECK_EQ(auto_tuner_.buffer_limit(), 0);
        }

        mutex_lock parent_l(parent_mu_);
        mutex_lock l(mu_);
        return input_impl_->GetNext(ctx, out_tensors, end_of_sequence);
      }

     protected:
      Status SaveInternal(IteratorStateWriter* writer) override {
        // Acquire both locks to ensure that the prefetch thread and
        // all GetNext threads are blocked.
        mutex_lock parent_l(parent_mu_);
        mutex_lock l(mu_);
        TF_RETURN_IF_ERROR(SaveParent(writer, input_impl_));
        TF_RETURN_IF_ERROR(
            writer->WriteScalar(full_name("buffer_size"), buffer_.size()));
        for (size_t i = 0; i < buffer_.size(); i++) {
          auto& buffer_element = buffer_[i];
          TF_RETURN_IF_ERROR(WriteStatus(writer, i, buffer_element.status));
          if (buffer_element.status.ok()) {
            TF_RETURN_IF_ERROR(writer->WriteScalar(
                full_name(strings::StrCat("buffer[", i, "].size")),
                buffer_element.value.size()));
            for (size_t j = 0; j < buffer_element.value.size(); j++) {
              TF_RETURN_IF_ERROR(writer->WriteTensor(
                  full_name(strings::StrCat("buffer[", i, "][", j, "]")),
                  buffer_element.value[j]));
            }
          }
        }
        return Status::OK();
      }

      Status RestoreInternal(IteratorContext* ctx,
                             IteratorStateReader* reader) override {
        mutex_lock parent_l(parent_mu_);
        mutex_lock l(mu_);
        buffer_.clear();
        TF_RETURN_IF_ERROR(RestoreParent(ctx, reader, input_impl_));
        size_t buffer_size;
        {
          int64 temp;
          TF_RETURN_IF_ERROR(
              reader->ReadScalar(full_name("buffer_size"), &temp));
          buffer_size = static_cast<size_t>(temp);
        }
        for (size_t i = 0; i < buffer_size; i++) {
          buffer_.emplace_back();
          auto& buffer_element = buffer_.back();
          TF_RETURN_IF_ERROR(ReadStatus(reader, i, &buffer_element.status));
          if (buffer_element.status.ok()) {
            size_t value_size;
            {
              int64 temp;
              TF_RETURN_IF_ERROR(reader->ReadScalar(
                  full_name(strings::StrCat("buffer[", i, "].size")), &temp));
              value_size = static_cast<size_t>(temp);
            }
            buffer_element.value.reserve(value_size);
            for (size_t j = 0; j < value_size; j++) {
              buffer_element.value.emplace_back();
              TF_RETURN_IF_ERROR(reader->ReadTensor(
                  full_name(strings::StrCat("buffer[", i, "][", j, "]")),
                  &buffer_element.value.back()));
            }
          }
        }
        return Status::OK();
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

      Status Consume(std::vector<Tensor>* out_tensors, bool* end_of_sequence)
          EXCLUSIVE_LOCKS_REQUIRED(mu_) {
        // A new element is available. Forward the status from computing it, and
        // (if we successfully got an element) the output values.
        Status s = buffer_.front().status;
        if (s.ok()) {
          *out_tensors = std::move(buffer_.front().value);
        }
        buffer_.pop_front();
        *end_of_sequence = false;

        // Wake the prefetch thread, in case it has been waiting for space
        // in the buffer. Also wake up threads from other calls to GetNext.
        //
        // TODO(mrry): Consider using different condition variables for
        // GetNext and Prefetch.
        cond_var_.notify_all();
        return s;
      }

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
            while (!cancelled_ &&
                   buffer_.size() >= auto_tuner_.buffer_limit()) {
              cond_var_.wait(l);
            }

            if (cancelled_) {
              return;
            }
          }

          // 2. Read the next element.
          // Acquire the parent lock since we will be reading an element
          // from the input iterator. Note that we do not wish to release
          // this lock till we have added the fetched element to the
          // `buffer_` else there will be local state that may be missed
          // by SaveInternal.
          mutex_lock parent_l(parent_mu_);
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

      Status WriteStatus(IteratorStateWriter* writer, size_t index,
                         const Status& status) EXCLUSIVE_LOCKS_REQUIRED(mu_) {
        TF_RETURN_IF_ERROR(writer->WriteScalar(
            CodeKey(index), static_cast<int64>(status.code())));
        if (!status.ok()) {
          TF_RETURN_IF_ERROR(writer->WriteScalar(ErrorMessageKey(index),
                                                 status.error_message()));
        }
        return Status::OK();
      }

      Status ReadStatus(IteratorStateReader* reader, size_t index,
                        Status* status) EXCLUSIVE_LOCKS_REQUIRED(mu_) {
        int64 code_int;
        TF_RETURN_IF_ERROR(reader->ReadScalar(CodeKey(index), &code_int));
        error::Code code = static_cast<error::Code>(code_int);

        if (code != error::Code::OK) {
          string error_message;
          TF_RETURN_IF_ERROR(
              reader->ReadScalar(ErrorMessageKey(index), &error_message));
          *status = Status(code, error_message);
        } else {
          *status = Status::OK();
        }
        return Status::OK();
      }

      string CodeKey(size_t index) {
        return full_name(strings::StrCat("status[", index, "].code"));
      }

      string ErrorMessageKey(size_t index) {
        return full_name(strings::StrCat("status[", index, "].error_message"));
      }

      // This mutex is used to ensure exclusivity between multiple threads
      // reading/writing this iterator's local state.
      mutex mu_;
      // This mutex is used to ensure exclusivity between multiple threads
      // accessing the parent iterator. We keep this separate from `mu_` to
      // allow prefetching to run in parallel with GetNext calls.
      mutex parent_mu_ ACQUIRED_BEFORE(mu_);
      std::unique_ptr<IteratorBase> input_impl_ GUARDED_BY(parent_mu_);
      condition_variable cond_var_;
      PrefetchAutotuner auto_tuner_ GUARDED_BY(mu_);
      std::deque<BufferElement> buffer_ GUARDED_BY(mu_);
      std::unique_ptr<Thread> prefetch_thread_ GUARDED_BY(mu_);
      bool cancelled_ GUARDED_BY(mu_) = false;
      bool prefetch_thread_finished_ GUARDED_BY(mu_) = false;
    };

    const DatasetBase* const input_;
    const int64 buffer_size_;
  };
};

REGISTER_KERNEL_BUILDER(Name("PrefetchDataset").Device(DEVICE_CPU),
                        PrefetchDatasetOp);
REGISTER_KERNEL_BUILDER(Name("PrefetchDataset")
                            .Device(DEVICE_GPU)
                            .HostMemory("buffer_size")
                            .HostMemory("input_dataset")
                            .HostMemory("handle"),
                        PrefetchDatasetOp);
}  // namespace

}  // namespace tensorflow
