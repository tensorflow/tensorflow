/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/core/kernels/data/parallel_map_iterator.h"

#include <deque>
#include <functional>
#include <utility>
#include <vector>

namespace tensorflow {
namespace {

class ParallelMapIterator : public DatasetBaseIterator {
 public:
  explicit ParallelMapIterator(
      const typename DatasetBaseIterator::BaseParams& params,
      const DatasetBase* input_dataset,
      std::function<Status(IteratorContext*)> init_func,
      ParallelMapIteratorFunction map_func, int32 num_parallel_calls)
      : DatasetBaseIterator(params),
        input_dataset_(input_dataset),
        init_func_(std::move(init_func)),
        map_func_(std::move(map_func)),
        num_parallel_calls_(num_parallel_calls) {}

  ~ParallelMapIterator() override {
    // TODO(mrry): Replace this cancellation logic with a
    // CancellationManager. The syntax would be more heavyweight,
    // but it would be possible to thread a cancellation manager
    // through the IteratorContext to upstream,
    // potentially-blocking iterators, when we add these.
    mutex_lock l(mu_);
    // Cancel the runner thread.
    cancelled_ = true;
    cond_var_.notify_all();
    // Wait for all in-flight calls to complete.
    while (num_calls_ > 0) {
      cond_var_.wait(l);
    }
  }

  Status Initialize(IteratorContext* ctx) override {
    TF_RETURN_IF_ERROR(
        input_dataset_->MakeIterator(ctx, prefix(), &input_impl_));
    if (init_func_) {
      TF_RETURN_IF_ERROR(init_func_(ctx));
    }
    return Status::OK();
  }

  Status GetNextInternal(IteratorContext* ctx, std::vector<Tensor>* out_tensors,
                         bool* end_of_sequence) override {
    std::shared_ptr<InvocationResult> result;
    {
      mutex_lock l(mu_);
      EnsureRunnerThreadStarted(ctx);
      while (invocation_results_.empty()) {
        cond_var_.wait(l);
      }
      std::swap(result, invocation_results_.front());
      invocation_results_.pop_front();
    }
    cond_var_.notify_all();
    result->notification.WaitForNotification();
    return ProcessResult(result, out_tensors, end_of_sequence);
  }

 protected:
  Status SaveInternal(IteratorStateWriter* writer) override {
    mutex_lock l(mu_);
    // Wait for all in-flight calls to complete.
    while (num_calls_ > 0) {
      cond_var_.wait(l);
    }
    CHECK_EQ(num_calls_, 0);
    TF_RETURN_IF_ERROR(SaveInput(writer, input_impl_));
    TF_RETURN_IF_ERROR(
        writer->WriteScalar(full_name("invocation_results.size"),
                            invocation_results_.size()));
    for (size_t i = 0; i < invocation_results_.size(); i++) {
      std::shared_ptr<InvocationResult> result = invocation_results_[i];
      TF_RETURN_IF_ERROR(WriteStatusLocked(writer, i, result->status));
      TF_RETURN_IF_ERROR(writer->WriteScalar(
          full_name(strings::StrCat("invocation_results[", i, "].size")),
          result->return_values.size()));
      for (size_t j = 0; j < result->return_values.size(); j++) {
        TF_RETURN_IF_ERROR(
            writer->WriteTensor(full_name(strings::StrCat(
                                    "invocation_results[", i, "][", j, "]")),
                                result->return_values[j]));
      }
      if (result->end_of_input) {
        TF_RETURN_IF_ERROR(writer->WriteScalar(
            full_name(
                strings::StrCat("invocation_results[", i, "].end_of_input")),
            ""));
      }
    }
    return Status::OK();
  }

  Status RestoreInternal(IteratorContext* ctx,
                         IteratorStateReader* reader) override {
    mutex_lock l(mu_);
    TF_RETURN_IF_ERROR(RestoreInput(ctx, reader, input_impl_));
    int64 invocation_results_size;
    TF_RETURN_IF_ERROR(reader->ReadScalar(
        full_name("invocation_results.size"), &invocation_results_size));
    for (size_t i = 0; i < invocation_results_size; i++) {
      std::shared_ptr<InvocationResult> result(new InvocationResult());
      invocation_results_.push_back(result);
      TF_RETURN_IF_ERROR(ReadStatusLocked(reader, i, &result->status));
      size_t num_return_values;
      {
        int64 size;
        TF_RETURN_IF_ERROR(
            reader->ReadScalar(full_name(strings::StrCat(
                                   "invocation_results[", i, "].size")),
                               &size));
        num_return_values = static_cast<size_t>(size);
        if (num_return_values != size) {
          return errors::InvalidArgument(strings::StrCat(
              full_name(
                  strings::StrCat("invocation_results[", i, "].size")),
              ": ", size, " is not a valid value of type size_t."));
        }
      }
      result->return_values.reserve(num_return_values);
      for (size_t j = 0; j < num_return_values; j++) {
        result->return_values.emplace_back();
        TF_RETURN_IF_ERROR(
            reader->ReadTensor(full_name(strings::StrCat(
                                   "invocation_results[", i, "][", j, "]")),
                               &result->return_values.back()));
      }
      result->end_of_input = reader->Contains(full_name(
          strings::StrCat("invocation_results[", i, "].end_of_input")));
      result->notification.Notify();
    }
    return Status::OK();
  }

 private:
  struct InvocationResult {
    Notification notification;
    Status status;
    std::vector<Tensor> return_values;
    bool end_of_input;
  };

  void EnsureRunnerThreadStarted(IteratorContext* ctx)
      EXCLUSIVE_LOCKS_REQUIRED(mu_) {
    if (!runner_thread_) {
      std::shared_ptr<IteratorContext> ctx_copy(new IteratorContext(*ctx));
      runner_thread_.reset(ctx->env()->StartThread(
          {}, "runner_thread",
          std::bind(&ParallelMapIterator::RunnerThread, this, ctx_copy)));
    }
  }

  void CallCompleted(const std::shared_ptr<InvocationResult>& result)
      LOCKS_EXCLUDED(mu_) {
    {
      mutex_lock l(mu_);
      num_calls_--;
    }
    result->notification.Notify();
    cond_var_.notify_all();
  }

  void CallFunction(const std::shared_ptr<IteratorContext>& ctx,
                    const std::shared_ptr<InvocationResult>& result)
      LOCKS_EXCLUDED(mu_) {
    // Get the next input element.
    std::vector<Tensor> input_element;
    result->status =
        input_impl_->GetNext(ctx.get(), &input_element, &result->end_of_input);
    if (result->end_of_input || !result->status.ok()) {
      CallCompleted(result);
      return;
    }

    // Call `func_(input_element)`, store the result in
    // `result->return_values`, and notify `result->notification` to unblock
    // a consumer.
    auto done = [this, result](Status status) {
      result->status.Update(status);
      CallCompleted(result);
    };

    map_func_(ctx.get(), std::move(input_element), &result->return_values,
              std::move(done));
  }

  int64 MaxInvocationResults() { return num_parallel_calls_; }

  Status ProcessResult(const std::shared_ptr<InvocationResult>& result,
                       std::vector<Tensor>* out_tensors,
                       bool* end_of_sequence) {
    if (!result->end_of_input && result->status.ok()) {
      *out_tensors = std::move(result->return_values);
      *end_of_sequence = false;
      return Status::OK();
    }
    if (errors::IsOutOfRange(result->status)) {
      // `f` may deliberately raise `errors::OutOfRange` to indicate that we
      // should terminate the iteration early.
      *end_of_sequence = true;
      return Status::OK();
    }
    *end_of_sequence = result->end_of_input;
    return result->status;
  }

  void RunnerThread(const std::shared_ptr<IteratorContext>& ctx) {
    std::vector<std::shared_ptr<InvocationResult>> new_calls;
    new_calls.reserve(num_parallel_calls_);
    while (true) {
      {
        mutex_lock l(mu_);
        while (!cancelled_ &&
               (num_calls_ >= num_parallel_calls_ ||
                invocation_results_.size() >= MaxInvocationResults())) {
          cond_var_.wait(l);
        }
        if (cancelled_) {
          return;
        }
        while (num_calls_ < num_parallel_calls_ &&
               invocation_results_.size() < MaxInvocationResults()) {
          invocation_results_.emplace_back(new InvocationResult());
          new_calls.push_back(invocation_results_.back());
          num_calls_++;
        }
      }
      cond_var_.notify_all();
      for (const auto& call : new_calls) {
        CallFunction(ctx, call);
      }
      new_calls.clear();
    }
  }

  Status WriteStatusLocked(IteratorStateWriter* writer, size_t index,
                           const Status& status) EXCLUSIVE_LOCKS_REQUIRED(mu_) {
    TF_RETURN_IF_ERROR(
        writer->WriteScalar(CodeKey(index), static_cast<int64>(status.code())));
    if (!status.ok()) {
      TF_RETURN_IF_ERROR(
          writer->WriteScalar(ErrorMessageKey(index), status.error_message()));
    }
    return Status::OK();
  }

  Status ReadStatusLocked(IteratorStateReader* reader, size_t index,
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
    return full_name(
        strings::StrCat("invocation_results[", index, "].code"));
  }

  string ErrorMessageKey(size_t index) {
    return full_name(
        strings::StrCat("invocation_results[", index, "].error_message"));
  }

  const DatasetBase* const input_dataset_;  // Not owned.
  const std::function<Status(IteratorContext*)> init_func_;
  const ParallelMapIteratorFunction map_func_;
  const int32 num_parallel_calls_;
  // Used for coordination between the main thread and the runner thread.
  mutex mu_;
  // Used for coordination between the main thread and the runner thread. In
  // particular, the runner thread should only schedule new calls when the
  // number of in-flight calls is less than the user specified level of
  // parallelism and there are slots available in the `invocation_results_`
  // buffer.
  condition_variable cond_var_;
  // Counts the number of outstanding calls.
  int64 num_calls_ GUARDED_BY(mu_) = 0;
  std::unique_ptr<IteratorBase> input_impl_;
  // Buffer for storing the invocation results.
  std::deque<std::shared_ptr<InvocationResult>> invocation_results_
      GUARDED_BY(mu_);
  std::unique_ptr<Thread> runner_thread_ GUARDED_BY(mu_);
  bool cancelled_ GUARDED_BY(mu_) = false;
};

}  // namespace

std::unique_ptr<IteratorBase> NewParallelMapIterator(
    const DatasetBaseIterator::BaseParams& params,
    const DatasetBase* input_dataset, ParallelMapIteratorFunction map_func,
    int32 num_parallel_calls) {
  return NewParallelMapIterator(params, input_dataset, nullptr,
                                std::move(map_func), num_parallel_calls);
}

std::unique_ptr<IteratorBase> NewParallelMapIterator(
    const DatasetBaseIterator::BaseParams& params,
    const DatasetBase* input_dataset,
    std::function<Status(IteratorContext*)> init_func,
    ParallelMapIteratorFunction map_func, int32 num_parallel_calls) {
  return std::unique_ptr<IteratorBase>(
      new ParallelMapIterator(params, input_dataset, std::move(init_func),
                              std::move(map_func), num_parallel_calls));
}

}  // namespace tensorflow
