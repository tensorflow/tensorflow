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

#include <atomic>
#include <deque>
#include <functional>
#include <memory>
#include <utility>
#include <vector>

#include "tensorflow/core/framework/stats_aggregator.h"
#include "tensorflow/core/lib/gtl/cleanup.h"
#include "tensorflow/core/platform/cpu_info.h"

namespace tensorflow {
namespace data {
namespace {

class ParallelMapIterator : public DatasetBaseIterator {
 public:
  struct Params {
    Params(std::unique_ptr<ParallelMapFunctor> parallel_map_functor,
           int32 num_parallel_calls, bool sloppy, bool preserve_cardinality)
        : parallel_map_functor(std::move(parallel_map_functor)),
          num_parallel_calls(num_parallel_calls),
          sloppy(sloppy),
          preserve_cardinality(preserve_cardinality) {}

    std::unique_ptr<ParallelMapFunctor> parallel_map_functor;
    int32 num_parallel_calls;
    bool sloppy;
    bool preserve_cardinality;
  };

  ParallelMapIterator(
      const typename DatasetBaseIterator::BaseParams& base_params,
      const DatasetBase* input_dataset, Params params)
      : DatasetBaseIterator(base_params),
        input_dataset_(input_dataset),
        parallel_map_functor_(std::move(params.parallel_map_functor)),
        mu_(std::make_shared<mutex>()),
        cond_var_(std::make_shared<condition_variable>()),
        num_parallel_calls_(std::make_shared<model::SharedState>(
            params.num_parallel_calls, mu_, cond_var_)),
        sloppy_(params.sloppy),
        preserve_cardinality_(params.preserve_cardinality) {
    std::vector<string> components =
        str_util::Split(base_params.prefix, "::", str_util::SkipEmpty());
    key_prefix_ = components.back();
  }

  ~ParallelMapIterator() override {
    mutex_lock l(*mu_);
    // Cancel the runner thread.
    cancelled_ = true;
    cond_var_->notify_all();
    // Wait for all in-flight calls to complete.
    while (num_calls_ > 0) {
      cond_var_->wait(l);
    }
  }

  Status Initialize(IteratorContext* ctx) override {
    mutex_lock l(*mu_);
    if (num_parallel_calls_->value == model::kAutoTune) {
      num_parallel_calls_->value = ctx->runner_threadpool_size();
    }
    TF_RETURN_IF_ERROR(
        input_dataset_->MakeIterator(ctx, prefix(), &input_impl_));
    return parallel_map_functor_->InitFunc(ctx);
  }

  Status GetNextInternal(IteratorContext* ctx, std::vector<Tensor>* out_tensors,
                         bool* end_of_sequence) override {
    std::shared_ptr<InvocationResult> result;
    {
      mutex_lock l(*mu_);
      EnsureRunnerThreadStarted(ctx);
      while (ShouldWait(&result)) {
        RecordStop(ctx);
        cond_var_->wait(l);
        RecordStart(ctx);
      }
    }
    RecordStop(ctx);
    result->notification.WaitForNotification();
    RecordStart(ctx);
    return ProcessResult(ctx, result, out_tensors, end_of_sequence);
  }

 protected:
  std::shared_ptr<model::Node> CreateNode(
      IteratorContext* ctx, model::Node::Args args) const override {
    return model::MakeAsyncKnownRatioNode(
        std::move(args),
        /*ratio=*/1,
        {model::MakeParameter("parallelism", num_parallel_calls_, /*min=*/1,
                              /*max=*/ctx->runner_threadpool_size())});
  }

  Status SaveInternal(IteratorStateWriter* writer) override {
    mutex_lock l(*mu_);
    // Wait for all in-flight calls to complete.
    while (num_calls_ > 0) {
      cond_var_->wait(l);
    }
    CHECK_EQ(num_calls_, 0);
    TF_RETURN_IF_ERROR(SaveInput(writer, input_impl_));
    TF_RETURN_IF_ERROR(writer->WriteScalar(full_name("invocation_results.size"),
                                           invocation_results_.size()));
    for (size_t i = 0; i < invocation_results_.size(); i++) {
      const auto& result = *(invocation_results_[i]);
      TF_RETURN_IF_ERROR(WriteStatusLocked(writer, i, result.status));
      TF_RETURN_IF_ERROR(writer->WriteScalar(
          full_name(strings::StrCat("invocation_results[", i, "].size")),
          result.return_values.size()));
      for (size_t j = 0; j < result.return_values.size(); j++) {
        TF_RETURN_IF_ERROR(writer->WriteTensor(
            full_name(strings::StrCat("invocation_results[", i, "][", j, "]")),
            result.return_values[j]));
      }
      if (result.end_of_input) {
        TF_RETURN_IF_ERROR(
            writer->WriteScalar(full_name(strings::StrCat("invocation_results[",
                                                          i, "].end_of_input")),
                                ""));
      }
    }
    return Status::OK();
  }

  Status RestoreInternal(IteratorContext* ctx,
                         IteratorStateReader* reader) override {
    mutex_lock l(*mu_);
    TF_RETURN_IF_ERROR(RestoreInput(ctx, reader, input_impl_));
    int64 invocation_results_size;
    TF_RETURN_IF_ERROR(reader->ReadScalar(full_name("invocation_results.size"),
                                          &invocation_results_size));
    for (size_t i = 0; i < invocation_results_size; i++) {
      invocation_results_.push_back(std::make_shared<InvocationResult>());
      auto& result = *invocation_results_.back();
      TF_RETURN_IF_ERROR(ReadStatusLocked(reader, i, &result.status));
      size_t num_return_values;
      {
        int64 size;
        TF_RETURN_IF_ERROR(reader->ReadScalar(
            full_name(strings::StrCat("invocation_results[", i, "].size")),
            &size));
        num_return_values = static_cast<size_t>(size);
        if (num_return_values != size) {
          return errors::InvalidArgument(strings::StrCat(
              full_name(strings::StrCat("invocation_results[", i, "].size")),
              ": ", size, " is not a valid value of type size_t."));
        }
      }
      result.return_values.reserve(num_return_values);
      for (size_t j = 0; j < num_return_values; j++) {
        result.return_values.emplace_back();
        TF_RETURN_IF_ERROR(reader->ReadTensor(
            full_name(strings::StrCat("invocation_results[", i, "][", j, "]")),
            &result.return_values.back()));
      }
      result.end_of_input = reader->Contains(full_name(
          strings::StrCat("invocation_results[", i, "].end_of_input")));
      result.notification.Notify();
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
      EXCLUSIVE_LOCKS_REQUIRED(*mu_) {
    if (!runner_thread_) {
      auto ctx_copy = std::make_shared<IteratorContext>(*ctx);
      runner_thread_.reset(ctx->env()->StartThread(
          {}, "tf_data_parallel_map",
          std::bind(&ParallelMapIterator::RunnerThread, this, ctx_copy)));
    }
  }

  void CallCompleted(const std::shared_ptr<IteratorContext>& ctx,
                     const std::shared_ptr<InvocationResult>& result)
      LOCKS_EXCLUDED(*mu_) {
    mutex_lock l(*mu_);
    num_calls_--;
    const auto& stats_aggregator = ctx->stats_aggregator();
    if (stats_aggregator) {
      stats_aggregator->AddScalar(
          strings::StrCat(key_prefix_, "::thread_utilization"),
          static_cast<float>(num_calls_) /
              static_cast<float>(num_parallel_calls_->value));
    }
    RecordBufferEnqueue(ctx.get(), result->return_values);
    result->notification.Notify();
    cond_var_->notify_all();
  }

  void CallFunction(const std::shared_ptr<IteratorContext>& ctx,
                    const std::shared_ptr<InvocationResult>& result)
      LOCKS_EXCLUDED(*mu_) {
    // Get the next input element.
    std::vector<Tensor> input_element;
    result->status =
        input_impl_->GetNext(ctx.get(), &input_element, &result->end_of_input);
    if (result->end_of_input || !result->status.ok()) {
      CallCompleted(ctx, result);
      return;
    }

    auto done = [this, ctx, result](Status status) {
      result->status.Update(status);
      CallCompleted(ctx, result);
    };

    // Apply the map function on `input_element`, storing the result in
    // `result->return_values`, and invoking `done` when finished.
    parallel_map_functor_->MapFunc(ctx.get(), prefix(),
                                   std::move(input_element),
                                   &result->return_values, std::move(done));
  }

  Status ProcessResult(IteratorContext* ctx,
                       const std::shared_ptr<InvocationResult>& result,
                       std::vector<Tensor>* out_tensors, bool* end_of_sequence)
      LOCKS_EXCLUDED(*mu_) {
    if (!result->end_of_input && result->status.ok()) {
      *out_tensors = std::move(result->return_values);
      RecordBufferDequeue(ctx, *out_tensors);
      *end_of_sequence = false;
      return Status::OK();
    }
    if (errors::IsOutOfRange(result->status)) {
      if (preserve_cardinality_) {
        // To guarantee that the transformation preserves the cardinality of the
        // dataset, we convert `OutOfRange` to `InvalidArgument` as the former
        // may be interpreted by a caller as the end of sequence.
        return errors::InvalidArgument(
            "Function invocation produced OutOfRangeError: ",
            result->status.error_message());
      } else {
        // `f` may deliberately raise `errors::OutOfRange` to indicate
        // that we should terminate the iteration early.
        *end_of_sequence = true;
        return Status::OK();
      }
    }
    *end_of_sequence = result->end_of_input;
    return result->status;
  }

  void RunnerThread(const std::shared_ptr<IteratorContext>& ctx)
      LOCKS_EXCLUDED(*mu_) {
    RecordStart(ctx.get());
    auto cleanup = gtl::MakeCleanup([this, ctx] { RecordStop(ctx.get()); });
    std::vector<std::shared_ptr<InvocationResult>> new_calls;
    {
      tf_shared_lock l(*mu_);  // mu_ == num_parallel_calls_->mu
      new_calls.reserve(num_parallel_calls_->value);
    }
    auto busy = [this]() EXCLUSIVE_LOCKS_REQUIRED(*mu_) -> bool {
      int64 num_parallel_calls = num_parallel_calls_->value;
      return num_calls_ >= num_parallel_calls ||
             invocation_results_.size() >= num_parallel_calls;
    };
    while (true) {
      {
        mutex_lock l(*mu_);
        while (!cancelled_ && busy()) {
          RecordStop(ctx.get());
          cond_var_->wait(l);
          RecordStart(ctx.get());
        }
        if (cancelled_) {
          return;
        }
        while (!busy()) {
          invocation_results_.push_back(std::make_shared<InvocationResult>());
          new_calls.push_back(invocation_results_.back());
          num_calls_++;
        }
        const auto& stats_aggregator = ctx->stats_aggregator();
        if (stats_aggregator) {
          stats_aggregator->AddScalar(
              strings::StrCat(key_prefix_, "::thread_utilization"),
              static_cast<float>(num_calls_) /
                  static_cast<float>(num_parallel_calls_->value));
        }
        cond_var_->notify_all();
      }
      for (const auto& call : new_calls) {
        CallFunction(ctx, call);
      }
      new_calls.clear();
    }
  }

  // Determines whether the caller needs to wait for a result. Upon returning
  // false, `result` will point to the result.
  bool ShouldWait(std::shared_ptr<InvocationResult>* result)
      EXCLUSIVE_LOCKS_REQUIRED(*mu_) {
    if (sloppy_) {
      for (auto it = invocation_results_.begin();
           it != invocation_results_.end(); ++it) {
        if ((*it)->notification.HasBeenNotified() &&
            (it == invocation_results_.begin() || !(*it)->end_of_input)) {
          std::swap(*result, *it);
          invocation_results_.erase(it);
          cond_var_->notify_all();
          return false;
        }
      }
    } else if (!invocation_results_.empty()) {
      std::swap(*result, invocation_results_.front());
      invocation_results_.pop_front();
      cond_var_->notify_all();
      return false;
    }
    return true;
  }

  Status WriteStatusLocked(IteratorStateWriter* writer, size_t index,
                           const Status& status)
      EXCLUSIVE_LOCKS_REQUIRED(*mu_) {
    TF_RETURN_IF_ERROR(
        writer->WriteScalar(CodeKey(index), static_cast<int64>(status.code())));
    if (!status.ok()) {
      TF_RETURN_IF_ERROR(
          writer->WriteScalar(ErrorMessageKey(index), status.error_message()));
    }
    return Status::OK();
  }

  Status ReadStatusLocked(IteratorStateReader* reader, size_t index,
                          Status* status) EXCLUSIVE_LOCKS_REQUIRED(*mu_) {
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
    return full_name(strings::StrCat("invocation_results[", index, "].code"));
  }

  string ErrorMessageKey(size_t index) {
    return full_name(
        strings::StrCat("invocation_results[", index, "].error_message"));
  }

  const DatasetBase* const input_dataset_;  // Not owned.
  std::unique_ptr<ParallelMapFunctor> parallel_map_functor_;
  // Used for coordination between the main thread and the runner thread.
  const std::shared_ptr<mutex> mu_;
  // Used for coordination between the main thread and the runner thread. In
  // particular, the runner thread should only schedule new calls when the
  // number of in-flight calls is less than the user specified level of
  // parallelism and there are slots available in the `invocation_results_`
  // buffer.
  const std::shared_ptr<condition_variable> cond_var_;
  // Identifies the maximum number of parallel calls.
  const std::shared_ptr<model::SharedState> num_parallel_calls_;
  // Determines whether outputs can be produced in non-deterministic order.
  const bool sloppy_;
  const bool preserve_cardinality_;
  // Counts the number of outstanding calls.
  int64 num_calls_ GUARDED_BY(*mu_) = 0;
  std::unique_ptr<IteratorBase> input_impl_;
  // Buffer for storing the invocation results.
  std::deque<std::shared_ptr<InvocationResult>> invocation_results_
      GUARDED_BY(*mu_);
  std::unique_ptr<Thread> runner_thread_ GUARDED_BY(*mu_);
  bool cancelled_ GUARDED_BY(*mu_) = false;
  string key_prefix_;
};

}  // namespace

std::unique_ptr<IteratorBase> NewParallelMapIterator(
    const DatasetBaseIterator::BaseParams& params,
    const DatasetBase* input_dataset,
    std::unique_ptr<ParallelMapFunctor> parallel_map_functor,
    int32 num_parallel_calls, bool sloppy, bool preserve_cardinality) {
  return absl::make_unique<ParallelMapIterator>(
      params, input_dataset,
      ParallelMapIterator::Params{std::move(parallel_map_functor),
                                  num_parallel_calls, sloppy,
                                  preserve_cardinality});
}

}  // namespace data
}  // namespace tensorflow
