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
#include <atomic>
#include <deque>
#include <utility>

#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/framework/partial_tensor_shape.h"
#include "tensorflow/core/framework/stats_aggregator.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/kernels/data/captured_function.h"
#include "tensorflow/core/kernels/data/dataset_utils.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/gtl/cleanup.h"
#include "tensorflow/core/lib/random/random.h"
#include "tensorflow/core/util/ptr_util.h"

namespace tensorflow {
namespace data {
namespace {

// See documentation in ../../ops/dataset_ops.cc for a high-level
// description of the following op.

// The motivation for creating an alternative implementation of parallel
// interleave is to decouple the degree of parallelism from the cycle length.
// This makes it possible to change the degree of parallelism (e.g. through
// auto-tuning) without changing the cycle length (which would change the order
// in which elements are produced).
//
// Furthermore, this class favors modularity over extended functionality. In
// particular, it refrains from implementing configurable buffering of output
// elements and prefetching of input iterators, relying on other parts of
// tf.data to provide this functionality if necessary.
//
// The above design choices were made with automated optimizations in mind,
// isolating the degree of parallelism as the single tunable knob of this
// implementation.
class ParallelInterleaveDatasetOp : public UnaryDatasetOpKernel {
 public:
  explicit ParallelInterleaveDatasetOp(OpKernelConstruction* ctx)
      : UnaryDatasetOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("f", &interleave_func_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("output_types", &output_types_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("output_shapes", &output_shapes_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("sloppy", &sloppy_));
  }

  void MakeDataset(OpKernelContext* ctx, DatasetBase* input,
                   DatasetBase** output) override {
    int64 cycle_length = 0;
    OP_REQUIRES_OK(ctx,
                   ParseScalarArgument(ctx, "cycle_length", &cycle_length));
    OP_REQUIRES(ctx, cycle_length > 0,
                errors::InvalidArgument("`cycle_length` must be > 0"));

    int64 block_length = 0;
    OP_REQUIRES_OK(ctx,
                   ParseScalarArgument(ctx, "block_length", &block_length));
    OP_REQUIRES(ctx, block_length > 0,
                errors::InvalidArgument("`block_length` must be > 0"));

    int64 num_parallel_calls;
    OP_REQUIRES_OK(ctx, ParseScalarArgument(ctx, "num_parallel_calls",
                                            &num_parallel_calls));
    OP_REQUIRES(
        ctx, num_parallel_calls > 0 || num_parallel_calls == model::kAutoTune,
        errors::InvalidArgument(
            "num_parallel_calls must be greater than zero."));
    OP_REQUIRES(
        ctx, num_parallel_calls <= cycle_length,
        errors::InvalidArgument(
            "num_parallel_calls must less than or equal to cycle_length."));

    std::unique_ptr<CapturedFunction> captured_func;
    OP_REQUIRES_OK(
        ctx, CapturedFunction::Create(interleave_func_, ctx, "other_arguments",
                                      &captured_func));

    *output =
        new Dataset(ctx, input, interleave_func_, std::move(captured_func),
                    cycle_length, block_length, num_parallel_calls, sloppy_,
                    output_types_, output_shapes_);
  }

 private:
  class Dataset : public DatasetBase {
   public:
    Dataset(OpKernelContext* ctx, const DatasetBase* input,
            const NameAttrList& func,
            std::unique_ptr<CapturedFunction> captured_func, int64 cycle_length,
            int64 block_length, int64 num_parallel_calls, bool sloppy,
            const DataTypeVector& output_types,
            const std::vector<PartialTensorShape>& output_shapes)
        : DatasetBase(DatasetContext(ctx)),
          input_(input),
          interleave_func_(func),
          captured_func_(std::move(captured_func)),
          cycle_length_(cycle_length),
          block_length_(block_length),
          num_parallel_calls_(num_parallel_calls),
          sloppy_(sloppy),
          output_types_(output_types),
          output_shapes_(output_shapes) {
      input_->Ref();
    }

    ~Dataset() override { input_->Unref(); }

    std::unique_ptr<IteratorBase> MakeIteratorInternal(
        const string& prefix) const override {
      return MakeUnique<ParallelInterleaveIterator>(
          ParallelInterleaveIterator::Params{
              this, strings::StrCat(prefix, "::ParallelInterleaveV2")},
          sloppy_);
    }

    const DataTypeVector& output_dtypes() const override {
      return output_types_;
    }

    const std::vector<PartialTensorShape>& output_shapes() const override {
      return output_shapes_;
    }

    string DebugString() const override {
      return "ParallelInterleaveDatasetV2Op::Dataset";
    }

   protected:
    Status AsGraphDefInternal(SerializationContext* ctx,
                              DatasetGraphDefBuilder* b,
                              Node** output) const override {
      TF_RETURN_IF_ERROR(b->AddFunction(ctx, interleave_func_.name()));
      Node* input_node;
      TF_RETURN_IF_ERROR(b->AddInputDataset(ctx, input_, &input_node));
      Node* cycle_length_node;
      TF_RETURN_IF_ERROR(b->AddScalar(cycle_length_, &cycle_length_node));
      Node* block_length_node;
      TF_RETURN_IF_ERROR(b->AddScalar(block_length_, &block_length_node));
      Node* num_parallel_calls_node;
      TF_RETURN_IF_ERROR(
          b->AddScalar(num_parallel_calls_, &num_parallel_calls_node));
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
      b->BuildAttrValue(interleave_func_, &f);
      AttrValue other_arguments_types_attr;
      b->BuildAttrValue(other_arguments_types, &other_arguments_types_attr);
      AttrValue sloppy_attr;
      b->BuildAttrValue(sloppy_, &sloppy_attr);

      TF_RETURN_IF_ERROR(
          b->AddDataset(this,
                        {{0, input_node},
                         {2, cycle_length_node},
                         {3, block_length_node},
                         {4, num_parallel_calls_node}},
                        {{1, other_arguments}},
                        {{"f", f},
                         {"Targuments", other_arguments_types_attr},
                         {"sloppy", sloppy_attr}},
                        output));
      return Status::OK();
    }

   private:
    class ParallelInterleaveIterator : public DatasetIterator<Dataset> {
     public:
      explicit ParallelInterleaveIterator(const Params& params, bool sloppy)
          : DatasetIterator<Dataset>(params),
            mu_(std::make_shared<mutex>()),
            cond_var_(std::make_shared<condition_variable>()),
            num_parallel_calls_(std::make_shared<model::SharedState>(
                params.dataset->num_parallel_calls_, mu_, cond_var_)),
            sloppy_(sloppy),
            args_list_(params.dataset->cycle_length_),
            current_elements_(params.dataset->cycle_length_),
            element_in_use_(params.dataset->cycle_length_, false),
            thread_pool_(new thread::ThreadPool(
                Env::Default(), ThreadOptions(),
                "data_parallel_interleave_worker_pool",
                dataset()->cycle_length_ /* num_threads */,
                false /* low_latency_hint */)) {
        std::vector<string> components =
            str_util::Split(params.prefix, "::", str_util::SkipEmpty());
        prefix_end_ = components.back();
      }

      ~ParallelInterleaveIterator() override {
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
          num_parallel_calls_->value = dataset()->cycle_length_;
        }
        TF_RETURN_IF_ERROR(
            dataset()->input_->MakeIterator(ctx, prefix(), &input_impl_));
        return dataset()->captured_func_->Instantiate(
            ctx, &instantiated_captured_func_);
      }

      Status GetNextInternal(IteratorContext* ctx,
                             std::vector<Tensor>* out_tensors,
                             bool* end_of_sequence) override {
        std::shared_ptr<InvocationResult> result;
        do {
          result.reset();
          {
            mutex_lock l(*mu_);
            EnsureRunnerThreadStarted(ctx);
            while (ShouldWait(&result)) {
              RecordStop(ctx);
              cond_var_->wait(l);
              RecordStart(ctx);
            }
            if (!result) {
              *end_of_sequence = true;
              return Status::OK();
            }
          }
          RecordStop(ctx);
          result->notification.WaitForNotification();
          RecordStart(ctx);
        } while (result->skip);

        if (result->status.ok()) {
          *out_tensors = std::move(result->return_values);
          RecordBufferDequeue(ctx, *out_tensors);
        }
        *end_of_sequence = false;
        return result->status;
      }

     protected:
      std::shared_ptr<model::Node> CreateNode(
          IteratorContext* ctx, model::Node::Args args) const override {
        return model::MakeAsyncInterleaveManyNode(
            std::move(args),
            {model::MakeParameter("parallelism", num_parallel_calls_, /*min=*/1,
                                  /*max=*/dataset()->cycle_length_)});
      }

      Status SaveInternal(IteratorStateWriter* writer) override {
        mutex_lock l(*mu_);
        // Wait for all in-flight calls to complete.
        while (num_calls_ > 0) {
          cond_var_->wait(l);
        }
        CHECK_EQ(num_calls_, 0);
        TF_RETURN_IF_ERROR(SaveInput(writer, input_impl_));
        TF_RETURN_IF_ERROR(writer->WriteScalar(
            full_name("invocation_results.size"), invocation_results_.size()));
        for (size_t i = 0; i < invocation_results_.size(); i++) {
          std::shared_ptr<InvocationResult> result = invocation_results_[i];
          TF_RETURN_IF_ERROR(WriteStatusLocked(writer, i, result->status));
          TF_RETURN_IF_ERROR(writer->WriteScalar(
              full_name(strings::StrCat("invocation_results[", i, "].size")),
              result->return_values.size()));
          for (size_t j = 0; j < result->return_values.size(); j++) {
            TF_RETURN_IF_ERROR(writer->WriteTensor(
                full_name(
                    strings::StrCat("invocation_results[", i, "][", j, "]")),
                result->return_values[j]));
          }
          if (result->skip) {
            TF_RETURN_IF_ERROR(writer->WriteScalar(
                full_name(strings::StrCat("invocation_results[", i, "].skip")),
                ""));
          }
        }
        TF_RETURN_IF_ERROR(
            writer->WriteScalar(full_name("cycle_index"), cycle_index_));
        if (end_of_input_) {
          TF_RETURN_IF_ERROR(
              writer->WriteScalar(full_name("end_of_input"), ""));
        }
        TF_RETURN_IF_ERROR(
            writer->WriteScalar(full_name("num_open"), num_open_));
        TF_RETURN_IF_ERROR(WriteCurrentElements(writer));
        return Status::OK();
      }

      Status RestoreInternal(IteratorContext* ctx,
                             IteratorStateReader* reader) override {
        mutex_lock l(*mu_);
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
            TF_RETURN_IF_ERROR(reader->ReadScalar(
                full_name(strings::StrCat("invocation_results[", i, "].size")),
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
          result->skip = reader->Contains(
              full_name(strings::StrCat("invocation_results[", i, "].skip")));
          result->notification.Notify();
        }
        TF_RETURN_IF_ERROR(
            reader->ReadScalar(full_name("cycle_index"), &cycle_index_));
        if (reader->Contains(full_name("end_of_input"))) end_of_input_ = true;
        TF_RETURN_IF_ERROR(
            reader->ReadScalar(full_name("num_open"), &num_open_));
        TF_RETURN_IF_ERROR(ReadCurrentElements(ctx, reader));
        return Status::OK();
      }

     private:
      struct InvocationResult {
        Notification notification;  // used for coordination with the consumer
        Status status;              // the invocation status
        std::vector<Tensor> return_values;  // the invocation result values
        bool skip;  // if set the result should be skipped
      };

      void EnsureRunnerThreadStarted(IteratorContext* ctx)
          EXCLUSIVE_LOCKS_REQUIRED(*mu_) {
        if (!runner_thread_) {
          std::shared_ptr<IteratorContext> new_ctx(new IteratorContext(*ctx));
          runner_thread_.reset(ctx->env()->StartThread(
              {}, "tf_data_parallel_interleave_runner",
              [this, new_ctx]() { RunnerThread(new_ctx); }));
        }
      }

      // Fetches up to `results.size()` outputs from the cycle element at
      // position `cycle_index`.
      //
      // If end of input is encountered, the `skip` field of the invocation
      // result is used to identify results that should be skipped.
      void FetchOutputs(
          const std::shared_ptr<IteratorContext>& ctx, int64 cycle_index,
          const std::vector<std::shared_ptr<InvocationResult>>& results)
          LOCKS_EXCLUDED(*mu_) {
        RecordStart(ctx.get());
        auto cleanup = gtl::MakeCleanup([this, ctx] { RecordStop(ctx.get()); });
        bool end_of_input = false;
        for (auto& result : results) {
          if (!end_of_input) {
            result->status = current_elements_[cycle_index]->GetNext(
                ctx.get(), &result->return_values, &end_of_input);
          }
          if (end_of_input) {
            result->skip = true;
          }
          RecordBufferEnqueue(ctx.get(), result->return_values);
          {
            mutex_lock l(*mu_);
            result->notification.Notify();
            cond_var_->notify_all();
          }
          if (!result->status.ok()) {
            break;
          }
        }

        // Release the ownership of the cycle element iterator, closing the
        // iterator if end of input was encountered.
        if (end_of_input) {
          current_elements_[cycle_index].reset();
        }
        mutex_lock l(*mu_);
        element_in_use_[cycle_index] = false;
        num_calls_--;
        const auto& stats_aggregator = ctx->stats_aggregator();
        if (stats_aggregator) {
          stats_aggregator->AddScalar(
              strings::StrCat(prefix_end_, "::active_parallel_calls"),
              static_cast<float>(num_calls_));
        }
        if (end_of_input) {
          args_list_[cycle_index].clear();
          num_open_--;
        }
        cond_var_->notify_all();
      }

      // Method responsible for 1) creating iterators out of input elements, 2)
      // determining the order in which elements are fetched from the iterators,
      // and 3) scheduling the fetching of the elements to a threadpool.
      //
      // This method runs in the `runner_thread` background thread.
      void RunnerThread(const std::shared_ptr<IteratorContext>& ctx) {
        RecordStart(ctx.get());
        auto cleanup = gtl::MakeCleanup([this, ctx] { RecordStop(ctx.get()); });
        auto busy = [this]() EXCLUSIVE_LOCKS_REQUIRED(*mu_) -> bool {
          return element_in_use_[cycle_index_] ||
                 num_calls_ >= num_parallel_calls_->value ||
                 invocation_results_.size() >=
                     dataset()->cycle_length_ * dataset()->block_length_;
        };
        while (true) {
          mutex_lock l(*mu_);
          // Wait until this thread is cancelled, the end of input has been
          // reached, or the cycle element at the `cycle_index_` position is
          // not in use and there is space in the `invocation_results_` queue.
          while (!cancelled_ && (!end_of_input_ || num_open_ > 0) && busy()) {
            RecordStop(ctx.get());
            cond_var_->wait(l);
            RecordStart(ctx.get());
          }

          if (cancelled_ || (end_of_input_ && num_open_ == 0)) {
            return;
          }

          while ((!end_of_input_ || num_open_ > 0) && !busy()) {
            if (!current_elements_[cycle_index_]) {
              // Try to create a new iterator from the next input element.
              Status status = input_impl_->GetNext(
                  ctx.get(), &args_list_[cycle_index_], &end_of_input_);
              if (!status.ok()) {
                invocation_results_.emplace_back(new InvocationResult());
                std::shared_ptr<InvocationResult>& result =
                    invocation_results_.back();
                result->status.Update(status);
                result->notification.Notify();
                break;
              }
              if (!end_of_input_) {
                Status status = MakeIteratorFromInputElement(
                    ctx.get(), args_list_[cycle_index_], cycle_index_,
                    *instantiated_captured_func_, prefix(),
                    &current_elements_[cycle_index_]);
                if (!status.ok()) {
                  invocation_results_.emplace_back(new InvocationResult());
                  std::shared_ptr<InvocationResult>& result =
                      invocation_results_.back();
                  result->status.Update(status);
                  result->notification.Notify();
                  break;
                }
                ++num_open_;
              }
            }
            if (current_elements_[cycle_index_]) {
              // Pre-allocate invocation results for outputs to be fetched
              // and then fetch the outputs asynchronously.
              std::vector<std::shared_ptr<InvocationResult>> results;
              results.reserve(dataset()->block_length_);
              for (int i = 0; i < dataset()->block_length_; ++i) {
                invocation_results_.emplace_back(new InvocationResult());
                results.push_back(invocation_results_.back());
              }
              num_calls_++;
              element_in_use_[cycle_index_] = true;
              thread_pool_->Schedule(
                  std::bind(&ParallelInterleaveIterator::FetchOutputs, this,
                            ctx, cycle_index_, std::move(results)));
            }
            cycle_index_ = (cycle_index_ + 1) % dataset()->cycle_length_;
          }
          const auto& stats_aggregator = ctx->stats_aggregator();
          if (stats_aggregator) {
            // TODO(shivaniagrawal): add `parallel_calls_utilization` in the
            // monitoring code or as histogram at fixed time intervals.
            stats_aggregator->AddScalar(
                strings::StrCat(prefix_end_, "::active_parallel_calls"),
                static_cast<float>(num_calls_));
            stats_aggregator->AddScalar(
                strings::StrCat(prefix_end_, "::num_parallel_calls"),
                static_cast<float>(num_parallel_calls_->value));
          }
          cond_var_->notify_all();
        }
      }

      // Determines whether the caller needs to wait for a result. Upon
      // returning false, `result` will either be NULL if end of input has been
      // reached or point to the result.
      bool ShouldWait(std::shared_ptr<InvocationResult>* result)
          EXCLUSIVE_LOCKS_REQUIRED(*mu_) {
        if (sloppy_) {
          for (auto it = invocation_results_.begin();
               it != invocation_results_.end(); ++it) {
            if ((*it)->notification.HasBeenNotified()) {
              std::swap(*result, *it);
              invocation_results_.erase(it);
              cond_var_->notify_all();
              return false;
            }
          }
          return !invocation_results_.empty() ||
                 (!end_of_input_ || num_open_ > 0);
        } else {
          if (!invocation_results_.empty()) {
            std::swap(*result, invocation_results_.front());
            invocation_results_.pop_front();
            cond_var_->notify_all();
            return false;
          }
          return (!end_of_input_ || num_open_ > 0);
        }
      }

      Status WriteStatusLocked(IteratorStateWriter* writer, size_t index,
                               const Status& status)
          EXCLUSIVE_LOCKS_REQUIRED(*mu_) {
        TF_RETURN_IF_ERROR(writer->WriteScalar(
            CodeKey(index), static_cast<int64>(status.code())));
        if (!status.ok()) {
          TF_RETURN_IF_ERROR(writer->WriteScalar(ErrorMessageKey(index),
                                                 status.error_message()));
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
        return full_name(
            strings::StrCat("invocation_results[", index, "].code"));
      }

      string ErrorMessageKey(size_t index) {
        return full_name(
            strings::StrCat("invocation_results[", index, "].error_message"));
      }

      Status WriteCurrentElements(IteratorStateWriter* writer)
          EXCLUSIVE_LOCKS_REQUIRED(*mu_) {
        for (int idx = 0; idx < current_elements_.size(); idx++) {
          if (current_elements_[idx]) {
            TF_RETURN_IF_ERROR(SaveInput(writer, current_elements_[idx]));
            TF_RETURN_IF_ERROR(writer->WriteScalar(
                full_name(strings::StrCat("args_size[", idx, "]")),
                args_list_[idx].size()));
            for (int i = 0; i < args_list_[idx].size(); i++) {
              TF_RETURN_IF_ERROR(writer->WriteTensor(
                  full_name(strings::StrCat("args_list_[", idx, "][", i, "]")),
                  args_list_[idx][i]));
            }
          }
        }
        return Status::OK();
      }

      Status ReadCurrentElements(IteratorContext* ctx,
                                 IteratorStateReader* reader)
          EXCLUSIVE_LOCKS_REQUIRED(*mu_) {
        for (int idx = 0; idx < current_elements_.size(); idx++) {
          if (reader->Contains(
                  full_name(strings::StrCat("args_size[", idx, "]")))) {
            int64 args_size;
            TF_RETURN_IF_ERROR(reader->ReadScalar(
                full_name(strings::StrCat("args_size[", idx, "]")),
                &args_size));
            args_list_[idx].resize(args_size);
            for (int i = 0; i < args_size; i++) {
              TF_RETURN_IF_ERROR(reader->ReadTensor(
                  full_name(strings::StrCat("args_list_[", idx, "][", i, "]")),
                  &args_list_[idx][i]));
            }
            TF_RETURN_IF_ERROR(MakeIteratorFromInputElement(
                ctx, args_list_[idx], idx, *instantiated_captured_func_.get(),
                prefix(), &current_elements_[idx]));
            TF_RETURN_IF_ERROR(
                RestoreInput(ctx, reader, current_elements_[idx]));
          } else {
            current_elements_[idx].reset();
          }
        }
        return Status::OK();
      }

      // Used for coordination between the main thread, the runner thread, and
      // the worker threads.
      const std::shared_ptr<mutex> mu_;

      // Used for coordination between the main thread, the runner thread, and
      // the worker threads. In particular, the runner thread should only
      // schedule new calls when the number of in-flight calls is less than the
      // user specified level of parallelism, there are slots available in the
      // `invocation_results_` buffer, the current cycle element is not in use,
      // and there are elements left to be fetched.
      const std::shared_ptr<condition_variable> cond_var_;

      // Identifies the maximum number of parallel calls.
      const std::shared_ptr<model::SharedState> num_parallel_calls_;

      // Determines whether outputs can be produced in non-deterministic order.
      const bool sloppy_;

      // Iterator for input elements.
      std::unique_ptr<IteratorBase> input_impl_ GUARDED_BY(*mu_);

      // Identifies current cycle element.
      int64 cycle_index_ = 0;

      // Arguments for creating an iterator for cycle elements.
      std::vector<std::vector<Tensor>> args_list_ GUARDED_BY(*mu_);

      // Iterators for the current cycle elements. Concurrent access is
      // protected by `element_in_use_`.
      std::vector<std::unique_ptr<IteratorBase>> current_elements_;

      // Identifies cycle elements that are in use by worker threads.
      std::vector<bool> element_in_use_ GUARDED_BY(*mu_);

      // Buffer for storing the invocation results.
      std::deque<std::shared_ptr<InvocationResult>> invocation_results_
          GUARDED_BY(*mu_);

      // Identifies whether end of input has been reached.
      bool end_of_input_ GUARDED_BY(*mu_) = false;

      // Identifies the number of open iterators.
      int64 num_open_ GUARDED_BY(*mu_) = 0;

      // Identifies the number of outstanding calls.
      int64 num_calls_ GUARDED_BY(*mu_) = 0;

      std::unique_ptr<thread::ThreadPool> thread_pool_;
      std::unique_ptr<Thread> runner_thread_ GUARDED_BY(*mu_);

      // Identifies whether background activity should be cancelled.
      bool cancelled_ GUARDED_BY(*mu_) = false;
      string prefix_end_;
      std::unique_ptr<InstantiatedCapturedFunction> instantiated_captured_func_;
    };

    const DatasetBase* const input_;
    const NameAttrList interleave_func_;
    const std::unique_ptr<CapturedFunction> captured_func_;
    const int64 cycle_length_;
    const int64 block_length_;
    const int64 num_parallel_calls_;
    const bool sloppy_;
    const DataTypeVector output_types_;
    const std::vector<PartialTensorShape> output_shapes_;
  };

  bool sloppy_;
  DataTypeVector output_types_;
  std::vector<PartialTensorShape> output_shapes_;
  NameAttrList interleave_func_;
};

REGISTER_KERNEL_BUILDER(Name("ParallelInterleaveDatasetV2").Device(DEVICE_CPU),
                        ParallelInterleaveDatasetOp);

}  // namespace
}  // namespace data
}  // namespace tensorflow
