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
#include <memory>
#include <utility>

#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/common_runtime/metrics.h"
#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/framework/partial_tensor_shape.h"
#include "tensorflow/core/framework/stats_aggregator.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/kernels/data/captured_function.h"
#include "tensorflow/core/kernels/data/dataset_utils.h"
#include "tensorflow/core/kernels/data/stats_utils.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/gtl/cleanup.h"
#include "tensorflow/core/lib/random/random.h"
#include "tensorflow/core/platform/cpu_info.h"

namespace tensorflow {
namespace data {
namespace {

constexpr char kDatasetName[] = "ParallelInterleaveV2";

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
// elements and prefetching of input iterators.
class ParallelInterleaveDatasetOp : public UnaryDatasetOpKernel {
 public:
  explicit ParallelInterleaveDatasetOp(OpKernelConstruction* ctx)
      : UnaryDatasetOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("f", &interleave_func_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("output_types", &output_types_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("output_shapes", &output_shapes_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("sloppy", &sloppy_));
    OP_REQUIRES_OK(ctx,
                   CreateFunctionLibraryDefinition(
                       ctx->function_library()->GetFunctionLibraryDefinition(),
                       interleave_func_.name(), &lib_def_));
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
    CapturedFunction::Params params;
    params.lib_def = lib_def_;
    OP_REQUIRES_OK(
        ctx, CapturedFunction::Create(interleave_func_, ctx, "other_arguments",
                                      std::move(params), &captured_func));

    if (num_parallel_calls == model::kAutoTune) {
      metrics::RecordTFDataAutotune(kDatasetName);
    }

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
      return absl::make_unique<ParallelInterleaveIterator>(
          ParallelInterleaveIterator::Params{
              this, strings::StrCat(prefix, "::", kDatasetName)},
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
      Node* input_node;
      TF_RETURN_IF_ERROR(b->AddInputDataset(ctx, input_, &input_node));
      Node* cycle_length_node;
      TF_RETURN_IF_ERROR(b->AddScalar(cycle_length_, &cycle_length_node));
      Node* block_length_node;
      TF_RETURN_IF_ERROR(b->AddScalar(block_length_, &block_length_node));
      Node* num_parallel_calls_node;
      TF_RETURN_IF_ERROR(
          b->AddScalar(num_parallel_calls_, &num_parallel_calls_node));
      std::vector<Node*> other_arguments;
      DataTypeVector other_arguments_types;
      TF_RETURN_IF_ERROR(captured_func_->AddToGraph(ctx, b, &other_arguments,
                                                    &other_arguments_types));
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
      ParallelInterleaveIterator(const Params& params, bool sloppy)
          : DatasetIterator<Dataset>(params),
            mu_(std::make_shared<mutex>()),
            cond_var_(std::make_shared<condition_variable>()),
            num_parallel_calls_(std::make_shared<model::SharedState>(
                params.dataset->num_parallel_calls_, mu_, cond_var_)),
            sloppy_(sloppy),
            current_elements_(params.dataset->cycle_length_),
            thread_pool_(absl::make_unique<thread::ThreadPool>(
                Env::Default(), ThreadOptions(),
                "data_parallel_interleave_worker_pool",
                port::NumSchedulableCPUs() /* num_threads */,
                false /* low_latency_hint */)) {
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
        std::shared_ptr<Result> result;
        {
          mutex_lock l(*mu_);
          EnsureThreadsStarted(ctx);
          while (!Consume(&result)) {
            RecordStop(ctx);
            cond_var_->wait(l);
            RecordStart(ctx);
          }
        }
        if (!result) {
          *end_of_sequence = true;
          return Status::OK();
        }
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
                                  /*max=*/port::NumSchedulableCPUs())});
      }

      Status SaveInternal(IteratorStateWriter* writer) override {
        mutex_lock l(*mu_);
        // Wait for all in-flight calls to complete.
        while (num_calls_ > 0) {
          cond_var_->wait(l);
        }
        DCHECK_EQ(num_calls_, 0);
        TF_RETURN_IF_ERROR(SaveInput(writer, input_impl_));
        TF_RETURN_IF_ERROR(
            writer->WriteScalar(full_name("block_index"), block_index_));
        TF_RETURN_IF_ERROR(
            writer->WriteScalar(full_name("cycle_index"), cycle_index_));
        if (end_of_input_) {
          TF_RETURN_IF_ERROR(
              writer->WriteScalar(full_name("end_of_input"), ""));
        }
        TF_RETURN_IF_ERROR(writer->WriteScalar(full_name("element_id_counter"),
                                               element_id_counter_));
        TF_RETURN_IF_ERROR(
            writer->WriteScalar(full_name("num_open"), num_open_));
        TF_RETURN_IF_ERROR(WriteCurrentElements(writer));
        TF_RETURN_IF_ERROR(WriteFutureElements(writer));
        return Status::OK();
      }

      Status RestoreInternal(IteratorContext* ctx,
                             IteratorStateReader* reader) override {
        mutex_lock l(*mu_);
        TF_RETURN_IF_ERROR(RestoreInput(ctx, reader, input_impl_));
        TF_RETURN_IF_ERROR(
            reader->ReadScalar(full_name("block_index"), &block_index_));
        TF_RETURN_IF_ERROR(
            reader->ReadScalar(full_name("cycle_index"), &cycle_index_));
        TF_RETURN_IF_ERROR(reader->ReadScalar(full_name("element_id_counter"),
                                              &element_id_counter_));
        if (reader->Contains(full_name("end_of_input"))) end_of_input_ = true;
        TF_RETURN_IF_ERROR(
            reader->ReadScalar(full_name("num_open"), &num_open_));
        TF_RETURN_IF_ERROR(ReadCurrentElements(ctx, reader));
        TF_RETURN_IF_ERROR(ReadFutureElements(ctx, reader));
        return Status::OK();
      }

     private:
      // Represents the result of fetching an element from a dataset.
      struct Result {
        Status status;
        std::vector<Tensor> return_values;
        // Indicates whether the result is ready to be consumed.
        bool is_ready = false;
      };

      // The interleave transformation repeatedly inputs elements, applies the
      // user-provided function to transform the input elements to datasets, and
      // interleaves the elements of these datasets as its output.
      //
      // This structure represents an input element and derived state.
      struct Element {
        // Unique identifier, needed to support checkpointing.
        int64 id;
        // The actual input element.
        std::vector<Tensor> inputs;
        // Iterator created from the input element.
        std::unique_ptr<IteratorBase> iterator;
        mutex mu;
        // Buffer for storing the outputs of `iterator`.
        std::deque<std::shared_ptr<Result>> results GUARDED_BY(mu);
        // Indicates whether the element is used by a worker thread.
        bool in_use = false;
      };

      // Advances the position in the interleave cycle to the next cycle
      // element.
      void AdvanceToNextInCycle() EXCLUSIVE_LOCKS_REQUIRED(*mu_) {
        block_index_ = 0;
        cycle_index_ = (cycle_index_ + 1) % dataset()->cycle_length_;
      }

      // Advances the position in the interleave cycle by one.
      void AdvancePosition() EXCLUSIVE_LOCKS_REQUIRED(*mu_) {
        ++block_index_;
        if (block_index_ == dataset()->block_length_) {
          AdvanceToNextInCycle();
        }
      }

      // Consumes a result (if available), returning an indication of whether
      // a result is available. If `true` is returned, `result` either
      // points to a valid result or is null if end of input has been reached.
      bool Consume(std::shared_ptr<Result>* result)
          EXCLUSIVE_LOCKS_REQUIRED(*mu_) {
        if (!sloppy_) {
          return ConsumeHelper(result);
        }
        // If we are allowed to be sloppy (i.e. return results out of order),
        // try to find an element in the cycle that has a result available.
        for (int i = 0; i < dataset()->cycle_length_; ++i) {
          if (ConsumeHelper(result)) {
            return true;
          }
          AdvanceToNextInCycle();
        }
        return false;
      }

      bool ConsumeHelper(std::shared_ptr<Result>* result)
          EXCLUSIVE_LOCKS_REQUIRED(*mu_) {
        while (true) {
          std::shared_ptr<Element> element = current_elements_[cycle_index_];
          if (element) {
            mutex_lock l(element->mu);
            if (!element->results.empty()) {
              if (element->results.front()->is_ready) {
                // We found a result.
                std::swap(*result, element->results.front());
                element->results.pop_front();
                AdvancePosition();
                cond_var_->notify_all();
                return true;
              } else {
                // Wait for the result to become ready.
                return false;
              }
            } else if (!element->iterator) {
              // We reached the end of input for this element. Reset
              // it and move on to the next cycle element.
              current_elements_[cycle_index_].reset();
              AdvanceToNextInCycle();
              cond_var_->notify_all();
              continue;
            } else {
              // Wait for the iterator to produce a result.
              return false;
            }
          } else {
            if (!future_elements_.empty() || !end_of_input_) {
              // Wait for an element to be created.
              return false;
            }
            // No new elements will be created; try to find a
            // non-empty element in the cycle.
            for (int i = 0; i < dataset()->cycle_length_; ++i) {
              AdvanceToNextInCycle();
              if (current_elements_[cycle_index_]) {
                break;
              }
            }
            if (current_elements_[cycle_index_]) {
              continue;
            }
            // End of input has been reached.
            return true;
          }
        }
      }

      // Manages current cycle elements, creating new iterators as needed and
      // asynchronously fetching results from existing iterators.
      //
      // This method runs in the `current_elements_manager_` background thread.
      void CurrentElementsManager(const std::shared_ptr<IteratorContext>& ctx) {
        RecordStart(ctx.get());
        auto cleanup = gtl::MakeCleanup([this, ctx] { RecordStop(ctx.get()); });
        auto busy = [this]() EXCLUSIVE_LOCKS_REQUIRED(*mu_) -> bool {
          const bool has_more_elements =
              !future_elements_.empty() || !end_of_input_;
          const int block_length = dataset()->block_length_;
          bool all_elements_busy = true;
          for (auto& element : current_elements_) {
            if (!element) {
              if (has_more_elements) {
                all_elements_busy = false;
                break;
              }
            } else {
              mutex_lock l(element->mu);
              if (!element->in_use && element->iterator &&
                  element->results.size() < block_length) {
                all_elements_busy = false;
                break;
              }
            }
          }
          return all_elements_busy || num_calls_ >= num_parallel_calls_->value;
        };
        while (true) {
          mutex_lock l(*mu_);

          // Wait until this thread is cancelled, the end of input has been
          // reached.
          while (!cancelled_ && (!end_of_input_ || num_open_ > 0) && busy()) {
            RecordStop(ctx.get());
            cond_var_->wait(l);
            RecordStart(ctx.get());
          }

          if (cancelled_ ||
              (future_elements_.empty() && end_of_input_ && num_open_ == 0)) {
            return;
          }

          for (int i = 0; i < dataset()->cycle_length_; ++i) {
            int idx = (cycle_index_ + i) % dataset()->cycle_length_;
            if (!current_elements_[idx]) {
              if (!future_elements_.empty()) {
                current_elements_[idx] = std::move(future_elements_.back());
                future_elements_.pop_back();
              } else {
                current_elements_[idx] = MakeElement(ctx);
                if (!current_elements_[idx]) {
                  continue;
                }
              }
            }
            std::shared_ptr<Element> element = current_elements_[idx];
            if (!element->in_use && element->iterator) {
              int64 num_results;
              {
                mutex_lock l(element->mu);
                num_results =
                    dataset()->block_length_ - element->results.size();
              }
              if (num_results > 0) {
                num_calls_++;
                element->in_use = true;
                thread_pool_->Schedule(
                    std::bind(&ParallelInterleaveIterator::FetchResults, this,
                              ctx, std::move(element), num_results));
              }
            }
          }
          const auto& stats_aggregator = ctx->stats_aggregator();
          if (stats_aggregator) {
            stats_aggregator->AddScalar(
                stats_utils::ThreadUtilizationScalarName(
                    dataset()->node_name()),
                static_cast<float>(num_calls_) /
                    static_cast<float>(num_parallel_calls_->value),
                num_elements());
          }
          cond_var_->notify_all();
        }
      }

      void EnsureThreadsStarted(IteratorContext* ctx)
          EXCLUSIVE_LOCKS_REQUIRED(*mu_) {
        if (!current_elements_manager_) {
          auto new_ctx = std::make_shared<IteratorContext>(*ctx);
          current_elements_manager_ = ctx->StartThread(
              "tf_data_parallel_interleave_current",
              [this, new_ctx]() { CurrentElementsManager(new_ctx); });
        }
        if (!future_elements_manager_) {
          auto new_ctx = std::make_shared<IteratorContext>(*ctx);
          future_elements_manager_ = ctx->StartThread(
              "tf_data_parallel_interleave_future",
              [this, new_ctx]() { FutureElementsManager(new_ctx); });
        }
      }

      // Fetches up to `dataset()->block_length_` results from `element`.
      void FetchResults(const std::shared_ptr<IteratorContext>& ctx,
                        const std::shared_ptr<Element>& element,
                        int64 num_results) LOCKS_EXCLUDED(*mu_) {
        RecordStart(ctx.get());
        auto cleanup = gtl::MakeCleanup([this, ctx] { RecordStop(ctx.get()); });
        bool end_of_input = false;
        for (int64 i = 0; i < num_results; ++i) {
          auto result = std::make_shared<Result>();
          result->status = element->iterator->GetNext(
              ctx.get(), &result->return_values, &end_of_input);
          if (end_of_input) {
            break;
          }
          RecordBufferEnqueue(ctx.get(), result->return_values);
          mutex_lock l(*mu_);
          mutex_lock l2(element->mu);
          element->results.push_back(result);
          result->is_ready = true;
          cond_var_->notify_all();
        }

        mutex_lock l(*mu_);
        // Release the ownership of the cycle element iterator.
        element->in_use = false;
        if (end_of_input) {
          // Close the iterator if end of input was encountered.
          element->iterator.reset();
          element->inputs.clear();
          --num_open_;
        }
        --num_calls_;
        const auto& stats_aggregator = ctx->stats_aggregator();
        if (stats_aggregator) {
          stats_aggregator->AddScalar(
              stats_utils::ThreadUtilizationScalarName(dataset()->node_name()),
              static_cast<float>(num_calls_) /
                  static_cast<float>(num_parallel_calls_->value),
              num_elements());
        }
        cond_var_->notify_all();
      }

      // Manages futures cycle elements, creating new iterators as needed and
      // asynchronously fetching results from existing iterators.
      //
      // This method runs in the `future_elements_manager_` background thread.
      void FutureElementsManager(const std::shared_ptr<IteratorContext>& ctx) {
        RecordStart(ctx.get());
        auto cleanup = gtl::MakeCleanup([this, ctx] { RecordStop(ctx.get()); });
        auto busy = [this]() EXCLUSIVE_LOCKS_REQUIRED(*mu_) -> bool {
          // TODO(jsimsa): Autotune the buffer size.
          return num_calls_ >= num_parallel_calls_->value ||
                 future_elements_.size() >= 2 * dataset()->cycle_length_;
        };
        while (true) {
          mutex_lock l(*mu_);

          // Wait until this thread is cancelled, the end of input has been
          // reached, or the cycle element at the `cycle_index_` position is
          // not in use.
          while (!cancelled_ && !end_of_input_ && busy()) {
            RecordStop(ctx.get());
            cond_var_->wait(l);
            RecordStart(ctx.get());
          }

          if (cancelled_ || end_of_input_) {
            return;
          }

          while (!end_of_input_ && !busy()) {
            std::shared_ptr<Element> element = MakeElement(ctx);
            if (!element) {
              break;
            }
            future_elements_.push_front(element);
            if (!element->iterator) {
              continue;
            }
            ++num_calls_;
            element->in_use = true;
            thread_pool_->Schedule(
                std::bind(&ParallelInterleaveIterator::FetchResults, this, ctx,
                          std::move(element), dataset()->block_length_));
          }
          const auto& stats_aggregator = ctx->stats_aggregator();
          if (stats_aggregator) {
            stats_aggregator->AddScalar(
                stats_utils::ThreadUtilizationScalarName(
                    dataset()->node_name()),
                static_cast<float>(num_calls_) /
                    static_cast<float>(num_parallel_calls_->value),
                num_elements());
          }
          cond_var_->notify_all();
        }
      }

      // Creates a new element.
      std::shared_ptr<Element> MakeElement(
          const std::shared_ptr<IteratorContext>& ctx)
          EXCLUSIVE_LOCKS_REQUIRED(*mu_) {
        auto element = std::make_shared<Element>();
        element->id = element_id_counter_++;
        Status status =
            input_impl_->GetNext(ctx.get(), &element->inputs, &end_of_input_);
        if (!status.ok()) {
          auto result = std::make_shared<Result>();
          result->is_ready = true;
          result->status = status;
          mutex_lock l(element->mu);
          element->results.push_back(std::move(result));
          return element;
        }
        if (!end_of_input_) {
          Status status = MakeIteratorFromInputElement(
              ctx.get(), element->inputs, element->id,
              *instantiated_captured_func_, prefix(), &element->iterator);
          if (!status.ok()) {
            auto result = std::make_shared<Result>();
            result->is_ready = true;
            result->status = status;
            mutex_lock l(element->mu);
            element->results.push_back(std::move(result));
            return element;
          }
          ++num_open_;
        } else {
          element.reset();
        }
        return element;
      }

      Status WriteStatusLocked(IteratorStateWriter* writer,
                               const string& key_prefix, size_t idx,
                               const Status& status)
          EXCLUSIVE_LOCKS_REQUIRED(*mu_) {
        TF_RETURN_IF_ERROR(writer->WriteScalar(
            CodeKey(key_prefix, idx), static_cast<int64>(status.code())));
        if (!status.ok()) {
          TF_RETURN_IF_ERROR(writer->WriteScalar(
              ErrorMessageKey(key_prefix, idx), status.error_message()));
        }
        return Status::OK();
      }

      Status ReadStatusLocked(IteratorStateReader* reader,
                              const string& key_prefix, size_t idx,
                              Status* status) EXCLUSIVE_LOCKS_REQUIRED(*mu_) {
        int64 code_int;
        TF_RETURN_IF_ERROR(
            reader->ReadScalar(CodeKey(key_prefix, idx), &code_int));
        error::Code code = static_cast<error::Code>(code_int);

        if (code != error::Code::OK) {
          string error_message;
          TF_RETURN_IF_ERROR(reader->ReadScalar(
              ErrorMessageKey(key_prefix, idx), &error_message));
          *status = Status(code, error_message);
        } else {
          *status = Status::OK();
        }
        return Status::OK();
      }

      string CodeKey(const string& key_prefix, size_t idx) {
        return full_name(
            strings::StrCat(key_prefix, ".results[", idx, "].code"));
      }

      string ErrorMessageKey(const string& key_prefix, size_t idx) {
        return full_name(
            strings::StrCat(key_prefix, ".results[", idx, "].error_message"));
      }

      Status WriteElement(std::shared_ptr<Element> element, int idx,
                          const string& key_prefix, IteratorStateWriter* writer)
          EXCLUSIVE_LOCKS_REQUIRED(*mu_) {
        if (element->iterator) {
          TF_RETURN_IF_ERROR(SaveInput(writer, element->iterator));
          TF_RETURN_IF_ERROR(writer->WriteScalar(
              full_name(strings::StrCat(key_prefix, "[", idx, "].id")),
              element->id));
          TF_RETURN_IF_ERROR(writer->WriteScalar(
              full_name(strings::StrCat(key_prefix, "[", idx, "].inputs.size")),
              element->inputs.size()));
          for (int i = 0; i < element->inputs.size(); i++) {
            TF_RETURN_IF_ERROR(writer->WriteTensor(
                full_name(
                    strings::StrCat(key_prefix, "[", idx, "].inputs[", i, "]")),
                element->inputs[i]));
          }
        }
        mutex_lock l(element->mu);
        TF_RETURN_IF_ERROR(writer->WriteScalar(
            full_name(strings::StrCat(key_prefix, "[", idx, "].results.size")),
            element->results.size()));
        for (size_t i = 0; i < element->results.size(); i++) {
          std::shared_ptr<Result> result = element->results[i];
          TF_RETURN_IF_ERROR(WriteStatusLocked(
              writer, strings::StrCat(key_prefix, "[", idx, "]"), i,
              result->status));
          TF_RETURN_IF_ERROR(writer->WriteScalar(
              full_name(strings::StrCat(key_prefix, "[", idx, "].results[", i,
                                        "].size")),
              result->return_values.size()));
          for (size_t j = 0; j < result->return_values.size(); j++) {
            TF_RETURN_IF_ERROR(writer->WriteTensor(
                full_name(strings::StrCat(key_prefix, "[", idx, "].results[", i,
                                          "][", j, "]")),
                result->return_values[j]));
          }
          if (result->is_ready) {
            TF_RETURN_IF_ERROR(writer->WriteScalar(
                full_name(strings::StrCat(key_prefix, "[", idx, "].results[", i,
                                          "].is_ready")),
                ""));
          }
        }
        return Status::OK();
      }

      Status WriteCurrentElements(IteratorStateWriter* writer)
          EXCLUSIVE_LOCKS_REQUIRED(*mu_) {
        TF_RETURN_IF_ERROR(writer->WriteScalar(
            full_name("current_elements.size"), current_elements_.size()));
        for (int idx = 0; idx < current_elements_.size(); idx++) {
          if (current_elements_[idx]) {
            TF_RETURN_IF_ERROR(WriteElement(current_elements_[idx], idx,
                                            "current_elements", writer));
          }
        }
        return Status::OK();
      }

      Status WriteFutureElements(IteratorStateWriter* writer)
          EXCLUSIVE_LOCKS_REQUIRED(*mu_) {
        TF_RETURN_IF_ERROR(writer->WriteScalar(
            full_name("future_elements.size"), future_elements_.size()));
        for (int idx = 0; idx < future_elements_.size(); idx++) {
          if (future_elements_[idx]) {
            TF_RETURN_IF_ERROR(WriteElement(future_elements_[idx], idx,
                                            "future_elements", writer));
          }
        }
        return Status::OK();
      }

      Status ReadElement(IteratorContext* ctx, IteratorStateReader* reader,
                         int idx, const string& key_prefix,
                         std::shared_ptr<Element>* out)
          EXCLUSIVE_LOCKS_REQUIRED(*mu_) {
        if (!reader->Contains(full_name(
                strings::StrCat(key_prefix, "[", idx, "].results.size")))) {
          return Status::OK();
        }
        auto element = std::make_shared<Element>();
        mutex_lock l(element->mu);
        int64 results_size;
        TF_RETURN_IF_ERROR(reader->ReadScalar(
            full_name(strings::StrCat(key_prefix, "[", idx, "].results.size")),
            &results_size));
        element->results.resize(results_size);
        for (size_t i = 0; i < results_size; i++) {
          auto result = std::make_shared<Result>();
          TF_RETURN_IF_ERROR(ReadStatusLocked(
              reader, strings::StrCat(key_prefix, "[", idx, "]"), i,
              &result->status));
          int64 num_return_values;
          TF_RETURN_IF_ERROR(reader->ReadScalar(
              full_name(strings::StrCat(key_prefix, "[", idx, "].results[", i,
                                        "].size")),
              &num_return_values));
          result->return_values.reserve(num_return_values);
          for (size_t j = 0; j < num_return_values; j++) {
            result->return_values.emplace_back();
            TF_RETURN_IF_ERROR(reader->ReadTensor(
                full_name(strings::StrCat(key_prefix, "[", idx, "].results[", i,
                                          "][", j, "]")),
                &result->return_values.back()));
          }
          result->is_ready = reader->Contains(full_name(strings::StrCat(
              key_prefix, "[", idx, "].results[", i, "].is_ready")));
          element->results[i] = std::move(result);
        }
        if (!reader->Contains(full_name(
                strings::StrCat(key_prefix, "[", idx, "].inputs.size")))) {
          element->iterator.reset();
          *out = std::move(element);
          return Status::OK();
        }
        int64 inputs_size;
        TF_RETURN_IF_ERROR(reader->ReadScalar(
            full_name(strings::StrCat(key_prefix, "[", idx, "].inputs.size")),
            &inputs_size));
        element->inputs.resize(inputs_size);
        for (int i = 0; i < inputs_size; i++) {
          TF_RETURN_IF_ERROR(reader->ReadTensor(
              full_name(
                  strings::StrCat(key_prefix, "[", idx, "].inputs[", i, "]")),
              &element->inputs[i]));
        }
        TF_RETURN_IF_ERROR(reader->ReadScalar(
            full_name(strings::StrCat(key_prefix, "[", idx, "].id")),
            &element->id));
        TF_RETURN_IF_ERROR(MakeIteratorFromInputElement(
            ctx, element->inputs, element->id,
            *instantiated_captured_func_.get(), prefix(), &element->iterator));
        TF_RETURN_IF_ERROR(RestoreInput(ctx, reader, element->iterator));
        *out = std::move(element);
        return Status::OK();
      }

      Status ReadCurrentElements(IteratorContext* ctx,
                                 IteratorStateReader* reader)
          EXCLUSIVE_LOCKS_REQUIRED(*mu_) {
        int64 size;
        TF_RETURN_IF_ERROR(
            reader->ReadScalar(full_name("current_elements.size"), &size));
        DCHECK_EQ(current_elements_.size(), size);
        for (int idx = 0; idx < current_elements_.size(); idx++) {
          TF_RETURN_IF_ERROR(ReadElement(ctx, reader, idx, "current_elements",
                                         &current_elements_[idx]));
        }
        return Status::OK();
      }

      Status ReadFutureElements(IteratorContext* ctx,
                                IteratorStateReader* reader)
          EXCLUSIVE_LOCKS_REQUIRED(*mu_) {
        int64 size;
        TF_RETURN_IF_ERROR(
            reader->ReadScalar(full_name("future_elements.size"), &size));
        future_elements_.resize(size);
        for (int idx = 0; idx < future_elements_.size(); idx++) {
          TF_RETURN_IF_ERROR(ReadElement(ctx, reader, idx, "future_elements",
                                         &future_elements_[idx]));
        }
        return Status::OK();
      }

      // Used for coordination between the main thread, the runner thread, and
      // the worker threads.
      const std::shared_ptr<mutex> mu_;

      // Used for coordination between the main thread, the manager threads, and
      // the threadpool threads. In particular, the managers thread should only
      // schedule new calls into the threadpool when the number of in-flight
      // calls is less than the user specified level of parallelism and there
      // are slots available in the element `results` buffer.
      const std::shared_ptr<condition_variable> cond_var_;

      // Identifies the maximum number of parallel calls.
      const std::shared_ptr<model::SharedState> num_parallel_calls_;

      // Determines whether outputs can be produced in non-deterministic order.
      const bool sloppy_;

      // Iterator for input elements.
      std::unique_ptr<IteratorBase> input_impl_ GUARDED_BY(*mu_);

      // Identifies position in the interleave cycle.
      int64 block_index_ GUARDED_BY(*mu_) = 0;
      int64 cycle_index_ GUARDED_BY(*mu_) = 0;

      // Elements of the current interleave cycle.
      std::vector<std::shared_ptr<Element>> current_elements_ GUARDED_BY(*mu_);

      // Elements to be used in the interleave cycle in the future.
      std::deque<std::shared_ptr<Element>> future_elements_ GUARDED_BY(*mu_);

      // Identifies whether the global end of input has been reached.
      bool end_of_input_ GUARDED_BY(*mu_) = false;

      // Identifies the number of open iterators.
      int64 num_open_ GUARDED_BY(*mu_) = 0;

      // Identifies the number of outstanding calls.
      int64 num_calls_ GUARDED_BY(*mu_) = 0;

      std::unique_ptr<thread::ThreadPool> thread_pool_;
      std::unique_ptr<Thread> current_elements_manager_ GUARDED_BY(*mu_);
      std::unique_ptr<Thread> future_elements_manager_ GUARDED_BY(*mu_);
      int64 element_id_counter_ GUARDED_BY(*mu_) = 0;

      // Identifies whether background threads should be cancelled.
      bool cancelled_ GUARDED_BY(*mu_) = false;
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
  std::shared_ptr<FunctionLibraryDefinition> lib_def_;
};

REGISTER_KERNEL_BUILDER(Name("ParallelInterleaveDatasetV2").Device(DEVICE_CPU),
                        ParallelInterleaveDatasetOp);

}  // namespace
}  // namespace data
}  // namespace tensorflow
