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
#include "tensorflow/core/kernels/data/parallel_interleave_dataset_op.h"

#include <atomic>
#include <deque>
#include <memory>
#include <utility>

#include "absl/strings/str_format.h"
#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/common_runtime/input_colocation_exemption_registry.h"
#include "tensorflow/core/common_runtime/metrics.h"
#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/framework/model.h"
#include "tensorflow/core/framework/partial_tensor_shape.h"
#include "tensorflow/core/framework/stats_aggregator.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/kernels/data/captured_function.h"
#include "tensorflow/core/kernels/data/dataset_utils.h"
#include "tensorflow/core/kernels/data/name_utils.h"
#include "tensorflow/core/kernels/data/stats_utils.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/gtl/cleanup.h"
#include "tensorflow/core/lib/random/random.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/cpu_info.h"

namespace tensorflow {
namespace data {

// See documentation in ../../ops/dataset_ops.cc for a high-level
// description of the following op.

/* static */ constexpr const char* const
    ParallelInterleaveDatasetOp::kDatasetType;
/* static */ constexpr const char* const
    ParallelInterleaveDatasetOp::kInputDataset;
/* static */ constexpr const char* const
    ParallelInterleaveDatasetOp::kOtherArguments;
/* static */ constexpr const char* const
    ParallelInterleaveDatasetOp::kCycleLength;
/* static */ constexpr const char* const
    ParallelInterleaveDatasetOp::kBlockLength;
/* static */ constexpr const char* const
    ParallelInterleaveDatasetOp::kNumParallelCalls;
/* static */ constexpr const char* const ParallelInterleaveDatasetOp::kFunc;
/* static */ constexpr const char* const
    ParallelInterleaveDatasetOp::kTarguments;
/* static */ constexpr const char* const
    ParallelInterleaveDatasetOp::kOutputTypes;
/* static */ constexpr const char* const
    ParallelInterleaveDatasetOp::kOutputShapes;
/* static */ constexpr const char* const ParallelInterleaveDatasetOp::kSloppy;

constexpr char kTfDataParallelInterleaveWorkerPool[] =
    "tf_data_parallel_interleave_worker_pool";
constexpr char kParallelism[] = "parallelism";
constexpr char kBlockIndex[] = "block_index";
constexpr char kCycleIndex[] = "cycle_index";
constexpr char kEndOfInput[] = "end_of_input";
constexpr char kElementIdCounter[] = "element_id_counter";
constexpr char kCurrentElements[] = "current_elements";
constexpr char kCurrentElementsSize[] = "current_elements.size";
constexpr char kFutureElements[] = "future_elements";
constexpr char kFutureElementsSize[] = "future_elements.size";
constexpr char kResultsSuffix[] = ".results";
constexpr char kCodeSuffix[] = ".code";
constexpr char kErrorMessageSuffix[] = ".error_message";
constexpr char kIdSuffix[] = ".id";
constexpr char kSizeSuffix[] = ".size";
constexpr char kInputsSuffix[] = ".inputs";
constexpr char kIsReadySuffix[] = ".is_ready";

// `kCyclePrefetchFactor * cycle_length` is the number of future cycle elements
// that will be prefetched ahead of time. The purpose of prefetching future
// cycle elements is to overlap expensive initialization (e.g. opening of a
// remote file) with other computation.
constexpr double kCyclePrefetchFactor = 2.0L;

// `kPerIteratorPrefetchFactor * block_length + 1` is the number of per-iterator
// results that will be prefetched ahead of time. The `+ 1` is to match the
// behavior of the original autotune implementation.
constexpr double kPerIteratorPrefetchFactor = 2.0L;

// The motivation for creating an alternative implementation of parallel
// interleave is to decouple the degree of parallelism from the cycle length.
// This makes it possible to change the degree of parallelism (e.g. through
// auto-tuning) without changing the cycle length (which would change the order
// in which elements are produced).
//
// Furthermore, this class favors modularity over extended functionality. In
// particular, it refrains from implementing configurable buffering of output
// elements and prefetching of input iterators.
class ParallelInterleaveDatasetOp::Dataset : public DatasetBase {
 public:
  Dataset(OpKernelContext* ctx, const DatasetBase* input,
          std::unique_ptr<CapturedFunction> captured_func, int64 cycle_length,
          int64 block_length, int64 num_parallel_calls, bool sloppy,
          const DataTypeVector& output_types,
          const std::vector<PartialTensorShape>& output_shapes)
      : DatasetBase(DatasetContext(ctx)),
        input_(input),
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
    name_utils::IteratorPrefixParams params;
    params.op_version = op_version_;
    return absl::make_unique<ParallelInterleaveIterator>(
        ParallelInterleaveIterator::Params{
            this,
            name_utils::IteratorPrefix(
                ParallelInterleaveDatasetOp::kDatasetType, prefix, params)},
        sloppy_);
  }

  const DataTypeVector& output_dtypes() const override { return output_types_; }

  const std::vector<PartialTensorShape>& output_shapes() const override {
    return output_shapes_;
  }

  string DebugString() const override {
    name_utils::DatasetDebugStringParams params;
    params.op_version = op_version_;
    return name_utils::DatasetDebugString(
        ParallelInterleaveDatasetOp::kDatasetType, params);
  }

  Status CheckExternalState() const override {
    TF_RETURN_IF_ERROR(captured_func_->CheckExternalState());
    return input_->CheckExternalState();
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
    b->BuildAttrValue(captured_func_->func(), &f);
    AttrValue other_arguments_types_attr;
    b->BuildAttrValue(other_arguments_types, &other_arguments_types_attr);
    AttrValue sloppy_attr;
    b->BuildAttrValue(sloppy_, &sloppy_attr);

    TF_RETURN_IF_ERROR(b->AddDataset(this,
                                     {{0, input_node},
                                      {2, cycle_length_node},
                                      {3, block_length_node},
                                      {4, num_parallel_calls_node}},
                                     {{1, other_arguments}},
                                     {{kFunc, f},
                                      {kTarguments, other_arguments_types_attr},
                                      {kSloppy, sloppy_attr}},
                                     output));
    return Status::OK();
  }

 private:
  class ParallelInterleaveIterator : public DatasetIterator<Dataset> {
   public:
    ParallelInterleaveIterator(const Params& params, bool sloppy)
        : DatasetIterator<Dataset>(params),
          per_iterator_prefetch_(
              static_cast<int>(params.dataset->block_length_ *
                               kPerIteratorPrefetchFactor) +
              1),
          future_elements_prefetch_(static_cast<int>(
              params.dataset->cycle_length_ * kCyclePrefetchFactor)),
          mu_(std::make_shared<mutex>()),
          num_parallel_calls_cond_var_(std::make_shared<condition_variable>()),
          num_parallel_calls_(std::make_shared<model::SharedState>(
              params.dataset->num_parallel_calls_, mu_,
              num_parallel_calls_cond_var_)),
          sloppy_(sloppy),
          current_elements_(params.dataset->cycle_length_) {}

    ~ParallelInterleaveIterator() override {
      mutex_lock l(*mu_);
      cancelled_ = true;
      // Wake up all threads so that they can exit. This will also wake up any
      // threads waiting in GetNextInternal.
      for (auto element : current_elements_) {
        if (element) {
          element->cond_var.notify_all();
        }
      }
      current_workers_cond_var_.notify_all();
      future_workers_cond_var_.notify_all();
      num_parallel_calls_cond_var_->notify_all();
      while (outstanding_threads_ > 0) {
        outstanding_threads_finished_cond_var_.wait(l);
      }
      sloppy_cond_var_.notify_all();
      zero_active_workers_cond_var_.notify_all();
    }

    string BuildTraceMeName() override {
      int64 parallelism = -1;
      // NOTE: We only set the parallelism value if the lock can be acquired
      // right away to avoid introducing tracing overhead.
      if (mu_->try_lock()) {
        parallelism = num_parallel_calls_->value;
        mu_->unlock();
      }
      return strings::StrCat(
          prefix(), "#parallelism=", parallelism,
          ",cycle_length=", dataset()->cycle_length_,
          ",block_length=", dataset()->block_length_,
          ",autotune=", dataset()->num_parallel_calls_ == model::kAutotune,
          ",deterministic=", !sloppy_, "#");
    }

    Status Initialize(IteratorContext* ctx) override {
      mutex_lock l(*mu_);
      // Note that if `ctx->thread_pool()` is non-null, then instead of creating
      // a dedicated thread pool of size `num_threads`, computation will be
      // scheduled into the shared threadpool. The threadpool is guaranteed to
      // support `num_threads` concurrent tasks without blocking indefinitely.
      //
      // Allocate one thread for the worker manager, `cycle_length_` threads for
      // the current workers, and `future_elements_prefetch_` for the future
      // workers.
      int max_current_workers = dataset()->cycle_length_;
      int future_workers = future_elements_prefetch_ + dataset()->cycle_length_;
      const int num_threads = 1 + max_current_workers + future_workers;
      thread_pool_ = ctx->CreateThreadPool(kTfDataParallelInterleaveWorkerPool,
                                           num_threads);
      if (num_parallel_calls_->value == model::kAutotune) {
        num_parallel_calls_->value = dataset()->cycle_length_;
      }
      ctx_ = std::make_unique<IteratorContext>(*ctx);
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
        EnsureInitialElementsCreated();
        EnsureThreadsStarted();
        while (!cancelled_ && !Consume(&result)) {
          RecordStop(ctx);
          if (sloppy_) {
            sloppy_cond_var_.wait(l);
          } else {
            VLOG(3) << "Blocked waiting for element "
                    << current_elements_[cycle_index_]->id;
            current_elements_[cycle_index_]->cond_var.wait(l);
          }
          RecordStart(ctx);
        }
        if (cancelled_) {
          return errors::Cancelled("Iterator was cancelled");
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
          {model::MakeParameter(kParallelism, num_parallel_calls_, /*min=*/1,
                                /*max=*/dataset()->cycle_length_)});
    }

    Status SaveInternal(IteratorStateWriter* writer) override {
      mutex_lock l(*mu_);
      wait_for_checkpoint_ = true;
      // Wait for all in-flight calls to complete.
      while (num_active_workers_ > 0) {
        RecordStop(ctx_.get());
        zero_active_workers_cond_var_.wait(l);
        RecordStart(ctx_.get());
      }
      // Initialize all elements and filter out elements with no input.
      InitializeInputs(element_id_counter_);
      for (auto& element : current_elements_) {
        if (element && element->no_input) {
          element.reset();
        }
      }
      while (!future_elements_.empty() && future_elements_.back()->no_input) {
        future_elements_.pop_back();
      }
      wait_for_checkpoint_ = false;
      DCHECK_EQ(num_active_workers_, 0);
      VLOG(4) << "State before save:\n" << DebugString();
      TF_RETURN_IF_ERROR(SaveInput(writer, input_impl_));
      TF_RETURN_IF_ERROR(
          writer->WriteScalar(full_name(kBlockIndex), block_index_));
      TF_RETURN_IF_ERROR(
          writer->WriteScalar(full_name(kCycleIndex), cycle_index_));
      if (end_of_input_) {
        TF_RETURN_IF_ERROR(writer->WriteScalar(full_name(kEndOfInput), ""));
      }
      TF_RETURN_IF_ERROR(writer->WriteScalar(full_name(kElementIdCounter),
                                             element_id_counter_));
      TF_RETURN_IF_ERROR(WriteCurrentElements(writer));
      TF_RETURN_IF_ERROR(WriteFutureElements(writer));
      // Wake workers back up.
      current_workers_cond_var_.notify_all();
      future_workers_cond_var_.notify_all();
      return Status::OK();
    }

    Status RestoreInternal(IteratorContext* ctx,
                           IteratorStateReader* reader) override {
      mutex_lock l(*mu_);
      DCHECK(!threads_initialized_);
      DCHECK(!initial_elements_created_);
      TF_RETURN_IF_ERROR(RestoreInput(ctx, reader, input_impl_));
      TF_RETURN_IF_ERROR(
          reader->ReadScalar(full_name(kBlockIndex), &block_index_));
      TF_RETURN_IF_ERROR(
          reader->ReadScalar(full_name(kCycleIndex), &cycle_index_));
      TF_RETURN_IF_ERROR(reader->ReadScalar(full_name(kElementIdCounter),
                                            &element_id_counter_));
      end_of_input_ = reader->Contains(full_name(kEndOfInput));
      TF_RETURN_IF_ERROR(ReadCurrentElements(ctx, reader));
      TF_RETURN_IF_ERROR(ReadFutureElements(ctx, reader));
      initial_elements_created_ = false;
      for (int i = 0; i < current_elements_.size(); ++i) {
        int index = (cycle_index_ + i) % current_elements_.size();
        auto element = current_elements_[index];
        if (element) {
          elements_to_process_.push_back(index);
          element->initialized = true;
          element->cycle_index = index;
          initial_elements_created_ = true;
        }
      }
      for (auto element : future_elements_) {
        element->initialized = true;
      }
      last_valid_current_element_ = current_elements_.size() - 1;
      while (last_valid_current_element_ >= 0 &&
             !current_elements_[last_valid_current_element_]) {
        last_valid_current_element_--;
      }
      VLOG(2) << "Parallel interleave iterator restored";
      VLOG(4) << "State after restore:\n" << DebugString();
      return Status::OK();
    }

   private:
    // Represents the result of fetching an element from a dataset.
    struct Result {
      Status status;
      std::vector<Tensor> return_values;
    };

    // The interleave transformation repeatedly inputs elements, applies the
    // user-provided function to transform the input elements to datasets, and
    // interleaves the elements of these datasets as its output.
    //
    // This structure represents an input element and derived state.
    struct Element {
      // Unique identifier, needed to support checkpointing.
      int64 id GUARDED_BY(&ParallelInterleaveIterator::mu_);
      // The actual input element.  Iterator created from the input element. A
      // null value indicates that the element either reached end of input or
      // hasn't been initialized yet.
      std::unique_ptr<std::vector<Tensor>> inputs
          GUARDED_BY(&ParallelInterleaveIterator::mu_);
      // Iterator created from the input element. A null value indicates that
      // the element either reached end of input or hasn't been initialized yet.
      std::unique_ptr<IteratorBase> iterator
          GUARDED_BY(&ParallelInterleaveIterator::mu_);
      // Buffer for storing the outputs of `iterator`.
      std::deque<std::shared_ptr<Result>> GUARDED_BY(
          &ParallelInterleaveIterator::mu_) results;
      // The element's index in the cycle, if it is in the current cycle.
      // -1 if the element is not in the current cycle.
      int64 cycle_index GUARDED_BY(&ParallelInterleaveIterator::mu_) = -1;
      // Whether the element is currently being processed by a worker thread.
      // This is used to ensure that only one thread at a time tries to process
      // an element.
      bool active GUARDED_BY(&ParallelInterleaveIterator::mu_) = false;
      // Whether the inputs and iterator have been initialized.
      bool initialized GUARDED_BY(&ParallelInterleaveIterator::mu_) = false;
      // Whether we tried to initialize the element, but the input interator
      // was exhausted so we could produce no inputs.
      bool no_input GUARDED_BY(&ParallelInterleaveIterator::mu_) = false;
      // Condition variable for communicating between current worker threads
      // and GetNext.
      condition_variable cond_var;

      std::string DebugString()
          EXCLUSIVE_LOCKS_REQUIRED(&ParallelInterleaveIterator::mu_) {
        return absl::StrFormat(
            "Element(id: %d, iterator_null: %d, results_size: %d, "
            "cycle_index: %d, active: %d, initialized: %d, no_input: %d)",
            id, iterator == nullptr, results.size(), cycle_index, active,
            initialized, no_input);
      }
    };

    void EnsureInitialElementsCreated() EXCLUSIVE_LOCKS_REQUIRED(mu_) {
      if (!initial_elements_created_) {
        for (int i = 0; i < dataset()->cycle_length_; ++i) {
          current_elements_[i] = MakeElement();
          if (!current_elements_[i]) {
            break;
          }
          current_elements_[i]->cycle_index = i;
          elements_to_process_.push_back(i);
          last_valid_current_element_ = i;
        }
        initial_elements_created_ = true;
      }
    }

    void EnsureThreadsStarted() EXCLUSIVE_LOCKS_REQUIRED(mu_) {
      if (!threads_initialized_) {
        IncrementOutstandingThreads();
        thread_pool_->Schedule([this]() { WorkerManagerThread(); });
        threads_initialized_ = true;
      }
    }

    // Advances the position in the interleave cycle to the next cycle
    // element.
    void AdvanceToNextInCycle() EXCLUSIVE_LOCKS_REQUIRED(mu_) {
      DCHECK_NE(last_valid_current_element_, -1);
      block_index_ = 0;
      cycle_index_ = (cycle_index_ + 1) % (last_valid_current_element_ + 1);
    }

    // Advances the position in the interleave cycle by one.
    void AdvancePosition() EXCLUSIVE_LOCKS_REQUIRED(mu_) {
      ++block_index_;
      if (block_index_ == dataset()->block_length_) {
        AdvanceToNextInCycle();
      }
    }

    // Consumes a result (if available), returning an indication of whether
    // a result is available. If `true` is returned, `result` either
    // points to a valid result or is null if end of input has been reached.
    bool Consume(std::shared_ptr<Result>* result)
        EXCLUSIVE_LOCKS_REQUIRED(mu_) {
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

    // Consumes a result (if available), returning an indication of whether
    // a result is available. If `true` is returned, `result` either
    // points to a valid result or is null if end of input has been reached.
    bool ConsumeHelper(std::shared_ptr<Result>* result)
        EXCLUSIVE_LOCKS_REQUIRED(mu_) {
      while (true) {
        if (last_valid_current_element_ == -1) {
          // Reached end of input.
          return true;
        }
        for (int64 i = 0; i < (last_valid_current_element_ + 1); ++i) {
          int64 index = (cycle_index_ + i) % (last_valid_current_element_ + 1);
          if (current_elements_[index]) {
            cycle_index_ = index;
            if (i > 0) {
              block_index_ = 0;
            }
            break;
          }
        }
        DCHECK(current_elements_[cycle_index_]);
        std::shared_ptr<Element> element = current_elements_[cycle_index_];
        if (!element->results.empty()) {
          // We found a result.
          std::swap(*result, element->results.front());
          element->results.pop_front();
          if (!element->active) {
            elements_to_process_.push_back(cycle_index_);
            current_workers_cond_var_.notify_one();
          }
          AdvancePosition();
          return true;
        }
        if (!element->initialized || element->iterator) {
          // The element is still producing results, so we wait.
          return false;
        }
        // We've consumed all results from the element. Get a new element from
        // future_elements, or create a new element if no future elements are
        // available.
        if (!future_elements_.empty()) {
          std::shared_ptr<Element> future_element =
              std::move(future_elements_.front());
          future_elements_.pop_front();
          if (future_element->iterator) {
            EnableAutotune(ctx_.get(), future_element->iterator.get());
          }
          future_element->cycle_index = cycle_index_;
          current_elements_[cycle_index_] = std::move(future_element);
          future_workers_cond_var_.notify_one();
          if (!current_elements_[cycle_index_]->active) {
            current_workers_cond_var_.notify_one();
          }
        } else {
          current_elements_[cycle_index_] = MakeElement();
          if (current_elements_[cycle_index_]) {
            current_elements_[cycle_index_]->cycle_index = cycle_index_;
            elements_to_process_.push_back(cycle_index_);
            element->cycle_index = cycle_index_;
            current_workers_cond_var_.notify_one();
          }
          while (last_valid_current_element_ >= 0 &&
                 !current_elements_[last_valid_current_element_]) {
            last_valid_current_element_--;
            if (cycle_index_ > last_valid_current_element_) {
              // We are about to move the cycle index below in
              // AdvanceToNextInCycle().
              cycle_index_ = last_valid_current_element_;
            }
          }
        }
        if (last_valid_current_element_ != -1) {
          AdvanceToNextInCycle();
        }
      }
    }

    // Creates a new element.
    std::shared_ptr<Element> MakeElement() EXCLUSIVE_LOCKS_REQUIRED(mu_) {
      if (end_of_input_) {
        return nullptr;
      }
      auto element = std::make_shared<Element>();
      element->id = element_id_counter_++;
      uninitialized_elements_.push_back(element);
      return element;
    }

    // Thread responsible for launching all worker threads. The thread stays
    // around after startup in case autotuning increases num_parallel_calls.
    void WorkerManagerThread() LOCKS_EXCLUDED(mu_) {
      int initial_current_workers;
      // When elements are moved from `future_elements_` to `current_elements_`,
      // the future worker which created the element may continue to process
      // the element for some time. That is why we need an additional
      // `cycle_length_` future workers to guarantee that whenever
      // `future_element_.size() < future_elements_prefetch_`, there will be a
      // future worker available to create a new future element.
      int future_workers = future_elements_prefetch_ + dataset()->cycle_length_;
      {
        mutex_lock l(*mu_);
        initial_current_workers = num_parallel_calls_->value;
        outstanding_threads_ += initial_current_workers + future_workers;
        num_current_workers_ += initial_current_workers;
        num_active_workers_ += initial_current_workers + future_workers;
        num_current_active_workers_ += initial_current_workers;
      }
      // Start current workers before future workers to improve startup time.
      for (int i = 0; i < initial_current_workers; ++i) {
        StartCurrentWorkerThread();
      }
      for (int i = 0; i < future_workers; ++i) {
        StartFutureWorkerThread();
      }
      while (true) {
        {
          mutex_lock l(*mu_);
          while (!cancelled_ &&
                 num_current_workers_ >= num_parallel_calls_->value) {
            RecordStop(ctx_.get());
            num_parallel_calls_cond_var_->wait(l);
            RecordStart(ctx_.get());
          }
          if (cancelled_ || end_of_input_) {
            DecrementOutstandingThreads();
            return;
          }
          IncrementOutstandingThreads();
          IncrementCurrentWorkers();
          IncrementActiveWorkers();
          IncrementCurrentActiveWorkers();
          StartCurrentWorkerThread();
        }
      }
    }

    void StartCurrentWorkerThread() {
      thread_pool_->Schedule([this]() { CurrentWorkerThread(); });
    }

    void StartFutureWorkerThread() {
      thread_pool_->Schedule([this]() { FutureWorkerThread(); });
    }

    // Current workers are responsible for keeping elements in
    // `current_elements_` processed. An element is processed if it is either
    // done or its `results` buffer is full (contains `kPerIteratorPrefetch`
    // elements).
    //
    // Current workers cycle between two phases: (1) finding an element and (2)
    // processing it. When a worker is processing an element, it will
    // claim the element by setting `element->active`, then continue to produce
    // results for the element until enough results have been computed for the
    // current cycle and the results buffer is full.
    void CurrentWorkerThread() LOCKS_EXCLUDED(mu_) {
      RecordStart(ctx_.get());
      auto done = [this]() EXCLUSIVE_LOCKS_REQUIRED(mu_) {
        RecordStop(ctx_.get());
        DecrementActiveWorkers();
        DecrementCurrentActiveWorkers();
        DecrementOutstandingThreads();
        DecrementCurrentWorkers();
      };
      while (true) {
        int element_index;
        std::shared_ptr<Element> element;
        // Find an element to process.
        {
          mutex_lock l(*mu_);
          // In case autotune changes num_parallel_calls.
          if (num_current_workers_ > num_parallel_calls_->value) {
            done();
            return;
          }
          // Look for an element that needs processing.
          element.reset();
          while (!cancelled_) {
            while (!elements_to_process_.empty() && !wait_for_checkpoint_) {
              int index = elements_to_process_.front();
              elements_to_process_.pop_front();
              auto& e = current_elements_[index];
              if (NeedsProcessing(e) && !e->active) {
                element_index = index;
                element = e;
                break;
              }
            }
            if (element) {
              break;
            }
            DecrementCurrentActiveWorkers();
            WaitWorkerThread(&current_workers_cond_var_, &l);
            IncrementCurrentActiveWorkers();
          }
          if (cancelled_) {
            done();
            return;
          }
          VLOG(3) << "Current worker woke up to process " << element->id;
          element->active = true;
        }
        // Loop on the element until we fill its results buffer or reach end of
        // input for the element.
        while (true) {
          ProcessElement(element);
          {
            mutex_lock l(*mu_);
            // Check whether we have produced enough results for the current
            // cycle.
            if (!NeedsProcessing(element)) {
              element->active = false;
              break;
            }
          }
        }
      }
    }

    // Future workers process elements after the current interleave cycle. A
    // future worker's job is to keep `future_elements_` filled with elements.
    // Elements in `future_elements` have had their first `kPerIteratorPrefetch`
    // results computed.
    void FutureWorkerThread() LOCKS_EXCLUDED(mu_) {
      RecordStart(ctx_.get());
      auto done = [this]() EXCLUSIVE_LOCKS_REQUIRED(mu_) {
        RecordStop(ctx_.get());
        DecrementActiveWorkers();
        DecrementOutstandingThreads();
      };
      std::shared_ptr<Element> element;
      while (true) {
        {
          mutex_lock l(*mu_);
          if (element) {
            element->active = false;
            if (element->cycle_index != -1) {
              element->cond_var.notify_one();
              // A current worker may need to process the element further.
              elements_to_process_.push_back(element->cycle_index);
              current_workers_cond_var_.notify_one();
            }
          }
          while (!cancelled_ &&
                 (future_elements_.size() >= future_elements_prefetch_ ||
                  wait_for_checkpoint_)) {
            WaitWorkerThread(&future_workers_cond_var_, &l);
          }
          if (cancelled_) {
            done();
            return;
          }
          element = MakeElement();
          if (!element) {
            done();
            return;
          }
          VLOG(3) << "Future worker created element " << element->id;
          element->active = true;
          future_elements_.push_back(element);
        }
        ProcessElement(element);
      }
    }

    // Generates results for the given element until the element's results
    // buffer is full or the element is done producing results.
    void ProcessElement(std::shared_ptr<Element> element) LOCKS_EXCLUDED(mu_) {
      DCHECK(element != nullptr);
      IteratorBase* iterator;
      // Initialize the inputs and iterator if necessary.
      {
        mutex_lock l(*mu_);
        DCHECK(element->active);
        if (!element->iterator) {
          InitializeInputs(element->id);
          if (!element->iterator) {
            return;
          }
        }
        // `iterator` will remain valid after releasing the lock because we have
        // marked the element as active, so no other thread will modify its
        // iterator.
        iterator = element->iterator.get();
      }
      DCHECK(iterator != nullptr);
      // Process until the results queue is full or we reach end of input.
      while (true) {
        auto result = std::make_shared<Result>();
        bool end_of_input = false;
        result->status = iterator->GetNext(ctx_.get(), &result->return_values,
                                           &end_of_input);
        if (end_of_input) {
          mutex_lock l(*mu_);
          element->iterator.reset();
          element->inputs.reset();
          NotifyElementUpdate(element);
          break;
        }
        RecordBufferEnqueue(ctx_.get(), result->return_values);
        mutex_lock l(*mu_);
        element->results.push_back(std::move(result));
        NotifyElementUpdate(element);
        if (element->results.size() == per_iterator_prefetch_) {
          break;
        }
      }
    }

    // Initialize inputs and create an iterator for all elements up to
    // element_id.
    void InitializeInputs(int element_id) EXCLUSIVE_LOCKS_REQUIRED(mu_) {
      while (!uninitialized_elements_.empty() &&
             uninitialized_elements_.front()->id <= element_id) {
        std::shared_ptr<Element> element = uninitialized_elements_.front();
        uninitialized_elements_.pop_front();
        element->initialized = true;
        // Check if we've already reached end of input.
        if (end_of_input_) {
          element->no_input = true;
          NotifyElementUpdate(element);
          continue;
        }
        std::vector<Tensor> inputs;
        Status status =
            input_impl_->GetNext(ctx_.get(), &inputs, &end_of_input_);
        if (!status.ok()) {
          AddErrorResult(element, status);
          continue;
        }
        if (end_of_input_) {
          element->no_input = true;
          NotifyElementUpdate(element);
          continue;
        }
        element->inputs =
            absl::make_unique<std::vector<Tensor>>(std::move(inputs));
        status = MakeIteratorFromInputElement(
            ctx_.get(), *element->inputs, element->id,
            *instantiated_captured_func_, prefix(), &element->iterator);
        if (!status.ok()) {
          element->inputs.reset();
          element->iterator.reset();
          AddErrorResult(element, status);
          continue;
        }
        if (element->cycle_index == -1) {
          DisableAutotune(ctx_.get(), element->iterator.get());
        }
      }
    }

    // Adds an error result for the given element.
    void AddErrorResult(std::shared_ptr<Element> element, Status status)
        EXCLUSIVE_LOCKS_REQUIRED(mu_) {
      auto result = std::make_shared<Result>();
      result->status = status;
      element->results.push_back(std::move(result));
      NotifyElementUpdate(element);
    }

    // Cancels all threads (including the manager) and waits for them to finish.
    void StopAllThreads(mutex_lock* l) EXCLUSIVE_LOCKS_REQUIRED(mu_) {
    }

    // Waits on the given cond_var in a worker thread.
    void WaitWorkerThread(condition_variable* cond_var, mutex_lock* l)
        EXCLUSIVE_LOCKS_REQUIRED(mu_) {
      DecrementActiveWorkers();
      RecordStop(ctx_.get());
      cond_var->wait(*l);
      RecordStart(ctx_.get());
      IncrementActiveWorkers();
    }

    void NotifyElementUpdate(std::shared_ptr<Element> element)
        EXCLUSIVE_LOCKS_REQUIRED(mu_) {
      if (sloppy_) {
        sloppy_cond_var_.notify_one();
      } else {
        element->cond_var.notify_one();
      }
    }

    bool NeedsProcessing(const std::shared_ptr<Element>& element)
        EXCLUSIVE_LOCKS_REQUIRED(mu_) {
      if (!element) {
        return false;
      }
      if (!element->initialized) {
        return true;
      }
      return element->iterator &&
             element->results.size() < per_iterator_prefetch_;
    }

    inline void IncrementCurrentWorkers() EXCLUSIVE_LOCKS_REQUIRED(mu_) {
      num_current_workers_++;
    }

    inline void DecrementCurrentWorkers() EXCLUSIVE_LOCKS_REQUIRED(mu_) {
      num_current_workers_--;
    }

    inline void IncrementActiveWorkers() EXCLUSIVE_LOCKS_REQUIRED(mu_) {
      num_active_workers_++;
    }

    inline void DecrementActiveWorkers() EXCLUSIVE_LOCKS_REQUIRED(mu_) {
      num_active_workers_--;
      if (num_active_workers_ == 0) {
        zero_active_workers_cond_var_.notify_one();
      }
    }

    inline void IncrementCurrentActiveWorkers() EXCLUSIVE_LOCKS_REQUIRED(mu_) {
      num_current_active_workers_++;
      UpdateThreadUtilizationStats();
    }

    inline void DecrementCurrentActiveWorkers() EXCLUSIVE_LOCKS_REQUIRED(mu_) {
      num_current_active_workers_--;
      UpdateThreadUtilizationStats();
    }

    inline void IncrementOutstandingThreads() EXCLUSIVE_LOCKS_REQUIRED(mu_) {
      outstanding_threads_++;
    }

    inline void DecrementOutstandingThreads() EXCLUSIVE_LOCKS_REQUIRED(mu_) {
      outstanding_threads_--;
      if (outstanding_threads_ == 0) {
        outstanding_threads_finished_cond_var_.notify_one();
      }
    }

    inline void UpdateThreadUtilizationStats() EXCLUSIVE_LOCKS_REQUIRED(mu_) {
      const auto& stats_aggregator = ctx_->stats_aggregator();
      if (stats_aggregator) {
        stats_aggregator->AddScalar(
            stats_utils::ThreadUtilizationScalarName(dataset()->node_name()),
            static_cast<float>(num_current_active_workers_) /
                static_cast<float>(num_parallel_calls_->value),
            num_elements());
      }
    }

    Status WriteStatusLocked(IteratorStateWriter* writer,
                             const string& key_prefix, size_t idx,
                             const Status& status)
        EXCLUSIVE_LOCKS_REQUIRED(mu_) {
      TF_RETURN_IF_ERROR(writer->WriteScalar(
          CodeKey(key_prefix, idx), static_cast<int64>(status.code())));
      if (!status.ok()) {
        TF_RETURN_IF_ERROR(writer->WriteScalar(ErrorMessageKey(key_prefix, idx),
                                               status.error_message()));
      }
      return Status::OK();
    }

    Status ReadStatusLocked(IteratorStateReader* reader,
                            const string& key_prefix, size_t idx,
                            Status* status) EXCLUSIVE_LOCKS_REQUIRED(mu_) {
      int64 code_int;
      TF_RETURN_IF_ERROR(
          reader->ReadScalar(CodeKey(key_prefix, idx), &code_int));
      error::Code code = static_cast<error::Code>(code_int);

      if (code != error::Code::OK) {
        tstring error_message;
        TF_RETURN_IF_ERROR(reader->ReadScalar(ErrorMessageKey(key_prefix, idx),
                                              &error_message));
        *status = Status(code, error_message);
      } else {
        *status = Status::OK();
      }
      return Status::OK();
    }

    string CodeKey(const string& key_prefix, size_t idx) {
      return full_name(strings::StrCat(key_prefix, kResultsSuffix, "[", idx,
                                       "]", kCodeSuffix));
    }

    string ErrorMessageKey(const string& key_prefix, size_t idx) {
      return full_name(strings::StrCat(key_prefix, kResultsSuffix, "[", idx,
                                       "]", kErrorMessageSuffix));
    }

    Status WriteElement(std::shared_ptr<Element> element, int idx,
                        const string& key_prefix, IteratorStateWriter* writer)
        EXCLUSIVE_LOCKS_REQUIRED(*mu_) {
      if (element->iterator) {
        TF_RETURN_IF_ERROR(SaveInput(writer, element->iterator));
        TF_RETURN_IF_ERROR(writer->WriteScalar(
            full_name(strings::StrCat(key_prefix, "[", idx, "]", kIdSuffix)),
            element->id));
        TF_RETURN_IF_ERROR(writer->WriteScalar(
            full_name(strings::StrCat(key_prefix, "[", idx, "]", kInputsSuffix,
                                      kSizeSuffix)),
            element->inputs->size()));
        for (int i = 0; i < element->inputs->size(); i++) {
          TF_RETURN_IF_ERROR(writer->WriteTensor(
              full_name(strings::StrCat(key_prefix, "[", idx, "]",
                                        kInputsSuffix, "[", i, "]")),
              element->inputs->at(i)));
        }
      }
      TF_RETURN_IF_ERROR(writer->WriteScalar(
          full_name(strings::StrCat(key_prefix, "[", idx, "]", kResultsSuffix,
                                    kSizeSuffix)),
          element->results.size()));
      for (size_t i = 0; i < element->results.size(); i++) {
        std::shared_ptr<Result> result = element->results[i];
        TF_RETURN_IF_ERROR(WriteStatusLocked(
            writer, strings::StrCat(key_prefix, "[", idx, "]"), i,
            result->status));
        TF_RETURN_IF_ERROR(writer->WriteScalar(
            full_name(strings::StrCat(key_prefix, "[", idx, "]", kResultsSuffix,
                                      "[", i, "]", kSizeSuffix)),
            result->return_values.size()));
        for (size_t j = 0; j < result->return_values.size(); j++) {
          TF_RETURN_IF_ERROR(writer->WriteTensor(
              full_name(strings::StrCat(key_prefix, "[", idx, "]",
                                        kResultsSuffix, "[", i, "][", j, "]")),
              result->return_values[j]));
        }
        TF_RETURN_IF_ERROR(writer->WriteScalar(
            full_name(strings::StrCat(key_prefix, "[", idx, "]", kResultsSuffix,
                                      "[", i, "]", kIsReadySuffix)),
            ""));
      }
      return Status::OK();
    }

    Status WriteCurrentElements(IteratorStateWriter* writer)
        EXCLUSIVE_LOCKS_REQUIRED(mu_) {
      TF_RETURN_IF_ERROR(writer->WriteScalar(full_name(kCurrentElementsSize),
                                             current_elements_.size()));
      for (int idx = 0; idx < current_elements_.size(); idx++) {
        if (current_elements_[idx]) {
          TF_RETURN_IF_ERROR(WriteElement(current_elements_[idx], idx,
                                          kCurrentElements, writer));
        }
      }
      return Status::OK();
    }

    Status WriteFutureElements(IteratorStateWriter* writer)
        EXCLUSIVE_LOCKS_REQUIRED(mu_) {
      TF_RETURN_IF_ERROR(writer->WriteScalar(full_name(kFutureElementsSize),
                                             future_elements_.size()));
      for (int idx = 0; idx < future_elements_.size(); idx++) {
        if (future_elements_[idx]) {
          TF_RETURN_IF_ERROR(WriteElement(future_elements_[idx], idx,
                                          kFutureElements, writer));
        }
      }
      return Status::OK();
    }

    Status ReadElement(IteratorContext* ctx, IteratorStateReader* reader,
                       int idx, const string& key_prefix,
                       std::shared_ptr<Element>* out)
        EXCLUSIVE_LOCKS_REQUIRED(mu_) {
      if (!reader->Contains(full_name(strings::StrCat(
              key_prefix, "[", idx, "]", kResultsSuffix, kSizeSuffix)))) {
        return Status::OK();
      }
      auto element = std::make_shared<Element>();
      int64 results_size;
      TF_RETURN_IF_ERROR(reader->ReadScalar(
          full_name(strings::StrCat(key_prefix, "[", idx, "]", kResultsSuffix,
                                    kSizeSuffix)),
          &results_size));
      element->results.resize(results_size);
      for (size_t i = 0; i < results_size; i++) {
        auto result = std::make_shared<Result>();
        TF_RETURN_IF_ERROR(
            ReadStatusLocked(reader, strings::StrCat(key_prefix, "[", idx, "]"),
                             i, &result->status));
        int64 num_return_values;
        TF_RETURN_IF_ERROR(reader->ReadScalar(
            full_name(strings::StrCat(key_prefix, "[", idx, "]", kResultsSuffix,
                                      "[", i, "]", kSizeSuffix)),
            &num_return_values));
        result->return_values.reserve(num_return_values);
        for (size_t j = 0; j < num_return_values; j++) {
          result->return_values.emplace_back();
          TF_RETURN_IF_ERROR(reader->ReadTensor(
              full_name(strings::StrCat(key_prefix, "[", idx, "]",
                                        kResultsSuffix, "[", i, "][", j, "]")),
              &result->return_values.back()));
        }
        element->results[i] = std::move(result);
      }
      if (!reader->Contains(full_name(strings::StrCat(
              key_prefix, "[", idx, "]", kInputsSuffix, kSizeSuffix)))) {
        element->iterator.reset();
        *out = std::move(element);
        return Status::OK();
      }
      int64 inputs_size;
      TF_RETURN_IF_ERROR(reader->ReadScalar(
          full_name(strings::StrCat(key_prefix, "[", idx, "]", kInputsSuffix,
                                    kSizeSuffix)),
          &inputs_size));
      element->inputs = std::make_unique<std::vector<Tensor>>(inputs_size);
      for (int i = 0; i < inputs_size; i++) {
        TF_RETURN_IF_ERROR(reader->ReadTensor(
            full_name(strings::StrCat(key_prefix, "[", idx, "]", kInputsSuffix,
                                      "[", i, "]")),
            &element->inputs->at(i)));
      }
      TF_RETURN_IF_ERROR(reader->ReadScalar(
          full_name(strings::StrCat(key_prefix, "[", idx, "]", kIdSuffix)),
          &element->id));
      TF_RETURN_IF_ERROR(MakeIteratorFromInputElement(
          ctx, *element->inputs, element->id,
          *instantiated_captured_func_.get(), prefix(), &element->iterator));
      TF_RETURN_IF_ERROR(RestoreInput(ctx, reader, element->iterator));
      *out = std::move(element);
      return Status::OK();
    }

    Status ReadCurrentElements(IteratorContext* ctx,
                               IteratorStateReader* reader)
        EXCLUSIVE_LOCKS_REQUIRED(mu_) {
      int64 size;
      TF_RETURN_IF_ERROR(
          reader->ReadScalar(full_name(kCurrentElementsSize), &size));
      DCHECK_EQ(current_elements_.size(), size);
      for (int idx = 0; idx < current_elements_.size(); idx++) {
        TF_RETURN_IF_ERROR(ReadElement(ctx, reader, idx, kCurrentElements,
                                       &current_elements_[idx]));
      }
      return Status::OK();
    }

    Status ReadFutureElements(IteratorContext* ctx, IteratorStateReader* reader)
        EXCLUSIVE_LOCKS_REQUIRED(mu_) {
      int64 size;
      TF_RETURN_IF_ERROR(
          reader->ReadScalar(full_name(kFutureElementsSize), &size));
      future_elements_.resize(size);
      for (int idx = 0; idx < future_elements_.size(); idx++) {
        TF_RETURN_IF_ERROR(ReadElement(ctx, reader, idx, kFutureElements,
                                       &future_elements_[idx]));
      }
      return Status::OK();
    }

    std::string DebugString() EXCLUSIVE_LOCKS_REQUIRED(mu_) {
      std::string result;
      result.append(strings::StrCat("Cycle index: ", cycle_index_, "\n"));
      result.append(strings::StrCat("Block index: ", block_index_, "\n"));
      result.append(strings::StrCat("End of input: ", end_of_input_, "\n"));
      {
        result.append("Current elements:\n");
        for (int i = 0; i < current_elements_.size(); ++i) {
          string element_string = "null";
          if (current_elements_[i]) {
            element_string = current_elements_[i]->DebugString();
          }
          result.append(absl::StrFormat("%d: %s\n", i, element_string));
        }
      }
      {
        result.append("Future elements:\n");
        for (int i = 0; i < future_elements_.size(); ++i) {
          string element_string = "null";
          if (future_elements_[i]) {
            element_string = future_elements_[i]->DebugString();
          }
          result.append(absl::StrFormat("%d: %s\n", i, element_string));
        }
      }
      return result;
    }

    // Indices of `current_elements_` which need to be processed by a current
    // worker.
    std::deque<int> elements_to_process_;

    // The last index in `current_elements_` containing a non-null element.
    // This allows us to optimize the situation when the cycle_length is large
    // but the input dataset doesn't have many elements. By tracking the index
    // of the last valid element, GetNext can avoid checking many null entries
    // each time through the cycle.
    // TODO(aaudibert): Generalize this optimization by removing null elements
    // from `current_elements_`, e.g. by compacting the vector when x% of
    // its elements are null.
    int64 last_valid_current_element_ GUARDED_BY(mu_) = -1;

    const int per_iterator_prefetch_;
    const int future_elements_prefetch_;

    // Identifies whether the current_elements_ vector has been initialized.
    bool initial_elements_created_ GUARDED_BY(mu_) = false;

    // Identifies whether the element threads have been initialized.
    bool threads_initialized_ GUARDED_BY(mu_) = false;

    // Used for coordination between the main thread, the manager threads, and
    // the worker threads.
    const std::shared_ptr<mutex> mu_;

    // Condition variable for waking up current workers.
    condition_variable current_workers_cond_var_;

    // Condition variable for waking up future workers.
    condition_variable future_workers_cond_var_;

    // Number of active worker threads which might be processing elements,
    // including both current workers and future workers. Used by
    // checkpointing to wait for outstanding work to finish.
    int num_active_workers_ GUARDED_BY(mu_) = 0;

    // Number of active current worker threads.
    int num_current_active_workers_ GUARDED_BY(mu_) = 0;

    // Condition variable notified whenever the total number of active workers
    // drops to zero. Used for checkpointing.
    condition_variable zero_active_workers_cond_var_;

    // Condition notified whenever num_parallel_calls_ changes. Shared so that
    // autotuning can notify us when num_parallel_calls_ changes.
    std::shared_ptr<condition_variable> num_parallel_calls_cond_var_;

    // Identifies the maximum number of parallel calls.
    const std::shared_ptr<model::SharedState> num_parallel_calls_;

    // The number of current workers currently alive or scheduled to be started.
    // This includes current workers which are blocked waiting for work.
    int num_current_workers_ GUARDED_BY(mu_) = 0;

    // Condition variable to signal that a result has been produced by some
    // element thread. Only used when `sloppy_` is true.
    condition_variable sloppy_cond_var_;

    // Determines whether outputs can be produced in non-deterministic order.
    const bool sloppy_;

    // Iterator for input elements.
    std::unique_ptr<IteratorBase> input_impl_ GUARDED_BY(mu_);

    // Identifies position in the interleave cycle.
    int64 block_index_ GUARDED_BY(mu_) = 0;
    // It is an invariant that either `last_valid_current_element_ == -1` or
    // `cycle_index_ <= last_valid_current_element_`.
    int64 cycle_index_ GUARDED_BY(mu_) = 0;

    // Elements of the current interleave cycle.
    std::vector<std::shared_ptr<Element>> current_elements_ GUARDED_BY(mu_);

    // Elements which still need their inputs and iterators to be initialized.
    // Elements at the front need to be initialized first.
    std::deque<std::shared_ptr<Element>> uninitialized_elements_
        GUARDED_BY(mu_);

    // Elements to be used in the interleave cycle in the future. The element
    // at the front is the next element to add to the interleave cycle when a
    // current element is exhausted.
    std::deque<std::shared_ptr<Element>> future_elements_ GUARDED_BY(mu_);

    // Identifies whether the global end of input has been reached.
    bool end_of_input_ GUARDED_BY(mu_) = false;

    // The number of outstanding element threads.
    int outstanding_threads_ GUARDED_BY(mu_) = 0;

    // Condition variable notified when outstanding_threads_ drops to 0.
    condition_variable outstanding_threads_finished_cond_var_;

    std::unique_ptr<thread::ThreadPool> thread_pool_;

    int64 element_id_counter_ GUARDED_BY(mu_) = 0;

    // Iterator context used in worker threads.
    std::unique_ptr<IteratorContext> ctx_;

    // Set to true during checkpointing to alert element threads that they
    // should pause operation. This is needed to prevent constantly-active
    // worker threads from blocking checkpointing indefinitely.
    bool wait_for_checkpoint_ = false;

    // Identifies whether background threads should be cancelled.
    bool cancelled_ GUARDED_BY(mu_) = false;
    std::unique_ptr<InstantiatedCapturedFunction> instantiated_captured_func_;
  };

  const DatasetBase* const input_;
  const std::unique_ptr<CapturedFunction> captured_func_;
  const int64 cycle_length_;
  const int64 block_length_;
  const int64 num_parallel_calls_;
  const int op_version_ = 2;
  const bool sloppy_;
  const DataTypeVector output_types_;
  const std::vector<PartialTensorShape> output_shapes_;
};

ParallelInterleaveDatasetOp::ParallelInterleaveDatasetOp(
    OpKernelConstruction* ctx)
    : UnaryDatasetOpKernel(ctx) {
  FunctionMetadata::Params params;
  params.is_multi_device_function = true;
  OP_REQUIRES_OK(ctx,
                 FunctionMetadata::Create(ctx, kFunc, params, &func_metadata_));
  OP_REQUIRES_OK(ctx, ctx->GetAttr(kOutputTypes, &output_types_));
  OP_REQUIRES_OK(ctx, ctx->GetAttr(kOutputShapes, &output_shapes_));
  OP_REQUIRES_OK(ctx, ctx->GetAttr(kSloppy, &sloppy_));
}

void ParallelInterleaveDatasetOp::MakeDataset(OpKernelContext* ctx,
                                              DatasetBase* input,
                                              DatasetBase** output) {
  int64 cycle_length = 0;
  OP_REQUIRES_OK(ctx, ParseScalarArgument(ctx, kCycleLength, &cycle_length));
  if (cycle_length == model::kAutotune) {
    cycle_length = port::NumSchedulableCPUs();
  }
  OP_REQUIRES(ctx, cycle_length > 0,
              errors::InvalidArgument("`cycle_length` must be > 0"));

  int64 block_length = 0;
  OP_REQUIRES_OK(ctx, ParseScalarArgument(ctx, kBlockLength, &block_length));
  OP_REQUIRES(ctx, block_length > 0,
              errors::InvalidArgument("`block_length` must be > 0"));

  int64 num_parallel_calls = 0;
  OP_REQUIRES_OK(
      ctx, ParseScalarArgument(ctx, kNumParallelCalls, &num_parallel_calls));
  OP_REQUIRES(
      ctx, num_parallel_calls > 0 || num_parallel_calls == model::kAutotune,
      errors::InvalidArgument("num_parallel_calls must be greater than zero."));
  OP_REQUIRES(
      ctx, num_parallel_calls <= cycle_length,
      errors::InvalidArgument(
          "num_parallel_calls must less than or equal to cycle_length."));

  std::unique_ptr<CapturedFunction> captured_func;
  OP_REQUIRES_OK(ctx,
                 CapturedFunction::Create(ctx, func_metadata_, kOtherArguments,
                                          &captured_func));

  if (num_parallel_calls == model::kAutotune) {
    metrics::RecordTFDataAutotune(kDatasetType);
  }

  *output = new Dataset(ctx, input, std::move(captured_func), cycle_length,
                        block_length, num_parallel_calls, sloppy_,
                        output_types_, output_shapes_);
}

namespace {
REGISTER_KERNEL_BUILDER(Name("ParallelInterleaveDatasetV2").Device(DEVICE_CPU),
                        ParallelInterleaveDatasetOp);
REGISTER_INPUT_COLOCATION_EXEMPTION("ParallelInterleaveDatasetV2");
}  // namespace
}  // namespace data
}  // namespace tensorflow
