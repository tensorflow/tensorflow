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
#include "tensorflow/core/kernels/data/experimental/parallel_interleave_dataset_op.h"

#include <atomic>
#include <deque>
#include <functional>
#include <string>
#include <utility>

#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/common_runtime/input_colocation_exemption_registry.h"
#include "tensorflow/core/data/dataset_utils.h"
#include "tensorflow/core/data/name_utils.h"
#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/framework/partial_tensor_shape.h"
#include "tensorflow/core/framework/stats_aggregator.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/gtl/cleanup.h"
#include "tensorflow/core/lib/random/random.h"
#include "tensorflow/core/platform/blocking_counter.h"
#include "tensorflow/core/platform/stringprintf.h"
#include "tensorflow/core/profiler/lib/traceme.h"
#include "tensorflow/core/profiler/lib/traceme_encode.h"

namespace tensorflow {
namespace data {
namespace experimental {

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
    ParallelInterleaveDatasetOp::kDeterministic;
/* static */ constexpr const char* const ParallelInterleaveDatasetOp::kSloppy;
/* static */ constexpr const char* const
    ParallelInterleaveDatasetOp::kBufferOutputElements;
/* static */ constexpr const char* const
    ParallelInterleaveDatasetOp::kPrefetchInputElements;
/* static */ constexpr const char* const ParallelInterleaveDatasetOp::kFunc;
/* static */ constexpr const char* const
    ParallelInterleaveDatasetOp::kTarguments;
/* static */ constexpr const char* const
    ParallelInterleaveDatasetOp::kOutputTypes;
/* static */ constexpr const char* const
    ParallelInterleaveDatasetOp::kOutputShapes;

constexpr char kInputExhausted[] = "input_exhausted";
constexpr char kNextIndex[] = "next_index";
constexpr char kBlockCount[] = "block_count";
constexpr char kWorkersSize[] = "workers_size";
constexpr char kInterleaveSize[] = "interleave_size";
constexpr char kInterleaveIndices[] = "interleave_indices";
constexpr char kStagingSize[] = "staging_size";
constexpr char kStagingIndices[] = "staging_indices";
constexpr char kWorkerThreadsRunning[] = "worker_threads_running";
constexpr char kDataParallelInterleaveWorker[] =
    "data_parallel_interleave_worker";
constexpr char kWorker[] = "worker";
constexpr char kInputSize[] = "input_size";
constexpr char kInput[] = "input";
constexpr char kOutputsSize[] = "outputs_size";
constexpr char kOutputs[] = "outputs";
constexpr char kIsProducing[] = "is_producing";
constexpr char kWorkerThread[] = "worker_thread";
constexpr char kIteratorExhausted[] = "iterator_exhausted";
constexpr char kIteratorCreationStatus[] = "iterator_creation_status";
constexpr char kOutput[] = "output";
constexpr char kEndOfSequence[] = "end_of_sequence";
constexpr char kStatus[] = "status";
constexpr char kOutputSize[] = "output_size";
constexpr char kCode[] = "code";
constexpr char KMessage[] = "msg";

class ParallelInterleaveDatasetOp::Dataset : public DatasetBase {
 public:
  Dataset(OpKernelContext* ctx, const DatasetBase* input,
          std::unique_ptr<CapturedFunction> captured_func, int64_t cycle_length,
          int64_t block_length, DeterminismPolicy deterministic,
          int64_t buffer_output_elements, int64_t prefetch_input_elements,
          const DataTypeVector& output_types,
          const std::vector<PartialTensorShape>& output_shapes, int op_version)
      : DatasetBase(DatasetContext(ctx)),
        input_(input),
        captured_func_(std::move(captured_func)),
        cycle_length_(cycle_length),
        block_length_(block_length),
        deterministic_(deterministic),
        buffer_output_elements_(buffer_output_elements),
        prefetch_input_elements_(prefetch_input_elements),
        output_types_(output_types),
        output_shapes_(output_shapes),
        traceme_metadata_(
            {{"block_length",
              strings::Printf("%lld", static_cast<long long>(block_length))},
             {"cycle_length",
              strings::Printf("%lld", static_cast<long long>(cycle_length))},
             {"deterministic",
              deterministic.IsDeterministic() || deterministic.IsDefault()
                  ? "true"
                  : "false"}}),
        op_version_(op_version) {
    input_->Ref();
  }

  ~Dataset() override { input_->Unref(); }

  std::unique_ptr<IteratorBase> MakeIteratorInternal(
      const string& prefix) const override {
    name_utils::IteratorPrefixParams params;
    params.op_version = op_version_;
    bool deterministic =
        deterministic_.IsDeterministic() || deterministic_.IsDefault();
    return std::make_unique<Iterator>(
        Iterator::Params{
            this, name_utils::IteratorPrefix(kDatasetType, prefix, params)},
        deterministic);
  }

  const DataTypeVector& output_dtypes() const override { return output_types_; }

  const std::vector<PartialTensorShape>& output_shapes() const override {
    return output_shapes_;
  }

  string DebugString() const override {
    name_utils::DatasetDebugStringParams params;
    params.op_version = op_version_;
    return name_utils::DatasetDebugString(kDatasetType, params);
  }

  absl::Status InputDatasets(
      std::vector<const DatasetBase*>* inputs) const override {
    inputs->push_back(input_);
    return absl::OkStatus();
  }

  absl::Status CheckExternalState() const override {
    TF_RETURN_IF_ERROR(captured_func_->CheckExternalState());
    return input_->CheckExternalState();
  }

 protected:
  absl::Status AsGraphDefInternal(SerializationContext* ctx,
                                  DatasetGraphDefBuilder* b,
                                  Node** output) const override {
    std::vector<std::pair<size_t, Node*>> inputs;
    std::vector<std::pair<size_t, absl::Span<Node* const>>> list_inputs;
    int input_index = 0;

    Node* input_node;
    TF_RETURN_IF_ERROR(b->AddInputDataset(ctx, input_, &input_node));
    inputs.emplace_back(input_index++, input_node);

    std::vector<Node*> other_arguments;
    DataTypeVector other_arguments_types;
    TF_RETURN_IF_ERROR(captured_func_->AddToGraph(ctx, b, &other_arguments,
                                                  &other_arguments_types));
    list_inputs.emplace_back(input_index++, other_arguments);

    Node* cycle_length_node;
    TF_RETURN_IF_ERROR(b->AddScalar(cycle_length_, &cycle_length_node));
    inputs.emplace_back(input_index++, cycle_length_node);

    Node* block_length_node;
    TF_RETURN_IF_ERROR(b->AddScalar(block_length_, &block_length_node));
    inputs.emplace_back(input_index++, block_length_node);

    if (op_version_ == 1) {
      Node* sloppy_node;
      TF_RETURN_IF_ERROR(
          b->AddScalar(deterministic_.IsNondeterministic(), &sloppy_node));
      inputs.emplace_back(input_index++, sloppy_node);
    }

    Node* buffer_output_elements_node;
    TF_RETURN_IF_ERROR(
        b->AddScalar(buffer_output_elements_, &buffer_output_elements_node));
    inputs.emplace_back(input_index++, buffer_output_elements_node);

    Node* prefetch_input_elements_node;
    TF_RETURN_IF_ERROR(
        b->AddScalar(prefetch_input_elements_, &prefetch_input_elements_node));
    inputs.emplace_back(input_index++, prefetch_input_elements_node);

    std::vector<std::pair<StringPiece, AttrValue>> attrs;

    AttrValue f;
    b->BuildAttrValue(captured_func_->func(), &f);
    attrs.emplace_back(kFunc, f);

    if (op_version_ == 2) {
      AttrValue deterministic_attr;
      b->BuildAttrValue(deterministic_.String(), &deterministic_attr);
      attrs.emplace_back(kDeterministic, deterministic_attr);
    }

    AttrValue other_arguments_types_attr;
    b->BuildAttrValue(other_arguments_types, &other_arguments_types_attr);
    attrs.emplace_back(kTarguments, other_arguments_types_attr);

    TF_RETURN_IF_ERROR(b->AddDataset(this, inputs, list_inputs, attrs, output));
    return absl::OkStatus();
  }

 private:
  int64_t num_threads() const {
    return cycle_length_ + prefetch_input_elements_;
  }

  // Parallel interleave's implementation is designed around a few principles:
  //  1. Thread creation is relatively expensive. (Not reusing
  //     threads causes a number of indirect costs such as poorer tcmalloc
  //     performance due to thread-local caches, etc.) We allocate a fixed
  //     number of threads at the start and never change. This is why we've
  //     fused functionality that is theoretically orthogonal (i.e.
  //     .prefetch()) into the implementation.
  //  2. Drop-in replacement for standard interleave. The goal will be to
  //     auto-opt people into an optimized implementation without any work
  //     on the customer's part. We thus go through great pains to maintain
  //     identical iteration orders, full determinism (disabled only via a
  //     flag, etc.)
  //  3. Performance across a variety of environments and I/O envelopes.
  //
  // The actual implementation centers around a collection of worker threads
  // and their corresponding worker state (tracked in the `workers_` vector).
  // Worker threads repeatedly receive a vector of Tensors that are used as
  // input to the flat-map function (`captured_func_`). The output of this
  // function must be a dataset. The worker thread then repeatedly calls
  // `GetNext()`, maintaining a buffer of elements to minimize the likelihood
  // that a caller will block waiting for an element to be produced.
  //
  // Pointers to these worker states are kept in 2 disjoint data structures:
  //  1. `interleave_indices_` is a vector containing indices of WorkerStates
  //     in `workers_` that we are interleaving. Worker threads backing these
  //     WorkerStates should be regularly producing values.
  //  2. `staging_indices_` is a deque containing indices of WorkerStates in
  //     `workers_` that we will move to `interleave_indices_` when an
  //     iterator in `interleave_indices_` is exhausted.
  //
  // The client calls `GetNext[Internal]()` to retrieve an output element. The
  // internal implementation updates the state of `interleave_indices_` and
  // `staging_indices_` as output iterators (run by the worker threads) are
  // exhausted.
  //
  // `input_impl_` is the input iterator that generates arguments for the
  // flat-map function (`captured_func_`). It is set to an iterator at
  // Iterator construction, and is fixed until we consume all input elements.
  // Once it is exhausted, we reset the unique_ptr to eagerly deallocate
  // memory.
  //
  // A few invariants are maintained:
  //  1. No element in interleave_indices_ should be a -1 unless
  //     `staging_indices_` is empty and `input_impl_` is empty.
  //  2. Every `worker_` element is pointed to by at most one element of the
  //     union of `interleave_indices_` and `staging_indices_`.
  //  3. Unless `input_impl_` is empty, every `worker_` must be pointed to by
  //     an element in `interleave_indices_` or `staging_indices_`.
  class Iterator : public DatasetIterator<Dataset> {
   public:
    explicit Iterator(const Params& params, bool deterministic)
        : DatasetIterator<Dataset>(params),
          deterministic_(deterministic),
          workers_(dataset()->num_threads()),
          worker_thread_states_(dataset()->num_threads()) {}

    ~Iterator() override {
      CancelThreads();
      if (deregister_fn_) deregister_fn_();
    }

    // TODO(jsimsa): Register cancellation callback once the implementation is
    // refactored not to hold mu_ while calling `GetNext` on the input.
    absl::Status Initialize(IteratorContext* ctx) override {
      cancellation_manager_ = std::make_unique<CancellationManager>();
      IteratorContext::Params params(ctx);
      params.cancellation_manager = cancellation_manager_.get();
      TF_RETURN_IF_ERROR(dataset()->input_->MakeIterator(
          IteratorContext(params), this, prefix(), &input_impl_));
      return dataset()->captured_func_->Instantiate(
          ctx, &instantiated_captured_func_);
    }

    // It is implemented so that it matches the deterministic interleave
    // unless getting the next element would block and we are allowed to be
    // nondeterministic.
    absl::Status GetNextInternal(IteratorContext* ctx,
                                 std::vector<Tensor>* out_tensors,
                                 bool* end_of_sequence) override {
      mutex_lock l(mu_);
      TF_RETURN_IF_ERROR(EnsureWorkerThreadsStarted(ctx));
      while (!cancelled_) {
        // Wait for an item to become available, blocking if necessary. If we
        // are allowed to be nondeterministic, we can skip over input datasets
        // that do not have an item readily available.
        bool can_produce_elements = false;
        bool must_wait_for_input = true;
        for (int64_t i = 0; i < interleave_indices_.size(); ++i) {
          int64_t index = (next_index_ + i) % interleave_indices_.size();
          int64_t current_worker_index = interleave_indices_[index];
          if (current_worker_index < 0) {
            continue;  // Empty interleave elements.
          }
          WorkerState* current_worker = &workers_[current_worker_index];
          can_produce_elements |= current_worker->MayHaveElements();
          if (!current_worker->outputs.empty()) {
            // We have an element!
            next_index_ = index;
            const bool element_acquired_sloppily = !deterministic_ && i > 1;
            if (!element_acquired_sloppily) {
              // If the element was acquired in the regular (deterministic)
              // order, then advance the current block and cycle pointers to
              // the next element in the regular order.
              block_count_++;
              if (block_count_ == dataset()->block_length_) {
                next_index_ = (index + 1) % interleave_indices_.size();
                block_count_ = 0;
              }
            } else {
              block_count_ = 0;
            }
            *end_of_sequence = false;
            absl::Status s = current_worker->outputs.front().status;
            tsl::profiler::TraceMe traceme([&] {
              return tsl::profiler::TraceMeEncode(
                  "ParallelInterleaveConsume",
                  {{"element_id", current_worker->outputs.front().id}});
            });
            current_worker->outputs.front().output.swap(*out_tensors);
            current_worker->outputs.pop_front();
            current_worker->cond_var.notify_one();
            return s;
          } else if (current_worker->is_producing && deterministic_) {
            // current_worker.outputs.empty(), and we must wait for this
            // iterator.
            if (next_index_ != index) {
              // We have advanced to a new iterator; reset block counts.
              next_index_ = index;
              block_count_ = 0;
            }
            break;
          } else if (!current_worker->is_producing) {
            // This iterator has reached end of input.
            interleave_indices_[index] = -1;
            if (input_impl_) {
              // Start prefetching a new iterator.
              std::vector<Tensor> args;
              bool end_of_input = false;
              absl::Status s = input_impl_->GetNext(ctx, &args, &end_of_input);
              if (end_of_input) {
                input_impl_.reset();
              } else {
                current_worker->SetInputs(s, std::move(args));
                staging_indices_.emplace_back(current_worker_index);
              }
            }

            if (!staging_indices_.empty()) {
              // Move a worker from `staging_indices_` to
              // `interleave_indices_`.
              interleave_indices_[index] = staging_indices_.front();
              staging_indices_.pop_front();
              next_index_ = (index + 1) % interleave_indices_.size();
              block_count_ = 0;
              // Restart the inner [for] loop
              can_produce_elements = true;
              must_wait_for_input = false;
              break;
            }
          }
        }

        if (!can_produce_elements && !input_impl_) {
          // No potential for future values.
          *end_of_sequence = true;
          return absl::OkStatus();
        }

        if (must_wait_for_input) {
          // Wait for elements to become available.
          RecordStop(ctx);
          if (deterministic_) {
            workers_[interleave_indices_[next_index_]].cond_var.wait(l);
          } else {
            any_element_available_cond_var_.wait(l);
          }
          RecordStart(ctx);
        }
      }
      return errors::Cancelled(
          "ParallelInterleaveDatasetOp::Dataset::Iterator::GetNext");
    }

   protected:
    std::shared_ptr<model::Node> CreateNode(
        IteratorContext* ctx, model::Node::Args args) const override {
      return model::MakeAsyncInterleaveManyNode(
          std::move(args), {model::MakeNonTunableParameter(
                                kCycleLength, dataset()->cycle_length_),
                            model::MakeNonTunableParameter(
                                kDeterministic, deterministic_ ? 1.0 : 0.0)});
    }

    absl::Status SaveInternal(SerializationContext* ctx,
                              IteratorStateWriter* writer) override {
      TF_RETURN_IF_ERROR(ctx->HandleCheckExternalStateStatus(
          dataset()->captured_func_->CheckExternalState()));
      // The order of locking is important here to avoid deadlock.
      mutex_lock l(mu_);
      mutex_lock ckpt_l(ckpt_mu_);
      if (input_impl_) {
        TF_RETURN_IF_ERROR(SaveInput(ctx, writer, input_impl_));
      } else {
        TF_RETURN_IF_ERROR(writer->WriteScalar(prefix(), kInputExhausted, ""));
      }
      TF_RETURN_IF_ERROR(
          writer->WriteScalar(prefix(), kNextIndex, next_index_));
      TF_RETURN_IF_ERROR(
          writer->WriteScalar(prefix(), kBlockCount, block_count_));
      TF_RETURN_IF_ERROR(
          writer->WriteScalar(prefix(), kWorkersSize, workers_.size()));
      for (int i = 0; i < workers_.size(); ++i) {
        TF_RETURN_IF_ERROR(WriteWorkerStateLocked(writer, i));
      }
      for (int i = 0; i < worker_thread_states_.size(); ++i) {
        TF_RETURN_IF_ERROR(WriteWorkerThreadStateLocked(ctx, writer, i));
      }
      TF_RETURN_IF_ERROR(writer->WriteScalar(prefix(), kInterleaveSize,
                                             interleave_indices_.size()));
      for (int i = 0; i < interleave_indices_.size(); ++i) {
        TF_RETURN_IF_ERROR(writer->WriteScalar(
            prefix(), strings::StrCat(kInterleaveIndices, "_", i),
            interleave_indices_[i]));
      }
      TF_RETURN_IF_ERROR(
          writer->WriteScalar(prefix(), kStagingSize, staging_indices_.size()));
      for (int i = 0; i < staging_indices_.size(); ++i) {
        TF_RETURN_IF_ERROR(writer->WriteScalar(
            prefix(), strings::StrCat(kStagingIndices, "_", i),
            staging_indices_[i]));
      }
      if (!worker_threads_.empty()) {
        TF_RETURN_IF_ERROR(
            writer->WriteScalar(prefix(), kWorkerThreadsRunning, ""));
      }
      return absl::OkStatus();
    }

    absl::Status RestoreInternal(IteratorContext* ctx,
                                 IteratorStateReader* reader) override {
      {
        // The order of locking is important here to avoid deadlock.
        mutex_lock l(mu_);
        mutex_lock ckpt_l(ckpt_mu_);
        if (!reader->Contains(prefix(), kInputExhausted)) {
          TF_RETURN_IF_ERROR(RestoreInput(ctx, reader, input_impl_));
        } else {
          input_impl_.reset();
        }
        int64_t temp;
        TF_RETURN_IF_ERROR(reader->ReadScalar(prefix(), kNextIndex, &temp));
        next_index_ = size_t(temp);
        TF_RETURN_IF_ERROR(reader->ReadScalar(prefix(), kBlockCount, &temp));
        block_count_ = size_t(temp);

        // Restore WorkerStates.
        TF_RETURN_IF_ERROR(reader->ReadScalar(prefix(), kWorkersSize, &temp));
        if (temp != dataset()->num_threads()) {
          return errors::Internal("Expected ", dataset()->num_threads(),
                                  " worker states but found ", temp, ".");
        }
        for (size_t i = 0; i < dataset()->num_threads(); ++i) {
          TF_RETURN_IF_ERROR(ReadWorkerStateLocked(ctx, reader, i));
        }
      }
      std::unique_ptr<thread::ThreadPool> threadpool = ctx->CreateThreadPool(
          "read_worker_thread_state", dataset()->num_threads());
      absl::Status s = absl::OkStatus();
      BlockingCounter counter(dataset()->num_threads());
      for (size_t i = 0; i < dataset()->num_threads(); ++i) {
        threadpool->Schedule([this, i, ctx, reader, &s, &counter] {
          WorkerThreadState state;
          absl::Status result =
              ReadWorkerThreadStateLocked(ctx, reader, i, &state);
          mutex_lock l(mu_);
          mutex_lock ckpt_l(ckpt_mu_);
          if (!result.ok()) {
            s.Update(result);
            counter.DecrementCount();
            return;
          }
          worker_thread_states_[i] = std::move(state);
          counter.DecrementCount();
        });
      }
      counter.Wait();
      if (!s.ok()) {
        return s;
      }

      mutex_lock l(mu_);
      mutex_lock ckpt_l(ckpt_mu_);
      // Restore `interleave_indices_`.
      std::set<int64_t> all_indices;
      {
        int64_t interleave_size;
        TF_RETURN_IF_ERROR(
            reader->ReadScalar(prefix(), kInterleaveSize, &interleave_size));
        interleave_indices_.reserve(interleave_size);
        for (int64_t i = 0; i < interleave_size; ++i) {
          int64_t temp;
          TF_RETURN_IF_ERROR(reader->ReadScalar(
              prefix(), strings::StrCat(kInterleaveIndices, "_", i), &temp));
          if (temp >= 0 && all_indices.find(temp) != all_indices.end()) {
            return errors::Internal(
                "Duplicate entry for ", temp,
                " found when reading interleave and staging indices.");
          }
          if (temp >= 0) {
            all_indices.insert(temp);
          }
          interleave_indices_.emplace_back(temp);
        }
      }

      // Restore `staging_indices_`.
      {
        int64_t staging_size;
        TF_RETURN_IF_ERROR(
            reader->ReadScalar(prefix(), kStagingSize, &staging_size));
        for (int i = 0; i < staging_size; ++i) {
          int64_t temp;
          TF_RETURN_IF_ERROR(reader->ReadScalar(
              prefix(), strings::StrCat(kStagingIndices, "_", i), &temp));
          if (all_indices.find(temp) != all_indices.end()) {
            return errors::Internal(
                "Duplicate entry for ", temp,
                " found when reading interleave and staging indices.");
          }
          if (temp >= 0) {
            all_indices.insert(temp);
          }
          staging_indices_.emplace_back(temp);
        }
      }

      // Start Worker threads.
      if (reader->Contains(prefix(), kWorkerThreadsRunning)) {
        worker_threads_.reserve(dataset()->num_threads());
        for (size_t i = 0; i < dataset()->num_threads(); ++i) {
          std::shared_ptr<IteratorContext> new_ctx(new IteratorContext(*ctx));
          worker_threads_.emplace_back(ctx->StartThread(
              strings::StrCat(kDataParallelInterleaveWorker, "_", i),
              [this, new_ctx, i]() { WorkerThread(new_ctx, i); }));
        }
      }
      return absl::OkStatus();
    }

    TraceMeMetadata GetTraceMeMetadata() const override {
      return dataset()->traceme_metadata_;
    }

   private:
    // OutputElem contains the information from a call to GetNext by an output
    // iterator.
    struct OutputElem {
      // The output iterator sets `status` if getting the output element
      // fails.
      absl::Status status;
      // The buffered data element.
      std::vector<Tensor> output;
      int64_t id = -1;

      explicit OutputElem(const absl::Status& s) : status(s) {}
      OutputElem(const absl::Status& s, int64_t id) : status(s), id(id) {}
    };

    // Worker threads operate on their relevant WorkerState structs.
    //
    // WorkerState's fields are all protected by mu_;
    struct WorkerState {
      // The arguments to be used to construct an output iterator.
      std::vector<Tensor> input;
      // The buffered output elements.
      std::deque<OutputElem> outputs;
      // Set to true iff the worker thread expects to append more elements to
      // outputs. is_producing can be false despite !outputs.empty().
      // Concretely, all output elements will have been consumed only when:
      // is_producing == false && outputs.empty();
      bool is_producing = false;
      // Condition variable used to coordinate between threads. The worker
      // thread waits on this condition variable when it is either (1) waiting
      // for the main thread to add arguments to `input`, or (2) waiting for
      // the main thread to consume an element of `outputs`. The main thread
      // waits on cond_var if it is waiting for the worker thread to produce
      // an element into `outputs` (this implies deterministic==true).
      condition_variable cond_var;

      inline bool MayHaveElements() const {
        return is_producing || !outputs.empty();
      }

      // Sets inputs for a worker thread and notifies it to start processing.
      void SetInputs(const absl::Status& s,
                     std::vector<Tensor> input_arguments) {
        if (s.ok()) {
          DCHECK(!MayHaveElements())
              << "Tried to start inputs, despite already producing!";
          input = std::move(input_arguments);
          is_producing = true;
          cond_var.notify_one();
        } else {
          outputs.emplace_back(s);
        }
      }
    };

    // The internal state of a worker thread that is not already captured
    // in its `WorkerState`.
    //
    // This is needed only for checkpointing purposes. We keep this
    // separate from `WorkerState` and guard its fields using a separate
    // lock `ckpt_mu_` so as to not affect the performance of main pipeline.
    struct WorkerThreadState {
      // The output element that has been produced from the input iterator
      // and is waiting to be added to `WorkerState.outputs`.
      OutputElem output_elem;

      // Whether the input iterator returned an `end_of_sequence`.
      bool end_of_sequence = false;

      // Status returned from `MakeIteratorFromInputElement`.
      absl::Status iterator_creation_status;

      // The arguments to be used to construct `iterator`.
      std::vector<Tensor> input;

      std::unique_ptr<IteratorBase> iterator;

      WorkerThreadState() : output_elem(absl::OkStatus()) {}
    };

    void CancelThreads() TF_LOCKS_EXCLUDED(mu_) {
      cancellation_manager_->StartCancel();
      mutex_lock l(mu_);
      cancelled_ = true;
      for (auto& worker : workers_) {
        worker.cond_var.notify_all();
      }
    }

    absl::Status EnsureWorkerThreadsStarted(IteratorContext* ctx)
        TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
      if (worker_threads_.empty() && input_impl_) {
        worker_threads_.reserve(dataset()->num_threads());
        for (int64_t i = 0; i < dataset()->num_threads(); ++i) {
          std::vector<Tensor> args;
          bool end_of_input = false;
          absl::Status s = input_impl_->GetNext(ctx, &args, &end_of_input);
          if (end_of_input) {
            input_impl_.reset();
            return absl::OkStatus();
          }
          if (i < dataset()->cycle_length_) {
            interleave_indices_.push_back(i);
          } else {
            staging_indices_.push_back(i);
          }
          workers_[i].SetInputs(s, std::move(args));
          std::shared_ptr<IteratorContext> new_ctx(new IteratorContext(*ctx));
          worker_threads_.push_back(ctx->StartThread(
              strings::StrCat(kDataParallelInterleaveWorker, "_", i),
              [this, new_ctx, i]() { WorkerThread(new_ctx, i); }));
        }
        DCHECK(interleave_indices_.size() == dataset()->cycle_length_);
        DCHECK(staging_indices_.size() == dataset()->prefetch_input_elements_);
      }
      return absl::OkStatus();
    }

    // Produces elements into the worker's output buffers.
    void WorkerThread(const std::shared_ptr<IteratorContext>& ctx,
                      const int64_t thread_index) {
      // Notes on checkpointing thread local state, i.e., `WorkerThreadState`:
      //
      // 1. Any local state that may need to be checkpointed should be kept
      //    in `worker_thread_states_[thread_index]`.
      // 2. `WorkerThreadState` should contain state that is needed only for
      //    checkpointing, i.e., if we were to remove checkpointing support,
      //    we could keep that state as local variables in this thread.
      // 3. This thread should only read/write state at `thread_index`
      //    and should not access other thread states.
      // 4. When restoring from checkpoint, threads are started only after
      //    the restore is complete.
      // 5. Once restored from a checkpoint, the local state is edited only
      //    by this thread. 3 & 4 allow making assumptions like temporarily
      //    caching local state in this thread and using it outside a lock
      //    e.g. `make_new_iterator`.
      // 6. `ckpt_mu_` should be wisely used to create *consistent*
      //    checkpoint markers.

      // std::function arguments are copy-constructable, so we pass raw
      // pointers, and then immediately wrap them to ensure correct ownership.
      RecordStart(ctx.get());
      auto cleanup = gtl::MakeCleanup([this, thread_index, ctx] {
        mutex_lock l(mu_);
        workers_[thread_index].cond_var.notify_all();
        RecordStop(ctx.get());
      });
      bool make_new_iterator;
      {
        tf_shared_lock l(ckpt_mu_);
        // Decide whether a new iterator should be built.
        // 1. If there is an existing iterator, we use it.
        // 2. If there was an error in iterator creation that could not be
        //    notified to the client we attempt to send that to the client
        //    first.
        make_new_iterator =
            worker_thread_states_[thread_index].iterator == nullptr &&
            worker_thread_states_[thread_index].iterator_creation_status.ok();
      }

      bool thread_potentially_in_staging = true;

      // Even though `make_new_iterator` has cached values from
      // `worker_thread_states_[thread_index]` which is guarded by ckpt_mu_,
      // it is safe to *read* `make_new_iterator`outside of a lock without
      // worrying about concurrent changes to values in
      // `worker_thread_states_[thread_index]`. See comment at the start of
      // this function for details.
      while (true) {
        // Whether creation of the iterator succeeded.
        absl::Status iterator_creation_status;
        // 1. Build a new iterator or use the existing one.
        if (make_new_iterator) {
          // 1a. Get new input tensors or use the exiting ones.
          bool read_new_input;
          {
            tf_shared_lock l(ckpt_mu_);
            // worker_thread_states_[thread_index].input will be non-empty
            // if checkpointing happened at CHECKPOINT_MARKER_A.
            read_new_input = worker_thread_states_[thread_index].input.empty();
          }

          if (read_new_input) {
            mutex_lock l(mu_);
            while (!cancelled_ && !workers_[thread_index].is_producing) {
              RecordStop(ctx.get());
              workers_[thread_index].cond_var.wait(l);
              RecordStart(ctx.get());
            }
            if (cancelled_) return;
            // Copy the input tensors so that we do not need to block on `mu_`
            // when building the iterator.
            // We keep a copy of the input tensors in
            // `WorkerThreadState.input` till the iterator is in use. This is
            // used in `RestoreInternal` to re-build the iterator.
            // TODO(b/78046638): Explore ways to avoid tracking the input
            // tensors.
            tf_shared_lock ckpt_l(ckpt_mu_);
            worker_thread_states_[thread_index].input.swap(
                workers_[thread_index].input);
            // CHECKPOINT_MARKER_A
            // We have the input tensors but have not built the iterator yet.
          }
          {
            mutex_lock l(mu_);
            thread_potentially_in_staging =
                absl::c_find(staging_indices_, thread_index) !=
                staging_indices_.end();
          }
          // 1b. Run the user defined function to produce a new iterator.
          {
            tf_shared_lock l(ckpt_mu_);
            worker_thread_states_[thread_index].iterator_creation_status =
                MakeIteratorFromInputElement(
                    ctx.get(), this, worker_thread_states_[thread_index].input,
                    thread_index, *instantiated_captured_func_, prefix(),
                    &worker_thread_states_[thread_index].iterator,
                    model_node());
            iterator_creation_status =
                worker_thread_states_[thread_index].iterator_creation_status;
            if (!iterator_creation_status.ok()) {
              worker_thread_states_[thread_index].input.clear();
            } else if (thread_potentially_in_staging) {
              // Disable auto tune modeling while we are paused.
              // TODO(wilsin): Write a unit test where we iterate through a
              // dataset, pause, and check the model proto autotune value.
              DisableAutotune(
                  ctx.get(),
                  worker_thread_states_[thread_index].iterator.get());
            }
            // CHECKPOINT_MARKER_B
            // Either an iterator has been successfully built and placed in
            // `worker_thread_states_[thread_index].iterator` or it failed and
            // a non-OK status has been put in
            // `worker_thread_states_[thread_index].iterator_creation_status`.
          }
        } else {
          tf_shared_lock l(ckpt_mu_);
          iterator_creation_status =
              worker_thread_states_[thread_index].iterator_creation_status;
          // Mark that we have used up the restored iterator.
          make_new_iterator = true;
        }
        // 2. Start producing elements or send error state to client if
        //    iterator creation failed.
        if (!iterator_creation_status.ok()) {
          mutex_lock l(mu_);
          // Wait for space in the prefetch queue.
          while (!cancelled_ && workers_[thread_index].outputs.size() ==
                                    dataset()->buffer_output_elements_) {
            RecordStop(ctx.get());
            workers_[thread_index].cond_var.wait(l);
            RecordStart(ctx.get());
          }
          if (cancelled_) return;
          tf_shared_lock ckpt_l(ckpt_mu_);
          workers_[thread_index].outputs.emplace_back(iterator_creation_status);
          workers_[thread_index].is_producing = false;
          worker_thread_states_[thread_index].iterator_creation_status =
              absl::OkStatus();
          // CHECKPOINT_MARKER_C
          // Non-OK iterator creation status has been notified to the
          // client.
          if (deterministic_) {
            workers_[thread_index].cond_var.notify_one();
          } else {
            any_element_available_cond_var_.notify_one();
          }
        } else {
          bool end_of_sequence = false;
          while (!end_of_sequence) {
            if (thread_potentially_in_staging) {
              // Check whether we have left the staging state and reenable
              // auto tune modeling.
              mutex_lock l(mu_);
              thread_potentially_in_staging =
                  absl::c_find(staging_indices_, thread_index) !=
                  staging_indices_.end();
              if (!thread_potentially_in_staging) {
                tf_shared_lock l(ckpt_mu_);
                EnableAutotune(
                    ctx.get(),
                    worker_thread_states_[thread_index].iterator.get());
              }
            }

            // 3.a Produce an element!
            {
              tf_shared_lock ckpt_l(ckpt_mu_);
              if (worker_thread_states_[thread_index].output_elem.status.ok() &&
                  worker_thread_states_[thread_index]
                      .output_elem.output.empty() &&
                  !worker_thread_states_[thread_index].end_of_sequence) {
                int64_t& id =
                    worker_thread_states_[thread_index].output_elem.id;
                tsl::profiler::TraceMe traceme(
                    [&] {
                      id = tsl::profiler::TraceMe::NewActivityId();
                      return tsl::profiler::TraceMeEncode(
                          "ParallelInterleaveProduce", {{"element_id", id}});
                    },
                    profiler::kInfo);
                worker_thread_states_[thread_index].output_elem.status =
                    worker_thread_states_[thread_index].iterator->GetNext(
                        ctx.get(),
                        &worker_thread_states_[thread_index].output_elem.output,
                        &worker_thread_states_[thread_index].end_of_sequence);
                end_of_sequence =
                    worker_thread_states_[thread_index].end_of_sequence;
              } else {
                end_of_sequence =
                    worker_thread_states_[thread_index].end_of_sequence;
              }
              // CHECKPOINT_MARKER_D
              // An element has been read or an error or end_of_sequence has
              // been received from the input iterator and is waiting to be
              // sent to client.
            }

            // 3.b Make it available to the client.
            {
              mutex_lock l(mu_);

              // Wait for space in the prefetch queue.
              while (!cancelled_ && workers_[thread_index].outputs.size() ==
                                        dataset()->buffer_output_elements_) {
                RecordStop(ctx.get());
                workers_[thread_index].cond_var.wait(l);
                RecordStart(ctx.get());
              }
              if (cancelled_) return;

              tf_shared_lock ckpt_l(ckpt_mu_);
              workers_[thread_index].is_producing = !end_of_sequence;

              // Output the element.

              // Move the temporary state in WorkerThreadState to WorkerState
              // and mark it as used.
              if (end_of_sequence) {
                worker_thread_states_[thread_index].iterator.reset();
                worker_thread_states_[thread_index].input.clear();
                worker_thread_states_[thread_index].end_of_sequence = false;
              } else {
                workers_[thread_index].outputs.emplace_back(
                    worker_thread_states_[thread_index].output_elem.status,
                    worker_thread_states_[thread_index].output_elem.id);
                workers_[thread_index].outputs.back().output.swap(
                    worker_thread_states_[thread_index].output_elem.output);
              }
              worker_thread_states_[thread_index].output_elem.status =
                  absl::OkStatus();
              if (deterministic_) {
                workers_[thread_index].cond_var.notify_one();
              } else {
                any_element_available_cond_var_.notify_one();
              }
              // CHECKPOINT_MARKER_E
              // Output element or iterator status has been sent to the
              // client.
            }
          }
        }
      }
    }

    absl::Status WriteWorkerStateLocked(IteratorStateWriter* writer, int index)
        TF_EXCLUSIVE_LOCKS_REQUIRED(mu_, ckpt_mu_) {
      string iterator_name =
          strings::StrCat(prefix(), "::", kWorker, "_", index);
      TF_RETURN_IF_ERROR(writer->WriteScalar(iterator_name, kInputSize,
                                             workers_[index].input.size()));
      for (int i = 0; i < workers_[index].input.size(); ++i) {
        TF_RETURN_IF_ERROR(writer->WriteTensor(iterator_name,
                                               strings::StrCat(kInput, "_", i),
                                               workers_[index].input[i]));
      }
      TF_RETURN_IF_ERROR(writer->WriteScalar(iterator_name, kOutputsSize,
                                             workers_[index].outputs.size()));
      for (int i = 0; i < workers_[index].outputs.size(); ++i) {
        TF_RETURN_IF_ERROR(WriteOutputElemLocked(
            writer, workers_[index].outputs[i], iterator_name,
            strings::StrCat(kOutputs, "_", i)));
      }
      if (workers_[index].is_producing) {
        TF_RETURN_IF_ERROR(
            writer->WriteScalar(iterator_name, kIsProducing, ""));
      }
      return absl::OkStatus();
    }

    absl::Status ReadWorkerStateLocked(IteratorContext* ctx,
                                       IteratorStateReader* reader, int index)
        TF_EXCLUSIVE_LOCKS_REQUIRED(mu_, ckpt_mu_) {
      string worker_prefix =
          strings::StrCat(prefix(), "::", kWorker, "_", index);
      // Restore inputs.
      int64_t input_size;
      TF_RETURN_IF_ERROR(
          reader->ReadScalar(worker_prefix, kInputSize, &input_size));
      workers_[index].input.reserve(input_size);
      for (int i = 0; i < input_size; ++i) {
        workers_[index].input.emplace_back();
        TF_RETURN_IF_ERROR(reader->ReadTensor(ctx->flr(), worker_prefix,
                                              strings::StrCat(kInput, "_", i),
                                              &workers_[index].input.back()));
      }
      int64_t outputs_size;
      TF_RETURN_IF_ERROR(
          reader->ReadScalar(worker_prefix, kOutputsSize, &outputs_size));
      for (int i = 0; i < outputs_size; ++i) {
        workers_[index].outputs.emplace_back(absl::OkStatus());
        TF_RETURN_IF_ERROR(ReadOutputElemLocked(
            ctx, reader, &workers_[index].outputs.back(), worker_prefix,
            strings::StrCat(kOutputs, "_", i)));
      }
      if (reader->Contains(worker_prefix, kIsProducing)) {
        workers_[index].is_producing = true;
      } else {
        workers_[index].is_producing = false;
      }
      return absl::OkStatus();
    }

    absl::Status WriteWorkerThreadStateLocked(SerializationContext* ctx,
                                              IteratorStateWriter* writer,
                                              int index)
        TF_EXCLUSIVE_LOCKS_REQUIRED(mu_, ckpt_mu_) {
      string iterator_name =
          strings::StrCat(prefix(), "::", kWorkerThread, "_", index);
      if (worker_thread_states_[index].iterator != nullptr) {
        TF_RETURN_IF_ERROR(
            SaveInput(ctx, writer, worker_thread_states_[index].iterator));
      } else {
        TF_RETURN_IF_ERROR(
            writer->WriteScalar(iterator_name, kIteratorExhausted, ""));
      }
      TF_RETURN_IF_ERROR(
          writer->WriteScalar(iterator_name, kInputSize,
                              worker_thread_states_[index].input.size()));
      for (int i = 0; i < worker_thread_states_[index].input.size(); ++i) {
        TF_RETURN_IF_ERROR(
            writer->WriteTensor(iterator_name, strings::StrCat(kInput, "_", i),
                                worker_thread_states_[index].input[i]));
      }
      TF_RETURN_IF_ERROR(WriteStatusLocked(
          writer, iterator_name, kIteratorCreationStatus,
          worker_thread_states_[index].iterator_creation_status));
      TF_RETURN_IF_ERROR(WriteOutputElemLocked(
          writer, worker_thread_states_[index].output_elem, iterator_name,
          kOutput));
      if (worker_thread_states_[index].end_of_sequence) {
        TF_RETURN_IF_ERROR(
            writer->WriteScalar(iterator_name, kEndOfSequence, ""));
      }
      return absl::OkStatus();
    }

    absl::Status ReadWorkerThreadStateLocked(IteratorContext* ctx,
                                             IteratorStateReader* reader,
                                             int index,
                                             WorkerThreadState* state) {
      string worker_prefix =
          strings::StrCat(prefix(), "::", kWorkerThread, "_", index);
      // Restore inputs.
      int64_t input_size;
      TF_RETURN_IF_ERROR(
          reader->ReadScalar(worker_prefix, kInputSize, &input_size));
      state->input.reserve(input_size);
      for (int i = 0; i < input_size; ++i) {
        state->input.emplace_back();
        TF_RETURN_IF_ERROR(reader->ReadTensor(ctx->flr(), worker_prefix,
                                              strings::StrCat(kInput, "_", i),
                                              &state->input.back()));
      }
      // Restore iterator
      if (reader->Contains(worker_prefix, kIteratorExhausted)) {
        state->iterator.reset();
      } else {
        std::unique_ptr<IteratorBase> iterator;
        // NOTE: We intentionally ignore resource modeling outside GetNext().
        TF_RETURN_IF_ERROR(MakeIteratorFromInputElement(
            ctx, this, state->input, index, *instantiated_captured_func_,
            prefix(), &iterator, /*node=*/nullptr));
        TF_RETURN_IF_ERROR(RestoreInput(ctx, reader, iterator));
        state->iterator.swap(iterator);
      }
      TF_RETURN_IF_ERROR(ReadStatusLocked(reader, worker_prefix,
                                          kIteratorCreationStatus,
                                          &state->iterator_creation_status));
      TF_RETURN_IF_ERROR(ReadOutputElemLocked(ctx, reader, &state->output_elem,
                                              worker_prefix, kOutput));
      if (reader->Contains(worker_prefix, kEndOfSequence)) {
        state->end_of_sequence = true;
      } else {
        state->end_of_sequence = false;
      }
      return absl::OkStatus();
    }

    absl::Status WriteOutputElemLocked(IteratorStateWriter* writer,
                                       const OutputElem& output_elem,
                                       const string& iterator_name,
                                       const string& prefix)
        TF_EXCLUSIVE_LOCKS_REQUIRED(mu_, ckpt_mu_) {
      TF_RETURN_IF_ERROR(WriteStatusLocked(
          writer, iterator_name, strings::StrCat(prefix, "_", kStatus),
          output_elem.status));
      TF_RETURN_IF_ERROR(writer->WriteScalar(
          iterator_name, strings::StrCat(prefix, "_", kOutputSize),
          output_elem.output.size()));
      for (int i = 0; i < output_elem.output.size(); ++i) {
        TF_RETURN_IF_ERROR(writer->WriteTensor(
            iterator_name, strings::StrCat(prefix, "_", kOutput, "_", i),
            output_elem.output[i]));
      }
      return absl::OkStatus();
    }

    absl::Status ReadOutputElemLocked(IteratorContext* ctx,
                                      IteratorStateReader* reader,
                                      OutputElem* output_elem,
                                      const string& iterator_name,
                                      const string& prefix) {
      TF_RETURN_IF_ERROR(ReadStatusLocked(reader, iterator_name,
                                          strings::StrCat(prefix, "_", kStatus),
                                          &output_elem->status));
      int64_t output_size;
      TF_RETURN_IF_ERROR(reader->ReadScalar(
          iterator_name, strings::StrCat(prefix, "_", kOutputSize),
          &output_size));
      output_elem->output.reserve(output_size);
      for (int i = 0; i < output_size; ++i) {
        output_elem->output.emplace_back();
        TF_RETURN_IF_ERROR(
            reader->ReadTensor(ctx->flr(), iterator_name,
                               strings::StrCat(prefix, "_", kOutput, "_", i),
                               &output_elem->output.back()));
      }
      return absl::OkStatus();
    }

    absl::Status WriteStatusLocked(IteratorStateWriter* writer,
                                   const string& iterator_name,
                                   const string& prefix,
                                   const absl::Status& status)
        TF_EXCLUSIVE_LOCKS_REQUIRED(mu_, ckpt_mu_) {
      TF_RETURN_IF_ERROR(writer->WriteScalar(
          iterator_name, strings::StrCat(prefix, "_", kCode),
          static_cast<int64_t>(status.code())));
      if (!status.ok()) {
        TF_RETURN_IF_ERROR(writer->WriteScalar(
            iterator_name, strings::StrCat(prefix, "_", KMessage),
            std::string(status.message())));
      }
      return absl::OkStatus();
    }

    absl::Status ReadStatusLocked(IteratorStateReader* reader,
                                  const string& iterator_name,
                                  const string& prefix, absl::Status* status) {
      int64_t code_int;
      TF_RETURN_IF_ERROR(reader->ReadScalar(
          iterator_name, strings::StrCat(prefix, "_", kCode), &code_int));
      absl::StatusCode code = static_cast<absl::StatusCode>(code_int);

      if (code != absl::StatusCode::kOk) {
        tstring error_message;
        TF_RETURN_IF_ERROR(reader->ReadScalar(
            iterator_name, strings::StrCat(prefix, "_", KMessage),
            &error_message));
        *status = absl::Status(code, error_message);
      } else {
        *status = absl::OkStatus();
      }
      return absl::OkStatus();
    }

    // Mutex & condition variable to guard mutable iterator internals and
    // coordinate among worker threads and client thread[s].
    mutex mu_ TF_ACQUIRED_BEFORE(ckpt_mu_);
    // The main thread waits on this condition variable if running in
    // nondeterministic mode and no values are available.
    condition_variable any_element_available_cond_var_;
    // Whether outputs must be produced in deterministic order.
    const bool deterministic_;
    // Mutex used to wait for a consistent state while checkpointing.
    // Only Save and Restore require an exclusive lock on this mutex. In
    // other scenarios we just acquire a shared lock so the pipeline's
    // performance should not be affected in the absence of checkpointing.
    // A thread must not wait on any condition variable while holding
    // `ckpt_mu_` in either shared or exclusive modes.
    mutex ckpt_mu_;

    // Controls cancellation of `input_impl_`. Must be ordered before
    // `input_impl_` so that `input_impl_` is destroyed first.
    std::unique_ptr<CancellationManager> cancellation_manager_;

    // The iterator producing elements which are converted to datasets by
    // the dataset()->captured_func_ then interleaved together.
    // input_impl_ is reset when we have exhausted its input.
    std::unique_ptr<IteratorBase> input_impl_ TF_GUARDED_BY(mu_);

    std::unique_ptr<InstantiatedCapturedFunction> instantiated_captured_func_;

    // The WorkerState structs the worker threads operate on.
    // workers_ elements are in at most one of interleave_ and staging_.
    std::vector<WorkerState> workers_ TF_GUARDED_BY(mu_);

    // Stores the temporary state of WorkerThreads which is not stored in
    // WorkerState. This is used for checkpointing purposes only.
    std::vector<WorkerThreadState> worker_thread_states_
        TF_GUARDED_BY(ckpt_mu_);

    // Indices in `workers_` of iterators to interleave.
    std::vector<int64_t> interleave_indices_ TF_GUARDED_BY(mu_);
    // Indices in `workers_` of prefetched iterators.
    std::deque<int64_t> staging_indices_ TF_GUARDED_BY(mu_);

    // The index into output_elements_ for next element to produce.
    size_t next_index_ TF_GUARDED_BY(mu_) = 0;
    // The number of items produced so far within the block
    size_t block_count_ TF_GUARDED_BY(mu_) = 0;
    // Flag to instruct the worker threads to exit.
    bool cancelled_ TF_GUARDED_BY(mu_) = false;
    // The worker threads. This must be last to ensure the
    // threads have exited before any other members are deallocated.
    // TODO(b/65178177): Avoid allocating additional threads.
    std::vector<std::unique_ptr<Thread>> worker_threads_ TF_GUARDED_BY(mu_);

    // Method for deregistering the cancellation callback.
    std::function<void()> deregister_fn_;
  };

  const DatasetBase* const input_;
  const std::unique_ptr<CapturedFunction> captured_func_;
  const int64_t cycle_length_;
  const int64_t block_length_;
  const DeterminismPolicy deterministic_;
  const int64_t buffer_output_elements_;
  const int64_t prefetch_input_elements_;
  const DataTypeVector output_types_;
  const std::vector<PartialTensorShape> output_shapes_;
  const TraceMeMetadata traceme_metadata_;
  const int op_version_;
};

ParallelInterleaveDatasetOp::ParallelInterleaveDatasetOp(
    OpKernelConstruction* ctx)
    : UnaryDatasetOpKernel(ctx),
      op_version_(ctx->HasAttr(kDeterministic) ? 2 : 1) {
  OP_REQUIRES_OK(ctx, FunctionMetadata::Create(ctx, kFunc, /*params=*/{},
                                               &func_metadata_));
  if (op_version_ == 2) {
    std::string deterministic;
    OP_REQUIRES_OK(ctx, ctx->GetAttr(kDeterministic, &deterministic));
    OP_REQUIRES_OK(
        ctx, DeterminismPolicy::FromString(deterministic, &deterministic_));
  }
  OP_REQUIRES_OK(ctx, ctx->GetAttr(kOutputTypes, &output_types_));
  OP_REQUIRES_OK(ctx, ctx->GetAttr(kOutputShapes, &output_shapes_));
}

void ParallelInterleaveDatasetOp::MakeDataset(OpKernelContext* ctx,
                                              DatasetBase* input,
                                              DatasetBase** output) {
  int64_t cycle_length = 0;
  OP_REQUIRES_OK(ctx, ParseScalarArgument(ctx, kCycleLength, &cycle_length));
  OP_REQUIRES(ctx, cycle_length > 0,
              errors::InvalidArgument("`cycle_length` must be > 0"));

  int64_t block_length = 0;
  OP_REQUIRES_OK(ctx, ParseScalarArgument(ctx, kBlockLength, &block_length));
  OP_REQUIRES(ctx, block_length > 0,
              errors::InvalidArgument("`block_length` must be > 0"));

  if (op_version_ == 1) {
    bool sloppy = false;
    OP_REQUIRES_OK(ctx, ParseScalarArgument(ctx, kSloppy, &sloppy));
    if (sloppy) {
      deterministic_ =
          DeterminismPolicy(DeterminismPolicy::Type::kNondeterministic);
    } else {
      deterministic_ =
          DeterminismPolicy(DeterminismPolicy::Type::kDeterministic);
    }
  }

  int64_t buffer_output_elements = 0;
  OP_REQUIRES_OK(ctx, ParseScalarArgument(ctx, kBufferOutputElements,
                                          &buffer_output_elements));
  OP_REQUIRES(ctx, buffer_output_elements > 0,
              errors::InvalidArgument("`buffer_output_elements` must be > 0"));

  int64_t prefetch_input_elements = 0;
  OP_REQUIRES_OK(ctx, ParseScalarArgument(ctx, kPrefetchInputElements,
                                          &prefetch_input_elements));
  OP_REQUIRES(
      ctx, prefetch_input_elements >= 0,
      errors::InvalidArgument("`prefetch_input_elements` must be >= 0"));

  std::unique_ptr<CapturedFunction> captured_func;
  OP_REQUIRES_OK(ctx,
                 CapturedFunction::Create(ctx, func_metadata_, kOtherArguments,
                                          &captured_func));

  *output = new Dataset(ctx, input, std::move(captured_func), cycle_length,
                        block_length, deterministic_, buffer_output_elements,
                        prefetch_input_elements, output_types_, output_shapes_,
                        op_version_);
}

namespace {
REGISTER_KERNEL_BUILDER(Name("ParallelInterleaveDataset").Device(DEVICE_CPU),
                        ParallelInterleaveDatasetOp);
REGISTER_KERNEL_BUILDER(
    Name("ExperimentalParallelInterleaveDataset").Device(DEVICE_CPU),
    ParallelInterleaveDatasetOp);
REGISTER_KERNEL_BUILDER(
    Name("LegacyParallelInterleaveDatasetV2").Device(DEVICE_CPU),
    ParallelInterleaveDatasetOp);

REGISTER_INPUT_COLOCATION_EXEMPTION("ParallelInterleaveDataset");
REGISTER_INPUT_COLOCATION_EXEMPTION("ExperimentalParallelInterleaveDataset");
REGISTER_INPUT_COLOCATION_EXEMPTION("LegacyParallelInterleaveDatasetV2");

}  // namespace
}  // namespace experimental
}  // namespace data
}  // namespace tensorflow
