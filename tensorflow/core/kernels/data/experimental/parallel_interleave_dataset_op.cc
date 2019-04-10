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

namespace tensorflow {
namespace data {
namespace {

// See documentation in ../../ops/dataset_ops.cc for a high-level
// description of the following op.

class ParallelInterleaveDatasetOp : public UnaryDatasetOpKernel {
 public:
  explicit ParallelInterleaveDatasetOp(OpKernelConstruction* ctx)
      : UnaryDatasetOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("f", &interleave_func_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("output_types", &output_types_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("output_shapes", &output_shapes_));
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

    bool sloppy = false;
    OP_REQUIRES_OK(ctx, ParseScalarArgument(ctx, "sloppy", &sloppy));

    int64 buffer_output_elements = 0;
    OP_REQUIRES_OK(ctx, ParseScalarArgument(ctx, "buffer_output_elements",
                                            &buffer_output_elements));
    OP_REQUIRES(
        ctx, buffer_output_elements > 0,
        errors::InvalidArgument("`buffer_output_elements` must be > 0"));

    int64 prefetch_input_elements = 0;
    OP_REQUIRES_OK(ctx, ParseScalarArgument(ctx, "prefetch_input_elements",
                                            &prefetch_input_elements));
    OP_REQUIRES(
        ctx, prefetch_input_elements >= 0,
        errors::InvalidArgument("`prefetch_input_elements` must be >= 0"));

    std::unique_ptr<CapturedFunction> captured_func;
    CapturedFunction::Params params;
    params.lib_def = lib_def_;
    OP_REQUIRES_OK(
        ctx, CapturedFunction::Create(interleave_func_, ctx, "other_arguments",
                                      std::move(params), &captured_func));

    *output =
        new Dataset(ctx, input, interleave_func_, std::move(captured_func),
                    cycle_length, block_length, sloppy, buffer_output_elements,
                    prefetch_input_elements, output_types_, output_shapes_);
  }

 private:
  class Dataset : public DatasetBase {
   public:
    Dataset(OpKernelContext* ctx, const DatasetBase* input,
            const NameAttrList& func,
            std::unique_ptr<CapturedFunction> captured_func, int64 cycle_length,
            int64 block_length, bool sloppy, int64 buffer_output_elements,
            int64 prefetch_input_elements, const DataTypeVector& output_types,
            const std::vector<PartialTensorShape>& output_shapes)
        : DatasetBase(DatasetContext(ctx)),
          input_(input),
          interleave_func_(func),
          captured_func_(std::move(captured_func)),
          cycle_length_(cycle_length),
          block_length_(block_length),
          sloppy_(sloppy),
          buffer_output_elements_(buffer_output_elements),
          prefetch_input_elements_(prefetch_input_elements),
          output_types_(output_types),
          output_shapes_(output_shapes) {
      input_->Ref();
    }

    ~Dataset() override { input_->Unref(); }

    std::unique_ptr<IteratorBase> MakeIteratorInternal(
        const string& prefix) const override {
      return absl::make_unique<Iterator>(Iterator::Params{
          this, strings::StrCat(prefix, "::ParallelInterleave")});
    }

    const DataTypeVector& output_dtypes() const override {
      return output_types_;
    }

    const std::vector<PartialTensorShape>& output_shapes() const override {
      return output_shapes_;
    }

    string DebugString() const override {
      return "ParallelInterleaveDatasetOp::Dataset";
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
      Node* sloppy_node;
      TF_RETURN_IF_ERROR(b->AddScalar(sloppy_, &sloppy_node));
      Node* buffer_output_elements_node;
      TF_RETURN_IF_ERROR(
          b->AddScalar(buffer_output_elements_, &buffer_output_elements_node));
      Node* prefetch_input_elements_node;
      TF_RETURN_IF_ERROR(b->AddScalar(prefetch_input_elements_,
                                      &prefetch_input_elements_node));
      std::vector<Node*> other_arguments;
      DataTypeVector other_arguments_types;
      TF_RETURN_IF_ERROR(captured_func_->AddToGraph(ctx, b, &other_arguments,
                                                    &other_arguments_types));
      AttrValue f;
      b->BuildAttrValue(interleave_func_, &f);
      AttrValue other_arguments_types_attr;
      b->BuildAttrValue(other_arguments_types, &other_arguments_types_attr);

      TF_RETURN_IF_ERROR(b->AddDataset(
          this,
          {{0, input_node},
           {2, cycle_length_node},
           {3, block_length_node},
           {4, sloppy_node},
           {5, buffer_output_elements_node},
           {6, prefetch_input_elements_node}},
          {{1, other_arguments}},
          {{"f", f}, {"Targuments", other_arguments_types_attr}}, output));
      return Status::OK();
    }

   private:
    int64 num_threads() const {
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
      explicit Iterator(const Params& params)
          : DatasetIterator<Dataset>(params),
            workers_(dataset()->num_threads()),
            worker_thread_states_(dataset()->num_threads()) {}

      ~Iterator() override {
        mutex_lock l(mu_);
        cancelled_ = true;
        // Notify all workers in case they are blocked.
        for (auto& worker : workers_) {
          worker.cond_var.notify_all();
        }
      }

      Status Initialize(IteratorContext* ctx) override {
        TF_RETURN_IF_ERROR(
            dataset()->input_->MakeIterator(ctx, prefix(), &input_impl_));
        return dataset()->captured_func_->Instantiate(
            ctx, &instantiated_captured_func_);
      }

      // It is implemented so that it matches the deterministic interleave
      // unless getting the next element would block and we are allowed to be
      // sloppy.
      Status GetNextInternal(IteratorContext* ctx,
                             std::vector<Tensor>* out_tensors,
                             bool* end_of_sequence) override {
        mutex_lock l(mu_);
        TF_RETURN_IF_ERROR(EnsureWorkerThreadsStarted(ctx));
        while (!cancelled_) {
          // Wait for an item to become available, blocking if necessary. If we
          // are allowed to be sloppy, we can skip over input datasets that do
          // not have an item readily available.
          bool can_produce_elements = false;
          bool must_wait_for_input = true;
          for (int64 i = 0; i < interleave_indices_.size(); ++i) {
            int64 index = (next_index_ + i) % interleave_indices_.size();
            int64 current_worker_index = interleave_indices_[index];
            if (current_worker_index < 0) {
              continue;  // Empty interleave elements.
            }
            WorkerState* current_worker = &workers_[current_worker_index];
            can_produce_elements |= current_worker->MayHaveElements();
            if (!current_worker->outputs.empty()) {
              // We have an element!
              next_index_ = index;
              const bool element_acquired_sloppily =
                  dataset()->sloppy_ && i > 1;
              if (!element_acquired_sloppily) {
                // If the element was acquired in the regular (non-sloppy)
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
              Status s = current_worker->outputs.front().status;
              current_worker->outputs.front().output.swap(*out_tensors);
              current_worker->outputs.pop_front();
              current_worker->cond_var.notify_one();
              return s;
            } else if (current_worker->is_producing && !dataset()->sloppy_) {
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
                Status s = input_impl_->GetNext(ctx, &args, &end_of_input);
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
            return Status::OK();
          }

          if (must_wait_for_input) {
            // Wait for elements to become available.
            RecordStop(ctx);
            if (dataset()->sloppy_) {
              sloppy_cond_var_.wait(l);
            } else {
              workers_[interleave_indices_[next_index_]].cond_var.wait(l);
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
        return model::MakeAsyncInterleaveManyNode(std::move(args),
                                                  /*parameters=*/{});
      }

      Status SaveInternal(IteratorStateWriter* writer) override {
        // The order of locking is important here to avoid deadlock.
        mutex_lock l(mu_);
        mutex_lock ckpt_l(ckpt_mu_);
        if (input_impl_) {
          TF_RETURN_IF_ERROR(SaveInput(writer, input_impl_));
        } else {
          TF_RETURN_IF_ERROR(
              writer->WriteScalar(full_name("input_exhausted"), ""));
        }
        TF_RETURN_IF_ERROR(
            writer->WriteScalar(full_name("next_index"), next_index_));
        TF_RETURN_IF_ERROR(
            writer->WriteScalar(full_name("block_count"), block_count_));
        TF_RETURN_IF_ERROR(
            writer->WriteScalar(full_name("workers_size"), workers_.size()));
        for (int i = 0; i < workers_.size(); ++i) {
          TF_RETURN_IF_ERROR(WriteWorkerStateLocked(writer, i));
        }
        for (int i = 0; i < worker_thread_states_.size(); ++i) {
          TF_RETURN_IF_ERROR(WriteWorkerThreadStateLocked(writer, i));
        }
        TF_RETURN_IF_ERROR(writer->WriteScalar(full_name("interleave_size"),
                                               interleave_indices_.size()));
        for (int i = 0; i < interleave_indices_.size(); ++i) {
          TF_RETURN_IF_ERROR(writer->WriteScalar(
              full_name(strings::StrCat("interleave_indices_", i)),
              interleave_indices_[i]));
        }
        TF_RETURN_IF_ERROR(writer->WriteScalar(full_name("staging_size"),
                                               staging_indices_.size()));
        for (int i = 0; i < staging_indices_.size(); ++i) {
          TF_RETURN_IF_ERROR(writer->WriteScalar(
              full_name(strings::StrCat("staging_indices_", i)),
              staging_indices_[i]));
        }
        if (!worker_threads_.empty()) {
          TF_RETURN_IF_ERROR(
              writer->WriteScalar(full_name("worker_threads_running"), ""));
        }
        return Status::OK();
      }

      Status RestoreInternal(IteratorContext* ctx,
                             IteratorStateReader* reader) override {
        // The order of locking is important here to avoid deadlock.
        mutex_lock l(mu_);
        mutex_lock ckpt_l(ckpt_mu_);
        if (!reader->Contains(full_name("input_exhausted"))) {
          TF_RETURN_IF_ERROR(RestoreInput(ctx, reader, input_impl_));
        } else {
          input_impl_.reset();
        }
        int64 temp;
        TF_RETURN_IF_ERROR(reader->ReadScalar(full_name("next_index"), &temp));
        next_index_ = size_t(temp);
        TF_RETURN_IF_ERROR(reader->ReadScalar(full_name("block_count"), &temp));
        block_count_ = size_t(temp);

        // Restore WorkerStates.
        TF_RETURN_IF_ERROR(
            reader->ReadScalar(full_name("workers_size"), &temp));
        if (temp != dataset()->num_threads()) {
          return errors::Internal("Expected ", dataset()->num_threads(),
                                  " worker states but found ", temp, ".");
        }
        for (size_t i = 0; i < dataset()->num_threads(); ++i) {
          TF_RETURN_IF_ERROR(ReadWorkerStateLocked(reader, i, ctx));
        }
        for (size_t i = 0; i < dataset()->num_threads(); ++i) {
          TF_RETURN_IF_ERROR(ReadWorkerThreadStateLocked(reader, i, ctx));
        }

        // Restore `interleave_indices_`.
        std::set<int64> all_indices;
        {
          int64 interleave_size;
          TF_RETURN_IF_ERROR(reader->ReadScalar(full_name("interleave_size"),
                                                &interleave_size));
          interleave_indices_.reserve(interleave_size);
          for (int64 i = 0; i < interleave_size; ++i) {
            int64 temp;
            TF_RETURN_IF_ERROR(reader->ReadScalar(
                full_name(strings::StrCat("interleave_indices_", i)), &temp));
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
          int64 staging_size;
          TF_RETURN_IF_ERROR(
              reader->ReadScalar(full_name("staging_size"), &staging_size));
          for (int i = 0; i < staging_size; ++i) {
            int64 temp;
            TF_RETURN_IF_ERROR(reader->ReadScalar(
                full_name(strings::StrCat("staging_indices_", i)), &temp));
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
        if (reader->Contains(full_name("worker_threads_running"))) {
          worker_threads_.reserve(dataset()->num_threads());
          for (size_t i = 0; i < dataset()->num_threads(); ++i) {
            std::shared_ptr<IteratorContext> new_ctx(new IteratorContext(*ctx));
            worker_threads_.emplace_back(ctx->StartThread(
                strings::StrCat("tf_data_parallel_interleave_worker_", i),
                [this, new_ctx, i]() { WorkerThread(new_ctx, i); }));
          }
        }
        return Status::OK();
      }

     private:
      // OutputElem contains the information from a call to GetNext by an output
      // iterator.
      struct OutputElem {
        // The output iterator sets `status` if getting the output element
        // fails.
        Status status;
        // The buffered data element.
        std::vector<Tensor> output;

        explicit OutputElem(const Status& s) : status(s) {}
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
        // an element into `outputs` (this implies sloppy_==false).
        condition_variable cond_var;

        inline bool MayHaveElements() const {
          return is_producing || !outputs.empty();
        }

        // Sets inputs for a worker thread and notifies it to start processing.
        void SetInputs(const Status& s, std::vector<Tensor> input_arguments) {
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
        Status iterator_creation_status;

        // The arguments to be used to construct `iterator`.
        std::vector<Tensor> input;

        std::unique_ptr<IteratorBase> iterator;

        WorkerThreadState() : output_elem(Status::OK()) {}
      };

      Status EnsureWorkerThreadsStarted(IteratorContext* ctx)
          EXCLUSIVE_LOCKS_REQUIRED(mu_) {
        if (worker_threads_.empty()) {
          worker_threads_.reserve(dataset()->num_threads());
          for (int64 i = 0; i < dataset()->num_threads(); ++i) {
            std::vector<Tensor> args;
            bool end_of_input = false;
            Status s = input_impl_->GetNext(ctx, &args, &end_of_input);
            if (end_of_input) {
              input_impl_.reset();
              return Status::OK();
            }
            workers_[i].SetInputs(s, std::move(args));
            std::shared_ptr<IteratorContext> new_ctx(new IteratorContext(*ctx));
            worker_threads_.push_back(ctx->StartThread(
                strings::StrCat("tf_data_parallel_interleave_worker_", i),
                [this, new_ctx, i]() { WorkerThread(new_ctx, i); }));
            if (i < dataset()->cycle_length_) {
              interleave_indices_.push_back(i);
            } else {
              staging_indices_.push_back(i);
            }
          }
          DCHECK(interleave_indices_.size() == dataset()->cycle_length_);
          DCHECK(staging_indices_.size() ==
                 dataset()->prefetch_input_elements_);
        }
        return Status::OK();
      }

      // Produces elements into the worker's output buffers.
      void WorkerThread(const std::shared_ptr<IteratorContext>& ctx,
                        const int64 thread_index) {
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
        // Even though `make_new_iterator` has cached values from
        // `worker_thread_states_[thread_index]` which is guarded by ckpt_mu_,
        // it is safe to *read* `make_new_iterator`outside of a lock without
        // worrying about concurrent changes to values in
        // `worker_thread_states_[thread_index]`. See comment at the start of
        // this function for details.
        while (true) {
          // Whether creation of the iterator succeeded.
          Status iterator_creation_status;
          // 1. Build a new iterator or use the existing one.
          if (make_new_iterator) {
            // 1a. Get new input tensors or use the exiting ones.
            bool read_new_input;
            {
              tf_shared_lock l(ckpt_mu_);
              // worker_thread_states_[thread_index].input will be non-empty
              // if checkpointing happened at CHECKPOINT_MARKER_A.
              read_new_input =
                  worker_thread_states_[thread_index].input.empty();
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

            // 1b. Run the user defined function to produce a new iterator.
            {
              tf_shared_lock l(ckpt_mu_);
              worker_thread_states_[thread_index].iterator_creation_status =
                  MakeIteratorFromInputElement(
                      ctx.get(), worker_thread_states_[thread_index].input,
                      thread_index, *instantiated_captured_func_, prefix(),
                      &worker_thread_states_[thread_index].iterator);
              iterator_creation_status =
                  worker_thread_states_[thread_index].iterator_creation_status;
              if (!iterator_creation_status.ok()) {
                worker_thread_states_[thread_index].input.clear();
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
            workers_[thread_index].outputs.emplace_back(
                iterator_creation_status);
            workers_[thread_index].is_producing = false;
            worker_thread_states_[thread_index].iterator_creation_status =
                Status::OK();
            // CHECKPOINT_MARKER_C
            // Non-OK iterator creation status has been notified to the
            // client.
            workers_[thread_index].cond_var.notify_one();
          } else {
            bool end_of_sequence = false;
            while (!end_of_sequence) {
              // 3.a Produce an element!
              {
                tf_shared_lock ckpt_l(ckpt_mu_);
                if (worker_thread_states_[thread_index]
                        .output_elem.status.ok() &&
                    worker_thread_states_[thread_index]
                        .output_elem.output.empty() &&
                    !worker_thread_states_[thread_index].end_of_sequence) {
                  worker_thread_states_[thread_index].output_elem.status =
                      worker_thread_states_[thread_index].iterator->GetNext(
                          ctx.get(),
                          &worker_thread_states_[thread_index]
                               .output_elem.output,
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
                      worker_thread_states_[thread_index].output_elem.status);
                  workers_[thread_index].outputs.back().output.swap(
                      worker_thread_states_[thread_index].output_elem.output);
                }
                worker_thread_states_[thread_index].output_elem.status =
                    Status::OK();
                if (dataset()->sloppy_) {
                  sloppy_cond_var_.notify_one();
                } else {
                  workers_[thread_index].cond_var.notify_one();
                }
                // CHECKPOINT_MARKER_E
                // Output element or iterator status has been sent to the
                // client.
              }
            }
          }
        }
      }

      Status WriteWorkerStateLocked(IteratorStateWriter* writer, int index)
          EXCLUSIVE_LOCKS_REQUIRED(mu_, ckpt_mu_) {
        string prefix = strings::StrCat("worker_", index);
        TF_RETURN_IF_ERROR(writer->WriteScalar(
            full_name(strings::StrCat(prefix, "_input_size")),
            workers_[index].input.size()));
        for (int i = 0; i < workers_[index].input.size(); ++i) {
          TF_RETURN_IF_ERROR(writer->WriteTensor(
              full_name(strings::StrCat(prefix, "_input_", i)),
              workers_[index].input[i]));
        }
        TF_RETURN_IF_ERROR(writer->WriteScalar(
            full_name(strings::StrCat(prefix, "_outputs_size")),
            workers_[index].outputs.size()));
        for (int i = 0; i < workers_[index].outputs.size(); ++i) {
          TF_RETURN_IF_ERROR(WriteOutputElemLocked(
              writer, workers_[index].outputs[i],
              full_name(strings::StrCat(prefix, "_outputs_", i))));
        }
        if (workers_[index].is_producing) {
          TF_RETURN_IF_ERROR(writer->WriteScalar(
              full_name(strings::StrCat(prefix, "_is_producing")), ""));
        }
        return Status::OK();
      }

      Status ReadWorkerStateLocked(IteratorStateReader* reader, int index,
                                   IteratorContext* ctx)
          EXCLUSIVE_LOCKS_REQUIRED(mu_, ckpt_mu_) {
        string worker_prefix = strings::StrCat("worker_", index);
        // Restore inputs.
        int64 input_size;
        TF_RETURN_IF_ERROR(reader->ReadScalar(
            full_name(strings::StrCat(worker_prefix, "_input_size")),
            &input_size));
        workers_[index].input.reserve(input_size);
        for (int i = 0; i < input_size; ++i) {
          workers_[index].input.emplace_back();
          TF_RETURN_IF_ERROR(reader->ReadTensor(
              full_name(strings::StrCat(worker_prefix, "_input_", i)),
              &workers_[index].input.back()));
        }
        int64 outputs_size;
        TF_RETURN_IF_ERROR(reader->ReadScalar(
            full_name(strings::StrCat(worker_prefix, "_outputs_size")),
            &outputs_size));
        for (int i = 0; i < outputs_size; ++i) {
          workers_[index].outputs.emplace_back(Status::OK());
          TF_RETURN_IF_ERROR(ReadOutputElemLocked(
              reader, &workers_[index].outputs.back(),
              full_name(strings::StrCat(worker_prefix, "_outputs_", i))));
        }
        if (reader->Contains(
                full_name(strings::StrCat(worker_prefix, "_is_producing")))) {
          workers_[index].is_producing = true;
        } else {
          workers_[index].is_producing = false;
        }
        return Status::OK();
      }

      Status WriteWorkerThreadStateLocked(IteratorStateWriter* writer,
                                          int index)
          EXCLUSIVE_LOCKS_REQUIRED(mu_, ckpt_mu_) {
        string prefix = strings::StrCat("worker_thread_", index);
        if (worker_thread_states_[index].iterator != nullptr) {
          TF_RETURN_IF_ERROR(
              SaveInput(writer, worker_thread_states_[index].iterator));
        } else {
          TF_RETURN_IF_ERROR(writer->WriteScalar(
              full_name(strings::StrCat(prefix, "_iterator_exhausted")), ""));
        }
        TF_RETURN_IF_ERROR(writer->WriteScalar(
            full_name(strings::StrCat(prefix, "_input_size")),
            worker_thread_states_[index].input.size()));
        for (int i = 0; i < worker_thread_states_[index].input.size(); ++i) {
          TF_RETURN_IF_ERROR(writer->WriteTensor(
              full_name(strings::StrCat(prefix, "_input_", i)),
              worker_thread_states_[index].input[i]));
        }
        TF_RETURN_IF_ERROR(WriteStatusLocked(
            writer, strings::StrCat(prefix, "_iterator_creation_status"),
            worker_thread_states_[index].iterator_creation_status));
        TF_RETURN_IF_ERROR(WriteOutputElemLocked(
            writer, worker_thread_states_[index].output_elem,
            full_name(strings::StrCat(prefix, "_output"))));
        if (worker_thread_states_[index].end_of_sequence) {
          TF_RETURN_IF_ERROR(writer->WriteScalar(
              full_name(strings::StrCat(prefix, "_end_of_sequence")), ""));
        }
        return Status::OK();
      }

      Status ReadWorkerThreadStateLocked(IteratorStateReader* reader, int index,
                                         IteratorContext* ctx)
          EXCLUSIVE_LOCKS_REQUIRED(mu_, ckpt_mu_) {
        string worker_prefix = strings::StrCat("worker_thread_", index);
        // Restore inputs.
        int64 input_size;
        TF_RETURN_IF_ERROR(reader->ReadScalar(
            full_name(strings::StrCat(worker_prefix, "_input_size")),
            &input_size));
        worker_thread_states_[index].input.reserve(input_size);
        for (int i = 0; i < input_size; ++i) {
          worker_thread_states_[index].input.emplace_back();
          TF_RETURN_IF_ERROR(reader->ReadTensor(
              full_name(strings::StrCat(worker_prefix, "_input_", i)),
              &worker_thread_states_[index].input.back()));
        }
        // Restore iterator.
        if (reader->Contains(full_name(
                strings::StrCat(worker_prefix, "_iterator_exhausted")))) {
          worker_thread_states_[index].iterator.reset();
        } else {
          std::unique_ptr<IteratorBase> iterator;
          Status s = MakeIteratorFromInputElement(
              ctx, worker_thread_states_[index].input, index,
              *instantiated_captured_func_, prefix(), &iterator);
          TF_RETURN_IF_ERROR(RestoreInput(ctx, reader, iterator));
          worker_thread_states_[index].iterator.swap(iterator);
        }
        TF_RETURN_IF_ERROR(ReadStatusLocked(
            reader, strings::StrCat(worker_prefix, "_iterator_creation_status"),
            &worker_thread_states_[index].iterator_creation_status));
        TF_RETURN_IF_ERROR(ReadOutputElemLocked(
            reader, &worker_thread_states_[index].output_elem,
            full_name(strings::StrCat(worker_prefix, "_output"))));
        if (reader->Contains(full_name(
                strings::StrCat(worker_prefix, "_end_of_sequence")))) {
          worker_thread_states_[index].end_of_sequence = true;
        } else {
          worker_thread_states_[index].end_of_sequence = false;
        }
        return Status::OK();
      }

      Status WriteOutputElemLocked(IteratorStateWriter* writer,
                                   const OutputElem& output_elem,
                                   const string& prefix)
          EXCLUSIVE_LOCKS_REQUIRED(mu_, ckpt_mu_) {
        TF_RETURN_IF_ERROR(WriteStatusLocked(
            writer, strings::StrCat(prefix, "_status"), output_elem.status));
        TF_RETURN_IF_ERROR(
            writer->WriteScalar(strings::StrCat(prefix, "_output_size"),
                                output_elem.output.size()));
        for (int i = 0; i < output_elem.output.size(); ++i) {
          TF_RETURN_IF_ERROR(writer->WriteTensor(
              strings::StrCat(prefix, "_output_", i), output_elem.output[i]));
        }
        return Status::OK();
      }

      Status ReadOutputElemLocked(IteratorStateReader* reader,
                                  OutputElem* output_elem, const string& prefix)
          EXCLUSIVE_LOCKS_REQUIRED(mu_, ckpt_mu_) {
        TF_RETURN_IF_ERROR(ReadStatusLocked(
            reader, strings::StrCat(prefix, "_status"), &output_elem->status));
        int64 output_size;
        TF_RETURN_IF_ERROR(reader->ReadScalar(
            strings::StrCat(prefix, "_output_size"), &output_size));
        output_elem->output.reserve(output_size);
        for (int i = 0; i < output_size; ++i) {
          output_elem->output.emplace_back();
          TF_RETURN_IF_ERROR(
              reader->ReadTensor(strings::StrCat(prefix, "_output_", i),
                                 &output_elem->output.back()));
        }
        return Status::OK();
      }

      Status WriteStatusLocked(IteratorStateWriter* writer,
                               const string& prefix, const Status& status)
          EXCLUSIVE_LOCKS_REQUIRED(mu_, ckpt_mu_) {
        TF_RETURN_IF_ERROR(
            writer->WriteScalar(full_name(strings::StrCat(prefix, "_code")),
                                static_cast<int64>(status.code())));
        if (!status.ok()) {
          TF_RETURN_IF_ERROR(
              writer->WriteScalar(full_name(strings::StrCat(prefix, "_msg")),
                                  status.error_message()));
        }
        return Status::OK();
      }

      Status ReadStatusLocked(IteratorStateReader* reader, const string& prefix,
                              Status* status)
          EXCLUSIVE_LOCKS_REQUIRED(mu_, ckpt_mu_) {
        int64 code_int;
        TF_RETURN_IF_ERROR(reader->ReadScalar(
            full_name(strings::StrCat(prefix, "_code")), &code_int));
        error::Code code = static_cast<error::Code>(code_int);

        if (code != error::Code::OK) {
          string error_message;
          TF_RETURN_IF_ERROR(reader->ReadScalar(
              full_name(strings::StrCat(prefix, "_msg")), &error_message));
          *status = Status(code, error_message);
        } else {
          *status = Status::OK();
        }
        return Status::OK();
      }

      // Mutex & condition variable to guard mutable iterator internals and
      // coordinate among worker threads and client thread[s].
      mutex mu_ ACQUIRED_BEFORE(ckpt_mu_);
      // The main thread waits on this condition variable if running in sloppy
      // mode and no values are available.
      condition_variable sloppy_cond_var_;
      // Mutex used to wait for a consistent state while checkpointing.
      // Only Save and Restore require an exclusive lock on this mutex. In
      // other scenarios we just acquire a shared lock so the pipeline's
      // performance should not be affected in the absence of checkpointing.
      // A thread must not wait on any condition variable while holding
      // `ckpt_mu_` in either shared or exclusive modes.
      mutex ckpt_mu_;

      // The iterator producing elements which are converted to datasets by
      // the dataset()->captured_func_ then interleaved together.
      // input_impl_ is reset when we have exhausted its input.
      std::unique_ptr<IteratorBase> input_impl_ GUARDED_BY(mu_);

      std::unique_ptr<InstantiatedCapturedFunction> instantiated_captured_func_;

      // The WorkerState structs the worker threads operate on.
      // workers_ elements are in at most one of interleave_ and staging_.
      std::vector<WorkerState> workers_ GUARDED_BY(mu_);

      // Stores the temporary state of WorkerThreads which is not stored in
      // WorkerState. This is used for checkpointing purposes only.
      std::vector<WorkerThreadState> worker_thread_states_ GUARDED_BY(ckpt_mu_);

      // Indices in `workers_` of iterators to interleave.
      std::vector<int64> interleave_indices_ GUARDED_BY(mu_);
      // Indices in `workers_` of prefetched iterators.
      std::deque<int64> staging_indices_ GUARDED_BY(mu_);

      // The index into output_elements_ for next element to produce.
      size_t next_index_ GUARDED_BY(mu_) = 0;
      // The number of items produced so far within the block
      size_t block_count_ GUARDED_BY(mu_) = 0;
      // Flag to instruct the worker threads to exit.
      bool cancelled_ GUARDED_BY(mu_) = false;
      // The worker threads. This must be last to ensure the
      // threads have exited before any other members are deallocated.
      // TODO(b/65178177): Avoid allocating additional threads.
      std::vector<std::unique_ptr<Thread>> worker_threads_ GUARDED_BY(mu_);
    };

    const DatasetBase* const input_;
    const NameAttrList interleave_func_;
    const std::unique_ptr<CapturedFunction> captured_func_;
    const int64 cycle_length_;
    const int64 block_length_;
    const bool sloppy_;
    const int64 buffer_output_elements_;
    const int64 prefetch_input_elements_;
    const DataTypeVector output_types_;
    const std::vector<PartialTensorShape> output_shapes_;
  };

  DataTypeVector output_types_;
  std::vector<PartialTensorShape> output_shapes_;
  NameAttrList interleave_func_;
  std::shared_ptr<FunctionLibraryDefinition> lib_def_;
};

REGISTER_KERNEL_BUILDER(
    Name("ExperimentalParallelInterleaveDataset").Device(DEVICE_CPU),
    ParallelInterleaveDatasetOp);

}  // namespace
}  // namespace data
}  // namespace tensorflow
