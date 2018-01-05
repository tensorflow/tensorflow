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

#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/framework/partial_tensor_shape.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/kernels/data/captured_function.h"
#include "tensorflow/core/kernels/data/dataset.h"
#include "tensorflow/core/kernels/data/dataset_utils.h"
#include "tensorflow/core/lib/gtl/cleanup.h"
#include "tensorflow/core/lib/random/random.h"

namespace tensorflow {

namespace {

// See documentation in ../ops/dataset_ops.cc for a high-level
// description of the following op.

class ParallelInterleaveDatasetOp : public UnaryDatasetOpKernel {
 public:
  explicit ParallelInterleaveDatasetOp(OpKernelConstruction* ctx)
      : UnaryDatasetOpKernel(ctx),
        graph_def_version_(ctx->graph_def_version()) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("f", &func_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("output_types", &output_types_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("output_shapes", &output_shapes_));
  }

  void MakeDataset(OpKernelContext* ctx, DatasetBase* input,
                   DatasetBase** output) override {
    OpInputList inputs;
    OP_REQUIRES_OK(ctx, ctx->input_list("other_arguments", &inputs));
    std::vector<Tensor> other_arguments;
    other_arguments.reserve(inputs.size());
    for (const Tensor& t : inputs) {
      other_arguments.push_back(t);
    }

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
    OP_REQUIRES_OK(ctx, CapturedFunction::Create(ctx, func_, graph_def_version_,
                                                 std::move(other_arguments),
                                                 &captured_func));

    *output =
        new Dataset(input, std::move(captured_func), cycle_length, block_length,
                    sloppy, buffer_output_elements, prefetch_input_elements,
                    output_types_, output_shapes_);
  }

 private:
  class Dataset : public DatasetBase {
   public:
    Dataset(const DatasetBase* input,
            std::unique_ptr<CapturedFunction> captured_func, int64 cycle_length,
            int64 block_length, bool sloppy, int64 buffer_output_elements,
            int64 prefetch_input_elements, const DataTypeVector& output_types,
            const std::vector<PartialTensorShape>& output_shapes)
        : input_(input),
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

    std::unique_ptr<IteratorBase> MakeIterator(
        const string& prefix) const override {
      return std::unique_ptr<IteratorBase>(new Iterator(
          {this, strings::StrCat(prefix, "::ParallelInterleave")}));
    }

    const DataTypeVector& output_dtypes() const override {
      return output_types_;
    }
    const std::vector<PartialTensorShape>& output_shapes() const override {
      return output_shapes_;
    }

    string DebugString() override {
      return "ParallelInterleaveDatasetOp::Dataset";
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
    //  1. `interleave_` is a vector containing pointers to `WorkerState`s that
    //  we
    //     are interleaving. Worker threads backing these WorkerStates should
    //     be regularly producing values.
    //  2. `staging_` is a deque containing pointers to WorkerStates that we
    //     will move to `interleave_` when an iterator in `interleave_` is
    //     exhausted.
    //
    // The client calls `GetNext[Internal]()` to retrieve an output element. The
    // internal implementation updates the state of `interleave_` and `staging_`
    // as output iterators (run by the worker threads) are exhausted.
    //
    // `input_impl_` is the input iterator that generates arguments for the
    // flat-map function (`captured_func_`). It is set to an iterator at
    // Iterator construction, and is fixed until we consume all input elements.
    // Once it is exhausted, we reset the unique_ptr to eagerly deallocate
    // memory.
    //
    // A few invariants are maintained:
    //  1. No element in interleave_ should be a nullptr unless `staging_` is
    //     empty and `input_impl_` is empty.
    //  2. Every `worker_` element is pointed to by at most one element of the
    //     union of `interleave_` and `staging_`.
    //  3. Unless `input_impl_` is empty, every `worker_` must be pointed to by
    //     an element in `interleave_` or `staging_`.
    class Iterator : public DatasetIterator<Dataset> {
     public:
      explicit Iterator(const Params& params)
          : DatasetIterator<Dataset>(params),
            input_impl_(params.dataset->input_->MakeIterator(params.prefix)),
            workers_(dataset()->num_threads()) {}

      ~Iterator() override {
        mutex_lock l(mu_);
        cancelled_ = true;
        // Notify all workers in case they are blocked.
        for (auto& worker : workers_) {
          worker.cond_var.notify_all();
        }
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
          for (int64 i = 0; i < interleave_.size(); ++i) {
            int64 index = (next_index_ + i) % interleave_.size();
            WorkerState* current_worker = interleave_[index];
            if (!current_worker) continue;  // Empty interleave elements.
            can_produce_elements |= current_worker->MayHaveElements();
            if (!current_worker->outputs.empty()) {
              // We have an element!
              next_index_ = index;
              if (i == 0) {
                block_count_++;
                if (block_count_ == dataset()->block_length_) {
                  next_index_ = (index + 1) % interleave_.size();
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
              interleave_[index] = nullptr;
              if (input_impl_) {
                // Start prefetching a new iterator.
                std::vector<Tensor> args;
                bool end_of_input = false;
                Status s = input_impl_->GetNext(ctx, &args, &end_of_input);
                if (end_of_input) {
                  input_impl_.reset();
                } else {
                  current_worker->SetInputs(s, std::move(args));
                  staging_.emplace_back(current_worker);
                }
              }

              if (!staging_.empty()) {
                // Move a worker from `staging_` to `interleave_`.
                interleave_[index] = staging_.front();
                staging_.pop_front();

                next_index_ = (index + 1) % interleave_.size();
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
            if (dataset()->sloppy_) {
              sloppy_cond_var_.wait(l);
            } else {
              interleave_[next_index_]->cond_var.wait(l);
            }
          }
        }
        return errors::Cancelled(
            "ParallelInterleaveDatasetOp::Dataset::Iterator::GetNext");
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
            worker_threads_.emplace_back(ctx->env()->StartThread(
                {}, "worker_thread",
                std::bind(&Iterator::WorkerThread, this,
                          new IteratorContext(*ctx), i)));
            if (i < dataset()->cycle_length_) {
              interleave_.push_back(&workers_[i]);
            } else {
              staging_.push_back(&workers_[i]);
            }
          }
          DCHECK(interleave_.size() == dataset()->cycle_length_);
          DCHECK(staging_.size() == dataset()->prefetch_input_elements_);
        }
        return Status::OK();
      }

      // Produces elements into the worker's output buffers.
      void WorkerThread(IteratorContext* ctx_ptr, const int64 thread_index) {
        // std::function arguments are copy-constructable, so we pass raw
        // pointers, and then immediately wrap them to ensure correct ownership.
        std::unique_ptr<IteratorContext> ctx(ctx_ptr);
        auto cleanup = gtl::MakeCleanup([this, thread_index] {
          mutex_lock l(mu_);
          workers_[thread_index].cond_var.notify_all();
        });

        while (true) {
          // 1. Wait for input.
          std::vector<Tensor> input;
          {
            mutex_lock l(mu_);
            while (!cancelled_ && !workers_[thread_index].is_producing) {
              workers_[thread_index].cond_var.wait(l);
            }
            if (cancelled_) return;
            input.swap(workers_[thread_index].input);
          }

          // 2. Run the user defined function to produce a new iterator.
          std::unique_ptr<IteratorBase> iterator;
          Status s = dataset::MakeIteratorFromInputElement(
              ctx.get(), input, thread_index, dataset()->captured_func_.get(),
              prefix(), &iterator);
          input.clear();  // Release memory as early as possible.

          if (!s.ok()) {
            mutex_lock l(mu_);
            workers_[thread_index].outputs.emplace_back(s);
            workers_[thread_index].is_producing = false;
            workers_[thread_index].cond_var.notify_one();
          } else {
            // 3. Produce elements
            bool end_of_sequence = false;
            while (!end_of_sequence) {
              // 3.a Produce an element!
              std::vector<Tensor> output_elem;
              s = iterator->GetNext(ctx.get(), &output_elem, &end_of_sequence);

              // 3.b Make it available to the client.
              {
                mutex_lock l(mu_);

                // Wait for space in the prefetch queue.
                while (!cancelled_ && workers_[thread_index].outputs.size() ==
                                          dataset()->buffer_output_elements_) {
                  workers_[thread_index].cond_var.wait(l);
                }
                if (cancelled_) return;

                // Output the element.
                workers_[thread_index].is_producing = !end_of_sequence;
                if (!end_of_sequence) {
                  workers_[thread_index].outputs.emplace_back(s);
                  workers_[thread_index].outputs.back().output.swap(
                      output_elem);
                }
                if (dataset()->sloppy_) {
                  sloppy_cond_var_.notify_one();
                } else {
                  workers_[thread_index].cond_var.notify_one();
                }
              }
            }
          }
        }
      }

      // Mutex & condition variable to guard mutable iterator internals and
      // coordinate among worker threads and client thread[s].
      mutex mu_;
      // The main thread waits on this condition variable if running in sloppy
      // mode and no values are available.
      condition_variable sloppy_cond_var_;

      // The iterator producing elements which are converted to datasets by
      // the dataset()->captured_func_ then interleaved together.
      // input_impl_ is reset when we have exhausted its input.
      std::unique_ptr<IteratorBase> input_impl_ GUARDED_BY(mu_);

      // The WorkerState structs the worker threads operate on.
      // workers_ elements are in at most one of interleave_ and staging_.
      std::vector<WorkerState> workers_ GUARDED_BY(mu_);

      // The iterators to interleave
      std::vector<WorkerState*> interleave_ GUARDED_BY(mu_);
      // Prefetched iterators
      std::deque<WorkerState*> staging_ GUARDED_BY(mu_);

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
    const std::unique_ptr<CapturedFunction> captured_func_;
    const int64 cycle_length_;
    const int64 block_length_;
    const bool sloppy_;
    const int64 buffer_output_elements_;
    const int64 prefetch_input_elements_;
    const DataTypeVector output_types_;
    const std::vector<PartialTensorShape> output_shapes_;
  };

  const int graph_def_version_;
  DataTypeVector output_types_;
  std::vector<PartialTensorShape> output_shapes_;
  NameAttrList func_;
};

REGISTER_KERNEL_BUILDER(Name("ParallelInterleaveDataset").Device(DEVICE_CPU),
                        ParallelInterleaveDatasetOp);

}  // namespace

}  // namespace tensorflow
