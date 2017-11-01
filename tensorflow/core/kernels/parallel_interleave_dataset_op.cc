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
#include "tensorflow/core/kernels/dataset.h"

#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/framework/partial_tensor_shape.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/kernels/captured_function.h"
#include "tensorflow/core/kernels/dataset_utils.h"
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

    int64 cycle_length;
    OP_REQUIRES_OK(ctx,
                   ParseScalarArgument(ctx, "cycle_length", &cycle_length));
    OP_REQUIRES(ctx, cycle_length > 0,
                errors::InvalidArgument("`cycle_length` must be > 0"));

    int64 block_length;
    OP_REQUIRES_OK(ctx,
                   ParseScalarArgument(ctx, "block_length", &block_length));
    OP_REQUIRES(ctx, block_length > 0,
                errors::InvalidArgument("`block_length` must be > 0"));

    bool sloppy;
    OP_REQUIRES_OK(ctx, ParseScalarArgument(ctx, "sloppy", &sloppy));

    std::unique_ptr<CapturedFunction> captured_func;
    OP_REQUIRES_OK(ctx, CapturedFunction::Create(ctx, func_, graph_def_version_,
                                                 std::move(other_arguments),
                                                 &captured_func));

    *output = new Dataset(input, std::move(captured_func), cycle_length,
                          block_length, sloppy, output_types_, output_shapes_);
  }

 private:
  class Dataset : public DatasetBase {
   public:
    Dataset(const DatasetBase* input,
            std::unique_ptr<CapturedFunction> captured_func, int64 cycle_length,
            int64 block_length, bool sloppy, const DataTypeVector& output_types,
            const std::vector<PartialTensorShape>& output_shapes)
        : input_(input),
          captured_func_(std::move(captured_func)),
          cycle_length_(cycle_length),
          block_length_(block_length),
          sloppy_(sloppy),
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
    class Iterator : public DatasetIterator<Dataset> {
     public:
      explicit Iterator(const Params& params)
          : DatasetIterator<Dataset>(params),
            input_impl_(params.dataset->input_->MakeIterator(params.prefix)),
            output_elements_(params.dataset->cycle_length_) {}

      ~Iterator() override {
        mutex_lock l(mu_);
        cancelled_ = true;
        // Notify all workers in case they are blocked.
        for (int64 i = 0; i < dataset()->cycle_length_; ++i) {
          output_elements_[i].cond_var.notify_all();
        }
      }

      // It is implemented so that it matches the deterministic interleave
      // unless we would block waiting for an element, at which point it skips
      // along to the next available value.
      Status GetNextInternal(IteratorContext* ctx,
                             std::vector<Tensor>* out_tensors,
                             bool* end_of_sequence) override {
        mutex_lock l(mu_);
        TF_RETURN_IF_ERROR(EnsureWorkerThreadsStarted(ctx));
        const int64 num_workers = worker_threads_.size();
        if (num_workers == 0) {
          *end_of_sequence = true;
          return Status::OK();
        }
        while (!cancelled_) {
          // Wait for an item to become available, blocking if necessary. If we
          // are allowed to be sloppy, we can skip over input datasets that do
          // not have an item readily available.
          const int64 n = dataset()->sloppy_ ? num_workers : 1LL;
          for (int64 i = 0; i < n; ++i) {
            int64 index = (next_index_ + i) % num_workers;
            if (output_elements_[index].is_produced) {
              next_index_ = index;
              if (i == 0) {
                block_count_++;
                if (block_count_ == dataset()->block_length_) {
                  next_index_ = (index + 1) % num_workers;
                  block_count_ = 0;
                }
              } else {
                block_count_ = 0;
              }
              // If we encounter an EoF, advance to the next iterator
              if (output_elements_[index].end_of_sequence) {
                output_elements_[index].is_produced = false;
                output_elements_[index].cond_var.notify_one();
                next_index_ = (index + 1) % num_workers;
                block_count_ = 0;
                i = -1;  // Restart the inner loop
                continue;
              }
              *end_of_sequence = false;
              if (output_elements_[index].output_status.ok()) {
                output_elements_[index].output_value.swap(*out_tensors);
              }
              output_elements_[index].is_produced = false;
              output_elements_[index].cond_var.notify_one();
              return output_elements_[index].output_status;
            }
          }

          if (num_active_threads_ == 0) {
            // No potential for future values.
            //
            // Note: this condition check must occur after checking the output
            // buffer, as its possible for there to be values in the output
            // buffer, even if the number of live threads is zero.
            *end_of_sequence = true;
            return Status::OK();
          }

          // If we are not allowed to be sloppy and
          // `worker_threads_[next_index]` has finished, advance `next_index`.
          if (!dataset()->sloppy_ && worker_threads_[next_index_].finished) {
            next_index_ = (next_index_ + 1) % num_workers;
            continue;
          }

          // No values available; wait until woken up.
          // TODO(jsimsa): Use slot-specific condition variable for
          // coordination of elements consumption.
          cond_var_.wait(l);
        }
        return errors::Cancelled(
            "ParallelInterleaveDatasetOp::Dataset::Iterator::GetNext");
      }

     private:
      // Internal structure to manage thread coordination. All values are
      // guarded by the enclosing Iterator's mu_.
      struct OutputBufferElement {
        // The producer must set `is_produced` to `true` after
        // `output_status` or `output_value` has been written.
        bool is_produced = false;
        // The producer sets `output_status` if either getting the input element
        // or applying the function to it fails.
        Status output_status;
        // Reached end of sequence for the underlying iterator.
        bool end_of_sequence = false;
        // The output data element.
        std::vector<Tensor> output_value;
        // The producer thread waits on this condition variable after having
        // produced an element. The reader thread notifies this condition
        // variable after reading the value.
        condition_variable cond_var;
      };

      struct ThreadStatus {
        // The underlying thread uses `finished` to communicate to the producer
        // that it has finished.
        bool finished = false;
        // The underlying thread object.
        std::unique_ptr<Thread> thread;

        explicit ThreadStatus(Thread* thread) : thread(thread) {}
      };

      Status EnsureWorkerThreadsStarted(IteratorContext* ctx)
          EXCLUSIVE_LOCKS_REQUIRED(mu_) {
        if (worker_threads_.empty()) {
          for (int64 i = 0; i < dataset()->cycle_length_; ++i) {
            // Serialize the creation of the workers and their corresponding
            // input elements to ensure we match the standard interleave when
            // the underlying iterators induce no delay.
            std::vector<Tensor> args;
            TF_RETURN_IF_ERROR(
                input_impl_->GetNext(ctx, &args, &end_of_input_));
            if (end_of_input_) {
              LOG(WARNING) << "Input iterator exhausted after " << i
                           << " elements; cannot start all "
                           << dataset()->cycle_length_ << " worker threads.";
              return Status::OK();
            }
            std::unique_ptr<IteratorBase> itr;
            TF_RETURN_IF_ERROR(dataset::MakeIteratorFromInputElement(
                ctx, args, i, dataset()->captured_func_.get(), prefix(), &itr));
            worker_threads_.emplace_back(ctx->env()->StartThread(
                {}, "worker_thread",
                std::bind(&Iterator::WorkerThread, this,
                          new IteratorContext(*ctx), i, itr.release())));
            num_active_threads_ = i + 1;
          }
        }
        return Status::OK();
      }

      void BlockAndUpdateOutputBuffer(mutex_lock* l, const int64 thread_index,
                                      const Status& status,
                                      bool end_of_sequence,
                                      std::vector<Tensor>* out_tensors)
          EXCLUSIVE_LOCKS_REQUIRED(mu_) {
        // We have produced an element; push it into the output buffer
        // when space is available.
        while (!cancelled_ && output_elements_[thread_index].is_produced) {
          output_elements_[thread_index].cond_var.wait(*l);
        }
        if (cancelled_) {
          return;
        }
        output_elements_[thread_index].is_produced = true;
        output_elements_[thread_index].output_status = status;
        output_elements_[thread_index].end_of_sequence = end_of_sequence;
        if (status.ok()) {
          output_elements_[thread_index].output_value.swap(*out_tensors);
        } else {
          output_elements_[thread_index].output_value.clear();
        }
        cond_var_.notify_one();
      }

      // Races to produce elements into the output queue buffers.
      void WorkerThread(IteratorContext* ctx_ptr, const int64 thread_index,
                        IteratorBase* out_iterator_ptr) {
        // std::function arguments are copy-constructable, so we pass raw
        // pointers, and then immediately wrap them to ensure correct ownership.
        std::unique_ptr<IteratorContext> ctx(ctx_ptr);
        std::unique_ptr<IteratorBase> out_iterator(out_iterator_ptr);
        auto cleanup = gtl::MakeCleanup([this, thread_index] {
          mutex_lock l(mu_);
          worker_threads_[thread_index].finished = true;
          num_active_threads_--;
          cond_var_.notify_all();
        });
        while (true) {
          // Attempt to produce an element.
          bool end_of_out_itr_input = false;
          std::vector<Tensor> out_tensors;
          Status element_status = out_iterator->GetNext(ctx.get(), &out_tensors,
                                                        &end_of_out_itr_input);
          // Handle output.
          {
            mutex_lock l(mu_);
            BlockAndUpdateOutputBuffer(&l, thread_index, element_status,
                                       end_of_out_itr_input, &out_tensors);
            if (end_of_out_itr_input) {
              // We have exhausted our current iterator; get a new iterator;
              // loop to handle errors.
              while (!cancelled_) {
                if (end_of_input_) {
                  // No more iterator inputs; we're done!
                  return;
                }
                std::vector<Tensor> args;
                // BlockAndUpdateOutputBuffer() sequences calls to
                // input_impl_->GetNext when the out_iterator doesn't cause
                // slopping.
                Status input_status =
                    input_impl_->GetNext(ctx.get(), &args, &end_of_input_);
                if (end_of_input_) {
                  // No more elements to produce, stop the worker thread.
                  return;
                }
                if (input_status.ok()) {
                  input_status = dataset::MakeIteratorFromInputElement(
                      ctx.get(), args, thread_index,
                      dataset()->captured_func_.get(), prefix(), &out_iterator);
                }
                if (input_status.ok()) {
                  // Successfully have a new out_iterator; restart the outer
                  // loop to produce an element.
                  break;
                }

                // We encountered an error; push the error to the output buffer.
                BlockAndUpdateOutputBuffer(&l, thread_index, input_status,
                                           /* end_of_sequence = */ false,
                                           &out_tensors);
              }
            }

            // Check if we should exit.
            if (cancelled_) {
              return;
            }
          }
        }
      }

      // Mutex & condition variable to guard mutable iterator internals and
      // coordinate among worker threads and client thread[s].
      mutex mu_;
      condition_variable cond_var_;
      // The iterator producing elements which are converted to datasets by
      // the dataset()->captured_func_ then interleaved together.
      const std::unique_ptr<IteratorBase> input_impl_ GUARDED_BY(mu_);
      // Whether the input_impl_ can produce future elements.
      bool end_of_input_ GUARDED_BY(mu_) = false;
      // The buffer of elements to be produced. Each worker thread operates
      // on a single OutputBufferElement.
      std::vector<OutputBufferElement> output_elements_ GUARDED_BY(mu_);
      // The index into output_elements_ for next element to produce.
      size_t next_index_ GUARDED_BY(mu_) = 0;
      // The number of items produced so far within the block
      size_t block_count_ GUARDED_BY(mu_) = 0;
      // Number of active threads.
      size_t num_active_threads_ GUARDED_BY(mu_) = 0;
      // Flag to instruct the worker threads to exit.
      bool cancelled_ GUARDED_BY(mu_) = false;
      // Pointers to the worker threads. This must be last to ensure the
      // threads have exited before any other members are deallocated.
      // TODO(b/65178177): Avoid allocating additional threads.
      std::vector<ThreadStatus> worker_threads_ GUARDED_BY(mu_);
    };

    const DatasetBase* const input_;
    const std::unique_ptr<CapturedFunction> captured_func_;
    const int64 cycle_length_;
    const int64 block_length_;
    const bool sloppy_;
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
