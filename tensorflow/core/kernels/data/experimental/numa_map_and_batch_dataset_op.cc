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
#define EIGEN_USE_THREADS

#include <atomic>
#include <utility>

#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/framework/partial_tensor_shape.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/kernels/data/captured_function.h"
#include "tensorflow/core/kernels/inplace_ops_functor.h"
#include "tensorflow/core/lib/core/blocking_counter.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/gtl/cleanup.h"
#include "tensorflow/core/lib/random/random.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/cpu_info.h"
#include "tensorflow/core/platform/numa.h"
#include "tensorflow/core/platform/tracing.h"
#include "tensorflow/core/util/ptr_util.h"

namespace tensorflow {
namespace data {
namespace {

// kWindowSize is the fixed constant controlling the number of batch outputs
// each NumaWorkerBlock may be processing at a time. This is currently a
// constant and not user configurable to enable future performance optimizations
// in the implementation.
const int64 kWindowSize = 10;

// Define a helper for more consistent logging.
#define WORKER_VLOG(verbose_level)                                           \
  VLOG(verbose_level) << "WorkerThread (" << numa_node << ", " << thread_num \
                      << "): "

// See documentation in ../ops/dataset_ops.cc for a high-level
// description of the following op.

class NumaMapAndBatchDatasetOp : public UnaryDatasetOpKernel {
 public:
  explicit NumaMapAndBatchDatasetOp(OpKernelConstruction* ctx)
      : UnaryDatasetOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("f", &func_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("output_types", &output_types_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("output_shapes", &output_shapes_));
    // TODO(saeta): Implement support for preserve_cardinality logic.
    OP_REQUIRES_OK(
        ctx, ctx->GetAttr("preserve_cardinality", &preserve_cardinality_));
  }

 protected:
  void MakeDataset(OpKernelContext* ctx, DatasetBase* input,
                   DatasetBase** output) override {
    int64 batch_size;
    OP_REQUIRES_OK(ctx, ParseScalarArgument(ctx, "batch_size", &batch_size));
    OP_REQUIRES(
        ctx, batch_size > 0,
        errors::InvalidArgument("batch_size must be greater than zero."));

    int64 num_parallel_calls;
    OP_REQUIRES_OK(ctx, ParseScalarArgument(ctx, "num_parallel_calls",
                                            &num_parallel_calls));
    OP_REQUIRES(
        ctx, num_parallel_calls > 0 || num_parallel_calls == model::kAutoTune,
        errors::InvalidArgument(
            "num_parallel_calls must be greater than zero."));

    bool drop_remainder;
    OP_REQUIRES_OK(ctx,
                   ParseScalarArgument(ctx, "drop_remainder", &drop_remainder));

    std::unique_ptr<CapturedFunction> captured_func;
    OP_REQUIRES_OK(
        ctx, CapturedFunction::Create(func_, ctx, "other_arguments",
                                      /* use_inter_op_parallelism = */ false,
                                      &captured_func));

    *output = new Dataset(ctx, input, batch_size, num_parallel_calls,
                          drop_remainder, output_types_, output_shapes_, func_,
                          std::move(captured_func));
  }

 private:
  class Dataset : public DatasetBase {
   public:
    Dataset(OpKernelContext* ctx, const DatasetBase* input, int64 batch_size,
            int64 num_parallel_calls, bool drop_remainder,
            const DataTypeVector& output_types,
            const std::vector<PartialTensorShape>& output_shapes,
            const NameAttrList& func,
            std::unique_ptr<CapturedFunction> captured_func)
        : DatasetBase(DatasetContext(ctx)),
          input_(input),
          batch_size_(batch_size),
          num_parallel_calls_(num_parallel_calls),
          drop_remainder_(drop_remainder),
          output_types_(output_types),
          output_shapes_(output_shapes),
          func_(func),
          captured_func_(std::move(captured_func)) {
      input_->Ref();
    }

    ~Dataset() override { input_->Unref(); }

    std::unique_ptr<IteratorBase> MakeIteratorInternal(
        const string& prefix) const override {
      return std::unique_ptr<IteratorBase>(
          new Iterator({this, strings::StrCat(prefix, "::NumaMapAndBatch")}));
    }

    const DataTypeVector& output_dtypes() const override {
      return output_types_;
    }

    const std::vector<PartialTensorShape>& output_shapes() const override {
      return output_shapes_;
    }

    string DebugString() const override {
      return "NumaMapAndBatchDatasetOp::Dataset";
    }

    // TODO(b/120482302): Note that this is inaccurate until
    // NumaMapAndBatchMapDataset modified to preserve cardinality.
    int64 Cardinality() const override {
      int64 n = input_->Cardinality();
      if (n == kInfiniteCardinality || n == kUnknownCardinality) {
        return n;
      }
      return n / batch_size_ +
             (n % batch_size_ == 0 || drop_remainder_ ? 0 : 1);
    }

   protected:
    Status AsGraphDefInternal(SerializationContext* ctx,
                              DatasetGraphDefBuilder* b,
                              Node** output) const override {
      TF_RETURN_IF_ERROR(b->AddFunction(ctx, func_.name()));
      Node* input_graph_node = nullptr;
      TF_RETURN_IF_ERROR(b->AddInputDataset(ctx, input_, &input_graph_node));
      Node* batch_size_node;
      TF_RETURN_IF_ERROR(b->AddScalar(batch_size_, &batch_size_node));
      Node* num_parallel_calls_node;
      TF_RETURN_IF_ERROR(
          b->AddScalar(num_parallel_calls_, &num_parallel_calls_node));
      Node* drop_remainder_node;
      TF_RETURN_IF_ERROR(b->AddScalar(drop_remainder_, &drop_remainder_node));

      DataTypeVector other_arguments_types;
      other_arguments_types.reserve(captured_func_->captured_inputs().size());
      std::vector<Node*> other_arguments;
      other_arguments.reserve(captured_func_->captured_inputs().size());
      for (const Tensor& t : captured_func_->captured_inputs()) {
        Node* node;
        DatasetBase* input;
        Status s = GetDatasetFromVariantTensor(t, &input);
        if (s.ok()) {
          TF_RETURN_IF_ERROR(b->AddInputDataset(ctx, input, &node));
        } else {
          TF_RETURN_IF_ERROR(b->AddTensor(t, &node));
        }
        other_arguments.emplace_back(node);
        other_arguments_types.emplace_back(t.dtype());
      }
      AttrValue f;
      b->BuildAttrValue(func_, &f);
      AttrValue other_arguments_types_attr;
      b->BuildAttrValue(other_arguments_types, &other_arguments_types_attr);

      TF_RETURN_IF_ERROR(b->AddDataset(
          this,
          {std::make_pair(0, input_graph_node),
           std::make_pair(2, batch_size_node),
           std::make_pair(3, num_parallel_calls_node),
           std::make_pair(4, drop_remainder_node)},  // Single tensor inputs.
          {std::make_pair(1, other_arguments)},      // Tensor list inputs.
          {std::make_pair("f", f),
           std::make_pair("Targuments", other_arguments_types_attr)},  // Attrs
          output));
      return Status::OK();
    }

   private:
    class Iterator : public DatasetIterator<Dataset> {
     public:
      explicit Iterator(const Params& params)
          : DatasetIterator<Dataset>(params),
            mu_(std::make_shared<mutex>()),
            autotune_cond_var_(std::make_shared<condition_variable>()),
            num_parallel_calls_(std::make_shared<model::SharedState>(
                params.dataset->num_parallel_calls_, mu_, autotune_cond_var_)) {
      }

      ~Iterator() override {
        mutex_lock l(*mu_);
        cancelled_ = true;
        VLOG(3) << "NumaMapAndBatchIterator::~Iterator: cancelling operations.";
        for (size_t i = 0; i < workers_.size(); ++i) {
          workers_[i]->manager.Cancel();
        }
        VLOG(3) << "NumaMapAndBatchIterator::~Iterator: waiting for threads to "
                   "shut down.";
      }

      Status Initialize(IteratorContext* ctx) override {
        mutex_lock l(*mu_);
        if (num_parallel_calls_->value == model::kAutoTune) {
          num_parallel_calls_->value = ctx->runner_threadpool_size();
        }
        TF_RETURN_IF_ERROR(
            dataset()->input_->MakeIterator(ctx, prefix(), &input_impl_));
        TF_RETURN_IF_ERROR(dataset()->captured_func_->Instantiate(
            ctx, &instantiated_captured_func_));
        return Status::OK();
      }

      Status GetNextInternal(IteratorContext* ctx,
                             std::vector<Tensor>* out_tensors,
                             bool* end_of_sequence) override {
        auto cleanup = gtl::MakeCleanup(
            [] { VLOG(3) << "GetNextInternal call returning."; });
        NumaWorkerBlock* worker = nullptr;
        {
          mutex_lock l(*mu_);
          VLOG(3) << "GetNextInternal call; current block: " << cur_block_;
          if (global_end_of_input_) {
            *end_of_sequence = true;
            return Status::OK();
          }
          TF_RETURN_IF_ERROR(EnsureBackgroundThreadsStarted(ctx));
          worker = workers_[cur_block_].get();
          cur_block_ = (cur_block_ + 1) % workers_.size();
        }
        bool global_end_of_input_local = false;
        Status s = worker->manager.GetBatch(ctx, dataset()->drop_remainder_,
                                            &global_end_of_input_local,
                                            out_tensors, end_of_sequence);
        if (global_end_of_input_local) {
          mutex_lock l(*mu_);
          global_end_of_input_ = global_end_of_input_local;
        }
        return s;
      }

     protected:
      std::shared_ptr<model::Node> CreateNode(
          IteratorContext* ctx, model::Node::Args args) const override {
        return model::MakeAsyncKnownRatioNode(
            std::move(args), dataset()->batch_size_,
            {model::MakeParameter("parallelism", num_parallel_calls_, /*min=*/1,
                                  /*max=*/ctx->runner_threadpool_size())});
      }

      Status SaveInternal(IteratorStateWriter* writer) override {
        mutex_lock l(*mu_);
        for (size_t i = 0; i < workers_.size(); ++i) {
          if (!workers_[i]->manager.Quiesce()) {
            return errors::Cancelled(
                "The iterator was deleted before it could reach a "
                "checkpointable state.");
          }
        }

        TF_RETURN_IF_ERROR(SaveInput(writer, input_impl_));
        TF_RETURN_IF_ERROR(
            writer->WriteScalar(full_name("num_workers"), workers_.size()));

        for (size_t i = 0; i < workers_.size(); ++i) {
          size_t index = (cur_block_ + i) % workers_.size();
          TF_RETURN_IF_ERROR(workers_[index]->manager.Save(writer, this, i));
        }
        return Status::OK();
      }

      Status RestoreInternal(IteratorContext* ctx,
                             IteratorStateReader* reader) override {
        mutex_lock l(*mu_);
        TF_RETURN_IF_ERROR(RestoreInput(ctx, reader, input_impl_));
        int64 num_workers = -1;
        TF_RETURN_IF_ERROR(
            reader->ReadScalar(full_name("num_workers"), &num_workers));
        // Note: num_workers can be 0 if the iterator wasn't started when
        // first checkpointed.
        if (num_workers < 0) {
          return errors::DataLoss(
              "When restoring from checkpoint, we encountered a data "
              "consistency error: num_workers has an invalid value: ",
              num_workers);
        }
        if (port::NUMAEnabled()) {
          int actual_numa_domains = port::NUMANumNodes();
          if (actual_numa_domains != num_workers && num_workers > 0) {
            LOG(WARNING) << "# NUMA domains mismatch when restoring from "
                            "checkpoint: checkpoint has "
                         << num_workers
                         << " NUMA domains, while this host has: "
                         << actual_numa_domains << " NUMA domains.";
          }
        }
        if (num_workers > 1 && !port::NUMAEnabled()) {
          LOG(WARNING) << "NUMA is not enabled for this process, but restoring "
                          "a checkpoint that assumes "
                       << num_workers << " NUMA domains.";
        }
        workers_.resize(num_workers);
        for (size_t i = 0; i < num_workers; ++i) {
          workers_[i] = MakeUnique<NumaWorkerBlock>(this);
          TF_RETURN_IF_ERROR(
              workers_[i]->manager.Restore(ctx, reader, this, i));
        }
        cur_block_ = 0;
        return Status::OK();
      }

     private:
      // NumaBlockManager manages all the state for a set of threads pinned to a
      // single NUMA domain.
      //
      // The methods can be divided into 3 categories based on who should call
      // them:
      //
      //  (1) RunnerThread: WaitForInputSpace, PushInputs, SetEndOfInput.
      //  (2) WorkerThread: RetrieveInput, GetBatchTensors.
      //      RecordBatchEntryComplete
      //  (3) Client threads: GetBatch, Cancel, Save, Restore.
      //
      // Internally, we manage state in a circular buffer of size `kWindowSize`.
      // There are 3 pointers into the circular buffer, and must maintain the
      // following order: (1) next_input_batch_ (corresponding to the next input
      // batch to be pulled from the input iterator), (2) next_input_
      // (corresponding to the batch the WorkerThreads should pull from for
      // their next inputs), and (3) next_output_ corresponding to the next
      // value to be consumed by the output iterator.
      //
      // Methods return errors::Cancelled if the iteration is cancelled before
      // completing.
      //
      // NumaBlockManager is thread safe.
      class NumaBlockManager {
       public:
        explicit NumaBlockManager(Iterator* itr) : itr_(itr) {}

        // WaitForInputSpace blocks until there is space in the circular buffer
        // to begin processing a new batch of elements.
        //
        // Returns true when there is space, false if the Iterator is cancelled.
        bool WaitForInputSpace(IteratorContext* ctx) {
          mutex_lock l(mu_);

          size_t next = (next_input_batch_ + 1) % kWindowSize;
          DCHECK(next < kWindowSize) << next;

          // Wait for space in the circular buffer.
          while (!cancelled_ && batches_[next].state != BatchState::kEmpty) {
            VLOG(3) << "Waiting for input space; next: " << next
                    << ", next_output_: " << next_output_
                    << ", next_input_batch_: " << next_input_batch_;
            itr_->RecordStop(ctx);
            runner_cond_var_.wait(l);
            itr_->RecordStart(ctx);
          }
          if (cancelled_) {
            VLOG(3) << "WaitForInputSpace cancelled.";
            return false;
          }

          DCHECK(batches_[next].state == BatchState::kEmpty);

          next_input_batch_ = next;
          return true;
        }

        // PushInputs sets the inputs for the next batch as retrieved from the
        // input iterator.
        void PushInputs(const Status& status,
                        std::vector<std::vector<Tensor>> inputs) {
          mutex_lock l(mu_);

          DCHECK(next_input_ < kWindowSize) << next_input_;
          DCHECK(batches_[next_input_batch_].state == BatchState::kEmpty);
          DCHECK(batches_[next_input_batch_].next_input_to_process == 0)
              << batches_[next_input_batch_].next_input_to_process;
          DCHECK(batches_[next_input_batch_].status.ok())
              << batches_[next_input_batch_].status;

          batches_[next_input_batch_].inputs.swap(inputs);
          batches_[next_input_batch_].state = BatchState::kInputsFilled;
          batches_[next_input_batch_].status.Update(status);
          if (batches_[next_input_batch_].status.ok()) {
            worker_cond_var_.notify_all();
          } else {
            client_cond_var_.notify_all();
            batches_[next_input_batch_].error_index = 0;
          }
        }

        // SetEndOfInput records the fact that we have reached the end of the
        // input iterator, and that we should return end_of_sequence = true when
        // we have exhaused all buffered batches.
        void SetEndOfInput() {
          mutex_lock l(mu_);
          reached_eof_ = true;
          worker_cond_var_.notify_all();
          client_cond_var_.notify_all();
        }

        // RetrieveInput gets the next input tuple to be mapped by a worker
        // thread.
        //
        // Returns true if an input was retrieved, false if the iterator has
        // been cancelled.
        bool RetrieveInput(IteratorContext* ctx, std::vector<Tensor>* input,
                           uint64* index, size_t* sequence_number) {
          mutex_lock l(mu_);

          // Wait for inputs to be ready.
          while (!cancelled_ &&
                 batches_[next_input_].state != BatchState::kInputsFilled) {
            itr_->RecordStop(ctx);
            worker_cond_var_.wait(l);
            itr_->RecordStart(ctx);
          }

          if (cancelled_) {
            return false;
          }

          DCHECK(batches_[next_input_].next_input_to_process <
                 batches_[next_input_].inputs.size())
              << "next_input_: " << next_input_ << ", next_input_to_process: "
              << batches_[next_input_].next_input_to_process
              << ", inputs.size(): " << batches_[next_input_].inputs.size()
              << ", state: " << static_cast<int32>(batches_[next_input_].state)
              << ", this: " << this;
          *index = batches_[next_input_].next_input_to_process;
          *sequence_number = next_input_;
          input->swap(batches_[next_input_]
                          .inputs[batches_[next_input_].next_input_to_process]);
          // Increment pointers.
          batches_[next_input_].next_input_to_process++;

          if (batches_[next_input_].next_input_to_process ==
              batches_[next_input_].inputs.size()) {
            batches_[next_input_].state = BatchState::kAllMapsStarted;
            next_input_ = (next_input_ + 1) % kWindowSize;
          }
          return true;
        }

        // GetBatchTensors returns a pointer to the output batch tensors for the
        // worker thread to copy into.
        //
        // allocate_output is a function taking a batch size, and a pointer to
        // the output tuple of Tensors to allocate them. The allocate_output
        // function is called at most once per output batch.
        std::vector<Tensor>* GetBatchTensors(
            size_t sequence_number,
            std::function<void(size_t, std::vector<Tensor>*)> allocate_output) {
          mutex_lock l(mu_);
          DCHECK(sequence_number < kWindowSize) << sequence_number;
          DCHECK(batches_[sequence_number].state == BatchState::kInputsFilled ||
                 batches_[sequence_number].state == BatchState::kAllMapsStarted)
              << sequence_number;

          if (batches_[sequence_number].outputs.empty()) {
            allocate_output(batches_[sequence_number].inputs.size(),
                            &batches_[sequence_number].outputs);
          }
          return &batches_[sequence_number].outputs;
        }

        // RecordBatchEntryComplete records an element of the batch has finished
        // copying into the output tensors.
        void RecordBatchEntryComplete(size_t sequence_number, uint64 index,
                                      Status s) {
          mutex_lock l(mu_);
          DCHECK(sequence_number < kWindowSize) << sequence_number;
          DCHECK(batches_[sequence_number].state == BatchState::kInputsFilled ||
                 batches_[sequence_number].state == BatchState::kAllMapsStarted)
              << sequence_number;

          batches_[sequence_number].num_outputs_complete++;
          if (!s.ok() && batches_[sequence_number].error_index > index) {
            batches_[sequence_number].status = s;
            batches_[sequence_number].error_index = index;
          }

          if (batches_[sequence_number].num_outputs_complete ==
              batches_[sequence_number].inputs.size()) {
            DCHECK(batches_[sequence_number].state ==
                   BatchState::kAllMapsStarted);
            batches_[sequence_number].state = BatchState::kOutputsComplete;
            batches_[sequence_number].inputs.clear();  // Eagerly save memory.
            batches_[sequence_number].inputs.shrink_to_fit();
            client_cond_var_.notify_all();
          }
        }

        // GetBatch retrieves the next output batch tensors.
        Status GetBatch(IteratorContext* ctx, bool drop_remainder,
                        bool* global_eof, std::vector<Tensor>* out_tensor,
                        bool* end_of_sequence) {
          mutex_lock l(mu_);
          // Wait until one of 3 conditions occurs:
          //  (1) we're cancelled.
          //  (2) the state becomes kOutputsComplete
          //  (3) state is empty && reached_eof.
          while (!cancelled_ &&
                 batches_[next_output_].state != BatchState::kOutputsComplete &&
                 !(reached_eof_ &&
                   batches_[next_output_].state == BatchState::kEmpty)) {
            VLOG(3) << "Waiting in GetBatch.";
            itr_->RecordStop(ctx);
            client_cond_var_.wait(l);
            itr_->RecordStart(ctx);
          }

          if (cancelled_) {
            return errors::Cancelled(
                "Cancelled in NumaMapAndBatch::GetNext call.");
          }

          if (reached_eof_ &&
              batches_[next_output_].state == BatchState::kEmpty) {
            VLOG(4) << "GetBatch returning end of sequence.";
            *end_of_sequence = true;
            *global_eof = true;
            return Status::OK();
          }

          VLOG(3) << "Returning output index: " << next_output_
                  << ", this: " << this;

          *end_of_sequence = false;
          Status s = batches_[next_output_].status;
          if (s.ok()) {
            out_tensor->swap(batches_[next_output_].outputs);
          }
          // Handle early termination.
          if (errors::IsOutOfRange(s)) {
            *global_eof = true;
            s = Status::OK();
            if (drop_remainder || batches_[next_output_].error_index == 0) {
              *end_of_sequence = true;
            } else {
              std::vector<Tensor> true_outputs;
              for (size_t i = 0; i < batches_[next_output_].outputs.size();
                   ++i) {
                TensorShape component_shape(
                    batches_[next_output_].outputs[i].shape());
                component_shape.set_dim(0, batches_[next_output_].error_index);
                AllocatorAttributes attr;
                attr.set_gpu_compatible(true);
                true_outputs.emplace_back(
                    ctx->allocator(attr),
                    batches_[next_output_].outputs[i].dtype(), component_shape);
                TF_RETURN_IF_ERROR(CopyPartialBatch(
                    &true_outputs.back(), batches_[next_output_].outputs[i],
                    batches_[next_output_].error_index));
              }
              out_tensor->swap(true_outputs);
            }
          }

          batches_[next_output_].Reset();
          next_output_ = (next_output_ + 1) % kWindowSize;
          runner_cond_var_.notify_all();

          return s;
        }

        void Cancel() {
          mutex_lock l(mu_);
          VLOG(3) << "Cancelling NUMA block.";
          cancelled_ = true;
          runner_cond_var_.notify_all();
          worker_cond_var_.notify_all();
          client_cond_var_.notify_all();
        }

        // Waits until all the worker threads have completed their work and all
        // internal state has reached a "safe-point" where we can safely
        // checkpoint.
        //
        // Returns true if completed successfully, false if cancelled while
        // waiting.
        bool Quiesce() {
          mutex_lock l(mu_);
          VLOG(3) << "Waiting until the operations have quiesced.";
          while (!cancelled_ && !AllMapOperationsFinished()) {
            client_cond_var_.wait(l);
          }
          if (cancelled_) {
            return false;
          }
          return true;
        }

        Status Save(IteratorStateWriter* writer, Iterator* itr, size_t index) {
          mutex_lock l(mu_);
          string prefix = itr->full_name(strings::StrCat("numa_block_", index));
          if (reached_eof_) {
            TF_RETURN_IF_ERROR(writer->WriteScalar(
                strings::StrCat(prefix, "_end_of_input"), ""));
          }
          for (size_t i = 0; i < kWindowSize; ++i) {
            size_t index = (next_output_ + i) % kWindowSize;
            if (batches_[index].state == BatchState::kEmpty) {
              break;
            }
            string batch_prefix = strings::StrCat(prefix, "_batch_", i);
            TF_RETURN_IF_ERROR(writer->WriteScalar(
                strings::StrCat(batch_prefix, "_code"),
                static_cast<int64>(batches_[index].status.code())));
            if (!batches_[index].status.ok()) {
              TF_RETURN_IF_ERROR(
                  writer->WriteScalar(strings::StrCat(batch_prefix, "_msg"),
                                      batches_[index].status.error_message()));
              TF_RETURN_IF_ERROR(writer->WriteScalar(
                  strings::StrCat(batch_prefix, "_error_index"),
                  batches_[index].error_index));
            }

            TF_RETURN_IF_ERROR(writer->WriteScalar(
                strings::StrCat(batch_prefix, "_output_size"),
                batches_[index].outputs.size()));
            for (size_t j = 0; j < batches_[index].outputs.size(); ++j) {
              string tensor_prefix =
                  strings::StrCat(batch_prefix, "_output_", j);
              if (!batches_[index].status.ok()) {
                DCHECK(batches_[index].error_index >= 0 &&
                       batches_[index].error_index <
                           itr_->dataset()->batch_size_);
                // If the batch is not full, we only store the first
                // `error_index` values. The rest of the batch tensor might not
                // be initialized, and accessing that will raise msan errors.
                TF_RETURN_IF_ERROR(writer->WriteTensor(
                    tensor_prefix, batches_[index].outputs[j].Slice(
                                       0, batches_[index].error_index)));
              } else {
                TF_RETURN_IF_ERROR(writer->WriteTensor(
                    tensor_prefix, batches_[index].outputs[j]));
              }
            }
          }
          return Status::OK();
        }

        Status Restore(IteratorContext* ctx, IteratorStateReader* reader,
                       Iterator* itr, size_t index) {
          mutex_lock l(mu_);
          if (reached_eof_) {
            return errors::FailedPrecondition(
                "Already reached the end of the sequence.");
          }
          string prefix = itr->full_name(strings::StrCat("numa_block_", index));
          reached_eof_ =
              reader->Contains(strings::StrCat(prefix, "_end_of_input"));
          for (size_t i = 0; i < kWindowSize; ++i) {
            string batch_prefix = strings::StrCat(prefix, "_batch_", i);
            if (!reader->Contains(strings::StrCat(batch_prefix, "_code"))) {
              break;
            }
            Batch batch;
            batch.state = BatchState::kOutputsComplete;
            int64 code_int;
            TF_RETURN_IF_ERROR(reader->ReadScalar(
                strings::StrCat(batch_prefix, "_code"), &code_int));
            error::Code code = static_cast<error::Code>(code_int);
            if (code != error::Code::OK) {
              string error_message;
              TF_RETURN_IF_ERROR(reader->ReadScalar(
                  strings::StrCat(batch_prefix, "_msg"), &error_message));
              batch.status = Status(code, error_message);
              int64 error_index_int = -1;
              TF_RETURN_IF_ERROR(reader->ReadScalar(
                  strings::StrCat(batch_prefix, "_error_index"),
                  &error_index_int));
              if (error_index_int < 0 ||
                  error_index_int > itr->dataset()->batch_size_) {
                return errors::FailedPrecondition(
                    "Error index out of bounds when restoring from checkpoint; "
                    "error index: ",
                    error_index_int);
              }
              batch.error_index = static_cast<size_t>(error_index_int);
            }
            int64 output_size = -1;
            TF_RETURN_IF_ERROR(reader->ReadScalar(
                strings::StrCat(batch_prefix, "_output_size"), &output_size));
            batch.outputs.reserve(output_size);
            for (size_t j = 0; j < output_size; ++j) {
              string tensor_name = strings::StrCat(batch_prefix, "_output_", j);
              Tensor t;
              TF_RETURN_IF_ERROR(reader->ReadTensor(tensor_name, &t));
              batch.outputs.emplace_back(std::move(t));
            }
            batches_[i] = std::move(batch);
          }
          return Status::OK();
        }

       private:
        bool AllMapOperationsFinished() EXCLUSIVE_LOCKS_REQUIRED(mu_) {
          for (size_t i = 0; i < kWindowSize; ++i) {
            if (batches_[i].state == BatchState::kInputsFilled ||
                batches_[i].state == BatchState::kAllMapsStarted) {
              return false;
            }
            if (batches_[i].state != BatchState::kOutputsComplete &&
                !reached_eof_) {
              return false;
            }
          }
          return true;
        }

        // Batches begin in the `kEmpty` state. Once the RunnerThread has
        // filled the `inputs` to a `Batch`, it transitions to the
        // `kInputsFilled` state. At this point, the Worker threads run the map
        // function and copy the outputs appropriately. Once all worker threads
        // have started, it transitions to `kAllMapsStarted`. After the outputs
        // are complete, the GetNext call can consume the outputs, and return
        // the batch to the kEmpty state.
        enum class BatchState {
          kEmpty,
          kInputsFilled,
          kAllMapsStarted,
          kOutputsComplete,
        };

        // Batch captures all the state of an output batch as it progresses
        // through the machinery. Once the RunnerThread fills inputs, it
        // transitions to `kInputsFilled`. At this point, the worker threads can
        // work on it, incrementing outputs_complete for every element of the
        // input set that is copied into the output Tensors. Once all the input
        // tuples have been processed (i.e. num_outputs_complete ==
        // inputs.size()), it transitions to the `kOutputsComplete` stage, where
        // it is ready to be returned by a `GetBatch` call (called from
        // `GetNextInternal`).
        struct Batch {
          BatchState state;
          // Aggregates the Status of the input iterator's GetNext
          // calls, in addition to the Status of the map function invocations.
          //
          // In the case where multiple non-OK statuses are encountered, we
          // return the first one encountered.
          Status status;
          // In order to return the correct error status, we keep track of the
          // error_index.
          size_t error_index;
          // The batch_size input tuples (or fewer in the case of the last
          // batch).
          // TODO(saeta): Avoid re-allocating vectors all the time!
          std::vector<std::vector<Tensor>> inputs;
          std::vector<Tensor> outputs;
          size_t next_input_to_process;
          size_t num_outputs_complete;

          Batch() { Reset(); }

          // Resets the Batch state (e.g. after consuming the outputs).
          void Reset() {
            state = BatchState::kEmpty;
            status = Status::OK();
            inputs.clear();
            inputs.shrink_to_fit();
            outputs.clear();
            outputs.shrink_to_fit();
            next_input_to_process = 0;
            num_outputs_complete = 0;
            error_index = -1;
          }
        };

        Iterator* itr_;  // Not owned.
        mutex mu_;
        Batch batches_[kWindowSize] GUARDED_BY(mu_);
        size_t next_input_batch_ GUARDED_BY(mu_) = -1;
        size_t next_input_ GUARDED_BY(mu_) = 0;
        size_t next_output_ GUARDED_BY(mu_) = 0;
        bool cancelled_ GUARDED_BY(mu_) = false;
        bool reached_eof_ GUARDED_BY(mu_) = false;

        // The runner thread waits on this condition variable for space to be
        // available. When the client thread takes a value out of the circular
        // buffer, it notifies this condition variable that space is now
        // available.
        condition_variable runner_cond_var_ GUARDED_BY(mu_);
        // The worker threads wait on this condition variable for available
        // inputs. When the runner thread makes new inputs available, it
        // notifies this condition variable.
        condition_variable worker_cond_var_ GUARDED_BY(mu_);
        // The client threads wait on this condition variable for avaiable
        // batched outputs. When worker threads complete a batch, they notify
        // this condition variable.
        condition_variable client_cond_var_ GUARDED_BY(mu_);
      };
      // Mark NumaBlockManager as a friend of Iterator in order to call
      // protected Iterator methods during checkpointing.
      friend NumaBlockManager;

      struct NumaWorkerBlock {
        NumaBlockManager manager;
        // TODO(saeta): Migrate to BackgroundWorker.
        std::vector<std::unique_ptr<Thread>> threads;

        explicit NumaWorkerBlock(Iterator* itr) : manager(itr) {}
      };

      static void CustomNumaWorkerBlockDeleter(NumaWorkerBlock* ptr) {
        ptr->~NumaWorkerBlock();
        port::NUMAFree(ptr, sizeof(NumaWorkerBlock));
      }
      static void DefaultNumaWorkerBlockDeleter(NumaWorkerBlock* ptr) {
        delete ptr;
      }

      static Status CopyPartialBatch(Tensor* output, const Tensor& value,
                                     int64 num_elements) {
        switch (value.dtype()) {
#define HANDLE_TYPE(type)                                         \
  case DataTypeToEnum<type>::value: {                             \
    auto output_t = output->flat_outer_dims<type>();              \
    auto value_t = value.flat_outer_dims<type>();                 \
    for (size_t i = 0; i < num_elements; i++) {                   \
      output_t.template chip<0>(i) = value_t.template chip<0>(i); \
    }                                                             \
    return Status::OK();                                          \
  }
          TF_CALL_DATASET_TYPES(HANDLE_TYPE);
#undef HANDLE_TYPE
          default:
            return errors::InvalidArgument("Unsupported data type: ",
                                           DataTypeString(value.dtype()));
        }
        return Status::OK();
      }

      Status EnsureBackgroundThreadsStarted(IteratorContext* ctx)
          EXCLUSIVE_LOCKS_REQUIRED(*mu_) {
        if (curr_num_parallel_calls_ >= num_parallel_calls_->value) {
          // All necessary threads have been started.
          curr_num_parallel_calls_ = num_parallel_calls_->value;
          return Status::OK();
        }

        VLOG(4) << "Starting workers";
        bool numa_enabled = port::NUMAEnabled();

        if (!numa_enabled) {
          LOG(INFO) << "NUMA not enabled on this host.";
        }

        int num_numa_nodes = port::NUMANumNodes();
        if (num_numa_nodes < 1) {
          return errors::Internal("The number of NUMA nodes is invalid: ",
                                  num_numa_nodes);
        }

        // Only resize when empty to support restoring from checkpoints.
        if (workers_.empty()) {
          VLOG(3) << "# NUMA Nodes: " << num_numa_nodes
                  << ", # Parallel Calls: " << num_parallel_calls_->value;
          workers_.resize(num_numa_nodes);
        } else {
          num_numa_nodes = workers_.size();
        }

        // Round up num_parallel_calls, with a minimum of 1.
        const size_t num_threads_per_block =
            std::max(1LL, (num_parallel_calls_->value + num_numa_nodes - 1) /
                              num_numa_nodes);

        VLOG(3) << "Starting " << num_threads_per_block * num_numa_nodes
                << " worker threads, with " << num_threads_per_block
                << " threads per block.";

        // Only allocate new_ctx if required.
        std::shared_ptr<IteratorContext> new_ctx;

        for (int i = 0; i < num_numa_nodes; ++i) {
          if (!workers_[i]) {
            if (numa_enabled) {
              // Allocate in appropriate NUMA domain.
              // 4k page align.
              void* ptr = port::NUMAMalloc(i, sizeof(NumaWorkerBlock), 0);
              if (ptr != nullptr) {
                NumaWorkerBlock* block = new (ptr) NumaWorkerBlock(this);
                workers_[i] =
                    std::unique_ptr<NumaWorkerBlock,
                                    std::function<void(NumaWorkerBlock*)>>(
                        block, CustomNumaWorkerBlockDeleter);
              } else {
                LOG(ERROR) << "Could not NUMA-allocate worker block: " << i;
              }
            }
            // If the NUMA allocation fails, or NUMA is not enabled.
            if (!workers_[i]) {
              workers_[i] =
                  std::unique_ptr<NumaWorkerBlock,
                                  std::function<void(NumaWorkerBlock*)>>(
                      new NumaWorkerBlock(this), DefaultNumaWorkerBlockDeleter);
            }
          }
          // Be sure to start threads if num_parallel_calls_ has changed.
          for (size_t j = workers_[i]->threads.size();
               j < num_threads_per_block; ++j) {
            VLOG(3) << "Starting worker " << i << ", " << j;
            if (!new_ctx) {
              new_ctx = std::make_shared<IteratorContext>(*ctx);
            }
            workers_[i]->threads.emplace_back(ctx->env()->StartThread(
                {}, strings::StrCat("tf_data_numa_map_and_batch_", i, "_", j),
                [this, new_ctx, i, j]() { WorkerThread(new_ctx, i, j); }));
            VLOG(3) << "Worker " << i << ", " << j << " successfully started.";
          }
        }
        if (!runner_thread_) {
          if (!new_ctx) {
            new_ctx = std::make_shared<IteratorContext>(*ctx);
          }
          runner_thread_.reset(ctx->env()->StartThread(
              {}, "tf_data_numa_map_and_batch",
              [this, new_ctx] { RunnerThread(new_ctx); }));
        }
        VLOG(3) << "All workers & runner thread started.";
        return Status::OK();
      }

      void AllocateOutput(IteratorContext* ctx, size_t batch_size,
                          const std::vector<Tensor>& map_fn_outputs,
                          std::vector<Tensor>* batch_outputs) {
        DCHECK(dataset()->output_dtypes().size() ==
               dataset()->output_shapes().size());
        DCHECK(map_fn_outputs.size() == dataset()->output_dtypes().size());
        for (size_t i = 0; i < dataset()->output_dtypes().size(); ++i) {
          TensorShape component_shape({static_cast<uint32>(batch_size)});
          component_shape.AppendShape(map_fn_outputs.at(i).shape());
          AllocatorAttributes attr;
          attr.set_gpu_compatible(true);
          batch_outputs->emplace_back(ctx->allocator(attr),
                                      map_fn_outputs.at(i).dtype(),
                                      component_shape);
        }
      }

      void RunnerThread(std::shared_ptr<IteratorContext> ctx)
          LOCKS_EXCLUDED(mu_) {
        RecordStart(ctx.get());
        auto cleanup = gtl::MakeCleanup([this, &ctx] {
          // Set end of input on all the managers in order to clean up in an
          // orderly fashion.
          VLOG(3) << "Setting End of Input on workers_[*]->manager";
          for (size_t i = 0; i < workers_.size(); ++i) {
            workers_[i]->manager.SetEndOfInput();
          }
          RecordStop(ctx.get());
        });

        const size_t num_blocks = workers_.size();

        while (true) {
          for (size_t block = 0; block < num_blocks; ++block) {
            VLOG(4) << "RunnerThread waiting for input space in block: "
                    << block;
            if (TF_PREDICT_FALSE(
                    !workers_[block]->manager.WaitForInputSpace(ctx.get()))) {
              VLOG(3) << "RunnerThread exiting due to cancellation.";
              return;
            }
            VLOG(4) << "RunnerThread has space; pulling on upstream for block "
                    << block;

            Status s;
            std::vector<std::vector<Tensor>> inputs;
            bool end_of_sequence = false;
            for (size_t i = 0; i < dataset()->batch_size_; ++i) {
              std::vector<Tensor> tuple;
              s.Update(
                  input_impl_->GetNext(ctx.get(), &tuple, &end_of_sequence));
              if (!s.ok()) {
                break;
              }
              if (end_of_sequence) {
                VLOG(4) << "Runner thread encountered end of sequence.";
                if (dataset()->drop_remainder_) {
                  return;
                }
                break;
              }
              inputs.push_back(std::move(tuple));
            }

            VLOG(4) << "Moving inputs to block " << block
                    << ", which has size: " << inputs.size();
            if (!s.ok() || !inputs.empty()) {
              workers_[block]->manager.PushInputs(s, std::move(inputs));
              VLOG(4) << "Inputs moved into block " << block;
            }
            if (end_of_sequence) {
              return;
            }
          }
        }
      }

      void WorkerThread(std::shared_ptr<IteratorContext> ctx,
                        const int numa_node, const int thread_num) {
        RecordStart(ctx.get());
        WORKER_VLOG(3) << "started.";
        auto stop_cleanup =
            gtl::MakeCleanup([this, numa_node, thread_num, &ctx]() {
              RecordStop(ctx.get());
              WORKER_VLOG(3) << "exiting.";
            });

        NumaWorkerBlock* block = workers_[numa_node].get();
        port::NUMASetThreadNodeAffinity(numa_node);
        const int num_numa_nodes = port::NUMANumNodes();
        const int minimum_num_parallel_calls = thread_num * num_numa_nodes;

        while (true) {
          // Put threads to sleep based on autotuner.
          {
            mutex_lock l(*mu_);
            while (minimum_num_parallel_calls >= num_parallel_calls_->value &&
                   !cancelled_) {
              RecordStop(ctx.get());
              autotune_cond_var_->wait(l);
              RecordStart(ctx.get());
            }
            if (cancelled_) {
              return;
            }
          }

          std::vector<Tensor> input;
          uint64 index = 0;
          size_t sequence_number = 0;
          WORKER_VLOG(4) << "retrieving input.";
          {
            tracing::ScopedActivity trace(
                "NumaMapAndBatch::Iterator::Worker::RetrieveInput");
            if (!block->manager.RetrieveInput(ctx.get(), &input, &index,
                                              &sequence_number)) {
              return;
            }
          }

          WORKER_VLOG(4) << "retrieved input; index: " << index
                         << ", sequence_number: " << sequence_number;

          std::vector<Tensor> return_values;
          Status s;
          {
            tracing::ScopedActivity trace(
                "NumaMapAndBatch::Iterator::Worker::FunctionExecution");
            s = instantiated_captured_func_->Run(ctx.get(), std::move(input),
                                                 &return_values);
          }
          WORKER_VLOG(4) << "ran function for index: " << index
                         << ", sequence_number: " << sequence_number;

          if (s.ok()) {
            std::vector<Tensor>* output = block->manager.GetBatchTensors(
                sequence_number,
                [this, ctx, &return_values](size_t batch_size,
                                            std::vector<Tensor>* output) {
                  AllocateOutput(ctx.get(), batch_size, return_values, output);
                });
            WORKER_VLOG(4) << "copying tensors to batch output.";
            {
              tracing::ScopedActivity trace(
                  "NumaMapAndBatch::Iterator::Worker::BatchCopy");
              for (size_t i = 0; i < return_values.size() && s.ok(); ++i) {
                Tensor& tensor = return_values.at(i);
                Tensor* batch = &output->at(i);
                if (tensor.NumElements() !=
                    (batch->NumElements() / batch->dim_size(0))) {
                  s.Update(errors::InvalidArgument(
                      "Cannot add tensor to the batch: number of elements does "
                      "not match. Shapes are: [tensor]: ",
                      tensor.shape().DebugString(),
                      ", [batch]: ", batch->shape().DebugString()));
                  break;
                }
                s.Update(batch_util::CopyElementToSlice(std::move(tensor),
                                                        batch, index));
              }
            }
          }

          block->manager.RecordBatchEntryComplete(sequence_number, index, s);
          WORKER_VLOG(4) << "finished index: " << index
                         << ", sequence_number: " << sequence_number;
        }
      }

      // mu_ protects shared internal state and is used to coordinate between
      // the auto-tuner, client threads, worker threads, and the runner thread.
      const std::shared_ptr<mutex> mu_;
      const std::shared_ptr<condition_variable> autotune_cond_var_;
      // The maximum number of parallel calls (can be auto-tuned).
      const std::shared_ptr<model::SharedState> num_parallel_calls_;
      std::unique_ptr<InstantiatedCapturedFunction> instantiated_captured_func_;

      // Caches the last-seen value of num_parallel_calls_->value to
      // short-circuit starting workers.
      int64 curr_num_parallel_calls_ GUARDED_BY(*mu_) = 0;

      std::unique_ptr<IteratorBase> input_impl_;
      int64 cur_block_ GUARDED_BY(*mu_) = 0;
      bool global_end_of_input_ GUARDED_BY(*mu_) = false;
      bool cancelled_ GUARDED_BY(*mu_) = false;
      std::vector<std::unique_ptr<NumaWorkerBlock,
                                  std::function<void(NumaWorkerBlock*)>>>
          workers_;  // Const after initialization.
      std::unique_ptr<Thread> runner_thread_ GUARDED_BY(*mu_);
    };

    const DatasetBase* const input_;
    const int64 batch_size_;
    const int64 num_parallel_calls_;
    const bool drop_remainder_;
    const DataTypeVector output_types_;
    const std::vector<PartialTensorShape> output_shapes_;
    const NameAttrList func_;
    const std::unique_ptr<CapturedFunction> captured_func_;
  };

  DataTypeVector output_types_;
  std::vector<PartialTensorShape> output_shapes_;
  NameAttrList func_;
  bool preserve_cardinality_;
};

REGISTER_KERNEL_BUILDER(
    Name("ExperimentalNumaMapAndBatchDataset").Device(DEVICE_CPU),
    NumaMapAndBatchDatasetOp);

}  // namespace
}  // namespace data
}  // namespace tensorflow
