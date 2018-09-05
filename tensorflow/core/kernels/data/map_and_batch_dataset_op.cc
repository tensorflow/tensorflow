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
#define EIGEN_USE_THREADS

#include <utility>

#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/framework/partial_tensor_shape.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/kernels/data/captured_function.h"
#include "tensorflow/core/kernels/data/dataset.h"
#include "tensorflow/core/kernels/inplace_ops_functor.h"
#include "tensorflow/core/lib/core/blocking_counter.h"
#include "tensorflow/core/lib/gtl/cleanup.h"
#include "tensorflow/core/lib/random/random.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/tracing.h"

namespace tensorflow {

namespace {

// See documentation in ../ops/dataset_ops.cc for a high-level
// description of the following op.

class MapAndBatchDatasetOp : public UnaryDatasetOpKernel {
 public:
  explicit MapAndBatchDatasetOp(OpKernelConstruction* ctx)
      : UnaryDatasetOpKernel(ctx),
        graph_def_version_(ctx->graph_def_version()),
        op_version_(ctx->def().op() == "MapAndBatchDataset" ? 1 : 2) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("f", &func_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("output_types", &output_types_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("output_shapes", &output_shapes_));
  }

 protected:
  void MakeDataset(OpKernelContext* ctx, DatasetBase* input,
                   DatasetBase** output) override {
    OpInputList inputs;
    OP_REQUIRES_OK(ctx, ctx->input_list("other_arguments", &inputs));
    std::vector<Tensor> other_arguments;
    other_arguments.reserve(inputs.size());
    for (const Tensor& t : inputs) {
      other_arguments.push_back(t);
    }

    int64 batch_size;
    OP_REQUIRES_OK(ctx, ParseScalarArgument(ctx, "batch_size", &batch_size));
    OP_REQUIRES(
        ctx, batch_size > 0,
        errors::InvalidArgument("batch_size must be greater than zero."));

    int64 num_parallel_calls;
    switch (op_version_) {
      case 1:
        int64 num_parallel_batches;
        OP_REQUIRES_OK(ctx, ParseScalarArgument(ctx, "num_parallel_batches",
                                                &num_parallel_batches));
        num_parallel_calls = num_parallel_batches * batch_size;
        OP_REQUIRES(ctx, num_parallel_batches > 0,
                    errors::InvalidArgument(
                        "num_parallel_batches must be greater than zero."));
        break;
      case 2:
        OP_REQUIRES_OK(ctx, ParseScalarArgument(ctx, "num_parallel_calls",
                                                &num_parallel_calls));
        OP_REQUIRES(ctx, num_parallel_calls > 0,
                    errors::InvalidArgument(
                        "num_parallel_calls must be greater than zero."));
        break;
      default:
        OP_REQUIRES(ctx, false,
                    errors::Unimplemented("Unsupported operation version %d.",
                                          op_version_));
    }

    bool drop_remainder;
    OP_REQUIRES_OK(ctx,
                   ParseScalarArgument(ctx, "drop_remainder", &drop_remainder));

    std::unique_ptr<CapturedFunction> captured_func;
    OP_REQUIRES_OK(ctx, CapturedFunction::Create(
                            func_, std::move(other_arguments), &captured_func));

    *output = new Dataset(ctx, input, batch_size, num_parallel_calls,
                          drop_remainder, output_types_, output_shapes_, func_,
                          std::move(captured_func), &ctx->eigen_cpu_device());
  }

 private:
  class Dataset : public DatasetBase {
   public:
    Dataset(OpKernelContext* ctx, const DatasetBase* input, int64 batch_size,
            int64 num_parallel_calls, bool drop_remainder,
            const DataTypeVector& output_types,
            const std::vector<PartialTensorShape>& output_shapes,
            const NameAttrList& func,
            std::unique_ptr<CapturedFunction> captured_func,
            const Eigen::ThreadPoolDevice* device)
        : DatasetBase(DatasetContext(ctx)),
          input_(input),
          batch_size_(batch_size),
          num_parallel_calls_(num_parallel_calls),
          drop_remainder_(drop_remainder),
          output_types_(output_types),
          output_shapes_(output_shapes),
          map_fn_(func),
          captured_func_(std::move(captured_func)),
          device_(device) {
      input_->Ref();
    }

    ~Dataset() override { input_->Unref(); }

    std::unique_ptr<IteratorBase> MakeIteratorInternal(
        const string& prefix) const override {
      return std::unique_ptr<IteratorBase>(
          new Iterator({this, strings::StrCat(prefix, "::MapAndBatch")}));
    }

    const DataTypeVector& output_dtypes() const override {
      return output_types_;
    }

    const std::vector<PartialTensorShape>& output_shapes() const override {
      return output_shapes_;
    }

    string DebugString() const override {
      return "MapAndBatchDatasetOp::Dataset";
    }

   protected:
    Status AsGraphDefInternal(SerializationContext* ctx,
                              DatasetGraphDefBuilder* b,
                              Node** output) const override {
      TF_RETURN_IF_ERROR(b->AddFunction(ctx, map_fn_.name()));
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
        TF_RETURN_IF_ERROR(b->AddTensor(t, &node));
        other_arguments.emplace_back(node);
        other_arguments_types.emplace_back(t.dtype());
      }
      AttrValue f;
      b->BuildAttrValue(map_fn_, &f);
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
          : DatasetIterator<Dataset>(params) {}

      ~Iterator() override {
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
            dataset()->input_->MakeIterator(ctx, prefix(), &input_impl_));
        return dataset()->captured_func_->Instantiate(ctx);
      }

      Status GetNextInternal(IteratorContext* ctx,
                             std::vector<Tensor>* out_tensors,
                             bool* end_of_sequence) override {
        std::shared_ptr<BatchResult> result;
        {
          mutex_lock l(mu_);
          EnsureRunnerThreadStarted(ctx);
          while (batch_results_.empty() ||
                 batch_results_.front()->num_calls > 0) {
            cond_var_.wait(l);
          }
          std::swap(result, batch_results_.front());
          batch_results_.pop_front();
        }
        cond_var_.notify_all();
        return ProcessResult(ctx, result, out_tensors, end_of_sequence);
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
            writer->WriteScalar(full_name("call_counter"), call_counter_));
        TF_RETURN_IF_ERROR(writer->WriteScalar(full_name("batch_results_size"),
                                               batch_results_.size()));
        for (size_t i = 0; i < batch_results_.size(); ++i) {
          TF_RETURN_IF_ERROR(WriteBatchResult(writer, i));
        }
        return Status::OK();
      }

      Status RestoreInternal(IteratorContext* ctx,
                             IteratorStateReader* reader) override {
        mutex_lock l(mu_);
        TF_RETURN_IF_ERROR(RestoreInput(ctx, reader, input_impl_));
        TF_RETURN_IF_ERROR(
            reader->ReadScalar(full_name("call_counter"), &call_counter_));
        int64 batch_results_size;
        TF_RETURN_IF_ERROR(reader->ReadScalar(full_name("batch_results_size"),
                                              &batch_results_size));
        for (int i = 0; i < batch_results_size; ++i) {
          TF_RETURN_IF_ERROR(ReadBatchResult(ctx, reader, i));
        }
        return Status::OK();
      }

     private:
      struct BatchResult {
        explicit BatchResult(int64 batch_size) {
          end_of_input = false;
          num_calls = batch_size;
          num_elements = 0;
          output_allocated = false;
          status = Status::OK();
        }

        void UpdateStatus(const Status& s) {
          mutex_lock l(mu);
          status.Update(s);
        }

        mutex mu;
        bool end_of_input GUARDED_BY(mu);
        int64 num_elements GUARDED_BY(mu);
        std::vector<Tensor> output;
        bool output_allocated GUARDED_BY(mu);
        Status status GUARDED_BY(mu);
        // Counts the number of outstanding calls for this batch.
        int64 num_calls;  // access guarded by owner's mutex
      };

      void Callback(const std::shared_ptr<IteratorContext>& ctx,
                    const std::shared_ptr<BatchResult>& result,
                    const std::shared_ptr<std::vector<Tensor>>& return_values,
                    int64 offset, const Status& status) LOCKS_EXCLUDED(mu_) {
        result->UpdateStatus(status);
        if (status.ok()) {
          EnsureOutputAllocated(ctx, result, return_values);
          for (size_t i = 0; i < return_values->size(); ++i) {
            const Tensor& tensor = return_values->at(i);
            Tensor* batch = &(result->output)[i];
            if (tensor.NumElements() !=
                (batch->NumElements() / batch->dim_size(0))) {
              TensorShape batch_shape = batch->shape();
              batch_shape.RemoveDim(0);
              result->UpdateStatus(errors::InvalidArgument(
                  "Cannot add tensor to the batch: number of elements does not "
                  "match. Shapes are: [tensor]: ",
                  tensor.shape().DebugString(),
                  ", [batch]: ", batch_shape.DebugString()));
              break;
            }
            // TODO(mrry): Add a version of DoParallelConcat that allows us to
            // move `tensor` where possible, to speed up string tensor batching.
            Status copy_status = ::tensorflow::functor::DoParallelConcat(
                *dataset()->device_, tensor, offset, batch);
            if (!copy_status.ok()) {
              result->UpdateStatus(copy_status);
              break;
            }
          }
          {
            mutex_lock l(result->mu);
            result->num_elements++;
          }
        }
        CallCompleted(result);
      }

      void CallCompleted(const std::shared_ptr<BatchResult>& result)
          LOCKS_EXCLUDED(mu_) {
        {
          mutex_lock l(mu_);
          num_calls_--;
          result->num_calls--;
        }
        cond_var_.notify_all();
      }

      void CallFunction(std::shared_ptr<IteratorContext> ctx,
                        const std::shared_ptr<BatchResult>& result,
                        int64 offset) LOCKS_EXCLUDED(mu_) {
        // Get the next input element.
        std::vector<Tensor> input_element;
        bool end_of_input;
        Status status =
            input_impl_->GetNext(ctx.get(), &input_element, &end_of_input);
        bool return_early;
        {
          mutex_lock l(result->mu);
          result->end_of_input = result->end_of_input || end_of_input;
          result->status.Update(status);
          return_early = result->end_of_input || !result->status.ok();
        }
        if (return_early) {
          CallCompleted(result);
          return;
        }

        // Call `captured_func_(input_element)`, using `Callback` to store the
        // result in `result`.
        (*ctx->runner())(std::bind(
            [this, result, offset](std::shared_ptr<IteratorContext> ctx,
                                   std::vector<Tensor> input_element) {
              std::shared_ptr<std::vector<Tensor>> return_values(
                  new std::vector<Tensor>());
              dataset()->captured_func_->RunAsync(
                  ctx.get(), std::move(input_element), return_values.get(),
                  [this, ctx, result, return_values, offset](Status status) {
                    Callback(ctx, result, return_values, offset, status);
                  });
            },
            ctx, std::move(input_element)));
      }

      Status CopyPartialBatch(Tensor* output, const Tensor& value,
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

      void EnsureRunnerThreadStarted(IteratorContext* ctx)
          EXCLUSIVE_LOCKS_REQUIRED(mu_) {
        if (!runner_thread_) {
          std::shared_ptr<IteratorContext> ctx_copy(new IteratorContext(*ctx));
          runner_thread_.reset(ctx->env()->StartThread(
              {}, "runner_thread",
              std::bind(&Iterator::RunnerThread, this, ctx_copy)));
        }
      }

      void EnsureOutputAllocated(
          const std::shared_ptr<IteratorContext>& ctx,
          const std::shared_ptr<BatchResult>& result,
          const std::shared_ptr<std::vector<Tensor>>& return_values) {
        mutex_lock l(result->mu);
        if (result->output_allocated) {
          return;
        }
        const size_t num_components = return_values->size();
        for (size_t i = 0; i < num_components; ++i) {
          TensorShape component_shape({dataset()->batch_size_});
          component_shape.AppendShape(return_values->at(i).shape());
          AllocatorAttributes attr;
          attr.set_gpu_compatible(true);
          Tensor component(ctx->allocator(attr), return_values->at(i).dtype(),
                           component_shape);
          result->output.emplace_back(std::move(component));
        }
        result->output_allocated = true;
      }

      int MaxBatchResults() EXCLUSIVE_LOCKS_REQUIRED(mu_) {
        return (dataset()->num_parallel_calls_ + dataset()->batch_size_ - 1) /
               dataset()->batch_size_;
      }

      Status ProcessResult(IteratorContext* ctx,
                           const std::shared_ptr<BatchResult>& result,
                           std::vector<Tensor>* out_tensors,
                           bool* end_of_sequence) {
        mutex_lock l(result->mu);
        if (result->num_elements == 0) {
          *end_of_sequence = true;
          return Status::OK();
        }
        // `f` may deliberately raise `errors::OutOfRange` to indicate that we
        // should terminate the iteration early.
        if (!result->status.ok() && !errors::IsOutOfRange(result->status)) {
          // Deallocate tensors allocated for the output.
          result->output.clear();
          *end_of_sequence = false;
          return result->status;
        }
        if (result->num_elements < dataset()->batch_size_) {
          if (dataset()->drop_remainder_) {
            // Deallocate tensors allocated for the output.
            result->output.clear();
            *end_of_sequence = true;
            return Status::OK();
          }
          const std::vector<Tensor>& output = result->output;
          for (size_t i = 0; i < output.size(); ++i) {
            TensorShape component_shape(result->output[i].shape());
            component_shape.set_dim(0, result->num_elements);
            AllocatorAttributes attr;
            attr.set_gpu_compatible(true);
            Tensor component(ctx->allocator(attr), output[i].dtype(),
                             component_shape);
            TF_RETURN_IF_ERROR(
                CopyPartialBatch(&component, output[i], result->num_elements));
            out_tensors->emplace_back(std::move(component));
          }
          // Deallocate tensors allocated for the output.
          result->output.clear();
        } else {
          *out_tensors = std::move(result->output);
        }
        *end_of_sequence = result->num_elements == 0;
        return Status::OK();
      }

      void RunnerThread(const std::shared_ptr<IteratorContext>& ctx)
          LOCKS_EXCLUDED(mu_) {
        std::vector<std::pair<std::shared_ptr<BatchResult>, int64>> new_calls;
        new_calls.reserve(dataset()->num_parallel_calls_);
        while (true) {
          {
            mutex_lock l(mu_);
            while (!cancelled_ &&
                   (num_calls_ >= dataset()->num_parallel_calls_ ||
                    batch_results_.size() > MaxBatchResults() ||
                    (batch_results_.size() == MaxBatchResults() &&
                     call_counter_ % dataset()->batch_size_ == 0))) {
              cond_var_.wait(l);
            }

            if (cancelled_) {
              return;
            }

            while (num_calls_ < dataset()->num_parallel_calls_ &&
                   (batch_results_.size() < MaxBatchResults() ||
                    (batch_results_.size() == MaxBatchResults() &&
                     call_counter_ % dataset()->batch_size_ != 0))) {
              if (call_counter_ % dataset()->batch_size_ == 0) {
                batch_results_.emplace_back(
                    new BatchResult(dataset()->batch_size_));
              }
              int64 offset = call_counter_++ % dataset()->batch_size_;
              new_calls.emplace_back(batch_results_.back(), offset);
              num_calls_++;
            }
          }

          for (const auto& call : new_calls) {
            CallFunction(ctx, call.first, call.second);
          }
          new_calls.clear();
        }
      }

      Status ReadBatchResult(IteratorContext* ctx, IteratorStateReader* reader,
                             size_t index) EXCLUSIVE_LOCKS_REQUIRED(mu_) {
        batch_results_.emplace_back(new BatchResult(dataset()->batch_size_));
        std::shared_ptr<BatchResult> result = batch_results_.back();
        string prefix = strings::StrCat("batch_results_", index);
        mutex_lock l(result->mu);
        result->end_of_input = reader->Contains(
            full_name(strings::StrCat(prefix, "_end_of_input")));
        TF_RETURN_IF_ERROR(
            reader->ReadScalar(full_name(strings::StrCat(prefix, "_num_calls")),
                               &result->num_calls));
        TF_RETURN_IF_ERROR(reader->ReadScalar(
            full_name(strings::StrCat(prefix, "_num_elements")),
            &result->num_elements));
        result->output_allocated = reader->Contains(
            full_name(strings::StrCat(prefix, "_output_allocated")));
        int64 output_size;
        TF_RETURN_IF_ERROR(reader->ReadScalar(
            full_name(strings::StrCat(prefix, "_output_size")), &output_size));
        result->output.reserve(output_size);
        for (int i = 0; i < output_size; i++) {
          Tensor t;
          TF_RETURN_IF_ERROR(reader->ReadTensor(
              full_name(strings::StrCat(prefix, "_output_", i)), &t));
          // If the batch was not full, we may have stored only the relevant
          // slice. Since tensors in `BatchResult.output` are expected to
          // have the leading dimension of size batch_size, we build a larger
          // tensor and copy the slice read from the checkpoint into it.
          if (t.dim_size(0) < dataset()->batch_size_) {
            TensorShape component_shape(t.shape());
            component_shape.set_dim(0, dataset()->batch_size_);
            AllocatorAttributes attr;
            attr.set_gpu_compatible(true);
            Tensor new_t(ctx->allocator(attr), t.dtype(), component_shape);
            TF_RETURN_IF_ERROR(CopyPartialBatch(&new_t, t, t.dim_size(0)));
            result->output.emplace_back(std::move(new_t));
          } else {
            result->output.emplace_back(std::move(t));
          }
        }
        TF_RETURN_IF_ERROR(ReadStatus(
            reader, strings::StrCat(prefix, "_status"), &result->status));
        return Status::OK();
      }

      Status ReadStatus(IteratorStateReader* reader, const string& prefix,
                        Status* status) EXCLUSIVE_LOCKS_REQUIRED(mu_) {
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

      Status WriteBatchResult(IteratorStateWriter* writer, size_t index)
          EXCLUSIVE_LOCKS_REQUIRED(mu_) {
        std::shared_ptr<BatchResult> result = batch_results_[index];
        string prefix = strings::StrCat("batch_results_", index);
        mutex_lock l(result->mu);
        if (result->end_of_input) {
          TF_RETURN_IF_ERROR(writer->WriteScalar(
              full_name(strings::StrCat(prefix, "_end_of_input")), ""));
        }
        TF_RETURN_IF_ERROR(writer->WriteScalar(
            full_name(strings::StrCat(prefix, "_num_calls")),
            result->num_calls));
        TF_RETURN_IF_ERROR(writer->WriteScalar(
            full_name(strings::StrCat(prefix, "_num_elements")),
            result->num_elements));
        if (result->output_allocated) {
          TF_RETURN_IF_ERROR(writer->WriteScalar(
              full_name(strings::StrCat(prefix, "_output_allocated")), ""));
        }
        TF_RETURN_IF_ERROR(writer->WriteScalar(
            full_name(strings::StrCat(prefix, "_output_size")),
            result->output.size()));
        for (int i = 0; i < result->output.size(); i++) {
          // If the batch is not full, we only store the first `num_elements`
          // values. The rest of the batch tensor is *uninitialized* and
          // accessing that will raise msan errors.
          if (result->num_elements < dataset()->batch_size_) {
            TF_RETURN_IF_ERROR(writer->WriteTensor(
                full_name(strings::StrCat(prefix, "_output_", i)),
                result->output[i].Slice(0, result->num_elements)));
          } else {
            TF_RETURN_IF_ERROR(writer->WriteTensor(
                full_name(strings::StrCat(prefix, "_output_", i)),
                result->output[i]));
          }
        }
        TF_RETURN_IF_ERROR(WriteStatus(
            writer, strings::StrCat(prefix, "_status"), result->status));
        return Status::OK();
      }

      Status WriteStatus(IteratorStateWriter* writer, const string& prefix,
                         const Status& status) EXCLUSIVE_LOCKS_REQUIRED(mu_) {
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

      // Used for coordination between the main thread, the runner thread, and
      // the callback threads.
      mutex mu_;
      // Used for coordination between the main thread, the runner thread, and
      // the callback threads. In particular, the runner thread should only
      // schedule new calls when the number of in-flight calls is less than the
      // user specified level of parallelism and there are slots available in
      // the `batch_results_` buffer.
      condition_variable cond_var_;
      // Counts the number of outstanding calls for this batch.
      int64 num_calls_ GUARDED_BY(mu_) = 0;
      // Counts the total number of calls.
      int64 call_counter_ GUARDED_BY(mu_) = 0;
      std::unique_ptr<IteratorBase> input_impl_;
      // Buffer for storing the (intermediate) batch results.
      std::deque<std::shared_ptr<BatchResult>> batch_results_ GUARDED_BY(mu_);
      std::unique_ptr<Thread> runner_thread_ GUARDED_BY(mu_);
      bool cancelled_ GUARDED_BY(mu_) = false;
    };

    const DatasetBase* const input_;
    const NameAttrList func_;
    const int64 batch_size_;
    const int64 num_parallel_calls_;
    const bool drop_remainder_;
    const DataTypeVector output_types_;
    const std::vector<PartialTensorShape> output_shapes_;
    const NameAttrList map_fn_;
    const std::unique_ptr<CapturedFunction> captured_func_;
    const Eigen::ThreadPoolDevice* device_;  // not owned
  };

  const int graph_def_version_;
  const int op_version_;
  DataTypeVector output_types_;
  std::vector<PartialTensorShape> output_shapes_;
  NameAttrList func_;
};

REGISTER_KERNEL_BUILDER(Name("MapAndBatchDataset").Device(DEVICE_CPU),
                        MapAndBatchDatasetOp);

REGISTER_KERNEL_BUILDER(Name("MapAndBatchDatasetV2").Device(DEVICE_CPU),
                        MapAndBatchDatasetOp);

}  // namespace

}  // namespace tensorflow
