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

#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/framework/partial_tensor_shape.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/kernels/data/captured_function.h"
#include "tensorflow/core/kernels/data/dataset.h"
#include "tensorflow/core/kernels/inplace_ops_functor.h"
#include "tensorflow/core/lib/core/blocking_counter.h"
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
        graph_def_version_(ctx->graph_def_version()) {
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

    int64 num_parallel_batches;
    OP_REQUIRES_OK(ctx, ParseScalarArgument(ctx, "num_parallel_batches",
                                            &num_parallel_batches));
    OP_REQUIRES(ctx, num_parallel_batches > 0,
                errors::InvalidArgument(
                    "num_parallel_batches must be greater than zero."));

    bool drop_remainder;
    OP_REQUIRES_OK(ctx,
                   ParseScalarArgument(ctx, "drop_remainder", &drop_remainder));

    std::unique_ptr<CapturedFunction> captured_func;
    OP_REQUIRES_OK(ctx, CapturedFunction::Create(
                            func_, std::move(other_arguments), &captured_func));

    *output = new Dataset(ctx, input, batch_size, num_parallel_batches,
                          drop_remainder, output_types_, output_shapes_, func_,
                          std::move(captured_func), &ctx->eigen_cpu_device());
  }

 private:
  class Dataset : public GraphDatasetBase {
   public:
    Dataset(OpKernelContext* ctx, const DatasetBase* input, int64 batch_size,
            int64 num_parallel_batches, bool drop_remainder,
            const DataTypeVector& output_types,
            const std::vector<PartialTensorShape>& output_shapes,
            const NameAttrList& func,
            std::unique_ptr<CapturedFunction> captured_func,
            const Eigen::ThreadPoolDevice* device)
        : GraphDatasetBase(ctx),
          input_(input),
          batch_size_(batch_size),
          num_parallel_batches_(num_parallel_batches),
          drop_remainder_(drop_remainder),
          output_types_(output_types),
          output_shapes_(output_shapes),
          map_fn_(func),
          captured_func_(std::move(captured_func)),
          device_(device) {
      input_->Ref();
    }

    ~Dataset() override { input_->Unref(); }

    std::unique_ptr<IteratorBase> MakeIterator(
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

    string DebugString() override { return "MapAndBatchDatasetOp::Dataset"; }

   protected:
    Status AsGraphDefInternal(OpKernelContext* ctx, DatasetGraphDefBuilder* b,
                              Node** output) const override {
      TF_RETURN_IF_ERROR(b->AddFunction(ctx, map_fn_.name()));
      Node* input_graph_node = nullptr;
      TF_RETURN_IF_ERROR(b->AddParentDataset(ctx, input_, &input_graph_node));
      Node* batch_size_node;
      TF_RETURN_IF_ERROR(b->AddScalar(batch_size_, &batch_size_node));
      Node* num_parallel_batches_node;
      TF_RETURN_IF_ERROR(
          b->AddScalar(num_parallel_batches_, &num_parallel_batches_node));
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
           std::make_pair(3, num_parallel_batches_node),
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
            input_impl_(params.dataset->input_->MakeIterator(params.prefix)),
            invocation_results_(params.dataset->batch_size_ *
                                params.dataset->num_parallel_batches_),
            batch_results_(params.dataset->num_parallel_batches_) {}

      ~Iterator() override {
        // TODO(mrry): Replace this cancellation logic with a
        // CancellationManager. The syntax would be more heavyweight,
        // but it would be possible to thread a cancellation manager
        // through the IteratorContext to upstream,
        // potentially-blocking iterators, when we add these.
        mutex_lock l(mu_);
        if (current_batch_index_ != -1) {
          for (size_t batch_index = 0;
               batch_index < dataset()->num_parallel_batches_; ++batch_index) {
            int64 num_elements;
            WaitForBatch(batch_index, &num_elements).IgnoreError();
            // Deallocate tensors allocated for the output.
            batch_results_[batch_index].output.clear();
          }
        }
      }

      // TODO(jsimsa): Implement and profile the following alternative design:
      //
      // 0. Set the number of in-flight batches and invocations independently
      // (though obviously the max number of in-flight invocations must be <
      // batch_size * num_parallel_batches). Maintain a current producing batch
      // index and offset.
      // 1. Issue invocations in order of batch and offset, as you do currently.
      // 2. When an invocation finishes, increment the current producing batch
      // and offset. If that invocation would start a new batch and give more
      // than num_parallel_batches in-flight, block; else start the new
      // invocation into that location.
      // 3. When a GetNext() call arrives, block until there's a full batch.
      // Before returning the batch, if the number of pending invocations is
      // less than the max, issue that number of invocations.
      Status GetNextInternal(IteratorContext* ctx,
                             std::vector<Tensor>* out_tensors,
                             bool* end_of_sequence) override {
        mutex_lock l(mu_);

        // One-time initialization.
        if (current_batch_index_ == -1) {
          current_batch_index_ = 0;
          for (size_t i = 0; i < dataset()->num_parallel_batches_; ++i) {
            StartInvocationBatch(ctx, i);
          }
        }

        int64 num_elements = 0;
        Status status = WaitForBatch(current_batch_index_, &num_elements);
        if (num_elements == 0) {
          *end_of_sequence = true;
          return Status::OK();
        }
        if (!status.ok()) {
          // Deallocate tensors allocated for the output.
          batch_results_[current_batch_index_].output.clear();
        } else {
          if (num_elements < dataset()->batch_size_) {
            if (dataset()->drop_remainder_) {
              // Deallocate tensors allocated for the output.
              batch_results_[current_batch_index_].output.clear();
              *end_of_sequence = true;
              return Status::OK();
            }
            const std::vector<Tensor>& output =
                batch_results_[current_batch_index_].output;
            for (size_t i = 0; i < output.size(); ++i) {
              TensorShape component_shape(
                  batch_results_[current_batch_index_].output[i].shape());
              component_shape.set_dim(0, num_elements);
              AllocatorAttributes attr;
              attr.set_gpu_compatible(true);
              Tensor component(ctx->allocator(attr), output[i].dtype(),
                               component_shape);
              TF_RETURN_IF_ERROR(
                  CopyPartialBatch(&component, output[i], num_elements));
              out_tensors->emplace_back(std::move(component));
            }
            // Deallocate tensors allocated for the output.
            batch_results_[current_batch_index_].output.clear();
          } else {
            *out_tensors =
                std::move(batch_results_[current_batch_index_].output);
          }
          *end_of_sequence = false;
        }
        StartInvocationBatch(ctx, current_batch_index_);
        current_batch_index_ =
            (current_batch_index_ + 1) % dataset()->num_parallel_batches_;
        return status;
      }

     protected:
      Status SaveInternal(IteratorStateWriter* writer) override {
        mutex_lock l(mu_);
        if (current_batch_index_ == -1) {
          // Iterator has not been used. Nothing to save.
          return Status::OK();
        }
        TF_RETURN_IF_ERROR(writer->WriteScalar(full_name("current_batch_index"),
                                               current_batch_index_));
        TF_RETURN_IF_ERROR(SaveParent(writer, input_impl_));
        TF_RETURN_IF_ERROR(writer->WriteScalar(
            full_name("invocation_results_size"), invocation_results_.size()));
        for (size_t i = 0; i < invocation_results_.size(); ++i) {
          TF_RETURN_IF_ERROR(WriteInvocationResultLocked(writer, i));
        }
        TF_RETURN_IF_ERROR(writer->WriteScalar(full_name("batch_results_size"),
                                               batch_results_.size()));
        for (size_t i = 0; i < batch_results_.size(); ++i) {
          TF_RETURN_IF_ERROR(WriteBatchResultLocked(writer, i));
        }
        return Status::OK();
      }

      Status RestoreInternal(IteratorContext* ctx,
                             IteratorStateReader* reader) override {
        mutex_lock l(mu_);
        if (!reader->Contains(full_name("current_batch_index"))) {
          // Iterator was never used so nothing to restore.
          return Status::OK();
        }
        {
          int64 temp;
          TF_RETURN_IF_ERROR(
              reader->ReadScalar(full_name("current_batch_index"), &temp));
          current_batch_index_ = static_cast<int32>(temp);
          if (current_batch_index_ != temp) {
            return errors::Internal("Invalid value for current_batch_index ",
                                    temp);
          }
        }
        TF_RETURN_IF_ERROR(RestoreParent(ctx, reader, input_impl_));
        size_t invocation_results_size;
        {
          int64 temp;
          TF_RETURN_IF_ERROR(
              reader->ReadScalar(full_name("invocation_results_size"), &temp));
          invocation_results_size = static_cast<size_t>(temp);
          if (invocation_results_size != temp) {
            return errors::Internal(
                "Invalid value for invocation_results_size ", temp);
          }
        }
        CHECK_EQ(invocation_results_.size(), invocation_results_size);
        for (size_t i = 0; i < invocation_results_size; ++i) {
          TF_RETURN_IF_ERROR(ReadInvocationResultLocked(reader, i));
        }
        size_t batch_results_size;
        {
          int64 temp;
          TF_RETURN_IF_ERROR(
              reader->ReadScalar(full_name("batch_results_size"), &temp));
          batch_results_size = static_cast<size_t>(temp);
          if (batch_results_size != temp) {
            return errors::Internal("Invalid value for batch_results_size ",
                                    temp);
          }
        }
        CHECK_EQ(batch_results_.size(), batch_results_size);
        for (size_t i = 0; i < batch_results_size; ++i) {
          TF_RETURN_IF_ERROR(ReadBatchResultLocked(reader, i));
        }
        return Status::OK();
      }

     private:
      struct BatchResult {
        mutex mu ACQUIRED_AFTER(mu_);
        bool output_allocated GUARDED_BY(mu);
        std::vector<Tensor> output;
        std::unique_ptr<BlockingCounter> counter;
      };

      struct InvocationResult {
        Status status;
        bool end_of_input;
        std::vector<Tensor> return_values;
      };

      int64 ComputeInvocationIndex(int64 batch_index, int64 offset) {
        return batch_index * dataset()->batch_size_ + offset;
      }

      Status CopyPartialBatch(Tensor* output, const Tensor& value,
                              int64 num_elements) {
        switch (value.dtype()) {
#define CASE(type)                                                \
  case DataTypeToEnum<type>::value: {                             \
    auto output_t = output->flat_outer_dims<type>();              \
    auto value_t = value.flat_outer_dims<type>();                 \
    for (size_t i = 0; i < num_elements; i++) {                   \
      output_t.template chip<0>(i) = value_t.template chip<0>(i); \
    }                                                             \
    return Status::OK();                                          \
  }
          TF_CALL_NUMBER_TYPES(CASE);
          TF_CALL_string(CASE);
          TF_CALL_variant(CASE);
#undef CASE
          default:
            return errors::InvalidArgument("Unsupported data type: ",
                                           value.dtype());
        }
        return Status::OK();
      }

      void EnsureOutputAllocated(IteratorContext* ctx,
                                 BatchResult* batch_result,
                                 const std::vector<Tensor>& return_values) {
        mutex_lock l(batch_result->mu);
        if (batch_result->output_allocated) {
          return;
        }
        const size_t num_components = return_values.size();
        for (size_t i = 0; i < num_components; ++i) {
          TensorShape component_shape({dataset()->batch_size_});
          component_shape.AppendShape(return_values[i].shape());
          AllocatorAttributes attr;
          attr.set_gpu_compatible(true);
          Tensor component(ctx->allocator(attr), return_values[i].dtype(),
                           component_shape);
          batch_result->output.emplace_back(std::move(component));
        }
        batch_result->output_allocated = true;
      }

      void InvokeFunctionLocked(IteratorContext* ctx, int64 batch_index,
                                int64 offset) EXCLUSIVE_LOCKS_REQUIRED(mu_) {
        size_t index = ComputeInvocationIndex(batch_index, offset);
        InvocationResult* result = &invocation_results_[index];
        BatchResult* batch_result = &batch_results_[batch_index];

        // Get the next input element.
        std::vector<Tensor> input_element;
        result->status =
            input_impl_->GetNext(ctx, &input_element, &result->end_of_input);
        if (result->end_of_input || !result->status.ok()) {
          batch_result->counter->DecrementCount();
          return;
        }

        // Call `captured_func_(input_element)`, store the result in
        // `result->return_values`, and notify `batch_result->counter`
        // to unblock a consumer.
        (*ctx->runner())(std::bind(
            [this, result, batch_result, offset](
                IteratorContext* ctx, std::vector<Tensor> input_element) {
              dataset()->captured_func_->RunAsync(
                  ctx, std::move(input_element), &result->return_values,
                  [this, ctx, result, batch_result, offset](Status ret_status) {
                    result->status.Update(ret_status);
                    if (ret_status.ok()) {
                      EnsureOutputAllocated(ctx, batch_result,
                                            result->return_values);
                      const size_t num_components =
                          result->return_values.size();
                      for (size_t i = 0; i < num_components; ++i) {
                        const Tensor& tensor = result->return_values[i];
                        Tensor* batch = &(batch_result->output)[i];
                        if (tensor.NumElements() !=
                            (batch->NumElements() / batch->dim_size(0))) {
                          TensorShape batch_shape = batch->shape();
                          batch_shape.RemoveDim(0);
                          result->status.Update(errors::InvalidArgument(
                              "Cannot add tensor to the batch: number of "
                              "elements does not match. Shapes are: [tensor]: ",
                              tensor.shape().DebugString(),
                              ", [batch]: ", batch_shape.DebugString()));
                          break;
                        }
                        // TODO(mrry): Add a version of DoParallelConcat that
                        // allows us to move `tensor` where possible, to speed
                        // up string tensor batching.
                        Status copy_status =
                            ::tensorflow::functor::DoParallelConcat(
                                *dataset()->device_, tensor, offset, batch);
                        if (!copy_status.ok()) {
                          result->status.Update(copy_status);
                          break;
                        }
                      }
                    }
                    delete ctx;
                    // NOTE(mrry): We clear the return values here to release
                    // any memory associated with them and to paralellize the
                    // destruction of the tensors (which can be surprisingly
                    // expensive for map functions with large numbers of return
                    // values).
                    result->return_values.clear();
                    batch_result->counter->DecrementCount();
                  });
            },
            new IteratorContext(*ctx), std::move(input_element)));
      }

      void StartInvocationBatch(IteratorContext* ctx, int64 batch_index)
          EXCLUSIVE_LOCKS_REQUIRED(mu_) {
        port::Tracing::TraceMe activity(strings::StrCat(prefix(), "::Start"));
        // Initialize batch result.
        {
          mutex_lock l(batch_results_[batch_index].mu);
          batch_results_[batch_index].output_allocated = false;
          batch_results_[batch_index].counter.reset(
              new BlockingCounter(dataset()->batch_size_));
        }
        // Initialize invocation results.
        for (size_t i = 0; i < dataset()->batch_size_; ++i) {
          size_t index = ComputeInvocationIndex(batch_index, i);
          InvocationResult* result = &invocation_results_[index];
          // Reset the state of `result`; `result->return_values` was cleared
          // when the previous invocation completed.
          result->end_of_input = false;
          result->status = Status::OK();
        }
        // Start individual invocations.
        for (size_t i = 0; i < dataset()->batch_size_; ++i) {
          InvokeFunctionLocked(ctx, batch_index, i);
        }
      }

      Status WaitForBatch(int64 batch_index, int64* num_elements)
          EXCLUSIVE_LOCKS_REQUIRED(mu_) {
        port::Tracing::TraceMe activity(strings::StrCat(prefix(), "::Wait"));
        batch_results_[batch_index].counter->Wait();
        Status status = Status::OK();
        for (size_t i = 0; i < dataset()->batch_size_; ++i, ++*num_elements) {
          size_t index = ComputeInvocationIndex(batch_index, i);
          InvocationResult* result = &invocation_results_[index];
          if (result->end_of_input) {
            VLOG(3) << "end of input encountered at element[" << i << "]: ";
            return Status::OK();
          }
          if (!result->status.ok()) {
            VLOG(3) << "failed to process element[" << i
                    << "]: " << result->status;
            status.Update(result->status);
          }
        }
        return status;
      }

      Status WriteInvocationResultLocked(IteratorStateWriter* writer,
                                         size_t index)
          EXCLUSIVE_LOCKS_REQUIRED(mu_) {
        const InvocationResult& result = invocation_results_[index];
        string prefix = strings::StrCat("invocation_results_", index);
        TF_RETURN_IF_ERROR(WriteStatusLocked(
            writer, full_name(strings::StrCat(prefix, "_status")),
            result.status));
        if (result.end_of_input) {
          TF_RETURN_IF_ERROR(writer->WriteScalar(
              full_name(strings::StrCat(prefix, "_end_of_input")), ""));
        }
        TF_RETURN_IF_ERROR(writer->WriteScalar(
            full_name(strings::StrCat(prefix, "_return_values_size")),
            result.return_values.size()));
        for (size_t i = 0; i < result.return_values.size(); i++) {
          TF_RETURN_IF_ERROR(writer->WriteTensor(
              full_name(strings::StrCat(prefix, "_return_values_", i)),
              result.return_values[i]));
        }
        return Status::OK();
      }

      Status ReadInvocationResultLocked(IteratorStateReader* reader,
                                        size_t index)
          EXCLUSIVE_LOCKS_REQUIRED(mu_) {
        InvocationResult* result = &invocation_results_[index];
        string prefix = strings::StrCat("invocation_results_", index);
        TF_RETURN_IF_ERROR(ReadStatusLocked(
            reader, full_name(strings::StrCat(prefix, "_status")),
            &result->status));
        result->end_of_input = reader->Contains(
            full_name(strings::StrCat(prefix, "_end_of_input")));
        size_t return_values_size;
        {
          int64 temp;
          TF_RETURN_IF_ERROR(reader->ReadScalar(
              full_name(strings::StrCat(prefix, "_return_values_size")),
              &temp));
          return_values_size = static_cast<size_t>(temp);
          if (temp != return_values_size) {
            return errors::Internal("Invalid value for return_values_size ",
                                    return_values_size);
          }
        }
        result->return_values.reserve(return_values_size);
        for (size_t i = 0; i < return_values_size; i++) {
          result->return_values.emplace_back();
          TF_RETURN_IF_ERROR(reader->ReadTensor(
              full_name(strings::StrCat(prefix, "_return_values_", i)),
              &result->return_values.back()));
        }
        return Status::OK();
      }

      Status WriteBatchResultLocked(IteratorStateWriter* writer, size_t index)
          EXCLUSIVE_LOCKS_REQUIRED(mu_) {
        // Wait for the map_fn dispatches made in `InvokeFunctionLocked` to
        // finish. This may delay saving a checkpoint by a bit but keeps the
        // code clean and also saves us from checkpointing the state of the
        // `BlockingCounter`.
        batch_results_[index].counter->Wait();
        const BatchResult& result = batch_results_[index];
        string prefix = strings::StrCat("batch_results_", index);
        {
          mutex_lock l(batch_results_[index].mu);
          if (result.output_allocated) {
            TF_RETURN_IF_ERROR(writer->WriteScalar(
                full_name(strings::StrCat(prefix, "_output_allocated")), ""));
          }
        }
        TF_RETURN_IF_ERROR(writer->WriteScalar(
            full_name(strings::StrCat(prefix, "_output_size")),
            result.output.size()));
        for (size_t i = 0; i < result.output.size(); i++) {
          TF_RETURN_IF_ERROR(writer->WriteTensor(
              full_name(strings::StrCat(prefix, "_output_", i)),
              result.output[i]));
        }
        return Status::OK();
      }

      Status ReadBatchResultLocked(IteratorStateReader* reader, size_t index)
          EXCLUSIVE_LOCKS_REQUIRED(mu_) {
        BatchResult* result = &batch_results_[index];
        string prefix = strings::StrCat("batch_results_", index);
        {
          mutex_lock l(batch_results_[index].mu);
          result->output_allocated = reader->Contains(
              full_name(strings::StrCat(prefix, "_output_allocated")));
          // Simulate that the batch was fully generated.
          batch_results_[index].counter.reset(new BlockingCounter(0));
        }
        size_t output_size;
        {
          int64 temp;
          TF_RETURN_IF_ERROR(reader->ReadScalar(
              full_name(strings::StrCat(prefix, "_output_size")), &temp));
          output_size = static_cast<size_t>(temp);
          if (temp != output_size) {
            return errors::Internal("Invalid value for output_size ",
                                    output_size);
          }
        }
        result->output.reserve(output_size);
        for (size_t i = 0; i < output_size; i++) {
          result->output.emplace_back();
          TF_RETURN_IF_ERROR(reader->ReadTensor(
              full_name(strings::StrCat(prefix, "_output_", i)),
              &result->output.back()));
        }
        return Status::OK();
      }

      Status WriteStatusLocked(IteratorStateWriter* writer,
                               const string& prefix, const Status& status)
          EXCLUSIVE_LOCKS_REQUIRED(mu_) {
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
      mutex mu_;
      int32 current_batch_index_ GUARDED_BY(mu_) = -1;
      const std::unique_ptr<IteratorBase> input_impl_ GUARDED_BY(mu_);
      std::vector<InvocationResult> invocation_results_ GUARDED_BY(mu_);
      std::vector<BatchResult> batch_results_ GUARDED_BY(mu_);
    };

    const DatasetBase* const input_;
    const NameAttrList func_;
    const int64 batch_size_;
    const int64 num_parallel_batches_;
    const bool drop_remainder_;
    const DataTypeVector output_types_;
    const std::vector<PartialTensorShape> output_shapes_;
    const NameAttrList map_fn_;
    const std::unique_ptr<CapturedFunction> captured_func_;
    const Eigen::ThreadPoolDevice* device_;  // not owned
  };

  const int graph_def_version_;
  DataTypeVector output_types_;
  std::vector<PartialTensorShape> output_shapes_;
  NameAttrList func_;
};

REGISTER_KERNEL_BUILDER(Name("MapAndBatchDataset").Device(DEVICE_CPU),
                        MapAndBatchDatasetOp);

}  // namespace

}  // namespace tensorflow
