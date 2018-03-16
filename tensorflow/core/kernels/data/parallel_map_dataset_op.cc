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
#include "tensorflow/core/lib/core/error_codes.pb.h"
#include "tensorflow/core/lib/random/random.h"

namespace tensorflow {

namespace {

// See documentation in ../ops/dataset_ops.cc for a high-level
// description of the following op.

class ParallelMapDatasetOp : public UnaryDatasetOpKernel {
 public:
  explicit ParallelMapDatasetOp(OpKernelConstruction* ctx)
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

    int32 num_parallel_calls;
    OP_REQUIRES_OK(ctx, ParseScalarArgument(ctx, "num_parallel_calls",
                                            &num_parallel_calls));
    OP_REQUIRES(ctx, num_parallel_calls > 0,
                errors::InvalidArgument(
                    "num_parallel_calls must be greater than zero."));

    std::unique_ptr<CapturedFunction> captured_func;
    OP_REQUIRES_OK(ctx, CapturedFunction::Create(
                            func_, std::move(other_arguments), &captured_func));

    *output = new Dataset(ctx, input, func_, num_parallel_calls, output_types_,
                          output_shapes_, std::move(captured_func));
  }

 private:
  class Dataset : public GraphDatasetBase {
   public:
    Dataset(OpKernelContext* ctx, const DatasetBase* input,
            const NameAttrList& func, int32 num_parallel_calls,
            const DataTypeVector& output_types,
            const std::vector<PartialTensorShape>& output_shapes,
            std::unique_ptr<CapturedFunction> captured_func)
        : GraphDatasetBase(ctx),
          input_(input),
          func_(func),
          num_parallel_calls_(num_parallel_calls),
          output_types_(output_types),
          output_shapes_(output_shapes),
          captured_func_(std::move(captured_func)) {
      input_->Ref();
    }

    ~Dataset() override { input_->Unref(); }

    std::unique_ptr<IteratorBase> MakeIterator(
        const string& prefix) const override {
      return std::unique_ptr<IteratorBase>(
          new Iterator({this, strings::StrCat(prefix, "::ParallelMap")}));
    }

    const DataTypeVector& output_dtypes() const override {
      return output_types_;
    }

    const std::vector<PartialTensorShape>& output_shapes() const override {
      return output_shapes_;
    }

    string DebugString() override { return "ParallelMapDatasetOp::Dataset"; }

   protected:
    Status AsGraphDefInternal(OpKernelContext* ctx, DatasetGraphDefBuilder* b,
                              Node** output) const override {
      // Input: input_dataset
      Node* input_graph_node = nullptr;
      TF_RETURN_IF_ERROR(b->AddParentDataset(ctx, input_, &input_graph_node));

      // Input: other_arguments
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

      // Input: num_parallel_calls
      Node* num_parallel_calls = nullptr;
      TF_RETURN_IF_ERROR(
          b->AddScalar(num_parallel_calls_, &num_parallel_calls));

      // Attr: f
      TF_RETURN_IF_ERROR(b->AddFunction(ctx, func_.name()));
      AttrValue f;
      b->BuildAttrValue(func_, &f);

      // Attr: Targuments
      AttrValue other_arguments_types_attr;
      b->BuildAttrValue(other_arguments_types, &other_arguments_types_attr);

      TF_RETURN_IF_ERROR(b->AddDataset(
          this,
          {std::make_pair(0, input_graph_node),
           std::make_pair(2, num_parallel_calls)},  // Single tensor inputs.
          {std::make_pair(1, other_arguments)},     // Tensor list inputs.
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
            invocation_results_(params.dataset->num_parallel_calls_) {}

      ~Iterator() override {
        // TODO(mrry): Replace this cancellation logic with a
        // CancellationManager. The syntax would be more heavyweight,
        // but it would be possible to thread a cancellation manager
        // through the IteratorContext to upstream,
        // potentially-blocking iterators, when we add these.
        {
          mutex_lock l(mu_);
          for (size_t i = 0; i < dataset()->num_parallel_calls_; ++i) {
            if (invocation_results_[i].notification) {
              invocation_results_[i].notification->WaitForNotification();
            }
          }
        }
      }

      Status GetNextInternal(IteratorContext* ctx,
                             std::vector<Tensor>* out_tensors,
                             bool* end_of_sequence) override {
        mutex_lock l(mu_);

        // Ensure that there are `dataset()->num_parallel_calls_`
        // invocations of `func_` outstanding at once.
        while (input_impl_ && (num_inputs_consumed_ - num_outputs_consumed_ <
                               dataset()->num_parallel_calls_)) {
          InvokeFunctionLocked(ctx);
        }

        if (!input_impl_ && num_inputs_consumed_ == num_outputs_consumed_) {
          *end_of_sequence = true;
          return Status::OK();
        }

        // Read the next result out of `invocation_results_`, which
        // acts as a circular buffer.
        const size_t result_index =
            num_outputs_consumed_ % dataset()->num_parallel_calls_;
        InvocationResult* result = &invocation_results_[result_index];
        *end_of_sequence = false;
        if (result->notification) {
          result->notification->WaitForNotification();
          if (result->status.ok()) {
            std::swap(*out_tensors, result->return_values);
          }
        }
        ++num_outputs_consumed_;
        if (errors::IsOutOfRange(result->status)) {
          // `f` may deliberately raise `errors::OutOfRange` to indicate
          // that we should terminate the iteration early.
          *end_of_sequence = true;
          return Status::OK();
        } else {
          return result->status;
        }
      }

     protected:
      Status SaveInternal(IteratorStateWriter* writer) override {
        mutex_lock l(mu_);
        if (input_impl_) {
          TF_RETURN_IF_ERROR(SaveParent(writer, input_impl_));
        } else {
          TF_RETURN_IF_ERROR(
              writer->WriteScalar(full_name("end_of_input"), ""));
        }
        TF_RETURN_IF_ERROR(writer->WriteScalar(full_name("num_inputs_consumed"),
                                               num_inputs_consumed_));
        TF_RETURN_IF_ERROR(writer->WriteScalar(
            full_name("num_outputs_consumed"), num_outputs_consumed_));

        for (size_t i = 0; i < dataset()->num_parallel_calls_; i++) {
          if (invocation_results_[i].notification) {
            invocation_results_[i].notification->WaitForNotification();
            TF_RETURN_IF_ERROR(
                WriteStatusLocked(writer, i, invocation_results_[i].status));
            TF_RETURN_IF_ERROR(writer->WriteScalar(
                full_name(strings::StrCat("invocation_results[", i, "].size")),
                invocation_results_[i].return_values.size()));
            for (size_t j = 0; j < invocation_results_[i].return_values.size();
                 j++) {
              TF_RETURN_IF_ERROR(writer->WriteTensor(
                  full_name(
                      strings::StrCat("invocation_results[", i, "][", j, "]")),
                  invocation_results_[i].return_values[j]));
            }
          } else {
            TF_RETURN_IF_ERROR(writer->WriteScalar(
                full_name(strings::StrCat("invocation_results[", i, "]_empty")),
                ""));
          }
        }

        return Status::OK();
      }

      Status RestoreInternal(IteratorContext* ctx,
                             IteratorStateReader* reader) override {
        mutex_lock l(mu_);
        if (reader->Contains(full_name("end_of_input"))) {
          input_impl_.reset();
        } else {
          TF_RETURN_IF_ERROR(RestoreParent(ctx, reader, input_impl_));
        }
        TF_RETURN_IF_ERROR(reader->ReadScalar(full_name("num_inputs_consumed"),
                                              &num_inputs_consumed_));
        TF_RETURN_IF_ERROR(reader->ReadScalar(full_name("num_outputs_consumed"),
                                              &num_outputs_consumed_));
        for (size_t i = 0; i < dataset()->num_parallel_calls_; i++) {
          InvocationResult* result = &invocation_results_[i];
          *result = InvocationResult();
          if (!reader->Contains(full_name(
                  strings::StrCat("invocation_results[", i, "]_empty")))) {
            result->notification.reset(new Notification);
            result->notification->Notify();
            TF_RETURN_IF_ERROR(ReadStatusLocked(reader, i, &result->status));
            size_t num_return_values;
            {
              int64 size;
              TF_RETURN_IF_ERROR(
                  reader->ReadScalar(full_name(strings::StrCat(
                                         "invocation_results[", i, "].size")),
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
              TF_RETURN_IF_ERROR(reader->ReadTensor(
                  full_name(
                      strings::StrCat("invocation_results[", i, "][", j, "]")),
                  &result->return_values.back()));
            }
          }
        }
        return Status::OK();
      }

     private:
      struct InvocationResult {
        Status status;
        std::unique_ptr<Notification> notification;
        std::vector<Tensor> return_values;
      };

      void InvokeFunctionLocked(IteratorContext* ctx)
          EXCLUSIVE_LOCKS_REQUIRED(mu_) {
        DCHECK(input_impl_);
        DCHECK(num_inputs_consumed_ - num_outputs_consumed_ <
               dataset()->num_parallel_calls_);

        // The result of invoking the function will be written into the next
        // slot in `invocation_results_`, which acts as a circular buffer.
        const size_t result_index =
            num_inputs_consumed_ % dataset()->num_parallel_calls_;
        InvocationResult* result = &invocation_results_[result_index];
        *result = InvocationResult();

        // Get the next input element.
        std::vector<Tensor> input_element;
        bool end_of_input = false;
        result->status =
            input_impl_->GetNext(ctx, &input_element, &end_of_input);
        if (end_of_input) {
          input_impl_.reset();
          result->status = errors::OutOfRange("");
        } else {
          ++num_inputs_consumed_;
        }

        if (result->status.ok()) {
          // Call `func_(input_element)`, store the result in
          // `result->return_values`, and notify `result->notification`
          // to unblock a consumer.
          result->notification.reset(new Notification);
          dataset()->captured_func_->RunAsync(
              ctx, std::move(input_element), &result->return_values,
              [result, result_index](Status ret_status) {
                result->status.Update(ret_status);
                result->notification->Notify();
              });
        }
      }

      Status WriteStatusLocked(IteratorStateWriter* writer, size_t index,
                               const Status& status)
          EXCLUSIVE_LOCKS_REQUIRED(mu_) {
        TF_RETURN_IF_ERROR(writer->WriteScalar(
            CodeKey(index), static_cast<int64>(status.code())));
        if (!status.ok()) {
          TF_RETURN_IF_ERROR(writer->WriteScalar(ErrorMessageKey(index),
                                                 status.error_message()));
        }
        return Status::OK();
      }

      Status ReadStatusLocked(IteratorStateReader* reader, size_t index,
                              Status* status) EXCLUSIVE_LOCKS_REQUIRED(mu_) {
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

      mutex mu_;
      std::unique_ptr<IteratorBase> input_impl_ GUARDED_BY(mu_);
      std::vector<InvocationResult> invocation_results_ GUARDED_BY(mu_);
      int64 num_inputs_consumed_ GUARDED_BY(mu_) = 0;
      int64 num_outputs_consumed_ GUARDED_BY(mu_) = 0;
    };

    const DatasetBase* const input_;
    const NameAttrList func_;
    const int32 num_parallel_calls_;
    const DataTypeVector output_types_;
    const std::vector<PartialTensorShape> output_shapes_;
    const std::unique_ptr<CapturedFunction> captured_func_;
  };

  const int graph_def_version_;
  DataTypeVector output_types_;
  std::vector<PartialTensorShape> output_shapes_;
  NameAttrList func_;
};

REGISTER_KERNEL_BUILDER(Name("ParallelMapDataset").Device(DEVICE_CPU),
                        ParallelMapDatasetOp);

}  // namespace

}  // namespace tensorflow
