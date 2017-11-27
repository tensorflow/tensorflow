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
#include <iterator>
#include <vector>

#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/framework/partial_tensor_shape.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/kernels/captured_function.h"
#include "tensorflow/core/kernels/dataset.h"
#include "tensorflow/core/lib/random/random.h"

namespace tensorflow {

namespace {

// See documentation in ../ops/dataset_ops.cc for a high-level
// description of the following op.

class ScanDatasetOp : public UnaryDatasetOpKernel {
 public:
  explicit ScanDatasetOp(OpKernelConstruction* ctx)
      : UnaryDatasetOpKernel(ctx),
        graph_def_version_(ctx->graph_def_version()) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("f", &func_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("Tstate", &state_types_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("output_types", &output_types_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("output_shapes", &output_shapes_));
  }

  void MakeDataset(OpKernelContext* ctx, DatasetBase* input,
                   DatasetBase** output) override {
    OpInputList initial_state_inputs;
    OP_REQUIRES_OK(ctx,
                   ctx->input_list("initial_state", &initial_state_inputs));
    std::vector<Tensor> initial_state;
    initial_state.reserve(initial_state_inputs.size());
    for (const Tensor& t : initial_state_inputs) {
      initial_state.push_back(t);
    }

    OpInputList inputs;
    OP_REQUIRES_OK(ctx, ctx->input_list("other_arguments", &inputs));
    std::vector<Tensor> other_arguments;
    other_arguments.reserve(inputs.size());
    for (const Tensor& t : inputs) {
      other_arguments.push_back(t);
    }

    std::unique_ptr<CapturedFunction> captured_func;
    OP_REQUIRES_OK(ctx, CapturedFunction::Create(ctx, func_, graph_def_version_,
                                                 std::move(other_arguments),
                                                 &captured_func));

    *output =
        new Dataset(input, std::move(initial_state), std::move(captured_func),
                    state_types_, output_types_, output_shapes_);
  }

 private:
  class Dataset : public DatasetBase {
   public:
    Dataset(const DatasetBase* input, std::vector<Tensor> initial_state,
            std::unique_ptr<CapturedFunction> captured_func,
            const DataTypeVector& state_types,
            const DataTypeVector& output_types,
            const std::vector<PartialTensorShape>& output_shapes)
        : input_(input),
          initial_state_(std::move(initial_state)),
          captured_func_(std::move(captured_func)),
          state_types_(state_types),
          output_types_(output_types),
          output_shapes_(output_shapes) {
      input_->Ref();
    }

    ~Dataset() override { input_->Unref(); }

    std::unique_ptr<IteratorBase> MakeIterator(
        const string& prefix) const override {
      return std::unique_ptr<IteratorBase>(
          new Iterator({this, strings::StrCat(prefix, "::Scan")}));
    }

    const DataTypeVector& output_dtypes() const override {
      return output_types_;
    }
    const std::vector<PartialTensorShape>& output_shapes() const override {
      return output_shapes_;
    }

    string DebugString() override { return "ScanDatasetOp::Dataset"; }

   private:
    class Iterator : public DatasetIterator<Dataset> {
     public:
      explicit Iterator(const Params& params)
          : DatasetIterator<Dataset>(params),
            input_impl_(params.dataset->input_->MakeIterator(params.prefix)),
            state_(params.dataset->initial_state_) {}

      Status GetNextInternal(IteratorContext* ctx,
                             std::vector<Tensor>* out_tensors,
                             bool* end_of_sequence) override {
        mutex_lock l(mu_);

        std::vector<Tensor> next_element;
        TF_RETURN_IF_ERROR(
            input_impl_->GetNext(ctx, &next_element, end_of_sequence));
        if (*end_of_sequence) {
          return Status::OK();
        }

        std::vector<Tensor> args;
        args.reserve(state_.size() + next_element.size());
        std::copy(state_.begin(), state_.end(), std::back_inserter(args));
        std::copy(next_element.begin(), next_element.end(),
                  std::back_inserter(args));

        FunctionLibraryRuntime::Options opts;
        opts.step_id = CapturedFunction::generate_step_id();
        ScopedStepContainer step_container(
            opts.step_id, [this, ctx](const string& name) {
              dataset()
                  ->captured_func_->resource_manager()
                  ->Cleanup(name)
                  .IgnoreError();
            });
        opts.step_container = &step_container;
        opts.runner = ctx->runner();
        std::vector<Tensor> state_and_output;
        state_and_output.reserve(dataset()->state_types_.size() +
                                 output_dtypes().size());
        Status s =
            dataset()->captured_func_->Run(opts, args, &state_and_output);
        if (s.ok()) {
          state_.clear();
          size_t i = 0;
          for (; i < dataset()->state_types_.size(); ++i) {
            if (state_and_output[i].dtype() != dataset()->state_types_[i]) {
              return errors::InvalidArgument(
                  "Got wrong type for scan_func return value ", i,
                  " (expected ", DataTypeString(dataset()->state_types_[i]),
                  ", got ", DataTypeString(state_and_output[i].dtype()), ").");
            }
            state_.push_back(std::move(state_and_output[i]));
          }
          for (; i < state_and_output.size(); ++i) {
            const size_t output_index = i - dataset()->state_types_.size();
            if (state_and_output[i].dtype() != output_dtypes()[output_index]) {
              return errors::InvalidArgument(
                  "Got wrong type for scan_func return value ", i,
                  " (expected ",
                  DataTypeString(dataset()->state_types_[output_index]),
                  ", got ", DataTypeString(state_and_output[i].dtype()), ").");
            }
            if (!output_shapes()[output_index].IsCompatibleWith(
                    state_and_output[i].shape())) {
              return errors::InvalidArgument(
                  "Got wrong shape for scan_func return value ", i,
                  " (expected ", output_shapes()[output_index].DebugString(),
                  ", got ", state_and_output[i].shape().DebugString(), ").");
            }

            out_tensors->push_back(std::move(state_and_output[i]));
          }
        } else if (errors::IsOutOfRange(s)) {
          // `f` may deliberately raise `errors::OutOfRange` to indicate
          // that we should terminate the iteration early.
          *end_of_sequence = true;
          return Status::OK();
        }
        return s;
      }

     private:
      mutex mu_;
      const std::unique_ptr<IteratorBase> input_impl_ GUARDED_BY(mu_);
      std::vector<Tensor> state_ GUARDED_BY(mu_);
    };

    const DatasetBase* const input_;
    const std::vector<Tensor> initial_state_;
    const std::unique_ptr<CapturedFunction> captured_func_;
    const DataTypeVector state_types_;
    const DataTypeVector output_types_;
    const std::vector<PartialTensorShape> output_shapes_;
  };

  const int graph_def_version_;
  DataTypeVector state_types_;
  DataTypeVector output_types_;
  std::vector<PartialTensorShape> output_shapes_;
  NameAttrList func_;
};

REGISTER_KERNEL_BUILDER(Name("ScanDataset").Device(DEVICE_CPU), ScanDatasetOp);

}  // namespace

}  // namespace tensorflow
