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
#include <map>

#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/framework/partial_tensor_shape.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/random/random.h"

#include "tensorflow/core/kernels/captured_function.h"
#include "tensorflow/core/kernels/dataset.h"
#include "tensorflow/core/kernels/window_dataset.h"

namespace tensorflow {

namespace {

// See documentation in ../ops/dataset_ops.cc for a high-level
// description of the following op.
class GroupByWindowDatasetOp : public UnaryDatasetOpKernel {
 public:
  explicit GroupByWindowDatasetOp(OpKernelConstruction* ctx)
      : UnaryDatasetOpKernel(ctx),
        graph_def_version_(ctx->graph_def_version()) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("key_func", &key_func_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("reduce_func", &reduce_func_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("window_size_func", &window_size_func_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("output_types", &output_types_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("output_shapes", &output_shapes_));
  }

  void MakeDataset(OpKernelContext* ctx, DatasetBase* input,
                   DatasetBase** output) override {
    // Get captured inputs for the key, reduce, and window_size functions.
    OpInputList key_func_other_argument_inputs;
    OP_REQUIRES_OK(ctx, ctx->input_list("key_func_other_arguments",
                                        &key_func_other_argument_inputs));
    std::vector<Tensor> key_func_other_arguments;
    key_func_other_arguments.reserve(key_func_other_argument_inputs.size());
    for (const Tensor& t : key_func_other_argument_inputs) {
      key_func_other_arguments.push_back(t);
    }
    OpInputList reduce_func_other_argument_inputs;
    OP_REQUIRES_OK(ctx, ctx->input_list("reduce_func_other_arguments",
                                        &reduce_func_other_argument_inputs));
    std::vector<Tensor> reduce_func_other_arguments;
    reduce_func_other_arguments.reserve(
        reduce_func_other_argument_inputs.size());
    for (const Tensor& t : reduce_func_other_argument_inputs) {
      reduce_func_other_arguments.push_back(t);
    }
    OpInputList window_size_func_other_argument_inputs;
    OP_REQUIRES_OK(ctx,
                   ctx->input_list("window_size_func_other_arguments",
                                   &window_size_func_other_argument_inputs));
    std::vector<Tensor> window_size_func_other_arguments;
    window_size_func_other_arguments.reserve(
        window_size_func_other_argument_inputs.size());
    for (const Tensor& t : window_size_func_other_argument_inputs) {
      window_size_func_other_arguments.push_back(t);
    }
    // TODO(mrry): Refactor CapturedFunction to share the runtime
    // state between multiple functions?
    std::unique_ptr<CapturedFunction> captured_key_func;
    OP_REQUIRES_OK(ctx,
                   CapturedFunction::Create(ctx, key_func_, graph_def_version_,
                                            std::move(key_func_other_arguments),
                                            &captured_key_func));
    std::unique_ptr<CapturedFunction> captured_reduce_func;
    OP_REQUIRES_OK(
        ctx, CapturedFunction::Create(ctx, reduce_func_, graph_def_version_,
                                      std::move(reduce_func_other_arguments),
                                      &captured_reduce_func));
    std::unique_ptr<CapturedFunction> captured_window_size_func;
    OP_REQUIRES_OK(ctx, CapturedFunction::Create(
                            ctx, window_size_func_, graph_def_version_,
                            std::move(window_size_func_other_arguments),
                            &captured_window_size_func));

    *output = new Dataset(
        input, std::move(captured_key_func), std::move(captured_reduce_func),
        std::move(captured_window_size_func), output_types_, output_shapes_);
  }

 private:
  class Dataset : public DatasetBase {
   public:
    Dataset(const DatasetBase* input,
            std::unique_ptr<CapturedFunction> captured_key_func,
            std::unique_ptr<CapturedFunction> captured_reduce_func,
            std::unique_ptr<CapturedFunction> captured_window_size_func,
            const DataTypeVector& output_types,
            const std::vector<PartialTensorShape>& output_shapes)
        : input_(input),
          captured_key_func_(std::move(captured_key_func)),
          captured_reduce_func_(std::move(captured_reduce_func)),
          captured_window_size_func_(std::move(captured_window_size_func)),
          output_types_(output_types),
          output_shapes_(output_shapes) {
      input_->Ref();
    }

    ~Dataset() override { input_->Unref(); }

    std::unique_ptr<IteratorBase> MakeIterator(
        const string& prefix) const override {
      return std::unique_ptr<IteratorBase>(
          new Iterator({this, strings::StrCat(prefix, "::GroupByWindow")}));
    }

    const DataTypeVector& output_dtypes() const override {
      return output_types_;
    }
    const std::vector<PartialTensorShape>& output_shapes() const override {
      return output_shapes_;
    }

    string DebugString() override { return "GroupByWindowDatasetOp::Dataset"; }

   private:
    class Iterator : public DatasetIterator<Dataset> {
     public:
      explicit Iterator(const Params& params)
          : DatasetIterator<Dataset>(params),
            input_impl_(params.dataset->input_->MakeIterator(params.prefix)) {}

      Status GetNextInternal(IteratorContext* ctx,
                             std::vector<Tensor>* out_tensors,
                             bool* end_of_sequence) override {
        mutex_lock l(mu_);
        do {
          if (current_group_iterator_) {
            // We are currently processing a group, so try to get the
            // next element.
            bool end_of_group;
            TF_RETURN_IF_ERROR(current_group_iterator_->GetNext(
                ctx, out_tensors, &end_of_group));
            if (!end_of_group) {
              // Produce the subelement as output.
              *end_of_sequence = false;
              return Status::OK();
            }
            // We have reached the end of the current group, so maybe move on
            // to the next group.
            current_group_iterator_.reset();
          }

          // Iterate through the input dataset until we get a full
          // group, or reach the end.
          while (!end_of_input_) {
            std::vector<Tensor> next_input_element;
            TF_RETURN_IF_ERROR(
                input_impl_->GetNext(ctx, &next_input_element, &end_of_input_));

            if (!end_of_input_) {
              FunctionLibraryRuntime::Options opts;
              opts.step_id = CapturedFunction::generate_step_id();
              opts.runner = ctx->runner();
              ScopedStepContainer step_container(
                  opts.step_id, [this, ctx](const string& name) {
                    dataset()
                        ->captured_key_func_->resource_manager()
                        ->Cleanup(name)
                        .IgnoreError();
                  });
              opts.step_container = &step_container;

              // Run the key function on the input element to identify its
              // group.
              std::vector<Tensor> key_func_output;
              TF_RETURN_IF_ERROR(dataset()->captured_key_func_->Run(
                  opts, next_input_element, &key_func_output));

              if (key_func_output.size() != 1 ||
                  key_func_output[0].dtype() != DT_INT64 ||
                  key_func_output[0].NumElements() != 1) {
                // TODO(mrry): Support non-int64 keys.
                return errors::InvalidArgument(
                    "`key_func` must return a scalar int64.");
              }
              const int64 key = key_func_output[0].scalar<int64>()();

              if (window_sizes_.find(key) == window_sizes_.end()) {
                // Run window_size function
                FunctionLibraryRuntime::Options opts2;
                opts2.step_id = CapturedFunction::generate_step_id();
                opts2.runner = ctx->runner();
                ScopedStepContainer step_container2(
                    opts2.step_id, [this, ctx](const string& name) {
                      dataset()
                          ->captured_window_size_func_->resource_manager()
                          ->Cleanup(name)
                          .IgnoreError();
                    });
                opts2.step_container = &step_container2;

                // Run the window size function on the key to identify its
                // window size.
                std::vector<Tensor> window_size_func_output;
                TF_RETURN_IF_ERROR(dataset()->captured_window_size_func_->Run(
                    opts2, key_func_output, &window_size_func_output));

                if (window_size_func_output.size() != 1 ||
                    window_size_func_output[0].dtype() != DT_INT64 ||
                    window_size_func_output[0].NumElements() != 1) {
                  // TODO(mrry): Support non-int64 window sizes.
                  return errors::InvalidArgument(
                      "`window_size_func` must return a scalar int64.");
                }
                const int64 window_size =
                    window_size_func_output[0].scalar<int64>()();
                window_sizes_[key] = window_size;
              }

              const int64 window_size = window_sizes_[key];

              std::vector<std::vector<Tensor>>& group = groups_[key];
              group.push_back(std::move(next_input_element));

              if (group.size() == window_size) {
                TF_RETURN_IF_ERROR(StartFlushingGroup(ctx, key));
                break;
              }
            }
          }

          if (end_of_input_) {
            if (!groups_.empty()) {
              // We have consumed all of the input, so flush an
              // arbitrarily chosen group.
              TF_RETURN_IF_ERROR(
                  StartFlushingGroup(ctx, groups_.begin()->first));
            }
          }
        } while (current_group_iterator_ || !end_of_input_);

        *end_of_sequence = true;
        return Status::OK();
      }

     private:
      Status StartFlushingGroup(IteratorContext* ctx, int64 key)
          EXCLUSIVE_LOCKS_REQUIRED(mu_) {
        FunctionLibraryRuntime::Options opts;
        opts.step_id = CapturedFunction::generate_step_id();
        opts.runner = ctx->runner();
        ScopedStepContainer step_container(
            opts.step_id, [this, ctx](const string& name) {
              dataset()
                  ->captured_reduce_func_->resource_manager()
                  ->Cleanup(name)
                  .IgnoreError();
            });
        opts.step_container = &step_container;

        DatasetBase* group_dataset;
        TF_RETURN_IF_ERROR(NewWindowDataset(
            std::move(groups_[key]), dataset()->input_->output_dtypes(),
            dataset()->input_->output_shapes(), &group_dataset));
        groups_.erase(key);

        Tensor key_arg(DT_INT64, TensorShape({}));
        key_arg.scalar<int64>()() = key;

        Tensor group_dataset_arg(DT_VARIANT, TensorShape({}));
        TF_RETURN_IF_ERROR(
            StoreDatasetInVariantTensor(group_dataset, &group_dataset_arg));

        std::vector<Tensor> args(
            {std::move(key_arg), std::move(group_dataset_arg)});
        std::vector<Tensor> return_values;

        TF_RETURN_IF_ERROR(
            dataset()->captured_reduce_func_->Run(opts, args, &return_values));

        if (!(return_values.size() == 1 &&
              return_values[0].dtype() == DT_VARIANT &&
              TensorShapeUtils::IsScalar(return_values[0].shape()))) {
          return errors::InvalidArgument(
              "`reduce_func` must return a single scalar of dtype "
              "DT_VARIANT.");
        }

        // Retrieve the dataset that was created in `f`.
        // `returned_dataset` is borrowed from the `return_values[0]`.
        DatasetBase* returned_dataset;
        TF_RETURN_IF_ERROR(
            GetDatasetFromVariantTensor(return_values[0], &returned_dataset));

        // Create an iterator for the dataset that was returned by `f`.
        current_group_iterator_ = returned_dataset->MakeIterator(prefix());
        return Status::OK();
      }

      const std::unique_ptr<IteratorBase> input_impl_;
      mutex mu_;
      // TODO(mrry): Optimize for dense key space if appropriate.
      bool end_of_input_ GUARDED_BY(mu_) = false;
      std::map<int64, std::vector<std::vector<Tensor>>> groups_ GUARDED_BY(mu_);
      std::unique_ptr<IteratorBase> current_group_iterator_ GUARDED_BY(mu_);
      std::map<int64, int64> window_sizes_ GUARDED_BY(mu_);
    };

    // A resource name for the temporary window dataset that is
    // created as the input to the reduce function.
    static constexpr const char* kWindowResourceName = "__window_dataset";

    const DatasetBase* const input_;
    const std::unique_ptr<CapturedFunction> captured_key_func_;
    const std::unique_ptr<CapturedFunction> captured_reduce_func_;
    const std::unique_ptr<CapturedFunction> captured_window_size_func_;
    const DataTypeVector output_types_;
    const std::vector<PartialTensorShape> output_shapes_;
  };

  const int graph_def_version_;
  DataTypeVector output_types_;
  std::vector<PartialTensorShape> output_shapes_;
  NameAttrList key_func_;
  NameAttrList reduce_func_;
  NameAttrList window_size_func_;
};

REGISTER_KERNEL_BUILDER(Name("GroupByWindowDataset").Device(DEVICE_CPU),
                        GroupByWindowDatasetOp);

}  // namespace

}  // namespace tensorflow
