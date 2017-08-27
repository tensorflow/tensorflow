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
    OP_REQUIRES_OK(ctx, ctx->GetAttr("output_types", &output_types_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("output_shapes", &output_shapes_));
  }

  void MakeDataset(OpKernelContext* ctx, DatasetBase* input,
                   DatasetBase** output) override {
    int64 window_size = 0;
    OP_REQUIRES_OK(
        ctx, ParseScalarArgument<int64>(ctx, "window_size", &window_size));
    OP_REQUIRES(
        ctx, window_size > 0,
        errors::InvalidArgument("Window size must be greater than zero."));

    // Get captured inputs for the key and reduce functions.
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

    *output = new Dataset(input, window_size, std::move(captured_key_func),
                          std::move(captured_reduce_func), output_types_,
                          output_shapes_);
  }

 private:
  class Dataset : public DatasetBase {
   public:
    Dataset(const DatasetBase* input, int64 window_size,
            std::unique_ptr<CapturedFunction> captured_key_func,
            std::unique_ptr<CapturedFunction> captured_reduce_func,
            const DataTypeVector& output_types,
            const std::vector<PartialTensorShape>& output_shapes)
        : input_(input),
          window_size_(window_size),
          captured_key_func_(std::move(captured_key_func)),
          captured_reduce_func_(std::move(captured_reduce_func)),
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
                  opts, next_input_element, &key_func_output, prefix()));

              if (key_func_output.size() != 1 ||
                  key_func_output[0].dtype() != DT_INT64 ||
                  key_func_output[0].NumElements() != 1) {
                // TODO(mrry): Support non-int64 keys.
                return errors::InvalidArgument(
                    "`key_func` must return a scalar int64.");
              }
              const int64 key = key_func_output[0].scalar<int64>()();

              std::vector<std::vector<Tensor>>& group = groups_[key];
              group.push_back(std::move(next_input_element));

              if (group.size() == dataset()->window_size_) {
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

        Tensor group_dataset_arg(DT_RESOURCE, TensorShape({}));

        // NOTE(mrry): We cannot use the core `MakeResourceHandle()`,
        // `LookupResource()` or `DeleteResource()` functions, because
        // we have an `IteratorContext*` and not an
        // `OpKernelContext*`, so we replicate the necessary
        // functionality here.
        ResourceHandle group_dataset_handle;
        group_dataset_handle.set_device(
            dataset()->captured_reduce_func_->device()->attributes().name());
        group_dataset_handle.set_container(step_container.name());
        group_dataset_handle.set_name(kWindowResourceName);
        auto type_index = MakeTypeIndex<DatasetBase>();
        group_dataset_handle.set_hash_code(type_index.hash_code());
        group_dataset_handle.set_maybe_type_name(type_index.name());
        // NOTE(mrry): Ownership of `group_dataset` transfers to
        // `step_container` here.
        TF_RETURN_IF_ERROR(dataset()
                               ->captured_reduce_func_->resource_manager()
                               ->Create<DatasetBase>(
                                   group_dataset_handle.container(),
                                   group_dataset_handle.name(), group_dataset));

        group_dataset_arg.scalar<ResourceHandle>()() = group_dataset_handle;

        std::vector<Tensor> args(
            {std::move(key_arg), std::move(group_dataset_arg)});
        std::vector<Tensor> return_values;

        TF_RETURN_IF_ERROR(dataset()->captured_reduce_func_->Run(
            opts, args, &return_values, prefix()));

        if (!(return_values.size() == 1 &&
              return_values[0].dtype() == DT_RESOURCE &&
              TensorShapeUtils::IsScalar(return_values[0].shape()))) {
          return errors::InvalidArgument(
              "`reduce_func` must return a single scalar of dtype "
              "DT_RESOURCE.");
        }

        // Retrieve the dataset that was created in `f`.
        DatasetBase* returned_dataset;
        const ResourceHandle& dataset_resource =
            return_values[0].scalar<ResourceHandle>()();
        if (type_index.hash_code() != dataset_resource.hash_code()) {
          return errors::InvalidArgument(
              "`reduce_func` must return a Dataset resource.");
        }
        TF_RETURN_IF_ERROR(
            dataset()->captured_reduce_func_->resource_manager()->Lookup(
                dataset_resource.container(), dataset_resource.name(),
                &returned_dataset));
        core::ScopedUnref unref_returned_dataset(returned_dataset);

        // Create an iterator for the dataset that was returned by
        // `f`. This transfers ownership of the dataset to the
        // iterator.
        current_group_iterator_ = returned_dataset->MakeIterator(prefix());
        return Status::OK();
      }

      const std::unique_ptr<IteratorBase> input_impl_;
      mutex mu_;
      // TODO(mrry): Optimize for dense key space if appropriate.
      bool end_of_input_ GUARDED_BY(mu_) = false;
      std::map<int64, std::vector<std::vector<Tensor>>> groups_ GUARDED_BY(mu_);
      std::unique_ptr<IteratorBase> current_group_iterator_ GUARDED_BY(mu_);
    };

    // A resource name for the temporary window dataset that is
    // created as the input to the reduce function.
    static constexpr const char* kWindowResourceName = "__window_dataset";

    const DatasetBase* const input_;
    const int64 window_size_;
    const std::unique_ptr<CapturedFunction> captured_key_func_;
    const std::unique_ptr<CapturedFunction> captured_reduce_func_;
    const DataTypeVector output_types_;
    const std::vector<PartialTensorShape> output_shapes_;
  };

  const int graph_def_version_;
  DataTypeVector output_types_;
  std::vector<PartialTensorShape> output_shapes_;
  const NameAttrList* key_func_;
  const NameAttrList* reduce_func_;
};

REGISTER_KERNEL_BUILDER(Name("GroupByWindowDataset").Device(DEVICE_CPU),
                        GroupByWindowDatasetOp);

}  // namespace

}  // namespace tensorflow
