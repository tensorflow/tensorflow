
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
#include "tensorflow/core/lib/random/random.h"

#include "tensorflow/core/kernels/captured_function.h"

namespace tensorflow {

namespace {

// See documentation in ../ops/dataset_ops.cc for a high-level
// description of the following op.

class InterleaveDatasetOp : public OpKernel {
 public:
  explicit InterleaveDatasetOp(OpKernelConstruction* ctx)
      : OpKernel(ctx), graph_def_version_(ctx->graph_def_version()) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("f", &func_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("output_types", &output_types_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("output_shapes", &output_shapes_));
  }

  void Compute(OpKernelContext* ctx) override {
    DatasetBase* input;
    OP_REQUIRES_OK(ctx, LookupResource(ctx, HandleFromInput(ctx, 0), &input));
    core::ScopedUnref unref_input(input);

    OpInputList inputs;
    OP_REQUIRES_OK(ctx, ctx->input_list("other_arguments", &inputs));
    std::vector<Tensor> other_arguments;
    other_arguments.reserve(inputs.size());
    for (const Tensor& t : inputs) {
      other_arguments.push_back(t);
    }

    const Tensor* cycle_length_t;
    OP_REQUIRES_OK(ctx, ctx->input("cycle_length", &cycle_length_t));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(cycle_length_t->shape()),
                errors::InvalidArgument("cycle_length must be a scalar."));
    const int64 cycle_length = cycle_length_t->flat<int64>()(0);
    OP_REQUIRES(
        ctx, cycle_length > 0,
        errors::InvalidArgument("cycle_length must be greater than zero."));

    const Tensor* block_length_t;
    OP_REQUIRES_OK(ctx, ctx->input("block_length", &block_length_t));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(block_length_t->shape()),
                errors::InvalidArgument("block_length must be a scalar."));
    const int64 block_length = block_length_t->flat<int64>()(0);
    OP_REQUIRES(
        ctx, block_length > 0,
        errors::InvalidArgument("block_length must be greater than zero."));

    std::unique_ptr<CapturedFunction> captured_func;
    OP_REQUIRES_OK(ctx, CapturedFunction::Create(ctx, func_, graph_def_version_,
                                                 std::move(other_arguments),
                                                 &captured_func));

    DatasetBase* dataset =
        new Dataset(input, std::move(captured_func), cycle_length, block_length,
                    output_types_, output_shapes_);

    Tensor* output = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape({}), &output));
    ResourceHandle handle = MakeResourceHandle<DatasetBase>(
        ctx, ctx->step_container()->name(), name());
    OP_REQUIRES_OK(ctx, CreateResource(ctx, handle, dataset));
    output->flat<ResourceHandle>()(0) = handle;
  }

 private:
  class Dataset : public DatasetBase {
   public:
    Dataset(const DatasetBase* input,
            std::unique_ptr<CapturedFunction> captured_func, int64 cycle_length,
            int64 block_length, const DataTypeVector& output_types,
            const std::vector<PartialTensorShape>& output_shapes)
        : input_(input),
          captured_func_(std::move(captured_func)),
          cycle_length_(cycle_length),
          block_length_(block_length),
          output_types_(output_types),
          output_shapes_(output_shapes) {
      input_->Ref();
    }

    ~Dataset() override { input_->Unref(); }

    std::unique_ptr<IteratorBase> MakeIterator(
        const string& prefix) const override {
      return std::unique_ptr<IteratorBase>(
          new Iterator({this, strings::StrCat(prefix, "::Interleave")}));
    }

    const DataTypeVector& output_dtypes() const override {
      return output_types_;
    }
    const std::vector<PartialTensorShape>& output_shapes() const override {
      return output_shapes_;
    }

    string DebugString() override { return "InterleaveDatasetOp::Dataset"; }

   private:
    class Iterator : public DatasetIterator<Dataset> {
     public:
      explicit Iterator(const Params& params)
          : DatasetIterator<Dataset>(params),
            input_impl_(params.dataset->input_->MakeIterator(params.prefix)),
            current_elements_(params.dataset->cycle_length_) {}

      void AdvanceToNextInCycle() EXCLUSIVE_LOCKS_REQUIRED(mu_) {
        block_index_ = 0;
        cycle_index_ = (cycle_index_ + 1) % dataset()->cycle_length_;
      }

      void AdvancePosition() EXCLUSIVE_LOCKS_REQUIRED(mu_) {
        ++block_index_;
        if (block_index_ == dataset()->block_length_) {
          AdvanceToNextInCycle();
        }
      }

      Status GetNextInternal(IteratorContext* ctx,
                             std::vector<Tensor>* out_tensors,
                             bool* end_of_sequence) override {
        mutex_lock l(mu_);
        while (!end_of_input_ || num_open_ > 0) {
          if (current_elements_[cycle_index_]) {
            // We are currently processing a mapped element, so try to get the
            // next subelement.
            bool end_of_element;
            TF_RETURN_IF_ERROR(current_elements_[cycle_index_]->GetNext(
                ctx, out_tensors, &end_of_element));
            if (!end_of_element) {
              // Produce the subelement as output.
              AdvancePosition();
              *end_of_sequence = false;
              return Status::OK();
            }
            // We have reached the end of the current element, so move
            // on to the next element in the cycle.
            current_elements_[cycle_index_].reset();
            --num_open_;
            AdvanceToNextInCycle();
          } else if (!end_of_input_) {
            // Get the next element from the input dataset, and create
            // an iterator from it.
            std::vector<Tensor> args;
            TF_RETURN_IF_ERROR(
                input_impl_->GetNext(ctx, &args, &end_of_input_));
            if (!end_of_input_) {
              TF_RETURN_IF_ERROR(MakeIteratorFromInputElement(
                  ctx, args, &current_elements_[cycle_index_]));
              ++num_open_;
            }
          } else {
            AdvanceToNextInCycle();
          }
        }

        *end_of_sequence = true;
        return Status::OK();
      }

     private:
      Status MakeIteratorFromInputElement(
          IteratorContext* ctx, const std::vector<Tensor>& input_element,
          std::unique_ptr<IteratorBase>* out_iterator)
          EXCLUSIVE_LOCKS_REQUIRED(mu_) {
        FunctionLibraryRuntime::Options opts;
        opts.runner = ctx->runner();
        opts.step_id = CapturedFunction::generate_step_id();
        ScopedStepContainer step_container(
            opts.step_id, [this, ctx](const string& name) {
              dataset()
                  ->captured_func_->resource_manager()
                  ->Cleanup(name)
                  .IgnoreError();
            });
        opts.step_container = &step_container;
        std::vector<Tensor> return_values;
        TF_RETURN_IF_ERROR(dataset()->captured_func_->Run(
            opts, input_element, &return_values, prefix()));

        if (!(return_values.size() == 1 &&
              return_values[0].dtype() == DT_RESOURCE &&
              TensorShapeUtils::IsScalar(return_values[0].shape()))) {
          return errors::InvalidArgument(
              "`f` must return a single scalar of dtype DT_RESOURCE.");
        }

        // Retrieve the dataset that was created in `f`.
        DatasetBase* returned_dataset;
        const ResourceHandle& dataset_resource =
            return_values[0].scalar<ResourceHandle>()();

        // NOTE(mrry): We cannot use the core `LookupResource()` or
        // `DeleteResource()` functions, because we have an
        // `IteratorContext*` and not an `OpKernelContext*`, so we
        // replicate the necessary functionality here.
        auto type_index = MakeTypeIndex<DatasetBase>();
        if (type_index.hash_code() != dataset_resource.hash_code()) {
          return errors::InvalidArgument("`f` must return a Dataset resource.");
        }
        TF_RETURN_IF_ERROR(
            dataset()->captured_func_->resource_manager()->Lookup(
                dataset_resource.container(), dataset_resource.name(),
                &returned_dataset));
        core::ScopedUnref unref_dataset(returned_dataset);

        // Create an iterator for the dataset that was returned by
        // `f`. This transfers ownership of the dataset to the
        // iterator, so we can delete it from the resource manager.
        *out_iterator = returned_dataset->MakeIterator(
            strings::StrCat(prefix(), "[", cycle_index_, "]"));
        TF_RETURN_IF_ERROR(
            dataset()->captured_func_->resource_manager()->Delete<DatasetBase>(
                dataset_resource.container(), dataset_resource.name()));
        return Status::OK();
      }

      mutex mu_;
      const std::unique_ptr<IteratorBase> input_impl_ GUARDED_BY(mu_);
      std::vector<std::unique_ptr<IteratorBase>> current_elements_
          GUARDED_BY(mu_);
      size_t cycle_index_ GUARDED_BY(mu_) = 0;
      int64 block_index_ GUARDED_BY(mu_) = 0;
      bool end_of_input_ GUARDED_BY(mu_) = false;
      size_t num_open_ GUARDED_BY(mu_) = 0;
    };

    const DatasetBase* const input_;
    const std::unique_ptr<CapturedFunction> captured_func_;
    const int64 cycle_length_;
    const int64 block_length_;
    const DataTypeVector output_types_;
    const std::vector<PartialTensorShape> output_shapes_;
  };

  const int graph_def_version_;
  DataTypeVector output_types_;
  std::vector<PartialTensorShape> output_shapes_;
  const NameAttrList* func_;
};

REGISTER_KERNEL_BUILDER(Name("InterleaveDataset").Device(DEVICE_CPU),
                        InterleaveDatasetOp);

}  // namespace

}  // namespace tensorflow
