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

class FlatMapDatasetOp : public UnaryDatasetOpKernel {
 public:
  explicit FlatMapDatasetOp(OpKernelConstruction* ctx)
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

    std::unique_ptr<CapturedFunction> captured_func;
    OP_REQUIRES_OK(ctx, CapturedFunction::Create(ctx, func_, graph_def_version_,
                                                 std::move(other_arguments),
                                                 &captured_func));

    *output = new Dataset(input, std::move(captured_func), output_types_,
                          output_shapes_);
  }

 private:
  class Dataset : public DatasetBase {
   public:
    Dataset(const DatasetBase* input,
            std::unique_ptr<CapturedFunction> captured_func,
            const DataTypeVector& output_types,
            const std::vector<PartialTensorShape>& output_shapes)
        : input_(input),
          captured_func_(std::move(captured_func)),
          output_types_(output_types),
          output_shapes_(output_shapes) {
      input_->Ref();
    }

    ~Dataset() override { input_->Unref(); }

    std::unique_ptr<IteratorBase> MakeIterator() const override {
      return std::unique_ptr<IteratorBase>(new Iterator(this));
    }

    const DataTypeVector& output_dtypes() const override {
      return output_types_;
    }
    const std::vector<PartialTensorShape>& output_shapes() const override {
      return output_shapes_;
    }

    string DebugString() override { return "FlatMapDatasetOp::Dataset"; }

   private:
    class Iterator : public DatasetIterator<Dataset> {
     public:
      explicit Iterator(const Dataset* dataset)
          : DatasetIterator<Dataset>(dataset),
            input_impl_(dataset->input_->MakeIterator()) {}

      Status GetNext(IteratorContext* ctx, std::vector<Tensor>* out_tensors,
                     bool* end_of_sequence) override {
        mutex_lock l(mu_);
        do {
          if (current_element_iterator_) {
            // We are currently precessing a mapped element, so try to get the
            // next subelement.
            bool end_of_element;
            TF_RETURN_IF_ERROR(current_element_iterator_->GetNext(
                ctx, out_tensors, &end_of_element));
            if (!end_of_element) {
              // Produce the subelement as output.
              *end_of_sequence = false;
              return Status::OK();
            }

            // We have reached the end of the current element, so maybe move on
            // to the next element.
            current_element_iterator_.reset();
          }

          // Get the next element from the input dataset.
          std::vector<Tensor> args;
          TF_RETURN_IF_ERROR(input_impl_->GetNext(ctx, &args, end_of_sequence));
          if (*end_of_sequence) {
            return Status::OK();
          }

          FunctionLibraryRuntime::Options opts;
          opts.runner = ctx->runner();
          // Choose a step ID that is guaranteed not to clash with any
          // Session-generated step ID. DirectSession only generates
          // non-negative step IDs (contiguous, starting from 0), and
          // MasterSession generates 56-bit random step IDs whose MSB
          // is always 0, so a negative random step ID should suffice.
          opts.step_id = -std::abs(static_cast<int64>(random::New64()));
          ScopedStepContainer step_container(
              opts.step_id, [this, ctx](const string& name) {
                dataset()
                    ->captured_func_->resource_manager()
                    ->Cleanup(name)
                    .IgnoreError();
              });
          opts.step_container = &step_container;
          std::vector<Tensor> return_values;
          TF_RETURN_IF_ERROR(
              dataset()->captured_func_->Run(opts, args, &return_values));

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
            return errors::InvalidArgument(
                "`f` must return a Dataset resource.");
          }
          TF_RETURN_IF_ERROR(
              dataset()->captured_func_->resource_manager()->Lookup(
                  dataset_resource.container(), dataset_resource.name(),
                  &returned_dataset));
          core::ScopedUnref unref_dataset(returned_dataset);

          // Create an iterator for the dataset that was returned by
          // `f`. This transfers ownership of the dataset to the
          // iterator, so we can delete it from the resource manager.
          current_element_iterator_ = returned_dataset->MakeIterator();
          TF_RETURN_IF_ERROR(
              dataset()
                  ->captured_func_->resource_manager()
                  ->Delete<DatasetBase>(dataset_resource.container(),
                                        dataset_resource.name()));
        } while (true);
      }

     private:
      mutex mu_;
      const std::unique_ptr<IteratorBase> input_impl_ GUARDED_BY(mu_);
      std::unique_ptr<IteratorBase> current_element_iterator_ GUARDED_BY(mu_);
    };

    const DatasetBase* const input_;
    const std::unique_ptr<CapturedFunction> captured_func_;
    const DataTypeVector output_types_;
    const std::vector<PartialTensorShape> output_shapes_;
  };

  const int graph_def_version_;
  DataTypeVector output_types_;
  std::vector<PartialTensorShape> output_shapes_;
  const NameAttrList* func_;
};

REGISTER_KERNEL_BUILDER(Name("FlatMapDataset").Device(DEVICE_CPU),
                        FlatMapDatasetOp);

}  // namespace

}  // namespace tensorflow
