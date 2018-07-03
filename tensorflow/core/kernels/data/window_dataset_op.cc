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

#include "tensorflow/core/framework/partial_tensor_shape.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/kernels/data/dataset.h"
#include "tensorflow/core/kernels/data/window_dataset.h"

namespace tensorflow {

namespace {

// See documentation in ../ops/dataset_ops.cc for a high-level
// description of the following op.

class WindowDatasetOp : public UnaryDatasetOpKernel {
 public:
  explicit WindowDatasetOp(OpKernelConstruction* ctx)
      : UnaryDatasetOpKernel(ctx) {}

  void MakeDataset(OpKernelContext* ctx, DatasetBase* input,
                   DatasetBase** output) override {
    int64 window_size = 0;
    OP_REQUIRES_OK(
        ctx, ParseScalarArgument<int64>(ctx, "window_size", &window_size));
    OP_REQUIRES(
        ctx, window_size > 0,
        errors::InvalidArgument("Window size must be greater than zero."));

    *output = new Dataset(ctx, window_size, input);
  }

 private:
  class Dataset : public GraphDatasetBase {
   public:
    Dataset(OpKernelContext* ctx, int64 window_size, const DatasetBase* input)
        : GraphDatasetBase(ctx), window_size_(window_size), input_(input) {
      input_->Ref();
    }

    ~Dataset() override { input_->Unref(); }

    std::unique_ptr<IteratorBase> MakeIteratorInternal(
        const string& prefix) const override {
      return std::unique_ptr<IteratorBase>(new Iterator(
          Iterator::Params{this, strings::StrCat(prefix, "::Window")}));
    }

    const DataTypeVector& output_dtypes() const override {
      static DataTypeVector* output_dtypes = new DataTypeVector({DT_VARIANT});
      return *output_dtypes;
    }

    const std::vector<PartialTensorShape>& output_shapes() const override {
      static std::vector<PartialTensorShape>* output_shapes =
          new std::vector<PartialTensorShape>({TensorShape({})});
      return *output_shapes;
    }

    string DebugString() const override {
      return strings::StrCat("WindowDatasetOp(", window_size_, ")::Dataset");
    }

   protected:
    Status AsGraphDefInternal(OpKernelContext* ctx, DatasetGraphDefBuilder* b,
                              Node** output) const override {
      Node* input_graph_node = nullptr;
      TF_RETURN_IF_ERROR(b->AddParentDataset(ctx, input_, &input_graph_node));
      Node* window_size = nullptr;
      TF_RETURN_IF_ERROR(b->AddScalar(window_size_, &window_size));
      TF_RETURN_IF_ERROR(
          b->AddDataset(this, {input_graph_node, window_size}, output));
      return Status::OK();
    }

   private:
    class Iterator : public DatasetIterator<Dataset> {
     public:
      explicit Iterator(const Params& params)
          : DatasetIterator<Dataset>(params) {}

      Status Initialize(IteratorContext* ctx) override {
        return dataset()->input_->MakeIterator(ctx, prefix(), &input_impl_);
      }

      Status GetNextInternal(IteratorContext* ctx,
                             std::vector<Tensor>* out_tensors,
                             bool* end_of_sequence) override {
        // Each row of `window_elements` is a tuple of tensors from the
        // input iterator.
        std::vector<std::vector<Tensor>> window_elements;
        {
          mutex_lock l(mu_);
          if (!input_impl_) {
            *end_of_sequence = true;
            return Status::OK();
          }
          window_elements.reserve(dataset()->window_size_);
          *end_of_sequence = false;
          for (int i = 0; i < dataset()->window_size_ && !*end_of_sequence;
               ++i) {
            std::vector<Tensor> window_element_tuple;
            TF_RETURN_IF_ERROR(input_impl_->GetNext(ctx, &window_element_tuple,
                                                    end_of_sequence));
            if (!*end_of_sequence) {
              window_elements.emplace_back(std::move(window_element_tuple));
            } else {
              input_impl_.reset();
            }
          }
        }

        if (window_elements.empty()) {
          DCHECK(*end_of_sequence);
          return Status::OK();
        }

        const size_t num_tuple_components = window_elements[0].size();
        const int64 num_window_elements = window_elements.size();
        for (size_t idx = 0; idx < num_tuple_components; ++idx) {
          DatasetBase* window_dataset;
          std::vector<std::vector<Tensor>> window_component_elements;
          window_component_elements.reserve(num_window_elements);
          // Build the output tuple component by copying one slice
          // from each input element in the window.
          for (size_t i = 0; i < num_window_elements; ++i) {
            std::vector<Tensor> component_element;
            component_element.push_back(std::move(window_elements[i][idx]));
            window_component_elements.push_back(component_element);
          }
          DataTypeVector output_types(
              {dataset()->input_->output_dtypes()[idx]});
          std::vector<PartialTensorShape> output_shapes(
              {dataset()->input_->output_shapes()[idx]});
          TF_RETURN_IF_ERROR(NewWindowDataset(window_component_elements,
                                              output_types, output_shapes,
                                              &window_dataset));
          out_tensors->emplace_back(DT_VARIANT, TensorShape({}));
          TF_RETURN_IF_ERROR(StoreDatasetInVariantTensor(window_dataset,
                                                         &out_tensors->back()));
        }
        *end_of_sequence = false;
        return Status::OK();
      }

     protected:
      Status SaveInternal(IteratorStateWriter* writer) override {
        mutex_lock l(mu_);
        if (!input_impl_) {
          TF_RETURN_IF_ERROR(
              writer->WriteScalar(full_name("input_impl_empty"), ""));
        } else {
          TF_RETURN_IF_ERROR(SaveParent(writer, input_impl_));
        }
        return Status::OK();
      }

      Status RestoreInternal(IteratorContext* ctx,
                             IteratorStateReader* reader) override {
        mutex_lock l(mu_);
        if (!reader->Contains(full_name("input_impl_empty"))) {
          TF_RETURN_IF_ERROR(RestoreParent(ctx, reader, input_impl_));
        } else {
          input_impl_.reset();
        }
        return Status::OK();
      }

     private:
      mutex mu_;
      std::unique_ptr<IteratorBase> input_impl_ GUARDED_BY(mu_);
    };

    const int64 window_size_;
    const DatasetBase* const input_;
  };
};

REGISTER_KERNEL_BUILDER(Name("WindowDataset").Device(DEVICE_CPU),
                        WindowDatasetOp);

}  // namespace

}  // namespace tensorflow
