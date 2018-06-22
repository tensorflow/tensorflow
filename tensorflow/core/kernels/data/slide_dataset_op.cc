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
#include "tensorflow/core/util/batch_util.h"

namespace tensorflow {

namespace {

// See documentation in ../ops/dataset_ops.cc for a high-level
// description of the following op.

class SlideDatasetOp : public UnaryDatasetOpKernel {
 public:
  explicit SlideDatasetOp(OpKernelConstruction* ctx)
      : UnaryDatasetOpKernel(ctx) {}

  void MakeDataset(OpKernelContext* ctx, DatasetBase* input,
                   DatasetBase** output) override {
    int64 window_size = 0;
    int64 stride = 0;
    OP_REQUIRES_OK(
        ctx, ParseScalarArgument<int64>(ctx, "window_size", &window_size));
    OP_REQUIRES_OK(ctx, ParseScalarArgument<int64>(ctx, "stride", &stride));
    OP_REQUIRES(
        ctx, window_size > 0,
        errors::InvalidArgument("Window size must be greater than zero."));
    OP_REQUIRES(
        ctx, stride > 0,
        errors::InvalidArgument("Stride must be greater than zero."));

    *output = new Dataset(ctx, window_size, stride, input);
  }

 private:
  class Dataset : public GraphDatasetBase {
   public:
    Dataset(OpKernelContext* ctx, int64 window_size, int64 stride,
            const DatasetBase* input)
        : GraphDatasetBase(ctx),
          window_size_(window_size),
          stride_(stride),
          input_(input) {
      input_->Ref();

      const auto& input_shapes = input_->output_shapes();
      output_shapes_.reserve(input_shapes.size());
      for (const auto& input_shape : input_shapes) {
        output_shapes_.emplace_back(
            PartialTensorShape({-1}).Concatenate(input_shape));
      }
    }

    ~Dataset() override { input_->Unref(); }

    std::unique_ptr<IteratorBase> MakeIteratorInternal(
        const string& prefix) const override {
      return std::unique_ptr<IteratorBase>(new Iterator(
          Iterator::Params{this, strings::StrCat(prefix, "::Slide")}));
    }

    const DataTypeVector& output_dtypes() const override {
      return input_->output_dtypes();
    }

    const std::vector<PartialTensorShape>& output_shapes() const override {
      return output_shapes_;
    }

    string DebugString() const override {
      return strings::StrCat("SlideDatasetOp(", window_size_, ", ", stride_,
                             ")::Dataset");
    }

   protected:
    Status AsGraphDefInternal(OpKernelContext* ctx, DatasetGraphDefBuilder* b,
                              Node** output) const override {
      Node* input_graph_node = nullptr;
      TF_RETURN_IF_ERROR(b->AddParentDataset(ctx, input_, &input_graph_node));
      Node* window_size = nullptr;
      Node* stride = nullptr;
      TF_RETURN_IF_ERROR(b->AddScalar(window_size_, &window_size));
      TF_RETURN_IF_ERROR(b->AddScalar(stride_, &stride));
      TF_RETURN_IF_ERROR(
          b->AddDataset(this, {input_graph_node, window_size, stride}, output));
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
        const int64 window_size = dataset()->window_size_;
        const int64 stride = dataset()->stride_;
        std::vector<std::vector<Tensor>> batch_elements;
        {
          mutex_lock l(mu_);
          if (!input_impl_) {
            *end_of_sequence = true;
            return Status::OK();
          }
          batch_elements.reserve(window_size);
          // Use cache if stride < window_size.
          if (stride < window_size) {
            const bool first_call = cache_.empty();
            if (first_call) {
              cache_.reserve(window_size);
            } else {
              // Reuse cache in the previous iteration.
              cache_.swap(batch_elements);
            }
          }
          // Fill up with new elements.
          *end_of_sequence = false;
          for (size_t i = batch_elements.size(); i < window_size && !*end_of_sequence;
              ++i) {
            std::vector<Tensor> batch_element_tuple;
            TF_RETURN_IF_ERROR(input_impl_->GetNext(ctx, &batch_element_tuple,
                                                    end_of_sequence));
            if (!*end_of_sequence) {
              batch_elements.push_back(std::move(batch_element_tuple));
            } else {
              input_impl_.reset();
            }
          }
          // Drop the final smaller blocks.
          if (batch_elements.size() < window_size) {
            DCHECK(*end_of_sequence);
            return Status::OK();
          }

          if (stride < window_size) {
            // Cache the data used for the next iteration.
            for (size_t i = stride; i < window_size; ++i) {
              cache_.emplace_back(batch_elements[i]);
            }
          } else if (stride > window_size) {
            // Drop the data before the next iteration.
            std::vector<Tensor> batch_element_tuple;
            for (size_t i = window_size; i < stride && !*end_of_sequence; ++i) {
              TF_RETURN_IF_ERROR(input_impl_->GetNext(ctx, &batch_element_tuple,
                                                      end_of_sequence));
              if (*end_of_sequence) {
                input_impl_.reset();
              }
            }
          }
        }

        // Construct output tensors.
        // Those codes below are copied from batch_dataset_op.cc.
        const size_t num_tuple_components = batch_elements[0].size();
        const int64 num_batch_elements = batch_elements.size();
        for (size_t component_index = 0; component_index < num_tuple_components;
             ++component_index) {
          const Tensor& first_element = batch_elements[0][component_index];
          TensorShape batch_component_shape({num_batch_elements});
          batch_component_shape.AppendShape(first_element.shape());
          Tensor batch_component(cpu_allocator(), first_element.dtype(),
                                 batch_component_shape);
          // Build the output tuple component by copying one slice
          // from each input element in the batch.
          for (size_t i = 0; i < num_batch_elements; ++i) {
            if (batch_elements[i][component_index].shape() !=
                first_element.shape()) {
              return errors::InvalidArgument(
                  "Cannot batch tensors with different shapes in component ",
                  component_index, ". First element had shape ",
                  first_element.shape().DebugString(), " and element ", i,
                  " had shape ",
                  batch_elements[i][component_index].shape().DebugString(),
                  ".");
            }
            TF_RETURN_IF_ERROR(batch_util::CopyElementToSlice(
                std::move(batch_elements[i][component_index]), &batch_component,
                i));
          }
          out_tensors->emplace_back(std::move(batch_component));
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
        // Save cache.
        TF_RETURN_IF_ERROR(
            writer->WriteScalar(strings::StrCat("cache_size"), cache_.size()));
        for (int64 i = 0; i < cache_.size(); i++) {
          TF_RETURN_IF_ERROR(writer->WriteScalar(
              strings::StrCat("cache[", i, "]_size"), cache_[i].size()));
          for (int64 j = 0; j < cache_[i].size(); j++) {
            TF_RETURN_IF_ERROR(writer->WriteTensor(
                strings::StrCat("cache[", i, "][", j, "]"), cache_[i][j]));
          }
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
        // Restore cache.
        int64 cache_size;
        TF_RETURN_IF_ERROR(
            reader->ReadScalar(strings::StrCat("cache_size"), &cache_size));
        cache_.resize(cache_size);
        for (int64 i = 0; i < cache_size; i++) {
          int64 vector_size;
          TF_RETURN_IF_ERROR(reader->ReadScalar(
              strings::StrCat("cache[", i, "]_size"), &vector_size));
          cache_[i].resize(vector_size);
          for (int64 j = 0; j < vector_size; j++) {
            TF_RETURN_IF_ERROR(reader->ReadTensor(
                strings::StrCat("cache[", i, "][", j, "]"), &cache_[i][j]));
          }
        }
        return Status::OK();
      }

     private:
      mutex mu_;
      std::vector<std::vector<Tensor>> cache_ GUARDED_BY(mu_);
      std::unique_ptr<IteratorBase> input_impl_ GUARDED_BY(mu_);
    };

    const int64 window_size_;
    const int64 stride_;
    const DatasetBase* const input_;
    std::vector<PartialTensorShape> output_shapes_;
  };
};

REGISTER_KERNEL_BUILDER(Name("SlideDataset").Device(DEVICE_CPU),
                        SlideDatasetOp);

}  // namespace

}  // namespace tensorflow
