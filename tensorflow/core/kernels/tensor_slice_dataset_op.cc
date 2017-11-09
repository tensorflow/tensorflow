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

#include "tensorflow/core/framework/partial_tensor_shape.h"
#include "tensorflow/core/framework/tensor.h"

namespace tensorflow {

namespace {

// See documentation in ../ops/dataset_ops.cc for a high-level
// description of the following op.

class TensorSliceDatasetOp : public DatasetOpKernel {
 public:
  explicit TensorSliceDatasetOp(OpKernelConstruction* ctx)
      : DatasetOpKernel(ctx) {}

  void MakeDataset(OpKernelContext* ctx, DatasetBase** output) override {
    // Create a new TensorDatasetOp::Dataset, insert it in the step
    // container, and return it as the output.
    OpInputList inputs;
    OP_REQUIRES_OK(ctx, ctx->input_list("components", &inputs));
    std::vector<Tensor> components;
    components.reserve(inputs.size());
    OP_REQUIRES(ctx, inputs[0].dims() > 0,
                errors::InvalidArgument(
                    "All components must be at least 1-dimensional"));
    const int64 num_slices = inputs[0].dim_size(0);
    for (const Tensor& t : inputs) {
      components.push_back(t);
      OP_REQUIRES(ctx, t.dims() > 0,
                  errors::InvalidArgument(
                      "All components must be at least 1-dimensional"));
      OP_REQUIRES(
          ctx, t.dim_size(0) == num_slices,
          errors::InvalidArgument(
              "All components must have the same size in the 0th dimension"));
    }
    *output = new Dataset(ctx, std::move(components));
  }

 private:
  class Dataset : public GraphDatasetBase {
   public:
    explicit Dataset(OpKernelContext* ctx, std::vector<Tensor> tensors)
        : GraphDatasetBase(ctx), tensors_(std::move(tensors)) {
      for (const Tensor& t : tensors_) {
        dtypes_.push_back(t.dtype());
        gtl::InlinedVector<int64, 4> partial_dim_sizes;
        // Handle scalar here. Check that everyone matches here? Or fail
        // at runtime?
        for (int i = 1; i < t.dims(); ++i) {
          partial_dim_sizes.push_back(t.dim_size(i));
        }
        shapes_.emplace_back(std::move(partial_dim_sizes));
      }
    }

    std::unique_ptr<IteratorBase> MakeIterator(
        const string& prefix) const override {
      return std::unique_ptr<IteratorBase>(
          new Iterator({this, strings::StrCat(prefix, "::TensorSlice")}));
    }

    const DataTypeVector& output_dtypes() const override { return dtypes_; }
    const std::vector<PartialTensorShape>& output_shapes() const override {
      return shapes_;
    }

    string DebugString() override { return "TensorSliceDatasetOp::Dataset"; }

   protected:
    Status AsGraphDefInternal(DatasetGraphDefBuilder* b,
                              Node** output) const override {
      std::vector<NodeBuilder::NodeOut> components;
      components.reserve(tensors_.size());
      for (const Tensor& t : tensors_) {
        Node* node;
        TF_RETURN_IF_ERROR(b->AddTensor(t, &node));
        components.emplace_back(node);
      }
      TF_RETURN_IF_ERROR(
          b->AddDatasetWithInputAsList(this, components, output));
      return Status::OK();
    }

   private:
    template <typename T>
    static Status HandleSliceToElement(const Tensor& parent, Tensor* element,
                                       int64 index) {
      DCHECK_NE(parent.dim_size(0), 0);
      DCHECK_GE(index, 0);
      if (element->NumElements() !=
          (parent.NumElements() / parent.dim_size(0))) {
        TensorShape chip_shape = parent.shape();
        chip_shape.RemoveDim(0);
        return errors::Internal(
            "HandleSliceToElement Cannot copy slice: number of elements does "
            "not match.  Shapes are: [element]: ",
            element->shape().DebugString(), ", [parent slice]: ",
            chip_shape.DebugString());
      }
      auto parent_as_matrix = parent.flat_outer_dims<T>();
      element->flat<T>() = parent_as_matrix.chip(index, 0);
      return Status::OK();
    }

    static Status CopySliceToElement(const Tensor& parent, Tensor* element,
                                     int64 index) {
#define HANDLE_TYPE(T)                                      \
  case DataTypeToEnum<T>::value: {                          \
    return HandleSliceToElement<T>(parent, element, index); \
  }

      switch (parent.dtype()) {
        TF_CALL_DATASET_TYPES(HANDLE_TYPE);
        default:
          return errors::Unimplemented(
              "CopySliceToElement Unhandled data type: ", element->dtype());
      }
    }

    class Iterator : public DatasetIterator<Dataset> {
     public:
      explicit Iterator(const Params& params)
          : DatasetIterator<Dataset>(params),
            i_(0),
            n_(params.dataset->tensors_[0].dim_size(0)) {}

      Status GetNextInternal(IteratorContext* ctx,
                             std::vector<Tensor>* out_tensors,
                             bool* end_of_sequence) override {
        mutex_lock l(mu_);
        if (i_ < n_) {
          out_tensors->clear();
          out_tensors->reserve(dataset()->tensors_.size());
          for (int i = 0; i < dataset()->tensors_.size(); ++i) {
            const Tensor& t = dataset()->tensors_[i];
            Tensor t_slice(cpu_allocator(), t.dtype(),
                           TensorShape(dataset()->shapes_[i].dim_sizes()));
            TF_RETURN_IF_ERROR(CopySliceToElement(t, &t_slice, i_));
            out_tensors->emplace_back(std::move(t_slice));
          }
          ++i_;
          *end_of_sequence = false;
        } else {
          *end_of_sequence = true;
        }
        return Status::OK();
      }

     protected:
      Status SaveInternal(IteratorStateWriter* writer) override {
        mutex_lock l(mu_);
        TF_RETURN_IF_ERROR(writer->WriteScalar(full_name("i"), i_));
        return Status::OK();
      }

      Status RestoreInternal(OpKernelContext* ctx,
                             IteratorStateReader* reader) override {
        mutex_lock l(mu_);
        TF_RETURN_IF_ERROR(reader->ReadScalar(full_name("i"), &i_));
        return Status::OK();
      }

     private:
      mutex mu_;
      int64 i_ GUARDED_BY(mu_);
      const int64 n_;
    };

    const std::vector<Tensor> tensors_;
    DataTypeVector dtypes_;
    std::vector<PartialTensorShape> shapes_;
  };
};

REGISTER_KERNEL_BUILDER(Name("TensorSliceDataset").Device(DEVICE_CPU),
                        TensorSliceDatasetOp);

}  // namespace

}  // namespace tensorflow
