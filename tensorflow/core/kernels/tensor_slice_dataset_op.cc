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
    *output = new Dataset(std::move(components));
  }

 private:
  class Dataset : public DatasetBase {
   public:
    explicit Dataset(std::vector<Tensor> tensors)
        : tensors_(std::move(tensors)) {
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

    std::unique_ptr<IteratorBase> MakeIterator() const override {
      return std::unique_ptr<IteratorBase>(new Iterator(this));
    }

    const DataTypeVector& output_dtypes() const override { return dtypes_; }
    const std::vector<PartialTensorShape>& output_shapes() const override {
      return shapes_;
    }

    string DebugString() override { return "TensorSliceDatasetOp::Dataset"; }

   private:
    template <DataType DT>
    static Status HandleSliceToElement(const Tensor& parent, Tensor* element,
                                       int64 index) {
      typedef typename EnumToDataType<DT>::Type T;
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
#define HANDLE_TYPE(DT)                                                   \
  if (parent.dtype() == DT) {                                             \
    TF_RETURN_IF_ERROR(HandleSliceToElement<DT>(parent, element, index)); \
    return Status::OK();                                                  \
  }
      HANDLE_TYPE(DT_FLOAT);
      HANDLE_TYPE(DT_HALF);
      HANDLE_TYPE(DT_DOUBLE);
      HANDLE_TYPE(DT_INT32);
      HANDLE_TYPE(DT_UINT8);
      HANDLE_TYPE(DT_INT16);
      HANDLE_TYPE(DT_INT8);
      HANDLE_TYPE(DT_STRING);
      HANDLE_TYPE(DT_COMPLEX64);
      HANDLE_TYPE(DT_COMPLEX128);
      HANDLE_TYPE(DT_INT64);
      HANDLE_TYPE(DT_BOOL);
      HANDLE_TYPE(DT_QINT8);
      HANDLE_TYPE(DT_QUINT8);
      HANDLE_TYPE(DT_QINT32);
      HANDLE_TYPE(DT_QINT16);
      HANDLE_TYPE(DT_QUINT16);
#undef HANDLE_TYPE
      return errors::Unimplemented("CopySliceToElement Unhandled data type: ",
                                   element->dtype());
    }

    class Iterator : public DatasetIterator<Dataset> {
     public:
      explicit Iterator(const Dataset* dataset)
          : DatasetIterator<Dataset>(dataset),
            i_(0),
            n_(dataset->tensors_[0].dim_size(0)) {}

      Status GetNext(IteratorContext* ctx, std::vector<Tensor>* out_tensors,
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

     private:
      mutex mu_;
      int i_ GUARDED_BY(mu_);
      const int n_;
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
