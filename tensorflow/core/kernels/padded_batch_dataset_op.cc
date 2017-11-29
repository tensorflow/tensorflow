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
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_util.h"

namespace tensorflow {

namespace {

// See documentation in ../ops/dataset_ops.cc for a high-level
// description of the following op.

// The following five functions are copied from padding_fifo_queue.cc.
// TODO(mrry): Reconcile these functions with the similar methods in the
// queue implementation.
Status ValidateElementToLargerSlice(const Tensor& element, Tensor* parent) {
  DCHECK_NE(parent->dim_size(0), 0);
  if (element.NumElements() > (parent->NumElements() / parent->dim_size(0))) {
    TensorShape chip_shape = parent->shape();
    chip_shape.RemoveDim(0);
    return errors::Internal(
        "HandleElementToLargerSlice Cannot copy slice: number of entries in "
        "element is greater than number of elements in parent slice.  ",
        "Shapes are: [element]: ", element.shape().DebugString(),
        ", [parent slice]: ", chip_shape.DebugString());
  }
  return Status::OK();
}

template <typename T, int NDIMS>
Status HandleElementToLargerSlice(const Tensor& element, Tensor* parent,
                                  int index) {
  TF_RETURN_IF_ERROR(ValidateElementToLargerSlice(element, parent));
  if (element.NumElements() == 0) {
    return Status::OK();
  }
  auto element_t = element.tensor<T, NDIMS>();
  auto parent_t = parent->tensor<T, NDIMS + 1>();
  Eigen::DSizes<Eigen::DenseIndex, NDIMS + 1> slice_indices;
  slice_indices[0] = index;
  Eigen::DSizes<Eigen::DenseIndex, NDIMS + 1> slice_size;
  slice_size[0] = 1;
  for (size_t i = 1; i < slice_size.size(); ++i) {
    slice_size[i] = element_t.dimension(i - 1);
  }
  parent_t.slice(slice_indices, slice_size) = element_t.reshape(slice_size);
  return Status::OK();
}

template <int NDIMS>
Status HandleElementToLargerSliceWithRank(const Tensor& element, Tensor* parent,
                                          int index) {
#define HANDLE_TYPE(T)                                                   \
  case DataTypeToEnum<T>::value: {                                       \
    return HandleElementToLargerSlice<T, NDIMS>(element, parent, index); \
  }

  switch (element.dtype()) {
    TF_CALL_ALL_TYPES(HANDLE_TYPE);
#undef HANDLE_TYPE
    default:
      return errors::Unimplemented(
          "HandleElementToLargerSliceWithRank Unhandled data type: ",
          element.dtype());
  }
}

Status CopyElementToLargerSlice(const Tensor& element, Tensor* parent,
                                int index) {
  if (parent->dims() != element.dims() + 1) {
    return errors::Internal(
        "Mismatched ranks.  Element's rank is: ", element.dims(),
        " but element is meant to be a slice in output Tensor having rank: ",
        parent->dims(), " (should be: ", element.dims() + 1, ")");
  }

#define HANDLE_DIMS(NDIMS)                                                  \
  case NDIMS: {                                                             \
    TF_RETURN_IF_ERROR(                                                     \
        HandleElementToLargerSliceWithRank<NDIMS>(element, parent, index)); \
    return Status::OK();                                                    \
  }

  switch (element.dims()) {
    HANDLE_DIMS(0);
    HANDLE_DIMS(1);
    HANDLE_DIMS(2);
    HANDLE_DIMS(3);
    HANDLE_DIMS(4);
#undef HANDLE_DIMS
    default:
      return errors::Unimplemented("CopyElementToLargerSlice Unhandled rank: ",
                                   element.dims());
  }
}

Status SetElementZero(Tensor* element, const Tensor& padding) {
#define HANDLE_TYPE(T)                                     \
  if (element->dtype() == DataTypeToEnum<T>::value) {      \
    element->flat<T>().setConstant(padding.scalar<T>()()); \
    return Status::OK();                                   \
  }
  TF_CALL_ALL_TYPES(HANDLE_TYPE);
#undef HANDLE_TYPE
  return errors::Unimplemented("SetElementZero Unhandled data type: ",
                               element->dtype());
}

class PaddedBatchDatasetOp : public UnaryDatasetOpKernel {
 public:
  explicit PaddedBatchDatasetOp(OpKernelConstruction* ctx)
      : UnaryDatasetOpKernel(ctx) {}

  void MakeDataset(OpKernelContext* ctx, DatasetBase* input,
                   DatasetBase** output) override {
    int64 batch_size;
    OP_REQUIRES_OK(ctx,
                   ParseScalarArgument<int64>(ctx, "batch_size", &batch_size));
    OP_REQUIRES(
        ctx, batch_size > 0,
        errors::InvalidArgument("Batch size must be greater than zero."));

    OpInputList padded_shape_tensors;
    OP_REQUIRES_OK(ctx,
                   ctx->input_list("padded_shapes", &padded_shape_tensors));
    std::vector<PartialTensorShape> padded_shapes;
    padded_shapes.reserve(padded_shape_tensors.size());
    OP_REQUIRES(ctx,
                padded_shape_tensors.size() == input->output_shapes().size(),
                errors::InvalidArgument("Number of padded shapes (",
                                        padded_shape_tensors.size(),
                                        ") must match the number of components "
                                        "in the input dataset's elements (",
                                        input->output_shapes().size(), ")"));
    for (const Tensor& padded_shape_t : padded_shape_tensors) {
      OP_REQUIRES(ctx, TensorShapeUtils::IsVector(padded_shape_t.shape()),
                  errors::InvalidArgument("All padded shapes must be vectors"));
      PartialTensorShape padded_shape;
      OP_REQUIRES_OK(ctx, PartialTensorShape::MakePartialShape(
                              padded_shape_t.vec<int64>().data(),
                              padded_shape_t.NumElements(), &padded_shape));
      padded_shapes.push_back(std::move(padded_shape));
    }
    OpInputList padding_values_list;
    OP_REQUIRES_OK(ctx,
                   ctx->input_list("padding_values", &padding_values_list));
    std::vector<Tensor> padding_values;
    OP_REQUIRES(ctx,
                padding_values_list.size() == input->output_shapes().size(),
                errors::InvalidArgument(
                    "Number of padding values (", padding_values_list.size(),
                    ") must match the number of components in the input "
                    "dataset's elements (",
                    input->output_shapes().size(), ")"));
    for (int i = 0; i < padding_values_list.size(); ++i) {
      const Tensor& padding_value_t = padding_values_list[i];
      OP_REQUIRES(
          ctx, TensorShapeUtils::IsScalar(padding_value_t.shape()),
          errors::InvalidArgument("All padding values must be scalars"));
      OP_REQUIRES(ctx, padding_value_t.dtype() == input->output_dtypes()[i],
                  errors::InvalidArgument(
                      "Mismatched type between padding value ", i,
                      " and input dataset's component ", i, ": ",
                      DataTypeString(padding_value_t.dtype()), " vs. ",
                      DataTypeString(input->output_dtypes()[i])));
      padding_values.push_back(tensor::DeepCopy(padding_value_t));
    }

    *output = new Dataset(batch_size, std::move(padded_shapes),
                          std::move(padding_values), input);
  }

 private:
  class Dataset : public DatasetBase {
   public:
    Dataset(int64 batch_size, std::vector<PartialTensorShape> padded_shapes,
            std::vector<Tensor> padding_values, const DatasetBase* input)
        : batch_size_(batch_size),
          padded_shapes_(std::move(padded_shapes)),
          padding_values_(std::move(padding_values)),
          input_(input) {
      input_->Ref();

      // NOTE(mrry): Currently we implement "batch up to"
      // semantics. If we could tell statically that the input dataset
      // is infinite, then we could always report `batch_size` as the
      // 0th dimension.
      // TODO(mrry): Need to validate that the input shape and the
      // padded shape are "compatible" (i.e. that padded shape is >=
      // input shape, with both static and dynamic checks as appropriate).
      const auto& input_shapes = input_->output_shapes();
      output_shapes_.reserve(input_shapes.size());
      for (size_t i = 0; i < input_shapes.size(); ++i) {
        output_shapes_.push_back(
            PartialTensorShape({-1}).Concatenate(padded_shapes_[i]));
      }
    }

    ~Dataset() override { input_->Unref(); }

    std::unique_ptr<IteratorBase> MakeIterator() const override {
      return std::unique_ptr<IteratorBase>(new Iterator(this));
    }

    const DataTypeVector& output_dtypes() const override {
      return input_->output_dtypes();
    }

    const std::vector<PartialTensorShape>& output_shapes() const override {
      return output_shapes_;
    }

    string DebugString() override {
      return strings::StrCat("PaddedBatchDatasetOp(", batch_size_,
                             ")::Dataset");
    }

   private:
    // Copies element into the index^th slice of parent (in the 0th dimension).
    //

    class Iterator : public DatasetIterator<Dataset> {
     public:
      explicit Iterator(const Dataset* dataset)
          : DatasetIterator<Dataset>(dataset),
            input_impl_(dataset->input_->MakeIterator()) {}

      Status GetNext(IteratorContext* ctx, std::vector<Tensor>* out_tensors,
                     bool* end_of_sequence) override {
        // Each row of `batch_elements` is a tuple of tensors from the
        // input iterator.
        std::vector<std::vector<Tensor>> batch_elements;
        batch_elements.reserve(dataset()->batch_size_);
        {
          mutex_lock l(mu_);
          *end_of_sequence = false;
          for (int i = 0; i < dataset()->batch_size_ && !*end_of_sequence;
               ++i) {
            std::vector<Tensor> batch_element_tuple;
            TF_RETURN_IF_ERROR(input_impl_->GetNext(ctx, &batch_element_tuple,
                                                    end_of_sequence));
            if (!*end_of_sequence) {
              batch_elements.push_back(std::move(batch_element_tuple));
            }
          }
        }

        if (batch_elements.empty()) {
          DCHECK(*end_of_sequence);
          return Status::OK();
        }

        // Copy the retrieved batch elements into one output tensor
        // per tuple component.
        // NOTE(mrry): If the input or output sizes are statically
        // known, we could potentially read the input values in-place
        // into their respective slice locations. This would require a
        // different GetNext() overload that supports zero-copy, and might
        // make sense in an optimization pass.
        const size_t num_tuple_components = batch_elements[0].size();
        const int64 num_batch_elements = batch_elements.size();
        for (size_t component_index = 0; component_index < num_tuple_components;
             ++component_index) {
          // 1. Determine the shape of the padded tensor.
          TensorShape batch_component_shape({num_batch_elements});
          const PartialTensorShape& padded_shape =
              dataset()->padded_shapes_[component_index];

          for (int dim = 0; dim < padded_shape.dims(); ++dim) {
            if (padded_shape.dim_size(dim) == -1) {
              batch_component_shape.AddDim(0);
            } else {
              batch_component_shape.AddDim(padded_shape.dim_size(dim));
            }
          }

          for (int64 i = 0; i < num_batch_elements; ++i) {
            const TensorShape& element_shape =
                batch_elements[i][component_index].shape();
            // TODO(mrry): Perform this check in the shape function if
            // enough static information is available to do so.
            if (element_shape.dims() != padded_shape.dims()) {
              return errors::InvalidArgument(
                  "All elements in a batch must have the same rank as the "
                  "padded shape for component",
                  component_index, ": expected rank ", padded_shape.dims(),
                  " but got element with rank ", element_shape.dims());
            }
            for (int dim = 0; dim < padded_shape.dims(); ++dim) {
              if (padded_shape.dim_size(dim) == -1) {
                // Take the max of all batch elements in this dimension.
                if (batch_elements[i][component_index].shape().dim_size(dim) >
                    batch_component_shape.dim_size(dim + 1)) {
                  batch_component_shape.set_dim(
                      dim + 1,
                      batch_elements[i][component_index].shape().dim_size(dim));
                }
              } else {
                if (batch_elements[i][component_index].shape().dim_size(dim) >
                    batch_component_shape.dim_size(dim + 1)) {
                  return errors::DataLoss(
                      "Attempted to pad to a smaller size than the input "
                      "element.");
                }
              }
            }
          }

          // 2. Copy each batch element to the appropriate location in
          // the output component tensor.
          Tensor batch_component(cpu_allocator(),
                                 output_dtypes()[component_index],
                                 batch_component_shape);
          TF_RETURN_IF_ERROR(SetElementZero(
              &batch_component, dataset()->padding_values_[component_index]));

          // Build the output tuple component by copying one slice
          // from each input element in the batch.
          for (int64 i = 0; i < num_batch_elements; ++i) {
            TF_RETURN_IF_ERROR(ValidateElementToLargerSlice(
                batch_elements[i][component_index], &batch_component));

            TF_RETURN_IF_ERROR(CopyElementToLargerSlice(
                batch_elements[i][component_index], &batch_component, i));
          }
          out_tensors->push_back(std::move(batch_component));
        }
        *end_of_sequence = false;
        return Status::OK();
      }

     private:
      mutex mu_;
      int64 i_ GUARDED_BY(mu_);
      std::unique_ptr<IteratorBase> input_impl_ GUARDED_BY(mu_);
    };

    const int64 batch_size_;
    const std::vector<PartialTensorShape> padded_shapes_;
    const std::vector<Tensor> padding_values_;
    const DatasetBase* const input_;
    std::vector<PartialTensorShape> output_shapes_;
  };
};

REGISTER_KERNEL_BUILDER(Name("PaddedBatchDataset").Device(DEVICE_CPU),
                        PaddedBatchDatasetOp);

}  // namespace

}  // namespace tensorflow
