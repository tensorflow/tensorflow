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
#include "tensorflow/core/framework/partial_tensor_shape.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/variant.h"
#include "tensorflow/core/kernels/data/dataset.h"

namespace tensorflow {

namespace {

// See documentation in ../ops/dataset_ops.cc for a high-level
// description of the following op.

class DenseToSparseBatchDatasetOp : public UnaryDatasetOpKernel {
 public:
  explicit DenseToSparseBatchDatasetOp(OpKernelConstruction* ctx)
      : UnaryDatasetOpKernel(ctx) {}

  void MakeDataset(OpKernelContext* ctx, DatasetBase* input,
                   DatasetBase** output) override {
    // Create a new DenseToSparseBatchDatasetOp::Dataset, insert it in the
    // step-local container, and return it as the output.
    OP_REQUIRES(
        ctx, input->output_dtypes().size() == 1,
        errors::InvalidArgument("DenseToSparseBatchDataset only supports "
                                "inputs with a single component."));

    int64 batch_size;
    OP_REQUIRES_OK(ctx,
                   ParseScalarArgument<int64>(ctx, "batch_size", &batch_size));
    OP_REQUIRES(
        ctx, batch_size > 0,
        errors::InvalidArgument("Batch size must be greater than zero."));

    const Tensor* row_shape_t;
    OP_REQUIRES_OK(ctx, ctx->input("row_shape", &row_shape_t));
    OP_REQUIRES(ctx, TensorShapeUtils::IsVector(row_shape_t->shape()),
                errors::InvalidArgument("row_shape must be a vector"));
    PartialTensorShape row_shape;
    OP_REQUIRES_OK(ctx, PartialTensorShape::MakePartialShape(
                            row_shape_t->vec<int64>().data(),
                            row_shape_t->NumElements(), &row_shape));

    *output = nullptr;

#define HANDLE_TYPE(T)                                           \
  case DataTypeToEnum<T>::value: {                               \
    *output = new Dataset<T>(ctx, batch_size, row_shape, input); \
    break;                                                       \
  }

    switch (input->output_dtypes()[0]) {
      TF_CALL_DATASET_TYPES(HANDLE_TYPE);
#undef HANDLE_TYPE
      default:
        OP_REQUIRES(ctx, false,
                    errors::Unimplemented(
                        "DenseToSparseBatchDataset unhandled data type: ",
                        input->output_dtypes()[0]));
    }
  }

 private:
  // TODO(mrry): Push the templated code down to the raw copying routine.
  template <class T>
  class Dataset : public GraphDatasetBase {
   public:
    Dataset(OpKernelContext* ctx, int64 batch_size,
            const PartialTensorShape& row_shape, const DatasetBase* input)
        : GraphDatasetBase(ctx),
          batch_size_(batch_size),
          row_shape_(row_shape),
          input_(input) {
      input_->Ref();

      output_shapes_.reserve(1);
      PartialTensorShape output_shape({-1});
      output_shape.AppendShape(row_shape_);
      output_shapes_.push_back(output_shape);
    }

    ~Dataset() override { input_->Unref(); }

    std::unique_ptr<IteratorBase> MakeIteratorInternal(
        const string& prefix) const override {
      return std::unique_ptr<IteratorBase>(new Iterator(
          {this, strings::StrCat(prefix, "::DenseToSparseBatch")}));
    }

    const DataTypeVector& output_dtypes() const override {
      static DataTypeVector* output_dtypes = new DataTypeVector({DT_VARIANT});
      return *output_dtypes;
    }

    const std::vector<PartialTensorShape>& output_shapes() const override {
      return output_shapes_;
    }

    string DebugString() const override {
      return strings::StrCat("DenseToSparseBatchDatasetOp(", batch_size_,
                             ")::Dataset");
    }

   protected:
    Status AsGraphDefInternal(OpKernelContext* ctx, DatasetGraphDefBuilder* b,
                              Node** output) const override {
      Node* input_node;
      TF_RETURN_IF_ERROR(b->AddParentDataset(ctx, input_, &input_node));
      Node* batch_size_node;
      TF_RETURN_IF_ERROR(b->AddScalar(batch_size_, &batch_size_node));
      Node* row_shape_node;
      std::vector<int64> row_shape;
      row_shape.reserve(
          row_shape_.dims());  // not an unknown rank PartialTensorShape
      for (int i = 0; i < row_shape_.dims(); i++)
        row_shape.emplace_back(row_shape_.dim_size(i));
      TF_RETURN_IF_ERROR(b->AddVector(row_shape, &row_shape_node));
      TF_RETURN_IF_ERROR(b->AddDataset(
          this, {input_node, batch_size_node, row_shape_node}, output));
      return Status::OK();
    }

   private:
    class Iterator : public DatasetIterator<Dataset<T>> {
     public:
      explicit Iterator(const typename Iterator::Params& params)
          : DatasetIterator<Dataset<T>>(params) {}

      Status Initialize(IteratorContext* ctx) override {
        return DatasetIterator<Dataset<T>>::dataset()->input_->MakeIterator(
            ctx, DatasetIterator<Dataset<T>>::prefix(), &input_impl_);
      }

      Status GetNextInternal(IteratorContext* ctx,
                             std::vector<Tensor>* out_tensors,
                             bool* end_of_sequence) override {
        // Each row of the output SparseTensor is an individual tensor
        // from the input iterator.
        std::vector<Tensor> batch_elements;
        int64 total_elements = 0;
        batch_elements.reserve(
            DatasetIterator<Dataset<T>>::dataset()->batch_size_);
        const PartialTensorShape& row_shape =
            DatasetIterator<Dataset<T>>::dataset()->row_shape_;
        const int row_ndims = row_shape.dims();

        // Determine the size of the output tensors:
        // * dense_shape will be [`row_shape + 1`].
        Tensor dense_shape(ctx->allocator({}), DT_INT64, {row_ndims + 1});
        auto dense_shape_vec = dense_shape.vec<int64>();
        for (size_t i = 0; i < row_ndims; ++i) {
          if (row_shape.dim_size(i) == -1) {
            dense_shape_vec(i + 1) = 0;
          } else {
            dense_shape_vec(i + 1) = row_shape.dim_size(i);
          }
        }

        {
          mutex_lock l(mu_);
          *end_of_sequence = false;
          for (int i = 0;
               i < DatasetIterator<Dataset<T>>::dataset()->batch_size_ &&
               !*end_of_sequence;
               ++i) {
            std::vector<Tensor> batch_element_tuple;
            TF_RETURN_IF_ERROR(input_impl_->GetNext(ctx, &batch_element_tuple,
                                                    end_of_sequence));
            if (!*end_of_sequence) {
              DCHECK_EQ(1, batch_element_tuple.size());
              batch_elements.push_back(std::move(batch_element_tuple[0]));
              total_elements += batch_element_tuple[0].NumElements();

              // TODO(mrry): Investigate how to hoist this check when we
              // have static information that renders it unnecessary.
              if (batch_element_tuple[0].shape().dims() != row_ndims) {
                return errors::InvalidArgument(
                    "Input element had shape (",
                    batch_element_tuple[0].shape().DebugString(),
                    ") that is incompatible with the row shape (",
                    row_shape.DebugString(), ").");
              }
              for (int j = 0; j < row_ndims; ++j) {
                // Take the maximum in the dimension if -1 is given.
                if (row_shape.dim_size(j) == -1) {
                  dense_shape_vec(j + 1) =
                      std::max(batch_element_tuple[0].dim_size(j),
                               dense_shape_vec(j + 1));
                } else if (batch_element_tuple[0].dim_size(j) >
                           row_shape.dim_size(j)) {
                  return errors::DataLoss(
                      "Input element had shape (",
                      batch_element_tuple[0].shape().DebugString(),
                      ") that is larger than the row shape (",
                      row_shape.DebugString(), ").");
                }
              }
            }
          }
        }

        if (batch_elements.empty()) {
          DCHECK(*end_of_sequence);
          return Status::OK();
        }

        // * indices will be [`total_elements`, `row_shape + 1`].
        // * values will be [`total_elements`].
        Tensor indices(ctx->allocator({}), DT_INT64,
                       {total_elements, row_ndims + 1});
        Tensor values(
            ctx->allocator({}),
            DatasetIterator<Dataset<T>>::dataset()->input_->output_dtypes()[0],
            {total_elements});
        auto indices_matrix = indices.matrix<int64>();
        auto values_flat = values.flat<T>();

        int64 current_position_in_values = 0;
        for (int64 i = 0; i < batch_elements.size(); ++i) {
          const Tensor& t = batch_elements[i];
          const auto& t_flat = t.flat<T>();
          // TODO(mrry): Replace with a memcpy or something more
          // efficient. (Maybe an Eigen assign op?)
          gtl::InlinedVector<int64, 4> strides(row_ndims);
          if (!strides.empty()) {
            strides[row_ndims - 1] = 1;
            for (int64_t row_dim = strides.size() - 2; row_dim >= 0;
                 --row_dim) {
              strides[row_dim] =
                  strides[row_dim + 1] * t.shape().dim_size(row_dim + 1);
            }
          }

          for (int64 j = 0; j < t.NumElements(); ++j) {
            values_flat(current_position_in_values) = t_flat(j);
            indices_matrix(current_position_in_values, 0) = i;
            int64 index = j;
            for (size_t k = 0; k < strides.size(); ++k) {
              indices_matrix(current_position_in_values, k + 1) =
                  index / strides[k];
              index %= strides[k];
            }
            ++current_position_in_values;
          }
        }

        dense_shape_vec(0) = batch_elements.size();

        Tensor serialized_sparse(DT_VARIANT, TensorShape({3}));
        auto serialized_sparse_t = serialized_sparse.vec<Variant>();
        serialized_sparse_t(0) = std::move(indices);
        serialized_sparse_t(1) = std::move(values);
        serialized_sparse_t(2) = std::move(dense_shape);
        out_tensors->push_back(std::move(serialized_sparse));

        *end_of_sequence = false;
        return Status::OK();
      }

     protected:
      Status SaveInternal(IteratorStateWriter* writer) override {
        mutex_lock l(mu_);
        TF_RETURN_IF_ERROR(Iterator::SaveParent(writer, input_impl_));
        return Status::OK();
      }

      Status RestoreInternal(IteratorContext* ctx,
                             IteratorStateReader* reader) override {
        mutex_lock l(mu_);
        TF_RETURN_IF_ERROR(Iterator::RestoreParent(ctx, reader, input_impl_));
        return Status::OK();
      }

     private:
      mutex mu_;
      std::unique_ptr<IteratorBase> input_impl_ GUARDED_BY(mu_);
    };

    const int64 batch_size_;
    const PartialTensorShape row_shape_;
    const DatasetBase* const input_;
    std::vector<PartialTensorShape> output_shapes_;
  };
};

REGISTER_KERNEL_BUILDER(Name("DenseToSparseBatchDataset").Device(DEVICE_CPU),
                        DenseToSparseBatchDatasetOp);

}  // namespace

}  // namespace tensorflow
