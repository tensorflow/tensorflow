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
#include <numeric>

#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/framework/partial_tensor_shape.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/util/sparse/sparse_tensor.h"

namespace tensorflow {
namespace data {
namespace {

// See documentation in ../../ops/dataset_ops.cc for a high-level
// description of the following op.

template <typename T>
class Dataset : public DatasetBase {
 public:
  explicit Dataset(OpKernelContext* ctx,
                   const sparse::SparseTensor& sparse_tensor)
      : DatasetBase(DatasetContext(ctx)),
        sparse_tensor_(sparse_tensor),
        dtypes_({DT_INT64, sparse_tensor.dtype(), DT_INT64}),
        shapes_({{-1, sparse_tensor.dims() - 1},
                 {-1},
                 {sparse_tensor.dims() - 1}}) {}

  std::unique_ptr<IteratorBase> MakeIteratorInternal(
      const string& prefix) const override {
    return absl::make_unique<Iterator>(typename Iterator::Params{
        this, strings::StrCat(prefix, "::SparseTensorSlice")});
  }

  const DataTypeVector& output_dtypes() const override { return dtypes_; }
  const std::vector<PartialTensorShape>& output_shapes() const override {
    return shapes_;
  }

  string DebugString() const override {
    return "SparseTensorSliceDatasetOp::Dataset";
  }

  int64 Cardinality() const override { return sparse_tensor_.shape()[0]; }

  Status InputDatasets(std::vector<const DatasetBase*>* inputs) const override {
    return Status::OK();
  }

  Status CheckExternalState() const override { return Status::OK(); }

 protected:
  Status AsGraphDefInternal(SerializationContext* ctx,
                            DatasetGraphDefBuilder* b,
                            Node** output) const override {
    Node* indices_node;
    TF_RETURN_IF_ERROR(b->AddTensor(sparse_tensor_.indices(), &indices_node));
    Node* value_node;
    TF_RETURN_IF_ERROR(b->AddTensor(sparse_tensor_.values(), &value_node));
    Node* dense_shape_node;
    std::vector<int64> dense_shape;
    dense_shape.reserve(sparse_tensor_.shape().size());
    for (int i = 0; i < sparse_tensor_.shape().size(); i++)
      dense_shape.emplace_back(sparse_tensor_.shape()[i]);
    TF_RETURN_IF_ERROR(b->AddVector(dense_shape, &dense_shape_node));
    AttrValue val_dtype;
    b->BuildAttrValue(sparse_tensor_.dtype(), &val_dtype);
    TF_RETURN_IF_ERROR(
        b->AddDataset(this, {indices_node, value_node, dense_shape_node},
                      {{"Tvalues", val_dtype}}, output));
    return Status::OK();
  }

 private:
  class Iterator : public DatasetIterator<Dataset<T>> {
   public:
    explicit Iterator(const typename Iterator::Params& params)
        : DatasetIterator<Dataset<T>>(params),
          num_elements_(params.dataset->sparse_tensor_.shape()[0]),
          dense_shape_(DT_INT64, {params.dataset->sparse_tensor_.dims() - 1}),
          group_iterable_(params.dataset->sparse_tensor_.group({0})),
          iter_(group_iterable_.begin()) {
      for (size_t i = 0; i < dense_shape_.NumElements(); ++i) {
        dense_shape_.vec<int64>()(i) =
            params.dataset->sparse_tensor_.shape()[i + 1];
      }
    }

    Status GetNextInternal(IteratorContext* ctx,
                           std::vector<Tensor>* out_tensors,
                           bool* end_of_sequence) override {
      mutex_lock l(mu_);
      if (i_ == num_elements_) {
        *end_of_sequence = true;
        return Status::OK();
      }

      out_tensors->clear();
      out_tensors->reserve(3);
      const int rank = Iterator::dataset()->sparse_tensor_.dims();

      if (i_ > next_non_empty_i_ && iter_ != group_iterable_.end()) {
        // We still have elements to consume from `group_iterable_`
        // and we have emitted all elements up to and including the
        // current position.
        sparse::Group group = *iter_;
        const auto indices = group.indices();
        const auto values = group.values<T>();
        const int64 num_entries = values.size();
        next_non_empty_i_ = indices(0, 0);

        next_indices_ = Tensor(DT_INT64, {num_entries, rank - 1});
        next_values_ = Tensor(DataTypeToEnum<T>::value, {num_entries});

        auto next_indices_t = next_indices_.matrix<int64>();
        auto next_values_t = next_values_.vec<T>();

        for (int64 i = 0; i < num_entries; ++i) {
          for (int d = 1; d < rank; ++d) {
            next_indices_t(i, d - 1) = indices(i, d);
          }
          next_values_t(i) = values(i);
        }

        ++iter_;
      }
      if (i_ == next_non_empty_i_) {
        // The current position is non-empty in the input
        // `SparseTensor`, and we have already read the value from the
        // `GroupIterable`.
        out_tensors->push_back(std::move(next_indices_));
        out_tensors->push_back(std::move(next_values_));
        out_tensors->push_back(dense_shape_);
        next_non_empty_i_ = kNextNonEmptyUnknown;
      } else {
        DCHECK(i_ < next_non_empty_i_ || iter_ == group_iterable_.end());
        // The current position is empty in the input `SparseTensor`,
        // so emit empty indices and values.
        out_tensors->push_back(Tensor(DT_INT64, TensorShape({0, rank - 1})));
        out_tensors->push_back(Tensor(DataTypeToEnum<T>::value, {0}));
        out_tensors->push_back(dense_shape_);
      }

      ++i_;
      *end_of_sequence = false;
      return Status::OK();
    }

   protected:
    std::shared_ptr<model::Node> CreateNode(
        IteratorContext* ctx, model::Node::Args args) const override {
      return model::MakeSourceNode(std::move(args));
    }

    Status SaveInternal(SerializationContext* ctx,
                        IteratorStateWriter* writer) override {
      mutex_lock l(mu_);
      TF_RETURN_IF_ERROR(writer->WriteScalar(Iterator::full_name("i"), i_));
      TF_RETURN_IF_ERROR(
          writer->WriteScalar(Iterator::full_name("iter_loc"), iter_.loc()));
      TF_RETURN_IF_ERROR(writer->WriteScalar(
          Iterator::full_name("next_non_empty_i_"), next_non_empty_i_));
      if (i_ <= next_non_empty_i_) {
        TF_RETURN_IF_ERROR(writer->WriteTensor(
            Iterator::full_name("next_indices_"), next_indices_));
        TF_RETURN_IF_ERROR(writer->WriteTensor(
            Iterator::full_name("next_values_"), next_values_));
      }
      return Status::OK();
    }

    Status RestoreInternal(IteratorContext* ctx,
                           IteratorStateReader* reader) override {
      mutex_lock l(mu_);
      TF_RETURN_IF_ERROR(reader->ReadScalar(Iterator::full_name("i"), &i_));
      int64 iter_loc;
      TF_RETURN_IF_ERROR(
          reader->ReadScalar(Iterator::full_name("iter_loc"), &iter_loc));
      iter_ = group_iterable_.at(iter_loc);
      TF_RETURN_IF_ERROR(reader->ReadScalar(
          Iterator::full_name("next_non_empty_i_"), &next_non_empty_i_));
      if (i_ <= next_non_empty_i_) {
        TF_RETURN_IF_ERROR(reader->ReadTensor(
            Iterator::full_name("next_indices_"), &next_indices_));
        TF_RETURN_IF_ERROR(reader->ReadTensor(
            Iterator::full_name("next_values_"), &next_values_));
      }
      return Status::OK();
    }

   private:
    const int64 num_elements_;

    Tensor dense_shape_;

    mutex mu_;
    sparse::GroupIterable group_iterable_ TF_GUARDED_BY(mu_);
    sparse::GroupIterable::IteratorStep iter_ TF_GUARDED_BY(mu_);
    int64 i_ TF_GUARDED_BY(mu_) = 0;
    const int64 kNextNonEmptyUnknown = -1;
    int64 next_non_empty_i_ TF_GUARDED_BY(mu_) = kNextNonEmptyUnknown;
    Tensor next_indices_ TF_GUARDED_BY(mu_);
    Tensor next_values_ TF_GUARDED_BY(mu_);
  };

  const sparse::SparseTensor sparse_tensor_;
  const DataTypeVector dtypes_;
  const std::vector<PartialTensorShape> shapes_;
};

template <typename T>
class SparseTensorSliceDatasetOp : public DatasetOpKernel {
 public:
  explicit SparseTensorSliceDatasetOp(OpKernelConstruction* ctx)
      : DatasetOpKernel(ctx) {}

  void MakeDataset(OpKernelContext* ctx, DatasetBase** output) override {
    // Create a new SparseTensorSliceDatasetOp::Dataset, insert it in
    // the step container, and return it as the output.
    const Tensor* indices;
    OP_REQUIRES_OK(ctx, ctx->input("indices", &indices));
    const Tensor* values;
    OP_REQUIRES_OK(ctx, ctx->input("values", &values));
    const Tensor* dense_shape;
    OP_REQUIRES_OK(ctx, ctx->input("dense_shape", &dense_shape));

    OP_REQUIRES(ctx, TensorShapeUtils::IsMatrix(indices->shape()),
                errors::InvalidArgument(
                    "Input indices should be a matrix but received shape ",
                    indices->shape().DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsVector(values->shape()),
                errors::InvalidArgument(
                    "Input values should be a vector but received shape ",
                    indices->shape().DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsVector(dense_shape->shape()),
                errors::InvalidArgument(
                    "Input shape should be a vector but received shape ",
                    dense_shape->shape().DebugString()));

    // We currently ensure that `sparse_tensor` is ordered in the
    // batch dimension.
    // TODO(mrry): Investigate ways to avoid this unconditional check
    // if we can be sure that the sparse tensor was produced in an
    // appropriate order (e.g. by `tf.parse_example()` or a Dataset
    // that batches elements into rows of a SparseTensor).
    int64 previous_batch_index = -1;
    for (int64 i = 0; i < indices->dim_size(0); ++i) {
      int64 next_batch_index = indices->matrix<int64>()(i, 0);
      OP_REQUIRES(
          ctx, next_batch_index >= previous_batch_index,
          errors::Unimplemented("The SparseTensor must be ordered in the batch "
                                "dimension; handling arbitrarily ordered input "
                                "is not currently supported."));
      previous_batch_index = next_batch_index;
    }
    gtl::InlinedVector<int64, 8> std_order(dense_shape->NumElements(), 0);
    sparse::SparseTensor tensor;
    OP_REQUIRES_OK(
        ctx, sparse::SparseTensor::Create(
                 *indices, *values, TensorShape(dense_shape->vec<int64>()),
                 std_order, &tensor));
    *output = new Dataset<T>(ctx, std::move(tensor));
  }

 private:
};

#define REGISTER_DATASET_KERNEL(type)                           \
  REGISTER_KERNEL_BUILDER(Name("SparseTensorSliceDataset")      \
                              .Device(DEVICE_CPU)               \
                              .TypeConstraint<type>("Tvalues"), \
                          SparseTensorSliceDatasetOp<type>);

TF_CALL_DATASET_TYPES(REGISTER_DATASET_KERNEL);
#undef REGISTER_DATASET_KERNEL

}  // namespace
}  // namespace data
}  // namespace tensorflow
