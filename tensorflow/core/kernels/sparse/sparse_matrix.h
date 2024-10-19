/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_KERNELS_SPARSE_SPARSE_MATRIX_H_
#define TENSORFLOW_CORE_KERNELS_SPARSE_SPARSE_MATRIX_H_

#define EIGEN_USE_THREADS

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#define EIGEN_USE_GPU
#endif

#include "unsupported/Eigen/CXX11/Tensor"  // from @eigen_archive
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/variant.h"
#include "tensorflow/core/framework/variant_encode_decode.h"
#include "tensorflow/core/framework/variant_op_registry.h"
#include "tensorflow/core/platform/errors.h"

namespace tensorflow {

class CSRSparseMatrix {
  // CreateCSRSparseMatrix is the main method used to construct a
  // CSRSparseMatrix.  The representations for both 2D and 3D
  // (batched) CSR Sparse Matrices are the same:
  //
  // dtype: The datatype of the values.
  // dense_shape: The dense shape of the matrix.
  //   * Host int64 vector, size 2 or 3.
  //   * Takes on values: (rows, cols) or (batch_size, rows, cols).
  // batch_pointers: Batch offset pointers into col_indices and values.
  //   * Host int32 vector, size (batch_size + 1).
  //   * Takes on values: (0, nnz[0], nnz[0] + nnz[1], ..., total_nnz).
  // row_pointers: Row offset pointers into col_indices and values.
  //   * Device int32 vector, size ((rows + 1) * batch_size).
  //   * Each block of size (rows + 1) takes on values:
  //     (0, num_rows{b}[0], num_rows{b}[0] + num_rows{b}[1], ..., nnz[b]).
  //     for b = 0 .. batch_size - 1.
  // col_indices: Column values for the given row and column index.
  //   * Device int32 vector, size total_nnz.
  // values: Actual values for the given row and column index.
  //   * Device dtype vector, size total_nnz.
  //
  // The storage agreement is such that for a given (batch, row, ix):
  //   offset = batch_pointers(batch) + row_pointers(batch * (rows + 1) + row)
  //   col = col_indices(offset + ix)
  //   val = values(offset + ix)
  // where ix < #nnz columns in (batch, row).
  // Then:
  //   matrix(batch, row, col) = val.
  //
  // All other elements in the dense representation are treated as 0 / empty.
  //
  // For example, for a 2D sparse matrix m shaped (3, 4) such that:
  //
  //   m[0, 0] = 1.0
  //   m[0, 1] = 2.0
  //   m[0, 2] = 3.0
  //   m[2, 2] = 4.0
  //   m[2, 3] = 5.0
  //
  // The corresponding representation is:
  //
  //   dtype: DT_FLOAT
  //   dense_shape: (3, 4)
  //   batch_pointers: (0, 5)
  //   row_pointers: (0, 3, 3, 5)
  //   col_indices: concat((0, 1, 2), (), (2, 3))
  //   values: concat((1.0, 2.0, 3.0), (), (4.0, 5.0))
  //
  // For a 3D sparse matrix m shaped (2, 3, 4) such that:
  //
  //   m[0, 0, 0] = 1.0
  //   m[0, 0, 2] = 2.0
  //   m[0, 2, 3] = 3.0
  //   m[1, 0, 3] = 4.0
  //   m[1, 1, 0] = 5.0
  //
  // The corresponding representation is:
  //   dtype: DT_FLOAT
  //   dense_shape: (2, 3, 4)
  //   batch_pointers: (0, 3, 5)
  //   row_pointers: concat((0, 2, 2, 3), (0, 1, 2, 2))
  //   col_indices: concat(concat((0, 2), (), (3,)),
  //                       concat((3,),   (), (0,)))
  //   values: concat(concat((1.0, 2.0), (3.0,), ()),
  ///                 concat((4.0,),     (5.0,), ()))
  //
 public:
  static constexpr const char kTypeName[] = "tensorflow::CSRSparseMatrix";

  CSRSparseMatrix() : metadata_{false, DT_INVALID} {}

  CSRSparseMatrix(const CSRSparseMatrix& rhs)
      : metadata_(rhs.metadata_),
        dense_shape_(rhs.dense_shape_),
        batch_pointers_(rhs.batch_pointers_),
        row_pointers_(rhs.row_pointers_),
        col_indices_(rhs.col_indices_),
        values_(rhs.values_) {
    SetupVecs();
  }

  CSRSparseMatrix(CSRSparseMatrix&& rhs)
      : metadata_(rhs.metadata_),
        dense_shape_(std::move(rhs.dense_shape_)),
        batch_pointers_(std::move(rhs.batch_pointers_)),
        row_pointers_(std::move(rhs.row_pointers_)),
        col_indices_(std::move(rhs.col_indices_)),
        values_(std::move(rhs.values_)) {
    SetupVecs();
    rhs.metadata_.validated = false;
    rhs.metadata_.dtype = DT_INVALID;
    rhs.ClearVecs();
  }

  CSRSparseMatrix& operator=(CSRSparseMatrix&& rhs) {
    if (this == &rhs) return *this;
    metadata_ = rhs.metadata_;
    metadata_.validated = rhs.metadata_.validated;
    dense_shape_ = std::move(rhs.dense_shape_);
    batch_pointers_ = std::move(rhs.batch_pointers_);
    row_pointers_ = std::move(rhs.row_pointers_);
    col_indices_ = std::move(rhs.col_indices_);
    values_ = std::move(rhs.values_);
    SetupVecs();
    rhs.metadata_ = {false, DT_INVALID};
    rhs.ClearVecs();
    return *this;
  }

  static absl::Status CreateCSRSparseMatrix(
      DataType dtype,
      const Tensor& dense_shape,     // on host
      const Tensor& batch_pointers,  // on host
      const Tensor& row_pointers, const Tensor& col_indices,
      const Tensor& values, CSRSparseMatrix* matrix) {
    *matrix = CSRSparseMatrix(dtype, dense_shape, batch_pointers, row_pointers,
                              col_indices, values);
    absl::Status s = matrix->Validate();
    matrix->metadata_.validated = s.ok();
    matrix->SetupVecs();
    return s;
  }

  absl::Status Validate() const {
    return ValidateTypesAndShapes(metadata_.dtype, dense_shape_,
                                  batch_pointers_, row_pointers_, col_indices_,
                                  values_);
  }

  void Clear() {
    metadata_ = {false, DT_INVALID};
    dense_shape_ = Tensor();
    batch_pointers_ = Tensor();
    row_pointers_ = Tensor();
    col_indices_ = Tensor();
    values_ = Tensor();
    ClearVecs();
  }

  bool valid() const {
    return metadata_.validated && dense_shape_.IsInitialized() &&
           batch_pointers_.IsInitialized() && row_pointers_.IsInitialized() &&
           col_indices_.IsInitialized() && values_.IsInitialized() &&
           dense_shape_.NumElements() > 1 &&
           batch_pointers_.NumElements() > 0 && row_pointers_.NumElements() > 0;
  }

  DataType dtype() const {
    DCHECK(valid());
    return metadata_.dtype;
  }

  inline int dims() const {
    DCHECK(valid());
    return dense_shape_.NumElements();
  }

  inline int nnz(int batch) const {
    DCHECK_LT(batch, batch_size());
    return (*batch_pointers_vec_)(batch + 1) - (*batch_pointers_vec_)(batch);
  }

  inline int batch_offset(int batch) const {
    DCHECK_LT(batch, batch_size());
    return (*batch_pointers_vec_)(batch);
  }

  inline int total_nnz() const {
    DCHECK(valid());
    return (*batch_pointers_vec_)(batch_size());
  }

  inline Tensor& dense_shape() {
    DCHECK(valid());
    return dense_shape_;
  }

  inline const Tensor& dense_shape() const {
    DCHECK(valid());
    return dense_shape_;
  }

  inline TTypes<int32>::UnalignedVec row_pointers_vec(int batch) {
    DCHECK(valid());
    DCHECK_LT(batch, batch_size());
    const int64_t rows = dense_shape().vec<int64_t>()((dims() == 2) ? 0 : 1);
    const int offset = batch * (rows + 1);
    return TTypes<int32>::UnalignedVec(row_pointers_vec_->data() + offset,
                                       rows + 1);
  }

  inline TTypes<int32>::UnalignedConstVec row_pointers_vec(int batch) const {
    DCHECK(valid());
    DCHECK_LT(batch, batch_size());
    const int64_t rows = dense_shape().vec<int64_t>()((dims() == 2) ? 0 : 1);
    const int offset = batch * (rows + 1);
    return TTypes<int32>::UnalignedConstVec(row_pointers_vec_->data() + offset,
                                            rows + 1);
  }

  inline TTypes<int32>::UnalignedVec col_indices_vec(int batch) {
    DCHECK(valid());
    DCHECK_LT(batch, batch_size());
    const int offset = (*batch_pointers_vec_)(batch);
    const int nnz_in_batch = nnz(batch);
    return TTypes<int32>::UnalignedVec(col_indices_vec_->data() + offset,
                                       nnz_in_batch);
  }

  inline TTypes<int32>::UnalignedConstVec col_indices_vec(int batch) const {
    DCHECK(valid());
    DCHECK_LT(batch, batch_size());
    const int offset = (*batch_pointers_vec_)(batch);
    const int nnz_in_batch = nnz(batch);
    return TTypes<int32>::UnalignedConstVec(col_indices_vec_->data() + offset,
                                            nnz_in_batch);
  }

  template <typename T>
  inline typename TTypes<T>::UnalignedVec values_vec(int batch) {
    DCHECK(valid());
    DCHECK_LT(batch, batch_size());
    const int offset = (*batch_pointers_vec_)(batch);
    const int nnz_in_batch = nnz(batch);
    return typename TTypes<T>::UnalignedVec(values().vec<T>().data() + offset,
                                            nnz_in_batch);
  }

  template <typename T>
  inline typename TTypes<T>::UnalignedConstVec values_vec(int batch) const {
    DCHECK(valid());
    DCHECK_LT(batch, batch_size());
    const int offset = (*batch_pointers_vec_)(batch);
    const int nnz_in_batch = nnz(batch);
    return typename TTypes<T>::UnalignedConstVec(
        values().vec<T>().data() + offset, nnz_in_batch);
  }

  inline Tensor& row_pointers() {
    DCHECK(valid());
    return row_pointers_;
  }

  inline const Tensor& row_pointers() const {
    DCHECK(valid());
    return row_pointers_;
  }

  inline Tensor& col_indices() {
    DCHECK(valid());
    return col_indices_;
  }

  inline const Tensor& col_indices() const {
    DCHECK(valid());
    return col_indices_;
  }

  inline Tensor& values() {
    DCHECK(valid());
    return values_;
  }

  inline const Tensor& values() const {
    DCHECK(valid());
    return values_;
  }

  inline Tensor& batch_pointers() {
    DCHECK(valid());
    return batch_pointers_;
  }

  inline const Tensor& batch_pointers() const {
    DCHECK(valid());
    return batch_pointers_;
  }

  std::string TypeName() const { return kTypeName; }

  // TODO(ebrevdo): A better debug string.
  std::string DebugString() const { return dense_shape_.DebugString(); }

  // Returns the number of elements.  This is equal to 1 if the
  // CSRSparseMatrix is a singleton matrix (dense_shape is length 2).
  int batch_size() const {
    DCHECK(valid());
    return batch_pointers_.NumElements() - 1;
  }

  bool Decode(const VariantTensorData& p) {
    if (p.tensors_.empty()) return false;
    Metadata metadata;
    if (!p.get_metadata(&metadata)) return false;
    const bool validated = metadata.validated;
    const DataType dtype = metadata.dtype;

    // p.tensors_ should contain tensors {dense_shape, batch_pointers,
    // row_pointers, col_indices, values}.
    if (p.tensors_.size() != 5) return false;

    Tensor dense_shape = p.tensors_[0];
    if (dense_shape.dtype() != DT_INT64) return false;
    if (dense_shape.dims() != 1) return false;
    int rank = dense_shape.dim_size(0);
    if (rank < 2 || rank > 3) return false;

    Tensor batch_pointers(p.tensors_[1]);
    Tensor row_pointers(p.tensors_[2]);
    Tensor col_indices(p.tensors_[3]);
    Tensor values(p.tensors_[4]);

    // Check that the validated bool is consistent with the data.
    absl::Status s = ValidateTypesAndShapes(dtype, dense_shape, batch_pointers,
                                            row_pointers, col_indices, values);
    if (s.ok() != validated) return false;

    // Save to this object.
    metadata_ = metadata;
    dense_shape_ = std::move(dense_shape);
    batch_pointers_ = std::move(batch_pointers);
    row_pointers_ = std::move(row_pointers);
    col_indices_ = std::move(col_indices);
    values_ = std::move(values);
    SetupVecs();
    return true;
  }

  void Encode(VariantTensorData* p) const {
    DCHECK(valid());

    // Store metadata_ to p's metadata
    p->set_metadata(metadata_);

    // Store dense_shape, row_pointers, col_indices, and values to p->tensors_.
    p->tensors_.reserve(5);
    p->tensors_.push_back(dense_shape_);
    p->tensors_.push_back(batch_pointers_);
    p->tensors_.push_back(row_pointers_);
    p->tensors_.push_back(col_indices_);
    p->tensors_.push_back(values_);
  }

  // This static method copies CSRSparseMatrices in all directions:
  //   Host->Device, Device->Host, and Device->Device.
  static absl::Status DeviceCopy(
      const CSRSparseMatrix& from, CSRSparseMatrix* to,
      const UnaryVariantOpRegistry::AsyncTensorDeviceCopyFn& copy) {
    VLOG(2) << "DeviceCopy from type: " << DataTypeString(from.dtype())
            << " and shape: " << from.dense_shape().DebugString();
    Tensor to_row_ptr(DT_INT32);
    Tensor to_col_ind(DT_INT32);
    Tensor to_values(from.dtype());
    TF_RETURN_IF_ERROR(copy(from.row_pointers(), &to_row_ptr));
    TF_RETURN_IF_ERROR(copy(from.col_indices(), &to_col_ind));
    TF_RETURN_IF_ERROR(copy(from.values(), &to_values));
    return CreateCSRSparseMatrix(from.dtype(),
                                 from.dense_shape(),     // Always on host.
                                 from.batch_pointers(),  // Always on host.
                                 to_row_ptr, to_col_ind, to_values, to);
  }

 private:
  CSRSparseMatrix(DataType dtype, const Tensor& dense_shape,
                  const Tensor& batch_pointers, const Tensor& row_pointers,
                  const Tensor& col_indices, const Tensor& values)
      : metadata_{false, dtype},
        dense_shape_(dense_shape),
        batch_pointers_(batch_pointers),
        row_pointers_(row_pointers),
        col_indices_(col_indices),
        values_(values) {}

  void SetupVecs() {
    if (!metadata_.validated) return;
    batch_pointers_vec_.reset(
        new TTypes<int32>::Vec(batch_pointers_.vec<int32>()));
    row_pointers_vec_.reset(new TTypes<int32>::Vec(row_pointers_.vec<int32>()));
    col_indices_vec_.reset(new TTypes<int32>::Vec(col_indices_.vec<int32>()));
  }

  void ClearVecs() {
    batch_pointers_vec_.reset();
    row_pointers_vec_.reset();
    col_indices_vec_.reset();
  }

  static absl::Status ValidateTypesAndShapes(DataType dtype,
                                             const Tensor& dense_shape,
                                             const Tensor& batch_pointers,
                                             const Tensor& row_pointers,
                                             const Tensor& col_indices,
                                             const Tensor& values) {
    // TODO(ebrevdo): Consider adding support for other floating point types
    // (namely, float16).
    if (dtype != DT_FLOAT && dtype != DT_DOUBLE && dtype != DT_COMPLEX64 &&
        dtype != DT_COMPLEX128) {
      return errors::InvalidArgument(
          "CSRSparseMatrix::Validate: dtype = ", DataTypeString(dtype),
          " not in {float32, float64, complex64, complex128}");
    }
    // dense_shape checks
    if (dense_shape.dtype() != DT_INT64) {
      return errors::InvalidArgument(
          "CSRSparseMatrix::Validate: dense_shape.dtype() = ",
          DataTypeString(dense_shape.dtype()), " != int64");
    }
    if (dense_shape.dims() != 1) {
      return errors::InvalidArgument(
          "CSRSparseMatrix::Validate: dense_shape should be a vector, but saw "
          "tensor: ",
          dense_shape.DebugString());
    }
    int rank = dense_shape.dim_size(0);
    if (rank < 2 || rank > 3) {
      return errors::InvalidArgument(
          "CSRSparseMatrix::Validate: dense_shape should be a 2- or 3- vector, "
          "but saw: ",
          dense_shape.SummarizeValue(5));
    }
    auto dense_shape_t = dense_shape.vec<int64_t>();
    const int64_t batch_size = (rank == 2) ? 1 : dense_shape_t(0);
    const int64_t num_rows = (rank == 2) ? dense_shape_t(0) : dense_shape_t(1);

    if (batch_pointers.dtype() != DT_INT32) {
      return errors::InvalidArgument(
          "CSRSparseMatrix::Validate: batch_pointers.dtype() = ",
          DataTypeString(batch_pointers.dtype()), " != int32");
    }
    if (batch_pointers.dims() != 1) {
      return errors::InvalidArgument(
          "CSRSparseMatrix::Validate: batch_indices is not a vector, saw "
          "shape: ",
          batch_pointers.shape().DebugString());
    }

    // batch size checks
    if (batch_size != batch_pointers.NumElements() - 1) {
      return errors::InvalidArgument(
          "CSRSparseMatrix::Validate: dense_shape is ",
          dense_shape.SummarizeValue(5),
          " but batch pointers implies batch size is ",
          batch_pointers.NumElements() - 1);
    }

    if (row_pointers.dtype() != DT_INT32) {
      return errors::InvalidArgument(
          "CSRSparseMatrix::Validate: row_pointers.dtype() = ",
          DataTypeString(row_pointers.dtype()), " != int32");
    }
    if (row_pointers.dims() != 1) {
      return errors::InvalidArgument(
          "CSRSparseMatrix::Validate: row_pointers is not a vector, saw "
          "shape: ",
          row_pointers.shape().DebugString());
    }
    if (row_pointers.dim_size(0) != batch_size * (num_rows + 1)) {
      return errors::InvalidArgument(
          "CSRSparseMatrix::Validate: row_pointers should have size batch_size "
          "* (num_rows + 1), saw shapes: ",
          dense_shape.DebugString(), " vs. ",
          row_pointers.shape().DebugString());
    }
    if (col_indices.dtype() != DT_INT32) {
      return errors::InvalidArgument(
          "CSRSparseMatrix::Validate: col_indices.dtype() = ",
          DataTypeString(col_indices.dtype()), " != int32");
    }
    if (col_indices.dims() != 1) {
      return errors::InvalidArgument(
          "CSRSparseMatrix::Validate: col_indices is not a vector, saw shape: ",
          col_indices.shape().DebugString());
    }
    if (values.dtype() != dtype) {
      return errors::InvalidArgument(
          "CSRSparseMatrix::Validate: values.dtype() = ",
          DataTypeString(values.dtype()),
          " != dtype = ", DataTypeString(dtype));
    }
    if (values.dims() != 1) {
      return errors::InvalidArgument(
          "CSRSparseMatrix::Validate: values is not a vector, saw shape: ",
          values.shape().DebugString());
    }
    if (col_indices.dim_size(0) != values.dim_size(0)) {
      return errors::InvalidArgument(
          "CSRSparseMatrix::Validate: size(col_indices) = ",
          col_indices.dim_size(0), " != size(values) = ", values.dim_size(0));
    }
    return absl::OkStatus();
  }

  struct Metadata {
    bool validated;
    DataType dtype;
  };
  Metadata metadata_;
  Tensor dense_shape_;
  Tensor batch_pointers_;
  Tensor row_pointers_;
  Tensor col_indices_;
  Tensor values_;
  std::unique_ptr<TTypes<int32>::Vec> batch_pointers_vec_;
  std::unique_ptr<TTypes<int32>::Vec> row_pointers_vec_;
  std::unique_ptr<TTypes<int32>::Vec> col_indices_vec_;
};

// Call BinaryFunctor<Device, T>()(ctx, a, b, c)
// where T depends on a.dtype().  T will be one of: float, double,
// complex64, complex128.
template <typename Device, template <typename, typename> class BinaryFunctor>
absl::Status CSRSparseMatrixBinaryHelper(OpKernelContext* ctx,
                                         const CSRSparseMatrix& a,
                                         const CSRSparseMatrix& b,
                                         CSRSparseMatrix* c) {
  DataType dt = a.dtype();
  if (dt != b.dtype()) {
    return errors::InvalidArgument(
        "CSRSparseMatrixBinaryHelper: Inconsistent dtypes for input matrices, "
        "a "
        "dtype: ",
        DataTypeString(dt), ", b dtype: ", DataTypeString(b.dtype()));
  }
  switch (dt) {
    case DT_FLOAT: {
      BinaryFunctor<Device, float> functor(ctx);
      return functor(a, b, c);
    }
    case DT_DOUBLE: {
      BinaryFunctor<Device, double> functor(ctx);
      return functor(a, b, c);
    }
    case DT_COMPLEX64: {
      BinaryFunctor<Device, complex64> functor(ctx);
      return functor(a, b, c);
    }
    case DT_COMPLEX128: {
      BinaryFunctor<Device, complex128> functor(ctx);
      return functor(a, b, c);
    }
    default:
      return errors::InvalidArgument(
          "CSRSparseMatrixBinaryHelper: a.dtype (", DataTypeString(dt),
          ") is not one of: float, double, complex64, complex128");
  }
}

// Call UnaryFunctor<Device, T>()(ctx, a, b)
// where T depends on a.dtype().  T will be one of: float, double,
// complex64, complex128.
template <typename Device, template <typename, typename> class UnaryFunctor>
absl::Status CSRSparseMatrixUnaryHelper(OpKernelContext* ctx,
                                        const CSRSparseMatrix& a,
                                        CSRSparseMatrix* b) {
  DataType dt = a.dtype();
  switch (dt) {
    case DT_FLOAT: {
      UnaryFunctor<Device, float> functor(ctx);
      return functor(a, b);
    }
    case DT_DOUBLE: {
      UnaryFunctor<Device, double> functor(ctx);
      return functor(a, b);
    }
    case DT_COMPLEX64: {
      UnaryFunctor<Device, complex64> functor(ctx);
      return functor(a, b);
    }
    case DT_COMPLEX128: {
      UnaryFunctor<Device, complex128> functor(ctx);
      return functor(a, b);
    }
    default:
      return errors::InvalidArgument(
          "CSRSparseMatrixUnaryHelper: a.dtype (", DataTypeString(dt),
          ") is not one of: float, double, complex64, complex128");
  }
}

template <typename T>
struct ConstCSRComponent {
  TTypes<int32>::UnalignedConstVec row_ptr;
  TTypes<int32>::UnalignedConstVec col_ind;
  typename TTypes<T>::UnalignedConstVec values;
  TTypes<int64_t>::ConstVec dense_shape_host;
};

template <typename T>
struct CSRComponent {
  TTypes<int32>::UnalignedVec row_ptr;
  TTypes<int32>::UnalignedVec col_ind;
  typename TTypes<T>::UnalignedVec values;
  TTypes<int64_t>::Vec dense_shape_host;
};

template <typename T>
absl::Status ExtractVariantFromInput(OpKernelContext* ctx, int index,
                                     const T** value) {
  const Tensor& input_t = ctx->input(index);
  if (!TensorShapeUtils::IsScalar(input_t.shape())) {
    return errors::InvalidArgument(
        "Invalid input matrix: Shape must be rank 0 but is rank ",
        input_t.dims());
  }
  const Variant& input_variant = input_t.scalar<Variant>()();
  *value = input_variant.get<T>();
  if (*value == nullptr) {
    return errors::InvalidArgument("Could not retrieve Variant input ", index);
  }
  if (!(*value)->valid()) {
    return errors::InvalidArgument("Variant input ", index, " is not valid.");
  }
  return absl::OkStatus();
}

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_SPARSE_SPARSE_MATRIX_H_
