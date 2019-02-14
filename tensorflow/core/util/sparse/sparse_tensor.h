/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_UTIL_SPARSE_SPARSE_TENSOR_H_
#define TENSORFLOW_CORE_UTIL_SPARSE_SPARSE_TENSOR_H_

#include <limits>
#include <numeric>
#include <vector>

#include "absl/base/macros.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/bounds_check.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/sparse/dim_comparator.h"
#include "tensorflow/core/util/sparse/group_iterator.h"

namespace tensorflow {
namespace sparse {

class SparseTensor {
 public:
  typedef typename gtl::ArraySlice<int64> VarDimArray;
  typedef typename gtl::InlinedVector<int64, 8> ShapeArray;

  static Status Create(Tensor ix, Tensor vals, const VarDimArray shape,
                       const VarDimArray order, SparseTensor* result) {
    if (ix.dtype() != DT_INT64) {
      return Status(
          error::INVALID_ARGUMENT,
          strings::StrCat("indices must be type int64 but got: ", ix.dtype()));
    }
    if (!TensorShapeUtils::IsVector(vals.shape())) {
      return Status(error::INVALID_ARGUMENT,
                    strings::StrCat("vals must be a vec, but got: ",
                                    vals.shape().DebugString()));
    }
    if (ix.shape().dim_size(0) != vals.shape().dim_size(0)) {
      return Status(error::INVALID_ARGUMENT,
                    strings::StrCat("indices and values rows (indexing "
                                    "dimension) must match. (indices = ",
                                    ix.shape().dim_size(0), ", values = ",
                                    vals.shape().dim_size(0), ")"));
    }
    int dims;
    TF_RETURN_IF_ERROR(GetDimsFromIx(ix, &dims));
    if (order.size() != dims) {
      return Status(error::INVALID_ARGUMENT,
                    "Order length must be SparseTensor rank.");
    }
    if (shape.size() != dims) {
      return Status(error::INVALID_ARGUMENT,
                    "Shape rank must be SparseTensor rank.");
    }

    *result = SparseTensor(ix, vals, shape, order);
    return Status();
  }

  static Status Create(Tensor ix, Tensor vals, const TensorShape& shape,
                       SparseTensor* result) {
    return Create(ix, vals, TensorShapeToVector(shape),
                  UndefinedOrder(TensorShapeToVector(shape)), result);
  }

  static Status Create(Tensor ix, Tensor vals, const VarDimArray shape,
                       SparseTensor* result) {
    return Create(ix, vals, shape, UndefinedOrder(shape), result);
  }

  static Status Create(Tensor ix, Tensor vals, const TensorShape& shape,
                       const VarDimArray order, SparseTensor* result) {
    return Create(ix, vals, TensorShapeToVector(shape), order, result);
  }

  SparseTensor() : dims_(0) {}

  ABSL_DEPRECATED("Use Create() functions instead of constructors directly.")
  SparseTensor(Tensor ix, Tensor vals, const TensorShape& shape)
      : SparseTensor(ix, vals, TensorShapeToVector(shape),
                     UndefinedOrder(TensorShapeToVector(shape))) {}

  ABSL_DEPRECATED("Use Create() functions instead of constructors directly.")
  SparseTensor(Tensor ix, Tensor vals, const VarDimArray shape)
      : SparseTensor(ix, vals, shape, UndefinedOrder(shape)) {}

  ABSL_DEPRECATED("use Create() functions instead of constructors directly.")
  SparseTensor(Tensor ix, Tensor vals, const TensorShape& shape,
               const VarDimArray order)
      : SparseTensor(ix, vals, TensorShapeToVector(shape), order) {}

  ABSL_DEPRECATED("Use Create() functions instead of constructors directly.")
  SparseTensor(Tensor ix, Tensor vals, const VarDimArray shape,
               const VarDimArray order)
      : ix_(ix),
        vals_(vals),
        shape_(shape.begin(), shape.end()),
        order_(order.begin(), order.end()),
        dims_(UnsafeGetDimsFromIx(ix)) {
    DCHECK_EQ(ix.dtype(), DT_INT64)
        << "indices must be type int64 but got: " << ix.dtype();
    DCHECK(TensorShapeUtils::IsVector(vals.shape()))
        << "vals must be a vec, but got: " << vals.shape().DebugString();
    DCHECK_EQ(ix.shape().dim_size(0), vals.shape().dim_size(0))
        << "indices and values rows (indexing dimension) must match.";
    DCHECK_EQ(order.size(), dims_) << "Order length must be SparseTensor rank.";
    DCHECK_EQ(shape.size(), dims_) << "Shape rank must be SparseTensor rank.";
  }

  SparseTensor(const SparseTensor& other)
      : SparseTensor(other.ix_, other.vals_, other.shape_, other.order_) {}

  SparseTensor(SparseTensor&& other)
      : SparseTensor(std::move(other.ix_), std::move(other.vals_),
                     std::move(other.shape_), std::move(other.order_)) {}

  SparseTensor& operator=(const SparseTensor& other) {
    ix_ = other.ix_;
    vals_ = other.vals_;
    shape_ = other.shape_;
    order_ = other.order_;
    dims_ = other.dims_;
    return *this;
  }

  SparseTensor& operator=(SparseTensor&& other) {
    ix_ = std::move(other.ix_);
    vals_ = std::move(other.vals_);
    shape_ = std::move(other.shape_);
    order_ = std::move(other.order_);
    dims_ = std::move(other.dims_);
    return *this;
  }

  std::size_t num_entries() const { return ix_.dim_size(0); }

  int dims() const { return shape_.size(); }

  const Tensor& indices() const { return ix_; }

  const Tensor& values() const { return vals_; }

  DataType dtype() const { return vals_.dtype(); }

  Status IndicesValid() const {
    const auto ix_t = ix_.matrix<int64>();
    for (int64 ord : order_) {
      if (ord < 0) {
        return errors::FailedPrecondition(
            "Order was not provided.  Provide an order at "
            "construction time or run ReorderInPlace");
      }
    }

    for (std::size_t n = 0; n < num_entries(); ++n) {
      TF_RETURN_IF_ERROR(IndexValid(ix_t, n));
    }

    return Status::OK();
  }

  VarDimArray shape() const { return shape_; }

  VarDimArray order() const { return order_; }

  // Resorts the indices and values according to the dimensions in order.
  template <typename T>
  void Reorder(const VarDimArray& order);

  // Returns a group iterable that can be used for clumping indices
  // and values according to the group indices of interest.
  //
  // Precondition: order()[0..group_ix.size()] == group_ix.
  //
  // See the README.md in this directory for more usage information.
  GroupIterable group(const VarDimArray& group_ix) const {
    DCHECK_LE(group_ix.size(), dims_);
    for (std::size_t di = 0; di < group_ix.size(); ++di) {
      DCHECK_GE(group_ix[di], 0) << "Group dimension out of range";
      DCHECK_LT(group_ix[di], dims_) << "Group dimension out of range";
      DCHECK_EQ(group_ix[di], order_[di])
          << "Group dimension does not match sorted order";
    }
    return GroupIterable(ix_, vals_, dims_, group_ix);
  }

  // Stores the sparse indices into the dense tensor out.
  // Preconditions:
  //   out->shape().dims() == shape().dims()
  //   out->shape().dim_size(d) >= shape(d) for all d
  //
  // Returns true on success.  False on failure (mismatched dimensions
  // or out-of-bounds indices).
  //
  // If initialize==True, ToDense first overwrites all coefficients in out to 0.
  //
  template <typename T>
  bool ToDense(Tensor* out, bool initialize = true);

  // Concat() will concatenate all the tensors according to their first order
  // dimension.  All tensors must have identical shape except for
  // the first order dimension.  All tensors orders' first dimension
  // must match.
  //
  // If all of the tensors have identical ordering, then the output
  // will have this ordering.  Otherwise the output is set as not
  // having any order and a Reorder<T>() should be called on it before
  // performing any subsequent operations.
  template <typename T>
  static SparseTensor Concat(const gtl::ArraySlice<SparseTensor>& tensors);

  // Split() will split the input SparseTensor into a list of num_split
  // SparseTensor given a splitting dimension. If the input dimension range
  // isn't an integer multiple of split_dim, we add one extra dimension for
  // each slice.
  template <typename T>
  static Status Split(const SparseTensor& tensor, const int split_dim,
                      const int num_split, std::vector<SparseTensor>* result);

  // Slice() will slice the input SparseTensor into a SparseTensor based on
  // specified start and size. Both start and size are 1-D array with each
  // element of the array representing one dimension. The start is the start
  // index at each dimension and the size is the size at each dimension.
  template <typename T>
  static SparseTensor Slice(const SparseTensor& tensor,
                            const gtl::ArraySlice<int64>& start,
                            const gtl::ArraySlice<int64>& size);

  // Picks out the dimensions according to `dim_indices`.
  std::vector<int64> PickDims(gtl::ArraySlice<int64> dim_indices) const {
    std::vector<int64> res(dim_indices.size());
    for (size_t i = 0; i < dim_indices.size(); ++i) {
      res[i] = shape_[dim_indices[i]];
    }
    return res;
  }

 private:
  static Status GetDimsFromIx(const Tensor& ix, int* result) {
    if (!TensorShapeUtils::IsMatrix(ix.shape())) {
      return Status(error::INVALID_ARGUMENT,
                    strings::StrCat("indices must be a matrix, but got: ",
                                    ix.shape().DebugString()));
    }
    *result = UnsafeGetDimsFromIx(ix);
    return Status();
  }

  static int UnsafeGetDimsFromIx(const Tensor& ix) {
    DCHECK(TensorShapeUtils::IsMatrix(ix.shape()));
    return ix.dim_size(1);
  }

  static inline ShapeArray UndefinedOrder(const VarDimArray shape) {
    return ShapeArray(shape.size(), -1);
  }

  static inline ShapeArray TensorShapeToVector(const TensorShape& shape) {
    ShapeArray vec(shape.dims());
    for (int i = 0; i < shape.dims(); ++i) vec[i] = shape.dim_size(i);
    return vec;
  }

  // Helper for IndicesValid()
  inline Status IndexValid(const TTypes<int64>::ConstMatrix& ix_t,
                           int n) const {
    bool valid = true;
    bool different = false;
    bool increasing = true;
    if (n == 0) {
      for (int di = 0; di < dims_; ++di) {
        if (ix_t(n, di) < 0 || ix_t(n, di) >= shape_[di]) valid = false;
      }
      different = true;
    } else {
      for (int di = 0; di < dims_; ++di) {
        if (ix_t(n, di) < 0 || ix_t(n, di) >= shape_[di]) valid = false;
        int64 diff = ix_t(n, order_[di]) - ix_t(n - 1, order_[di]);
        if (diff > 0) different = true;
        if (!different && diff < 0) increasing = false;
      }
    }
    if (TF_PREDICT_FALSE(!valid || !increasing || !different)) {
      string index = strings::StrCat("indices[", n, "] = [");
      for (int di = 0; di < dims_; ++di) {
        strings::StrAppend(&index, ix_t(n, di), di < dims_ - 1 ? "," : "]");
      }
      if (!valid) {
        return errors::InvalidArgument(index,
                                       " is out of bounds: need 0 <= index < [",
                                       str_util::Join(shape_, ","), "]");
      }
      if (!increasing) {
        return errors::InvalidArgument(index, " is out of order");
      }
      if (!different) {
        return errors::InvalidArgument(index, " is repeated");
      }
    }
    return Status::OK();
  }

  // Helper for ToDense<T>()
  template <typename T>
  bool ValidateAndInitializeToDense(Tensor* out, bool initialize);

  // Helper for Split() that returns the slice index.
  static inline int GetSliceIndex(const int dim, const int split_size,
                                  const int residual) {
    DCHECK_GT(split_size, 0);
    DCHECK_GE(dim, 0);
    if (residual == 0) return dim / split_size;
    const int offset = residual * (split_size + 1);
    if (dim < offset) {
      return dim / (split_size + 1);
    } else {
      return residual + ((dim - offset) / split_size);
    }
  }

  // Helper for Split() that returns the dimension in the slice.
  static inline int GetDimensionInSlice(const int dim, const int split_size,
                                        const int residual) {
    DCHECK_GT(split_size, 0);
    DCHECK_GE(dim, 0);
    if (residual == 0) return dim % split_size;
    const int offset = residual * (split_size + 1);
    if (dim < offset) {
      return dim % (split_size + 1);
    } else {
      return (dim - offset) % split_size;
    }
  }

  // Helper for Split() that returns the shape given a slice index.
  static inline int GetSliceShape(const int slice_index, const int split_size,
                                  const int residual) {
    DCHECK_GT(split_size, 0);
    DCHECK_GE(slice_index, 0);
    if (residual == 0) return split_size;
    if (slice_index < residual) {
      return split_size + 1;
    } else {
      return split_size;
    }
  }

  Tensor ix_;
  Tensor vals_;
  ShapeArray shape_;
  ShapeArray order_;
  int dims_;
};

// This operation updates the indices and values Tensor rows, so it is
// an in-place algorithm.  It requires O(N log N) time and O(N)
// temporary space.
template <typename T>
void SparseTensor::Reorder(const VarDimArray& order) {
  DCHECK_EQ(DataTypeToEnum<T>::v(), dtype())
      << "Reorder requested with the wrong datatype";
  DCHECK_EQ(order.size(), dims_) << "Order length must be SparseTensor rank";
  auto ix_t = ix_.matrix<int64>();
  auto vals_t = vals_.vec<T>();

  std::vector<int64> reorder(num_entries());
  std::iota(reorder.begin(), reorder.end(), 0);

  // Sort to get order of indices
  switch (order.size()) {
#define CASE_SORT(ORDER_SIZE)                                    \
  case ORDER_SIZE: {                                             \
    FixedDimComparator<ORDER_SIZE> sorter(ix_t, order, shape()); \
    std::sort(reorder.begin(), reorder.end(), sorter);           \
    break;                                                       \
  }
    CASE_SORT(0);
    CASE_SORT(1);
    CASE_SORT(2);
    CASE_SORT(3);
    CASE_SORT(4);
    CASE_SORT(5);
#undef CASE_SORT
    default: {
      DimComparator sorter(ix_t, order, shape());
      std::sort(reorder.begin(), reorder.end(), sorter);
    }
  }

  // We have a forward reordering, but what we'll need is a
  // permutation (the inverse).  This can be calculated with O(1)
  // additional
  // and O(n) time (INVPERM) but we just do the simple thing here.
  std::vector<size_t> permutation(reorder.size());
  for (std::size_t n = 0; n < reorder.size(); ++n) {
    permutation[reorder[n]] = n;
  }

  // Update indices & values by converting the permutations to
  // a product of transpositions.  Iterate over the cycles in the
  // permutation, and convert each of those into a product of
  // transpositions (swaps):
  //   https://en.wikipedia.org/wiki/Cyclic_permutation
  // This is N swaps, 2*N comparisons.
  for (std::size_t n = 0; n + 1 < permutation.size(); ++n) {
    while (n != permutation[n]) {
      std::size_t r = permutation[n];
      std::swap_ranges(&(ix_t(n, 0)), &(ix_t(n + 1, 0)), &(ix_t(r, 0)));
      std::swap(vals_t(n), vals_t(r));
      std::swap(permutation[n], permutation[r]);
    }
  }

  order_ = ShapeArray(order.begin(), order.end());
}

template <typename T>
bool SparseTensor::ValidateAndInitializeToDense(Tensor* out, bool initialize) {
  DCHECK_EQ(DataTypeToEnum<T>::v(), dtype())
      << "ToDense requested with the wrong datatype";

  DCHECK_EQ(out->shape().dims(), dims_)
      << "Incompatible dimensions between SparseTensor and output";

  DCHECK_EQ(out->dtype(), DataTypeToEnum<T>::v())
      << "Output must be type: " << DataTypeToEnum<T>::v()
      << " but got: " << out->dtype();

  // Make sure the dense output is the same rank and has room
  // to hold the SparseTensor.
  const auto& out_shape = out->shape();
  if (shape_.size() != out_shape.dims()) return false;
  for (int d = 0; d < shape_.size(); ++d) {
    if (shape_[d] > out_shape.dim_size(d)) return false;
  }

  if (initialize) {
    auto out_t = out->flat<T>();
    out_t.setConstant(T());
  }

  return true;
}

template <typename T>
bool SparseTensor::ToDense(Tensor* out, bool initialize) {
  if (!ValidateAndInitializeToDense<T>(out, initialize)) return false;

  auto out_t = out->flat<T>();
  auto ix_t = ix_.matrix<int64>();
  auto vals_t = vals_.vec<T>();

  std::vector<int64> strides(dims_);
  const auto& out_shape = out->shape();
  if (dims_ > 0) {
    strides[dims_ - 1] = 1;
  }
  for (int d = dims_ - 2; d >= 0; --d) {
    strides[d] = strides[d + 1] * out_shape.dim_size(d + 1);
  }

  for (int n = 0; n < vals_t.dimension(0); ++n) {
    bool invalid_dims = false;
    int64 ix = 0;
    for (int d = 0; d < dims_; ++d) {
      const int64 ix_n_d = internal::SubtleMustCopy(ix_t(n, d));
      if (!FastBoundsCheck(ix_n_d, out_shape.dim_size(d))) {
        invalid_dims = true;
      }
      ix += strides[d] * ix_n_d;
    }
    if (invalid_dims) return false;
    out_t(ix) = vals_t(n);
  }
  return true;
}

template <typename T>
SparseTensor SparseTensor::Concat(
    const gtl::ArraySlice<SparseTensor>& tensors) {
  DCHECK_GE(tensors.size(), size_t{1}) << "Cannot concat 0 SparseTensors";
  const int dims = tensors[0].dims_;
  DCHECK_GE(dims, 1) << "Cannot concat 0-dimensional SparseTensors";
  auto order_0 = tensors[0].order();
  const int primary_dim = order_0[0];
  ShapeArray final_order(order_0.begin(), order_0.end());
  ShapeArray final_shape(tensors[0].shape().begin(), tensors[0].shape().end());
  final_shape[primary_dim] = 0;  // We'll build this up as we go along.
  int num_entries = 0;

  bool fully_ordered = true;
  for (const SparseTensor& st : tensors) {
    DCHECK_EQ(st.dims_, dims) << "All SparseTensors must have the same rank.";
    DCHECK_EQ(DataTypeToEnum<T>::v(), st.dtype())
        << "Concat requested with the wrong data type";
    DCHECK_GE(st.order()[0], 0) << "SparseTensor must be ordered";
    DCHECK_EQ(st.order()[0], primary_dim)
        << "All SparseTensors' order[0] must match.  This is the concat dim.";
    if (st.order() != final_order) fully_ordered = false;
    const VarDimArray& st_shape = st.shape();
    for (int d = 0; d < dims - 1; ++d) {
      const int cdim = (d < primary_dim) ? d : d + 1;
      DCHECK_EQ(final_shape[cdim], st_shape[cdim])
          << "All SparseTensors' shapes must match except on the concat dim.  "
          << "Concat dim: " << primary_dim
          << ", mismatched shape at dim: " << cdim
          << ".  Expecting shape like: [" << str_util::Join(final_shape, ",")
          << "] but saw shape: [" << str_util::Join(st_shape, ",") << "]";
    }

    // Update dimension of final shape
    final_shape[primary_dim] =
        (final_shape[primary_dim] + st_shape[primary_dim]);

    num_entries += st.num_entries();  // Update number of entries
  }

  // If nonconsistent ordering among inputs, set final order to -1s.
  if (!fully_ordered) {
    final_order = UndefinedOrder(final_shape);
  }

  Tensor output_ix(DT_INT64, TensorShape({num_entries, dims}));
  Tensor output_vals(DataTypeToEnum<T>::v(), TensorShape({num_entries}));

  TTypes<int64>::Matrix ix_t = output_ix.matrix<int64>();
  typename TTypes<T>::Vec vals_t = output_vals.vec<T>();

  Eigen::DenseIndex offset = 0;
  int64 shape_offset = 0;
  for (const SparseTensor& st : tensors) {
    const int st_num_entries = st.num_entries();

    // Fill in indices & values.
    std::copy_n(&st.vals_.vec<T>()(0), st_num_entries, &vals_t(offset));

    const auto* st_ix = &st.ix_.matrix<int64>()(0, 0);
    auto* ix_out = &ix_t(offset, 0);
    for (std::size_t i = 0; i < st_num_entries * dims; ++i) {
      *ix_out++ = *st_ix++ + ((i % dims == primary_dim) ? shape_offset : 0);
    }

    offset += st_num_entries;
    shape_offset += st.shape()[primary_dim];
  }

  return SparseTensor(output_ix, output_vals, final_shape, final_order);
}

template <typename T>
Status SparseTensor::Split(const SparseTensor& input_tensor,
                           const int split_dim, const int num_split,
                           std::vector<SparseTensor>* result) {
  std::vector<Tensor> output_indices;
  std::vector<Tensor> output_values;
  std::vector<TensorShape> output_shapes;
  output_indices.reserve(num_split);
  output_values.reserve(num_split);
  output_shapes.reserve(num_split);

  std::vector<typename TTypes<int64>::Matrix> output_indices_t;
  std::vector<typename TTypes<T>::Vec> output_values_t;
  output_indices_t.reserve(num_split);
  output_values_t.reserve(num_split);
  auto input_values_t = input_tensor.values().vec<T>();
  auto input_indices_t = input_tensor.indices().matrix<int64>();

  std::vector<int> num_values(num_split, 0);
  const int num_dim = input_tensor.shape().size();
  const int split_dim_size = input_tensor.shape()[split_dim];
  const int split_size = split_dim_size / num_split;

  if (!(num_split > 0 && num_split <= split_dim_size)) {
    return Status(error::INVALID_ARGUMENT,
                  strings::StrCat("num_split must be in the interval (0, ",
                                  split_dim_size, "]"));
  }
  if (!(split_dim >= 0 && split_dim < num_dim)) {
    return Status(
        error::INVALID_ARGUMENT,
        strings::StrCat("num_dim must be in the interval [0, ", num_dim, ")"));
  }

  const int residual = split_dim_size % num_split;
  for (int i = 0; i < input_tensor.indices().dim_size(0); ++i) {
    const int dim = input_tensor.indices().matrix<int64>()(i, split_dim);
    int slice_index = GetSliceIndex(dim, split_size, residual);
    num_values[slice_index]++;
  }

  for (int i = 0; i < num_split; ++i) {
    // TODO(ataei): Pass an allocator to avoid allocating large memory buffer.
    output_indices.emplace_back(DT_INT64,
                                TensorShape({num_values[i], num_dim}));
    output_values.emplace_back(DataTypeToEnum<T>::v(),
                               TensorShape({num_values[i]}));
    output_shapes.emplace_back(input_tensor.shape());
    output_indices_t.emplace_back(output_indices[i].matrix<int64>());
    output_values_t.emplace_back(output_values[i].vec<T>());
    const int size = GetSliceShape(i, split_size, residual);
    output_shapes[i].set_dim(split_dim, size);
  }

  std::vector<int> values_inserted_in_slice(num_split, 0);
  for (int i = 0; i < input_tensor.indices().dim_size(0); ++i) {
    const int dim = input_indices_t(i, split_dim);
    const int slice_index = GetSliceIndex(dim, split_size, residual);
    const int slice_dim = values_inserted_in_slice[slice_index]++;
    output_values_t[slice_index](slice_dim) = input_values_t(i);
    for (int j = 0; j < num_dim; ++j) {
      const int64 original_dim = input_indices_t(i, j);
      output_indices_t[slice_index](slice_dim, j) =
          (j == split_dim)
              ? GetDimensionInSlice(original_dim, split_size, residual)
              : original_dim;
    }
  }

  result->clear();
  result->reserve(num_split);
  for (int i = 0; i < num_split; ++i) {
    SparseTensor tensor;
    Status create_status =
        Create(output_indices[i], output_values[i], output_shapes[i], &tensor);
    if (!create_status.ok()) {
      return create_status;
    }
    result->push_back(std::move(tensor));
  }
  return Status::OK();
}

template <typename T>
SparseTensor SparseTensor::Slice(const SparseTensor& input_tensor,
                                 const gtl::ArraySlice<int64>& start,
                                 const gtl::ArraySlice<int64>& size) {
  TensorShape output_shape(input_tensor.shape());

  const int dims = input_tensor.dims();
  for (int dim = 0; dim < dims; dim++) {
    int64 dim_size = start[dim] + size[dim] < output_shape.dim_size(dim)
                         ? size[dim]
                         : output_shape.dim_size(dim) - start[dim];
    output_shape.set_dim(dim, dim_size);
  }

  auto input_indices_t = input_tensor.indices().matrix<int64>();
  auto input_values_t = input_tensor.values().vec<T>();

  // Find the number of indices that fall inside start and size.
  int count = 0;
  for (int i = 0; i < input_tensor.indices().dim_size(0); i++) {
    // The following will check to see if an input is within the
    // range specified by start and size.
    // The for loop below iterates through all dimensions. In case
    // the index falls outside of the start and size at any dimension,
    // it will be considered as a "no hit" (hit = false). In this
    // case, it will not be counted as the index that fall inside
    // the range specified by start and size.
    bool hit = true;
    for (int dim = 0; dim < dims; dim++) {
      if (!(start[dim] <= input_indices_t(i, dim) &&
            input_indices_t(i, dim) < start[dim] + size[dim])) {
        hit = false;
        break;
      }
    }
    if (!hit) {
      continue;
    }
    count++;
  }

  Tensor output_values(DataTypeToEnum<T>::v(), TensorShape({count}));
  Tensor output_indices(DT_INT64, TensorShape({count, dims}));

  auto output_values_t = output_values.vec<T>();
  auto output_indices_t = output_indices.matrix<int64>();

  // Obtain the output indices that fall inside start and size.
  int index = 0;
  for (int i = 0; i < input_tensor.indices().dim_size(0) && index < count;
       i++) {
    // The logic here is similar as the above except that the above
    // only count the number of indices while here we actually generate
    // the output.
    bool hit = true;
    for (int dim = 0; dim < dims; dim++) {
      if (!(start[dim] <= input_indices_t(i, dim) &&
            input_indices_t(i, dim) < start[dim] + size[dim])) {
        hit = false;
        break;
      }
    }
    if (!hit) {
      continue;
    }
    output_values_t(index) = input_values_t(i);
    for (int dim = 0; dim < dims; dim++) {
      output_indices_t(index, dim) = input_indices_t(i, dim) - start[dim];
    }
    index++;
  }

  return SparseTensor(output_indices, output_values, output_shape);
}

}  // namespace sparse
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_UTIL_SPARSE_SPARSE_TENSOR_H_
