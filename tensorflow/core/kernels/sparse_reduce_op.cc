/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

// See docs in ../ops/sparse_ops.cc.

#define EIGEN_USE_THREADS

#include "absl/status/status.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/op_requires.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_util.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/util/sparse/sparse_tensor.h"

// TODO(b/31496047): Fix non-standard include order.
#include <numeric>  // clang-format off

using tensorflow::sparse::SparseTensor;
using tensorflow::gtl::ArraySlice;

namespace tensorflow {

struct ReduceDetails {
  // The dimensions to call Reorder() with.
  std::vector<int64_t> reorder_dims;

  // The dimensions to call group() with after Reorder().
  std::vector<int64_t> group_by_dims;

  // The shape after reduction.
  TensorShape reduced_shape;
};

// Compute common reduce parameters that'll be used for SparseTensor
// reductions. Usage:
// StatusOr<ReduceDetails> reduction =
//     SparseTensorReduceHelper(sp, axes, keep_dims);
// sp.Reorder(reduction->reorder_dims);
// for (const auto& g : sp.group(reduction->group_by_dims)) {
//   ...
// }
// // Set output shape to reduction->reduced_shape.
absl::StatusOr<ReduceDetails> SparseTensorReduceHelper(const SparseTensor &sp,
                                       absl::Span<const int32> axes_slice,
                                       bool keep_dims) {
  ReduceDetails reduction;

  std::vector<int32> reduction_axes(axes_slice.begin(), axes_slice.end());
  int ndims = sp.dims();
  for (int64_t i = 0; i < reduction_axes.size(); ++i) {
    reduction_axes[i] = (reduction_axes[i] + ndims) % ndims;
  }
  std::sort(reduction_axes.begin(), reduction_axes.end());

  // (0) Calculate the grouping dimensions:
  // group_by_dims == {0, .., NDIMS-1} \ reduction_axes.
  std::vector<int64_t> perm(ndims);
  std::iota(perm.begin(), perm.end(), 0);

  // Requires perm and reduction_axes_ be sorted; group_by_dims will be
  // sorted as well.
  std::set_difference(
      perm.begin(), perm.end(), reduction_axes.begin(), reduction_axes.end(),
      std::inserter(reduction.group_by_dims, reduction.group_by_dims.begin()));

  // Now append the rest of the axes (the complement of group_by_dims_);
  // result is used by Reorder().
  reduction.reorder_dims = reduction.group_by_dims;
  std::set_difference(perm.begin(), perm.end(), reduction.group_by_dims.begin(),
                      reduction.group_by_dims.end(),
                      std::back_inserter(reduction.reorder_dims));

  // (1) Calculate the shape after reduction.
  auto sp_shape = sp.shape();
  std::vector<int64_t> out_dim_sizes;
  if (keep_dims) {
    out_dim_sizes.reserve(ndims);
    auto beg = reduction.group_by_dims.begin();
    auto end = reduction.group_by_dims.end();
    for (int d = 0; d < ndims; ++d) {
      if (std::find(beg, end, d) == end) {
        out_dim_sizes.push_back(1);  // A reduced axis.
      } else {
        out_dim_sizes.push_back(sp_shape[d]);
      }
    }
  } else {
    out_dim_sizes = sp.PickDims(reduction.group_by_dims);
  }

  absl::Status success =
      TensorShape::BuildTensorShape(out_dim_sizes, &reduction.reduced_shape);
  if (!success.ok()) {
    return success;
  }
  return reduction;
}

absl::Status ValidateInputs(const Tensor *shape_t, const Tensor *reduction_axes_t) {
  // indices and values are validated in SparseTensor ctor.
  if (!TensorShapeUtils::IsVector(shape_t->shape())) {
    return errors::InvalidArgument(
        "Expected input_shape to be a vector; got shape: ",
        shape_t->shape().DebugString());
  }
  if (!TensorShapeUtils::IsScalar(reduction_axes_t->shape()) &&
      !TensorShapeUtils::IsVector(reduction_axes_t->shape())) {
    return errors::InvalidArgument(
        "Expected reduction_axes to be a scalar or a vector; got shape: ",
        reduction_axes_t->shape().DebugString());
  }

  const auto reduction_axes_flat = reduction_axes_t->flat<int32>();
  for (int64_t i = 0; i < reduction_axes_flat.size(); i++) {
    int32_t axis = reduction_axes_flat(i);
    if (axis < -shape_t->NumElements() || axis >= shape_t->NumElements()) {
      return errors::InvalidArgument("Invalid reduction dimension ", axis,
                                     ", for input with ",
                                     shape_t->NumElements(), " dimensions.");
    }
  }

  return absl::OkStatus();
}

struct SumOp {
  template <typename T>
  static void Run(OpKernelContext *ctx, typename TTypes<T>::Scalar &s, const typename TTypes<T>::UnalignedVec &v) {
      s.device(ctx->eigen_cpu_device()) = v.sum();
  }
  static absl::string_view Name() {
      return "sum";
  }
};

struct MaxOp {
  template <typename T>
  static void Run(OpKernelContext *ctx, typename TTypes<T>::Scalar &s, const typename TTypes<T>::UnalignedVec &v) {
      s.device(ctx->eigen_cpu_device()) = v.maximum();
  }
  static absl::string_view Name() {
      return "max";
  }
};

template <typename T, typename Op>
class SparseReduceOp : public OpKernel {
 public:
  explicit SparseReduceOp(OpKernelConstruction *ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("keep_dims", &keep_dims_));
  }

  void Compute(OpKernelContext *ctx) override {
    const Tensor *indices_t, *values_t, *shape_t, *reduction_axes_t;
    OP_REQUIRES_OK(ctx, ctx->input("input_indices", &indices_t));
    OP_REQUIRES_OK(ctx, ctx->input("input_values", &values_t));
    OP_REQUIRES_OK(ctx, ctx->input("input_shape", &shape_t));
    OP_REQUIRES_OK(ctx, ctx->input("reduction_axes", &reduction_axes_t));

    OP_REQUIRES_OK(ctx, ValidateInputs(shape_t, reduction_axes_t));

    // TODO(zongheng): we will call Reorder() below, which will modify
    // in-place the underlying indices and values buffers.  To avoid
    // surprises of this kernel being stateful, we work around the above by
    // making deep copies here.  Remove this if/when we change Reorder()'s
    // semantics.
    const auto shape_vec = shape_t->vec<int64_t>();
    TensorShape shape;
    OP_REQUIRES_OK(ctx, TensorShape::BuildTensorShape(shape_vec, &shape));

    SparseTensor sp;
    OP_REQUIRES_OK(ctx, SparseTensor::Create(
        tensor::DeepCopy(*indices_t), tensor::DeepCopy(*values_t),
                    shape, &sp));
    absl::StatusOr<ReduceDetails> reduction_or = SparseTensorReduceHelper(
        sp, reduction_axes_t->flat<int32>(), keep_dims_);
    OP_REQUIRES_OK(ctx, reduction_or.status());
    ReduceDetails reduction = *reduction_or;

    Tensor *out_values;
    OP_REQUIRES_OK(
        ctx, ctx->allocate_output(0, reduction.reduced_shape, &out_values));
    auto out_flat = out_values->flat<T>();
    out_flat.setZero();

    Tensor tmp_reduced_val;
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(DataTypeToEnum<T>::value,
                                           TensorShape({}), &tmp_reduced_val));
    auto reduced_val = tmp_reduced_val.scalar<T>();

    // Compute strides, and use it to convert coords to flat index.  The
    // coordinates returned by .group() have the same ndims as group_by_dims.
    absl::InlinedVector<int64_t, 8UL> output_strides(reduction.group_by_dims.size());
    if (!output_strides.empty()) {  // Do this iff we don't reduce all.
      output_strides.back() = 1;
      for (int d = output_strides.size() - 2; d >= 0; --d) {
        output_strides[d] =
            output_strides[d + 1] * shape_vec(reduction.group_by_dims[d + 1]);
      }
    }

    auto CoordinatesToFlatIndex = [](absl::Span<const int64_t> coords,
                                     absl::Span<const int64_t> strides) -> int64 {
      if (strides.empty()) {  // Reduce all.
        return 0;
      }
      CHECK_EQ(coords.size(), strides.size());
      int64_t idx = 0;
      for (int i = 0; i < coords.size(); ++i) {
        idx += coords[i] * strides[i];
      }
      return idx;
    };

    // Each group maps one-on-one onto a value in the reduced tensor.
    // g.group() provides the coordinates of a particular reduced value.
    sp.Reorder<T>(reduction.reorder_dims);
    for (const auto &g : sp.group(reduction.group_by_dims)) {
      Op::template Run<T>(ctx, reduced_val, g.template values<T>());
      OP_REQUIRES(ctx,
                  output_strides.empty() ||
                  (g.group().size() == output_strides.size()),
                  errors::Internal(
                      "Expected group size and output_strides size to match",
                      ", but got ", g.group().size(), " and ",
                      output_strides.size()));
      const int64_t idx = CoordinatesToFlatIndex(g.group(), output_strides);
      OP_REQUIRES(ctx,
                  idx >= 0 && idx < out_flat.size(),
                  errors::Internal(
                      "Obtained a write index of ", idx,
                      " which is outside of bounds of [0, ",
                      out_flat.size(), ")"));
      out_flat(idx) = reduced_val();
      VLOG(2) << "coords: " << absl::StrJoin(g.group(), ",")
              << "; idx: " << idx << "; group " << Op::Name() << ": "
              << reduced_val();
    }
  }

 private:
  // True if the number of dimensions should be maintained.
  bool keep_dims_;
};

#define REGISTER_KERNELS(T)                                              \
  REGISTER_KERNEL_BUILDER(                                               \
      Name("SparseReduceSum").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      SparseReduceOp<T, SumOp>)
TF_CALL_NUMBER_TYPES(REGISTER_KERNELS);
#undef REGISTER_KERNELS

#define REGISTER_KERNELS(T)                                              \
  REGISTER_KERNEL_BUILDER(                                               \
      Name("SparseReduceMax").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      SparseReduceOp<T, MaxOp>)
TF_CALL_REAL_NUMBER_TYPES(REGISTER_KERNELS);
#undef REGISTER_KERNELS

template <typename T, typename Op>
class SparseReduceSparseOp : public OpKernel {
 public:
  explicit SparseReduceSparseOp(OpKernelConstruction *ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("keep_dims", &keep_dims_));
  }

  void Compute(OpKernelContext *ctx) override {
    const Tensor *indices_t, *values_t, *shape_t, *reduction_axes_t;
    OP_REQUIRES_OK(ctx, ctx->input("input_indices", &indices_t));
    OP_REQUIRES_OK(ctx, ctx->input("input_values", &values_t));
    OP_REQUIRES_OK(ctx, ctx->input("input_shape", &shape_t));
    OP_REQUIRES_OK(ctx, ctx->input("reduction_axes", &reduction_axes_t));

    OP_REQUIRES_OK(ctx, ValidateInputs(shape_t, reduction_axes_t));

    TensorShape shape;
    OP_REQUIRES_OK(ctx, TensorShape::BuildTensorShape(shape_t->vec<int64_t>(),
                                                      &shape));
    SparseTensor sp;
    OP_REQUIRES_OK(ctx, SparseTensor::Create(tensor::DeepCopy(*indices_t),
                                         tensor::DeepCopy(*values_t),
                    shape, &sp));
    absl::StatusOr<ReduceDetails> reduction_or = SparseTensorReduceHelper(
        sp, reduction_axes_t->flat<int32>(), keep_dims_);
    OP_REQUIRES_OK(ctx, reduction_or.status());
    ReduceDetails reduction = *reduction_or;

    sp.Reorder<T>(reduction.reorder_dims);
    // Count nnzs in the output SparseTensor.
    int64_t nnz = 0;
    auto iter = sp.group(reduction.group_by_dims);
    for (auto it = iter.begin(); it != iter.end(); ++it) {
      nnz++;
    }

    Tensor *out_indices_t;
    OP_REQUIRES_OK(ctx,
                   ctx->allocate_output(
                       0, TensorShape({nnz, reduction.reduced_shape.dims()}),
                       &out_indices_t));
    typename TTypes<int64_t>::Matrix out_indices_mat =
        out_indices_t->matrix<int64_t>();
    // For keep_dims. We don't explicitly set dim fields for reduced dims below.
    out_indices_mat.setZero();

    Tensor *out_values_t;
    OP_REQUIRES_OK(ctx,
                   ctx->allocate_output(1, TensorShape({nnz}), &out_values_t));
    auto out_flat = out_values_t->flat<T>();

    Tensor tmp_reduced_val;
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(DataTypeToEnum<T>::value,
                                           TensorShape({}), &tmp_reduced_val));
    auto reduced_val = tmp_reduced_val.scalar<T>();
    int64_t i = 0;
    for (const auto &g : sp.group(reduction.group_by_dims)) {
      Op::template Run<T>(ctx, reduced_val, g.template values<T>());
      std::vector<int64_t> group = g.group();
      for (int64_t j = 0; j < group.size(); j++) {
        if (keep_dims_) {
          out_indices_mat(i, reduction.group_by_dims[j]) = group[j];
        } else {
          out_indices_mat(i, j) = group[j];
        }
      }
      out_flat(i) = reduced_val();
      i++;
      VLOG(2) << "coords: " << absl::StrJoin(g.group(), ",")
              << "; group " << Op::Name() << ": "
              << reduced_val();
    }

    Tensor *out_shape_t;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(
                            2, TensorShape({reduction.reduced_shape.dims()}),
                            &out_shape_t));
    auto out_shape_flat = out_shape_t->flat<int64_t>();
    auto out_dim_sizes = reduction.reduced_shape.dim_sizes();
    if (!out_dim_sizes.empty()) {
      std::copy(out_dim_sizes.begin(), out_dim_sizes.end(), &out_shape_flat(0));
    }
  }

 private:
  // True if the number of dimensions should be maintained.
  bool keep_dims_;
};

#define REGISTER_KERNELS(T)                                                    \
  REGISTER_KERNEL_BUILDER(                                                     \
      Name("SparseReduceSumSparse").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      SparseReduceSparseOp<T, SumOp>)
TF_CALL_NUMBER_TYPES(REGISTER_KERNELS);
#undef REGISTER_KERNELS

#define REGISTER_KERNELS(T)                                                    \
  REGISTER_KERNEL_BUILDER(                                                     \
      Name("SparseReduceMaxSparse").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      SparseReduceSparseOp<T, MaxOp>)
TF_CALL_REAL_NUMBER_TYPES(REGISTER_KERNELS);
#undef REGISTER_KERNELS

}  // namespace tensorflow
