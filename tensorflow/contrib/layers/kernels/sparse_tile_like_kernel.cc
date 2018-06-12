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

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_util.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/util/sparse/sparse_tensor.h"

using tensorflow::sparse::SparseTensor;

namespace tensorflow {

struct ReduceDetails {
  // The dimensions to call Reorder() with.
  std::vector<int64> reorder_dims;
  // The dimensions to call group() with after Reorder().
  std::vector<int64> group_by_dims;
  // The shape after reduction.
  TensorShape reduced_shape;
};

ReduceDetails SparseTensorReduceHelper(const SparseTensor& sp,
                                       gtl::ArraySlice<int32> axes_slice) {
  ReduceDetails reduction;

  std::vector<int32> reduction_axes(axes_slice.begin(), axes_slice.end());
  int ndims = sp.dims();
  for (int64 i = 0; i < reduction_axes.size(); ++i) {
    reduction_axes[i] = (reduction_axes[i] + ndims) % ndims;
  }
  std::sort(reduction_axes.begin(), reduction_axes.end());

  // (0) Calculate the grouping dimensions:
  // group_by_dims == {0, .., NDIMS-1} \ reduction_axes.
  std::vector<int64> perm(ndims);
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
  std::vector<int64> out_dim_sizes;
  out_dim_sizes = sp.PickDims(reduction.group_by_dims);

  reduction.reduced_shape = TensorShape(out_dim_sizes);
  return reduction;
}

std::vector<int64> GetRowOfTensor(const Tensor t, const int64 i) {
  std::vector<int64> res;
  auto t_values = t.matrix<int64>();
  for (int64 j = 0; j < t.shape().dim_size(1); j++) {
    res.push_back(t_values(i, j));
  }
  return res;
}

// This operator is used for tiling a to a new SparseTensor like b.
template <typename T>
class SparseTileLikeOp : public OpKernel {
 public:
  explicit SparseTileLikeOp(OpKernelConstruction* context)
      : OpKernel(context) {}

  void Compute(OpKernelContext* ctx) override {
    // Define the tensors.
    const Tensor *a_indices_t, *a_values_t, *a_shape_t, *b_indices_t,
        *b_values_t, *b_shape_t, *axes_t;

    OP_REQUIRES_OK(ctx, ctx->input("a_input_indices", &a_indices_t));
    OP_REQUIRES_OK(ctx, ctx->input("a_input_values", &a_values_t));
    OP_REQUIRES_OK(ctx, ctx->input("a_input_shape", &a_shape_t));
    OP_REQUIRES_OK(ctx, ctx->input("b_input_indices", &b_indices_t));
    OP_REQUIRES_OK(ctx, ctx->input("b_input_values", &b_values_t));
    OP_REQUIRES_OK(ctx, ctx->input("b_input_shape", &b_shape_t));
    OP_REQUIRES_OK(ctx, ctx->input("axes", &axes_t));

    ValidateInput(ctx, a_indices_t, a_values_t, a_shape_t, b_indices_t,
                  b_values_t, b_shape_t, axes_t);

    // Set values for out_values.
    std::vector<int64> group_axes;
    std::vector<int64> reduction_a;
    auto axes = axes_t->flat<int32>();
    int64 a = 0;
    for (int32 i = 0; i < b_shape_t->dim_size(0); i++) {
      if (i != axes(0)) {
        group_axes.push_back(i);
        reduction_a.push_back(a);
        a++;
      }
    }

    SparseTensor sp_b(tensor::DeepCopy(*b_indices_t),
                      tensor::DeepCopy(*b_values_t),
                      TensorShape(b_shape_t->vec<int64>()));
    ReduceDetails reduction =
        SparseTensorReduceHelper(sp_b, axes_t->flat<int32>());
    sp_b.Reorder<T>(reduction.reorder_dims);

    SparseTensor sp_a(tensor::DeepCopy(*a_indices_t),
                      tensor::DeepCopy(*a_values_t),
                      TensorShape(a_shape_t->vec<int64>()));
    sp_a.Reorder<T>(reduction_a);

    std::vector<T> out_values_vec;
    int64 h = 0;
    int64 output_shape0 = 0;
    int64 tmp_count = 0;
    std::vector<int64> output_ids;
    auto a_values = a_values_t->vec<T>();

    for (const auto& g : sp_b.group(group_axes)) {
      std::vector<int64> group = g.group();
      std::vector<int64> row = GetRowOfTensor(sp_a.indices(), h);
      int64 g_indice_len = g.values<T>().dimension(0);
      while (row < group) {
        if (h > a_values.size()) break;
        h++;
        row = GetRowOfTensor(sp_a.indices(), h);
      }
      if (row == group) {
        auto s = a_values(h);
        for (int64 j = 0; j < g_indice_len; j++) {
          out_values_vec.push_back(s);
          output_ids.push_back(tmp_count + j);
        }
        output_shape0 = output_shape0 + g_indice_len;
      }
      tmp_count = tmp_count + g_indice_len;
    }

    // Allocate output indices first.
    Tensor* output_indices_t;
    auto b_indices = sp_b.indices().matrix<int64>();
    auto b_indices_shape = b_indices.dimensions();
    OP_REQUIRES_OK(ctx, ctx->allocate_output(
                            0, TensorShape({output_shape0, b_indices_shape[1]}),
                            &output_indices_t));

    auto output_indices = output_indices_t->matrix<int64>();

    for (int64 j = 0; j < output_shape0; j++) {
      output_indices.chip<0>(j) = b_indices.chip<0>(output_ids[j]);
    }

    // Allocate output values.
    Tensor* out_values_t;
    const auto output_values_shape = TensorShape({output_shape0});
    OP_REQUIRES_OK(ctx,
                   ctx->allocate_output(1, output_values_shape, &out_values_t));
    auto out_flat = out_values_t->flat<T>();
    for (int64 j = 0; j < output_shape0; j++) {
      out_flat(j) = out_values_vec[j];
    }

    // Allocate output shape.
    ctx->set_output(2, *b_shape_t);
  }

 private:
  void ValidateInput(OpKernelContext* context, const Tensor* a_indices_t,
                     const Tensor* a_values_t, const Tensor* a_shape_t,
                     const Tensor* b_indices_t, const Tensor* b_values_t,
                     const Tensor* b_shape_t, const Tensor* axes_t) {
    OP_REQUIRES(context, TensorShapeUtils::IsMatrix(a_indices_t->shape()),
                errors::InvalidArgument(
                    "Input indices should be a matrix but received shape ",
                    a_indices_t->shape().DebugString(), " at position ", 0));
    OP_REQUIRES(context, TensorShapeUtils::IsVector(a_values_t->shape()),
                errors::InvalidArgument(
                    "Input values should be a std::vector but received shape ",
                    a_values_t->shape().DebugString(), " at position ", 1));
    OP_REQUIRES(context, TensorShapeUtils::IsVector(a_shape_t->shape()),
                errors::InvalidArgument(
                    "Input shapes should be a std::vector but received shape ",
                    a_shape_t->shape().DebugString(), " at position ", 2));

    OP_REQUIRES(context, TensorShapeUtils::IsMatrix(b_indices_t->shape()),
                errors::InvalidArgument(
                    "Input indices should be a matrix but received shape ",
                    b_indices_t->shape().DebugString(), " at position ", 3));
    OP_REQUIRES(context, TensorShapeUtils::IsVector(b_values_t->shape()),
                errors::InvalidArgument(
                    "Input values should be a std::vector but received shape ",
                    b_values_t->shape().DebugString(), " at position ", 4));
    OP_REQUIRES(context, TensorShapeUtils::IsVector(b_shape_t->shape()),
                errors::InvalidArgument(
                    "Input shapes should be a std::vector but received shape ",
                    b_shape_t->shape().DebugString(), " at position ", 5));
    OP_REQUIRES(
        context,
        b_shape_t->vec<int64>().size() == a_shape_t->vec<int64>().size() + 1,
        errors::InvalidArgument(
            "shape of tensor a should be one dim less than tensor b, but got ",
            a_shape_t->shape().DebugString(), " and ",
            b_shape_t->shape().DebugString()));
  }

};  

#define REGISTER_KERNELS(T)                                             \
  REGISTER_KERNEL_BUILDER(                                              \
      Name("SparseTileLike").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      SparseTileLikeOp<T>)
TF_CALL_REAL_NUMBER_TYPES(REGISTER_KERNELS);
#undef REGISTER_KERNELS

}  
