/* Copyright 2016 Google Inc. All Rights Reserved.

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

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_util.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/util/sparse/sparse_tensor.h"

using tensorflow::sparse::SparseTensor;
using tensorflow::gtl::ArraySlice;

namespace tensorflow {

template <typename T>
class SparseReduceSumOp : public OpKernel {
 public:
  explicit SparseReduceSumOp(OpKernelConstruction *ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("keep_dims", &keep_dims_));
  }

  void Compute(OpKernelContext *ctx) override {
    const Tensor *indices_t, *values_t, *shape_t, *reduction_axes_t;
    OP_REQUIRES_OK(ctx, ctx->input("input_indices", &indices_t));
    OP_REQUIRES_OK(ctx, ctx->input("input_values", &values_t));
    OP_REQUIRES_OK(ctx, ctx->input("input_shape", &shape_t));
    OP_REQUIRES_OK(ctx, ctx->input("reduction_axes", &reduction_axes_t));

    // indices and values are validated in SparseTensor ctor.
    OP_REQUIRES(ctx, TensorShapeUtils::IsVector(shape_t->shape()),
                errors::InvalidArgument(
                    "Expected input_shape to be a vector; got shape: ",
                    shape_t->shape().DebugString()));
    OP_REQUIRES(
        ctx, TensorShapeUtils::IsScalar(reduction_axes_t->shape()) ||
                 TensorShapeUtils::IsVector(reduction_axes_t->shape()),
        errors::InvalidArgument(
            "Expected reduction_axes to be a scalar or a vector; got shape: ",
            reduction_axes_t->shape().DebugString()));

    // TODO(zongheng): we will call Reorder() below, which will modify in-place
    // the underlying indices and values buffers.  To avoid surprises of this
    // kernel being stateful, we work around the above by making deep copies
    // here.  Remove this if/when we change Reorder()'s semantics.
    const auto shape_vec = shape_t->vec<int64>();
    SparseTensor sp(tensor::DeepCopy(*indices_t), tensor::DeepCopy(*values_t),
                    TensorShape(shape_vec));

    // Calculates group_by_dims == {0, .., NDIMS-1} \ reduction_axes.
    const int ndims = static_cast<int>(shape_t->NumElements());
    const int num_reduction_axes =
        static_cast<int>(reduction_axes_t->NumElements());

    std::vector<int32> perm(ndims);
    std::iota(perm.begin(), perm.end(), 0);

    std::vector<int32> axes(num_reduction_axes);
    std::copy_n(reduction_axes_t->flat<int32>().data(), num_reduction_axes,
                axes.begin());
    std::sort(axes.begin(), axes.end());

    std::vector<int64> group_by_dims;
    // Requires perm and axes be sorted; group_by_dims will be sorted as well.
    std::set_difference(perm.begin(), perm.end(), axes.begin(), axes.end(),
                        std::inserter(group_by_dims, group_by_dims.begin()));

    // Now append the rest of the axes (the complement of group_by_dims);
    // result is used by Reorder().
    std::vector<int64> reorder_dims(group_by_dims);
    std::set_difference(perm.begin(), perm.end(), group_by_dims.begin(),
                        group_by_dims.end(), std::back_inserter(reorder_dims));

    // Fills in the output shape and allocates the tensor.
    std::vector<int64> out_dim_sizes;
    if (keep_dims_) {
      out_dim_sizes.reserve(ndims);
      auto beg = group_by_dims.begin();
      auto end = group_by_dims.end();
      for (int d = 0; d < ndims; ++d) {
        if (std::find(beg, end, d) == end) {
          out_dim_sizes.push_back(1);  // A reduced axis.
        } else {
          out_dim_sizes.push_back(shape_vec(d));
        }
      }
    } else {
      out_dim_sizes = sp.PickDims(group_by_dims);
    }
    Tensor *out_values;
    OP_REQUIRES_OK(
        ctx, ctx->allocate_output(0, TensorShape(out_dim_sizes), &out_values));
    auto out_flat = out_values->flat<T>();
    out_flat.setZero();

    Tensor tmp_group_sum;
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(DataTypeToEnum<T>::value,
                                           TensorShape({}), &tmp_group_sum));
    auto group_sum = tmp_group_sum.scalar<T>();

    // Compute strides, and use it to convert coords to flat index.  The
    // coordinates returned by .group() have the same ndims as group_by_dims.
    gtl::InlinedVector<int64, 8> output_strides(group_by_dims.size());
    if (!output_strides.empty()) {  // Do this iff we don't reduce all.
      output_strides.back() = 1;
      for (int d = output_strides.size() - 2; d >= 0; --d) {
        output_strides[d] =
            output_strides[d + 1] * shape_vec(group_by_dims[d + 1]);
      }
    }

    auto CoordinatesToFlatIndex = [](ArraySlice<int64> coords,
                                     ArraySlice<int64> strides) {
      if (strides.empty()) {  // Reduce all.
        return 0LL;
      }
      CHECK_EQ(coords.size(), strides.size());
      int64 idx = 0;
      for (int i = 0; i < coords.size(); ++i) {
        idx += coords[i] * strides[i];
      }
      return idx;
    };

    // Each group maps one-on-one onto a value in the reduced tensor.
    // g.group() provides the coordinates of a particular reduced value.
    sp.Reorder<T>(reorder_dims);  // Necessary for .group().
    for (const auto &g : sp.group(group_by_dims)) {
      group_sum.device(ctx->eigen_cpu_device()) = g.template values<T>().sum();
      const int64 idx = CoordinatesToFlatIndex(g.group(), output_strides);
      out_flat(idx) = group_sum();
      VLOG(2) << "coords: " << str_util::Join(g.group(), ",")
              << "; idx: " << idx << "; group sum: " << group_sum();
    }
  }

 private:
  // True if the number of dimensions should be maintained.
  bool keep_dims_;
};

#define REGISTER_KERNELS(T)                                              \
  REGISTER_KERNEL_BUILDER(                                               \
      Name("SparseReduceSum").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      SparseReduceSumOp<T>)

TF_CALL_NUMBER_TYPES(REGISTER_KERNELS);

#undef REGISTER
#undef REGISTER_KERNELS

}  // namespace tensorflow
