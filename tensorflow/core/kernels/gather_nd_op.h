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

#ifndef TENSORFLOW_CORE_KERNELS_GATHER_ND_OP_H_
#define TENSORFLOW_CORE_KERNELS_GATHER_ND_OP_H_
// Functor definition for GatherOp, must be compilable by nvcc.

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/bounds_check.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/util.h"

namespace tsl {
class Status;
}
namespace tensorflow {
using tsl::Status;

class OpKernelContext;
class Tensor;

namespace functor {

template <typename Device, typename T, typename Index, int IXDIM>
struct GatherNdSlice {
  // Performs a slice gather op on (Tparams, Tindices), writing to Tout.
  // Returns an index to Tindices if the value at that index is out of range.
  // Returns -1 if all values of Tindices are in range.
  Index operator()(const Device& d, const Index slice_size,
                   typename TTypes<int32>::Scalar Tscratch,
                   typename TTypes<T, IXDIM + 1>::ConstTensor Tparams,
                   typename TTypes<Index>::ConstMatrix Tindices,
                   typename TTypes<T>::Matrix Tout);
};

template <typename Device, typename T, typename Index>
Status DoGatherNd(OpKernelContext* c, const Tensor& params,
                  const Tensor& indices, Tensor* out) {
  if (!TensorShapeUtils::IsVectorOrHigher(params.shape())) {
    return errors::InvalidArgument("params must be at least a vector");
  }
  if (!TensorShapeUtils::IsVectorOrHigher(indices.shape())) {
    return errors::InvalidArgument("indices must be at least a vector");
  }
  if (indices.dim_size(indices.dims() - 1) > params.dims()) {
    return errors::InvalidArgument(
        "index innermost dimension length must be <= params rank; saw: ",
        indices.dim_size(indices.dims() - 1), " vs. ", params.dims());
  }

  const TensorShape& indices_shape(indices.shape());
  const int64_t indices_nd = indices_shape.dim_size(indices_shape.dims() - 1);

  // Check that we have enough index space
  int64_t N_big = 1;
  for (int i = 0; i < indices_shape.dims() - 1; ++i) {
    N_big *= indices_shape.dim_size(i);
  }
  if (N_big > std::numeric_limits<int>::max()) {
    return errors::InvalidArgument(
        "indices has too many elements for int indexing: ", N_big, " > ",
        std::numeric_limits<int>::max());
  }
  if (params.NumElements() > std::numeric_limits<Index>::max()) {
    return errors::InvalidArgument("params.NumElements() too large for ",
                                   DataTypeString(DataTypeToEnum<Index>::v()),
                                   " indexing: ", params.NumElements(), " > ",
                                   std::numeric_limits<Index>::max());
  }

  // The result shape is
  //   indices.shape[:-1] + params.shape[indices.shape[-1]:]
  Index N_result = 1;
  for (int i = 0; i < indices_shape.dims() - 1; ++i) {
    N_result *= indices_shape.dim_size(i);
  }

  const TensorShape& params_shape(params.shape());
  Index total_nd = params_shape.dims();

  TensorShape result_shape(indices_shape);
  result_shape.RemoveLastDims(1);

  int64_t slice_size_big = 1;
  for (Index i = indices_nd; i < total_nd; ++i) {
    slice_size_big *= params_shape.dim_size(i);
    result_shape.AddDim(params_shape.dim_size(i));
  }

  if (slice_size_big > std::numeric_limits<Index>::max()) {
    return errors::InvalidArgument(
        "slice size is too large for indexing: ", slice_size_big, " > ",
        std::numeric_limits<Index>::max());
  }

  const Index slice_size = static_cast<Index>(slice_size_big);

  TF_RETURN_IF_ERROR(
      c->allocate_temp(DataTypeToEnum<T>::value, result_shape, out));

  if (N_result > 0) {
    if (params_shape.num_elements() == 0) {
      return errors::InvalidArgument(
          "Requested more than 0 entries, but "
          "params is empty.  Params shape: ",
          params_shape.DebugString());
    }

    auto indices_mat = indices.flat_inner_dims<Index>();

    Index bad_i = -1;

    // Request to copy slices / subtensors
    // Make out a matrix with the slices the col size.
    auto out_mat = out->shaped<T, 2>({N_result, slice_size});
    Tensor scratch;
    TF_RETURN_IF_ERROR(c->allocate_temp(DT_INT32, TensorShape(), &scratch));
    auto scratch_scalar = scratch.scalar<int32>();

    switch (indices_nd) {
#define PARAMS_CASE(IXDIM)                                              \
  case IXDIM: {                                                         \
    functor::GatherNdSlice<Device, T, Index, IXDIM> func;               \
    auto params_flat = params.flat_outer_dims<T, IXDIM + 1>();          \
    bad_i = func(c->eigen_device<Device>(), slice_size, scratch_scalar, \
                 params_flat, indices_mat, out_mat);                    \
  } break
      PARAMS_CASE(0);
      PARAMS_CASE(1);
      PARAMS_CASE(2);
      PARAMS_CASE(3);
      PARAMS_CASE(4);
      PARAMS_CASE(5);
      PARAMS_CASE(6);
      PARAMS_CASE(7);
#undef PARAMS_CASE
      default:
        return errors::InvalidArgument(
            "Only indices.shape[-1] values between 1 and 7 "
            "are currently supported.  Requested rank: ",
            indices_nd);
    }

    // bad_i will only return >= 0 on CPUs right now.
    if (bad_i >= 0) {
      auto shape = indices.shape();
      shape.RemoveLastDims(1);
      return errors::InvalidArgument(
          "indices", SliceDebugString(shape, bad_i), " = [",
          str_util::Join(
              gtl::ArraySlice<Index>(&indices_mat(bad_i, 0), indices_nd), ", "),
          "] does not index into param shape ", params.shape().DebugString(),
          ", node name: ", c->op_kernel().name());
    }
  }
  return OkStatus();
}

}  // namespace functor
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_GATHER_ND_OP_H_
