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

// See docs in ../ops/array_ops.cc.

#include "tensorflow/core/framework/bounds_check.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/variant.h"
#include "tensorflow/core/framework/variant_encode_decode.h"
#include "tensorflow/core/kernels/gather_functor.h"
#include "tensorflow/core/kernels/gather_functor_batched.h"
#include "tensorflow/core/platform/mem.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/util.h"

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;
typedef Eigen::DenseIndex IndexType;

template <typename Device, typename T, typename Index>
class GatherOp : public OpKernel {
 public:
  //   QUESTION: It'd be nice to support DT_INT16, DT_UINT8,
  //   etc. here for the type of the second input argument.  Should
  //   we have the framework do some sort of integer promotion
  //   automatically, or should that be something that users have to
  //   do explicitly with a conversion operator in the graph?
  explicit GatherOp(OpKernelConstruction* c) : OpKernel(c) {
    // Set batch_dims_ to 0 if the attribute does not exist.
    if (c->HasAttr("batch_dims")) {
      OP_REQUIRES_OK(c, c->GetAttr("batch_dims", &batch_dims_));
    } else {
      batch_dims_ = 0;
    }
  }

  void Compute(OpKernelContext* c) override {
    const Tensor& params = c->input(0);
    const Tensor& indices = c->input(1);
    OP_REQUIRES(
        c, TensorShapeUtils::IsVectorOrHigher(params.shape()),
        errors::InvalidArgument("params must be at least 1 dimensional"));

    // GatherV2 added an axis argument. For backwards compatibility with Gather,
    // fall back to axis 0 if the op does not have an axis input.
    int64 axis = 0;
    bool axis_is_set = false;  // Indicates whether the axis argument was set.
    if (c->num_inputs() == 3) {
      axis_is_set = true;
      const Tensor& axis_tensor = c->input(2);
      OP_REQUIRES(c, TensorShapeUtils::IsScalar(axis_tensor.shape()),
                  errors::InvalidArgument("axis must be scalar"));

      if (axis_tensor.dtype() == DT_INT32) {
        axis = axis_tensor.scalar<int32>()();
      } else if (axis_tensor.dtype() == DT_INT64) {
        axis = axis_tensor.scalar<int64>()();
      } else {
        OP_REQUIRES(c, false,
                    errors::InvalidArgument("axis must be int32 or int64."));
      }
    }

    int64 min_params_dim = axis < 0 ? -axis : axis + 1;
    OP_REQUIRES(
        c, params.dims() >= min_params_dim,
        errors::InvalidArgument("Shape must be at least rank ", min_params_dim,
                                " but is rank ", params.dims()));

    if (axis < 0) {
      axis = params.dims() + axis;
    }

    // Modify only a local copy of batch_dims_.
    int32 batch_dims = batch_dims_;
    if (batch_dims != 0) {
      OP_REQUIRES(c,
                  batch_dims >= -indices.dims() && batch_dims <= indices.dims(),
                  errors::InvalidArgument("Expected batch_dims in the range [",
                                          -indices.dims(), ", ", indices.dims(),
                                          "], but got ", batch_dims));

      if (batch_dims < 0) {
        batch_dims = indices.dims() + batch_dims;
      }

      if (!axis_is_set) axis = batch_dims;

      OP_REQUIRES(c, batch_dims < params.dims(),
                  errors::InvalidArgument("batch_dims (", batch_dims,
                                          ") must be less than rank(params) (",
                                          params.dims(), ")."));

      OP_REQUIRES(c, axis >= batch_dims,
                  errors::InvalidArgument("batch_dims (", batch_dims,
                                          ") must be less than or equal to ",
                                          "axis (", axis, ")."));
      for (int i = 0; i < batch_dims; ++i) {
        OP_REQUIRES(c, params.dim_size(i) == indices.dim_size(i),
                    errors::InvalidArgument(
                        "params.shape[", i, "]: ", params.dim_size(i),
                        " should be equal to indices.shape[", i,
                        "]: ", indices.dim_size(i)));
      }
    }

    // Check that we have enough index space
    int64 gather_dim_size = params.dim_size(axis);
    const int64 N = indices.NumElements();
    OP_REQUIRES(
        c, gather_dim_size <= std::numeric_limits<Index>::max(),
        errors::InvalidArgument("params.shape[", axis, "] too large for ",
                                DataTypeString(DataTypeToEnum<Index>::v()),
                                " indexing: ", gather_dim_size, " > ",
                                std::numeric_limits<Index>::max()));

    // The result shape is params.shape[:axis] + indices.shape[batch_dims:] +
    // params.shape[axis + 1:].
    TensorShape result_shape;
    int64 batch_size = 1;
    int64 outer_size = 1;
    int64 inner_size = 1;

    for (int i = 0; i < batch_dims; ++i) {
      result_shape.AddDim(params.dim_size(i));
      batch_size *= params.dim_size(i);
    }
    for (int i = batch_dims; i < axis; ++i) {
      result_shape.AddDim(params.dim_size(i));
      outer_size *= params.dim_size(i);
    }
    for (int i = batch_dims; i < indices.dims(); ++i) {
      result_shape.AddDim(indices.dim_size(i));
    }
    for (int i = axis + 1; i < params.dims(); ++i) {
      result_shape.AddDim(params.dim_size(i));
      inner_size *= params.dim_size(i);
    }

    Tensor* out = nullptr;
    OP_REQUIRES_OK(c, c->allocate_output(0, result_shape, &out));
    if (N == 0) return;
    if (inner_size == 0) return;

    int64 bad_i = -1;
    auto indices_flat = indices.flat<Index>();
    if (batch_dims > 0) {
      auto params_flat = params.shaped<T, 4>(
          {batch_size, outer_size, gather_dim_size, inner_size});
      auto out_flat = out->shaped<T, 4>(
          {batch_size, outer_size, N / batch_size, inner_size});

      functor::GatherFunctorBatched<Device, T, Index> functor;
      bad_i = functor(c, params_flat, indices_flat, out_flat);
    } else {
      auto params_flat =
          params.shaped<T, 3>({outer_size, gather_dim_size, inner_size});
      auto out_flat = out->shaped<T, 3>({outer_size, N, inner_size});

      functor::GatherFunctor<Device, T, Index> functor;
      bad_i = functor(c, params_flat, indices_flat, out_flat);
    }
    OP_REQUIRES(
        c, bad_i < 0,
        errors::InvalidArgument(
            "indices", SliceDebugString(indices.shape(), bad_i), " = ",
            indices_flat(bad_i), " is not in [0, ", gather_dim_size, ")"));
  }

 private:
  // The number of batch dimensions, as passed in the batch_dims attribute.
  // It must be less than or equal to rank(indices).
  int32 batch_dims_ = 0;
};

#define REGISTER_GATHER_FULL(dev, type, index_type)                    \
  REGISTER_KERNEL_BUILDER(Name("Gather")                               \
                              .Device(DEVICE_##dev)                    \
                              .TypeConstraint<type>("Tparams")         \
                              .TypeConstraint<index_type>("Tindices"), \
                          GatherOp<dev##Device, type, index_type>);    \
  REGISTER_KERNEL_BUILDER(Name("GatherV2")                             \
                              .Device(DEVICE_##dev)                    \
                              .TypeConstraint<type>("Tparams")         \
                              .TypeConstraint<index_type>("Tindices")  \
                              .HostMemory("axis"),                     \
                          GatherOp<dev##Device, type, index_type>)

#define REGISTER_GATHER_ALL_INDICES(dev, type) \
  REGISTER_GATHER_FULL(dev, type, int32);      \
  REGISTER_GATHER_FULL(dev, type, int64)

#define REGISTER_GATHER_CPU(type) REGISTER_GATHER_ALL_INDICES(CPU, type)

// Registration of the CPU implementations.
TF_CALL_ALL_TYPES(REGISTER_GATHER_CPU);
TF_CALL_QUANTIZED_TYPES(REGISTER_GATHER_CPU);
TF_CALL_quint16(REGISTER_GATHER_CPU);
TF_CALL_qint16(REGISTER_GATHER_CPU);

#undef REGISTER_GATHER_CPU

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM

// Registration of the GPU implementations.
#define REGISTER_GATHER_GPU(type) REGISTER_GATHER_ALL_INDICES(GPU, type)

TF_CALL_int32(REGISTER_GATHER_GPU);
TF_CALL_int64(REGISTER_GATHER_GPU);
TF_CALL_GPU_ALL_TYPES(REGISTER_GATHER_GPU);

#undef REGISTER_GATHER_GPU

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#undef REGISTER_GATHER_ALL_INDICES
#undef REGISTER_GATHER_FULL

}  // namespace tensorflow
