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

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/kernels/bounds_check.h"
#include "tensorflow/core/kernels/gather_functor.h"
#include "tensorflow/core/platform/mem.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/util.h"

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

template <typename Device, typename T, typename Index>
class GatherOp : public OpKernel {
 public:
  //   QUESTION: It'd be nice to support DT_INT16, DT_UINT8,
  //   etc. here for the type of the second input argument.  Should
  //   we have the framework do some sort of integer promotion
  //   automatically, or should that be something that users have to
  //   do explicitly with a conversion operator in the graph?
  explicit GatherOp(OpKernelConstruction* c) : OpKernel(c) {}

  void Compute(OpKernelContext* c) override {
    const Tensor& params = c->input(0);
    const Tensor& indices = c->input(1);
    OP_REQUIRES(
        c, TensorShapeUtils::IsVectorOrHigher(params.shape()),
        errors::InvalidArgument("params must be at least 1 dimensional"));

    // GatherV2 added an axis argument. For backwards compatibility with Gather,
    // fall back to axis 0 if the op does not have an axis input.
    int64 axis = 0;
    if (c->num_inputs() == 3) {
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

    OP_REQUIRES(
        c, axis >= -params.dims() && axis < params.dims(),
        errors::InvalidArgument("Expected axis in the range [", -params.dims(),
                                ", ", params.dims(), "), but got ", axis));
    if (axis < 0) {
      axis = params.dims() + axis;
    }

    // Check that we have enough index space
    const int64 gather_dim_size = params.dim_size(axis);
    const int64 N = indices.NumElements();
    OP_REQUIRES(
        c, gather_dim_size <= std::numeric_limits<Index>::max(),
        errors::InvalidArgument("params.shape[", axis, "] too large for ",
                                DataTypeString(DataTypeToEnum<Index>::v()),
                                " indexing: ", gather_dim_size, " > ",
                                std::numeric_limits<Index>::max()));

    // The result shape is params.shape[0:axis] + indices.shape +
    // params.shape[axis + 1:].
    TensorShape result_shape;
    int64 outer_size = 1;
    int64 inner_size = 1;
    for (int i = 0; i < axis; i++) {
      result_shape.AddDim(params.dim_size(i));
      outer_size *= params.dim_size(i);
    }
    result_shape.AppendShape(indices.shape());
    for (int i = axis + 1; i < params.dims(); i++) {
      result_shape.AddDim(params.dim_size(i));
      inner_size *= params.dim_size(i);
    }

    Tensor* out = nullptr;
    OP_REQUIRES_OK(c, c->allocate_output(0, result_shape, &out));
    if (N > 0 && outer_size > 0 && inner_size > 0) {
      auto params_flat =
          params.shaped<T, 3>({outer_size, gather_dim_size, inner_size});
      auto indices_flat = indices.flat<Index>();
      auto out_flat = out->shaped<T, 3>({outer_size, N, inner_size});

      functor::GatherFunctor<Device, T, Index> functor;
      int64 bad_i = functor(c, params_flat, indices_flat, out_flat);

      OP_REQUIRES(
          c, bad_i < 0,
          errors::InvalidArgument(
              "indices", SliceDebugString(indices.shape(), bad_i), " = ",
              indices_flat(bad_i), " is not in [0, ", gather_dim_size, ")"));
    }
  }
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

#if GOOGLE_CUDA

// Registration of the GPU implementations.
#define REGISTER_GATHER_GPU(type) REGISTER_GATHER_ALL_INDICES(GPU, type)

TF_CALL_GPU_NUMBER_TYPES(REGISTER_GATHER_GPU);
TF_CALL_complex64(REGISTER_GATHER_GPU);
TF_CALL_complex128(REGISTER_GATHER_GPU);

#undef REGISTER_GATHER_GPU

#endif  // GOOGLE_CUDA

#undef REGISTER_GATHER_ALL_INDICES
#undef REGISTER_GATHER_FULL

}  // namespace tensorflow
