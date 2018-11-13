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

#define EIGEN_USE_THREADS

#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#endif  // GOOGLE_CUDA

#include "tensorflow/core/kernels/strided_slice_op.h"
#include "tensorflow/core/kernels/dense_update_functor.h"
#include "tensorflow/core/kernels/slice_op.h"
#include "tensorflow/core/kernels/strided_slice_op_impl.h"

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/kernels/bounds_check.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/kernels/training_op_helpers.h"
#include "tensorflow/core/kernels/variable_ops.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/platform/prefetch.h"
#include "tensorflow/core/util/strided_slice_op.h"

namespace tensorflow {
namespace {

template <typename T>
struct MemCpyFunctor {
  // Returns true if the copy was made with memcpy, false otherwise.
  bool Copy(const Tensor& input, const gtl::InlinedVector<int64, 4>& begin,
            const gtl::InlinedVector<int64, 4>& end, Tensor* result) {
    if (DataTypeCanUseMemcpy(DataTypeToEnum<T>::v())) {
      auto in = input.tensor<T, 2>();
      auto output = result->tensor<T, 2>();
      // TODO(agarwal): Consider multi-threading if size[0] is large
      for (int row_in = begin[0], row_out = 0; row_in < end[0];
           ++row_in, ++row_out) {
        if (row_in + 1 < end[0]) {
          port::prefetch<port::PREFETCH_HINT_T0>(&output(row_in + 1, 0));
          port::prefetch<port::PREFETCH_HINT_T0>(&in(row_in + 1, begin[1]));
        }
        memcpy(&output(row_out, 0), &in(row_in, begin[1]),
               (end[1] - begin[1]) * sizeof(T));
      }
      return true;
    }
    return false;
  }
};

template <>
struct MemCpyFunctor<ResourceHandle> {
  bool Copy(const Tensor& input, const gtl::InlinedVector<int64, 4>& begin,
            const gtl::InlinedVector<int64, 4>& end, Tensor* result) {
    return false;
  }
};

}  // namespace

template <typename Device, typename T>
class StridedSliceOp : public OpKernel {
 public:
  explicit StridedSliceOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("begin_mask", &begin_mask));
    OP_REQUIRES_OK(context, context->GetAttr("end_mask", &end_mask));
    OP_REQUIRES_OK(context, context->GetAttr("ellipsis_mask", &ellipsis_mask));
    OP_REQUIRES_OK(context, context->GetAttr("new_axis_mask", &new_axis_mask));
    OP_REQUIRES_OK(context,
                   context->GetAttr("shrink_axis_mask", &shrink_axis_mask));
  }

  void Compute(OpKernelContext* context) override {
    TensorShape processing_shape, final_shape;
    bool is_identity = true;
    bool slice_dim0 = true;
    bool is_simple_slice = true;
    gtl::InlinedVector<int64, 4> begin;
    gtl::InlinedVector<int64, 4> end;
    gtl::InlinedVector<int64, 4> strides;

    OP_REQUIRES_OK(
        context, ValidateStridedSliceOp(
                     &context->input(1), &context->input(2), context->input(3),
                     context->input(0).shape(), begin_mask, end_mask,
                     ellipsis_mask, new_axis_mask, shrink_axis_mask,
                     &processing_shape, &final_shape, &is_identity,
                     &is_simple_slice, &slice_dim0, &begin, &end, &strides));
    const Tensor& input = context->input(0);

    // Optimization #1, slice is a no-op plus reshape
    if (is_identity) {
      VLOG(1) << "Strided slice identity ";
      Tensor tmp;
      OP_REQUIRES(context, tmp.CopyFrom(input, final_shape),
                  errors::Internal("Copy failed"));
      context->set_output(0, tmp);
      return;
    }

    // Optimization #2, slice is memory contiguous (only occurs in dim 0)
    if (slice_dim0 && IsDim0SliceAligned<T>(input.shape(), begin[0], end[0])) {
      OP_REQUIRES(context, input.dims() >= 1,
                  errors::InvalidArgument(
                      "Input must have rank at least 1, got: ", input.dims()));
      // Otherwise, is_identity should be true.
      VLOG(1) << "Strided slice dim 0: " << input.shape().DebugString();
      OP_REQUIRES(
          context, begin[0] <= end[0],
          errors::InvalidArgument("begin[0] (", begin[0],
                                  ") must less or equal to end[0] (", end[0]));
      Tensor slice = input.Slice(begin[0], end[0]);
      Tensor tmp;
      OP_REQUIRES(context, tmp.CopyFrom(slice, final_shape),
                  errors::Internal("Copy failed"));
      context->set_output(0, tmp);
      return;
    }

    Tensor* result = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, final_shape, &result));
    const int input_dims = input.dims();
    const int processing_dims = processing_shape.dims();

    if (processing_shape.num_elements() > 0) {
      // Optimization #3, slice has stride 1 in all dimensions
      // Optimization #3A, slice has only two dimensions
      // TODO(aselle): Here we are restricting to processing_shape and
      // final_shape being 2D. This isn't strictly necessary, but I don't
      // want to blow up code gen size, because to shape<> you need static
      // NDIM and T
      if (is_simple_slice && std::is_same<Device, CPUDevice>::value &&
          input_dims == 2 && processing_shape.dims() == 2 &&
          final_shape.dims() == 2) {
        MemCpyFunctor<T> functor;
        if (functor.Copy(input, begin, end, result)) {
          return;
        }
      }

#define HANDLE_DIM(NDIM)                                                       \
  if (processing_dims == NDIM) {                                               \
    HandleStridedSliceCase<Device, T, NDIM>(context, begin, end, strides,      \
                                            processing_shape, is_simple_slice, \
                                            result);                           \
    return;                                                                    \
  }

      HANDLE_DIM(1);
      HANDLE_DIM(2);
      HANDLE_DIM(3);
      HANDLE_DIM(4);
      HANDLE_DIM(5);
      HANDLE_DIM(6);
      HANDLE_DIM(7);

#undef HANDLE_DIM

      OP_REQUIRES(
          context, false,
          errors::Unimplemented("Unhandled input dimensions ", input_dims));
    }
  }

 private:
  int32 begin_mask, end_mask;
  int32 ellipsis_mask, new_axis_mask, shrink_axis_mask;
};

template <typename Device, typename T>
class StridedSliceGradOp : public OpKernel {
 public:
  explicit StridedSliceGradOp(OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("begin_mask", &begin_mask));
    OP_REQUIRES_OK(context, context->GetAttr("end_mask", &end_mask));
    OP_REQUIRES_OK(context, context->GetAttr("ellipsis_mask", &ellipsis_mask));
    OP_REQUIRES_OK(context, context->GetAttr("new_axis_mask", &new_axis_mask));
    OP_REQUIRES_OK(context,
                   context->GetAttr("shrink_axis_mask", &shrink_axis_mask));
  }

  void Compute(OpKernelContext* context) override {
    TensorShape processing_shape, final_shape;
    bool is_identity = true;
    bool slice_dim0 = true;
    bool is_simple_slice = true;
    gtl::InlinedVector<int64, 4> begin;
    gtl::InlinedVector<int64, 4> end;
    gtl::InlinedVector<int64, 4> strides;

    TensorShape input_shape;
    const Tensor& input_shape_tensor = context->input(0);
    OP_REQUIRES(
        context, input_shape_tensor.dims() == 1,
        errors::InvalidArgument("shape must be 1-D, got shape.shape = ",
                                input_shape_tensor.shape().DebugString()));
    if (input_shape_tensor.dtype() == DT_INT32) {
      OP_REQUIRES_OK(
          context, TensorShapeUtils::MakeShape(input_shape_tensor.vec<int32>(),
                                               &input_shape));
    } else if (input_shape_tensor.dtype() == DT_INT64) {
      OP_REQUIRES_OK(
          context, TensorShapeUtils::MakeShape(input_shape_tensor.vec<int64>(),
                                               &input_shape));
    } else {
      LOG(FATAL) << "shape must have type int32 or int64.";
    }

    OP_REQUIRES_OK(
        context,
        ValidateStridedSliceOp(
            &context->input(1), &context->input(2), context->input(3),
            input_shape, begin_mask, end_mask, ellipsis_mask, new_axis_mask,
            shrink_axis_mask, &processing_shape, &final_shape, &is_identity,
            &is_simple_slice, &slice_dim0, &begin, &end, &strides));

    // Check to make sure dy is consistent with the original slice
    TensorShape dy_shape = context->input(4).shape();
    OP_REQUIRES(
        context, final_shape == dy_shape,
        errors::InvalidArgument("shape of dy was ", dy_shape.DebugString(),
                                " instead of ", final_shape.DebugString()));

    if (!context->status().ok()) return;

    // const int input_dims = input.dims();
    const int processing_dims = processing_shape.dims();
    Tensor* result = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, input_shape, &result));

    if (processing_shape.dims() == 0) {
      auto in = context->input(4);
      OP_REQUIRES(context, result->CopyFrom(in, processing_shape),
                  errors::Internal("Copy failed"));
      return;
    }

#define HANDLE_DIM(NDIM)                                                      \
  if (processing_dims == NDIM) {                                              \
    HandleStridedSliceGradCase<Device, T, NDIM>(context, begin, end, strides, \
                                                processing_shape,             \
                                                is_simple_slice, result);     \
    return;                                                                   \
  }

    HANDLE_DIM(1);
    HANDLE_DIM(2);
    HANDLE_DIM(3);
    HANDLE_DIM(4);
    HANDLE_DIM(5);
    HANDLE_DIM(6);
    HANDLE_DIM(7);

#undef HANDLE_DIM
  }

 private:
  int32 begin_mask, end_mask;
  int32 ellipsis_mask, new_axis_mask, shrink_axis_mask;
};

template <typename Device, typename T>
class StridedSliceAssignOp : public OpKernel {
 public:
  explicit StridedSliceAssignOp(OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("begin_mask", &begin_mask));
    OP_REQUIRES_OK(context, context->GetAttr("end_mask", &end_mask));
    OP_REQUIRES_OK(context, context->GetAttr("ellipsis_mask", &ellipsis_mask));
    OP_REQUIRES_OK(context, context->GetAttr("new_axis_mask", &new_axis_mask));
    OP_REQUIRES_OK(context,
                   context->GetAttr("shrink_axis_mask", &shrink_axis_mask));
  }

  void Compute(OpKernelContext* context) override {
    TensorShape processing_shape, final_shape;
    bool is_identity = true;
    bool slice_dim0 = true;
    bool is_simple_slice = true;
    gtl::InlinedVector<int64, 4> begin;
    gtl::InlinedVector<int64, 4> end;
    gtl::InlinedVector<int64, 4> strides;

    Tensor* old_lhs = nullptr;
    Tensor tmp;
    if (context->input_dtype(0) == DT_RESOURCE) {
      Var* v;
      OP_REQUIRES_OK(context,
                     LookupResource(context, HandleFromInput(context, 0), &v));
      core::ScopedUnref scoped_unref(v);
      mutex_lock ml(*v->mu());
      OP_REQUIRES_OK(context,
                     PrepareToUpdateVariable<Device, T>(context, v->tensor()));
      old_lhs = v->tensor();
      OP_REQUIRES(context, old_lhs->dtype() == DataTypeToEnum<T>::value,
                  errors::InvalidArgument(
                      "l-value dtype ", DataTypeString(old_lhs->dtype()),
                      " does not match r-value dtype ",
                      DataTypeString(DataTypeToEnum<T>::value)));
    } else {
      context->forward_ref_input_to_ref_output(0, 0);
      tmp = context->mutable_input(0, true);
      old_lhs = &tmp;
    }

    OP_REQUIRES_OK(
        context, ValidateStridedSliceOp(
                     &context->input(1), &context->input(2), context->input(3),
                     old_lhs->shape(), begin_mask, end_mask, ellipsis_mask,
                     new_axis_mask, shrink_axis_mask, &processing_shape,
                     &final_shape, &is_identity, &is_simple_slice, &slice_dim0,
                     &begin, &end, &strides));

    if (processing_shape.num_elements()) {
      const Tensor& input = context->input(4);
      TensorShape input_shape = input.shape();
      TensorShape original_shape = old_lhs->shape();
      // TODO(aselle): This check is too strong, we only should need
      // input_shape to be broadcastable to final_shape
      OP_REQUIRES(
          context, final_shape == input_shape,
          errors::Unimplemented(
              "sliced l-value shape ", final_shape.DebugString(),
              " does not match r-value shape ", input_shape.DebugString(),
              ". Automatic broadcasting not ", "yet implemented."));
      const int processing_dims = processing_shape.dims();

      // 0-dimensional case implies the left and right are exactly the same
      // scalar shape

// Handle general dimensions
#define HANDLE_DIM(NDIM)                                                       \
  if (processing_dims == NDIM) {                                               \
    HandleStridedSliceAssignCase<Device, T, NDIM>()(context, begin, end,       \
                                                    strides, processing_shape, \
                                                    is_simple_slice, old_lhs); \
    return;                                                                    \
  }
      HANDLE_DIM(0);
      HANDLE_DIM(1);
      HANDLE_DIM(2);
      HANDLE_DIM(3);
      HANDLE_DIM(4);
      HANDLE_DIM(5);
      HANDLE_DIM(6);
      HANDLE_DIM(7);
#undef HANDLE_DIM

      OP_REQUIRES(context, false,
                  errors::Unimplemented("Unhandled input dimensions ",
                                        processing_dims));
    }
  }

 private:
  int32 begin_mask, end_mask;
  int32 ellipsis_mask, new_axis_mask, shrink_axis_mask;
};

#define REGISTER_STRIDED_SLICE(type)                             \
  REGISTER_KERNEL_BUILDER(Name("StridedSlice")                   \
                              .Device(DEVICE_CPU)                \
                              .TypeConstraint<type>("T")         \
                              .HostMemory("begin")               \
                              .HostMemory("end")                 \
                              .HostMemory("strides"),            \
                          StridedSliceOp<CPUDevice, type>)       \
  REGISTER_KERNEL_BUILDER(Name("StridedSliceGrad")               \
                              .Device(DEVICE_CPU)                \
                              .TypeConstraint<type>("T")         \
                              .HostMemory("shape")               \
                              .HostMemory("begin")               \
                              .HostMemory("end")                 \
                              .HostMemory("strides"),            \
                          StridedSliceGradOp<CPUDevice, type>)   \
  REGISTER_KERNEL_BUILDER(Name("StridedSliceAssign")             \
                              .Device(DEVICE_CPU)                \
                              .TypeConstraint<type>("T")         \
                              .HostMemory("begin")               \
                              .HostMemory("end")                 \
                              .HostMemory("strides"),            \
                          StridedSliceAssignOp<CPUDevice, type>) \
  REGISTER_KERNEL_BUILDER(Name("ResourceStridedSliceAssign")     \
                              .Device(DEVICE_CPU)                \
                              .TypeConstraint<type>("T")         \
                              .HostMemory("ref")                 \
                              .HostMemory("begin")               \
                              .HostMemory("end")                 \
                              .HostMemory("strides"),            \
                          StridedSliceAssignOp<CPUDevice, type>)

TF_CALL_ALL_TYPES(REGISTER_STRIDED_SLICE);

#undef REGISTER_STRIDED_SLICE

#if GOOGLE_CUDA

#define REGISTER_GPU(type)                                       \
  REGISTER_KERNEL_BUILDER(Name("StridedSlice")                   \
                              .Device(DEVICE_GPU)                \
                              .TypeConstraint<type>("T")         \
                              .HostMemory("begin")               \
                              .HostMemory("end")                 \
                              .HostMemory("strides"),            \
                          StridedSliceOp<GPUDevice, type>)       \
  REGISTER_KERNEL_BUILDER(Name("StridedSliceGrad")               \
                              .Device(DEVICE_GPU)                \
                              .TypeConstraint<type>("T")         \
                              .HostMemory("shape")               \
                              .HostMemory("begin")               \
                              .HostMemory("end")                 \
                              .HostMemory("strides"),            \
                          StridedSliceGradOp<GPUDevice, type>)   \
  REGISTER_KERNEL_BUILDER(Name("StridedSliceAssign")             \
                              .Device(DEVICE_GPU)                \
                              .TypeConstraint<type>("T")         \
                              .HostMemory("begin")               \
                              .HostMemory("end")                 \
                              .HostMemory("strides"),            \
                          StridedSliceAssignOp<GPUDevice, type>) \
  REGISTER_KERNEL_BUILDER(Name("ResourceStridedSliceAssign")     \
                              .Device(DEVICE_GPU)                \
                              .TypeConstraint<type>("T")         \
                              .HostMemory("ref")                 \
                              .HostMemory("begin")               \
                              .HostMemory("end")                 \
                              .HostMemory("strides"),            \
                          StridedSliceAssignOp<GPUDevice, type>)

TF_CALL_GPU_NUMBER_TYPES(REGISTER_GPU);
TF_CALL_complex64(REGISTER_GPU);
TF_CALL_complex128(REGISTER_GPU);
TF_CALL_int64(REGISTER_GPU);

// A special GPU kernel for int32.
// TODO(b/25387198): Also enable int32 in device memory. This kernel
// registration requires all int32 inputs and outputs to be in host memory.
REGISTER_KERNEL_BUILDER(Name("StridedSlice")
                            .Device(DEVICE_GPU)
                            .TypeConstraint<int32>("T")
                            .HostMemory("input")
                            .HostMemory("begin")
                            .HostMemory("end")
                            .HostMemory("strides")
                            .HostMemory("output"),
                        StridedSliceOp<CPUDevice, int32>);
REGISTER_KERNEL_BUILDER(Name("StridedSliceGrad")
                            .Device(DEVICE_GPU)
                            .TypeConstraint<int32>("T")
                            .HostMemory("shape")
                            .HostMemory("begin")
                            .HostMemory("end")
                            .HostMemory("strides")
                            .HostMemory("dy")
                            .HostMemory("output"),
                        StridedSliceGradOp<CPUDevice, int32>);
REGISTER_KERNEL_BUILDER(Name("StridedSliceAssign")
                            .Device(DEVICE_GPU)
                            .TypeConstraint<int32>("T")
                            .HostMemory("ref")
                            .HostMemory("begin")
                            .HostMemory("end")
                            .HostMemory("strides"),
                        StridedSliceAssignOp<CPUDevice, int32>)
REGISTER_KERNEL_BUILDER(Name("ResourceStridedSliceAssign")
                            .Device(DEVICE_GPU)
                            .TypeConstraint<int32>("T")
                            .HostMemory("ref")
                            .HostMemory("begin")
                            .HostMemory("end")
                            .HostMemory("strides"),
                        StridedSliceAssignOp<CPUDevice, int32>)
#undef REGISTER_GPU

#endif  // GOOGLE_CUDA

#ifdef TENSORFLOW_USE_SYCL
#define REGISTER_SYCL(type)                                       \
  REGISTER_KERNEL_BUILDER(Name("StridedSlice")                    \
                              .Device(DEVICE_SYCL)                \
                              .TypeConstraint<type>("T")          \
                              .HostMemory("begin")                \
                              .HostMemory("end")                  \
                              .HostMemory("strides"),             \
                          StridedSliceOp<SYCLDevice, type>)       \
  REGISTER_KERNEL_BUILDER(Name("StridedSliceGrad")                \
                              .Device(DEVICE_SYCL)                \
                              .TypeConstraint<type>("T")          \
                              .HostMemory("shape")                \
                              .HostMemory("begin")                \
                              .HostMemory("end")                  \
                              .HostMemory("strides"),             \
                          StridedSliceGradOp<SYCLDevice, type>)   \
  REGISTER_KERNEL_BUILDER(Name("StridedSliceAssign")              \
                              .Device(DEVICE_SYCL)                \
                              .TypeConstraint<type>("T")          \
                              .HostMemory("begin")                \
                              .HostMemory("end")                  \
                              .HostMemory("strides"),             \
                          StridedSliceAssignOp<SYCLDevice, type>) \
  REGISTER_KERNEL_BUILDER(Name("ResourceStridedSliceAssign")      \
                              .Device(DEVICE_SYCL)                \
                              .TypeConstraint<type>("T")          \
                              .HostMemory("ref")                  \
                              .HostMemory("begin")                \
                              .HostMemory("end")                  \
                              .HostMemory("strides"),             \
                          StridedSliceAssignOp<SYCLDevice, type>)

TF_CALL_GPU_NUMBER_TYPES_NO_HALF(REGISTER_SYCL);

REGISTER_KERNEL_BUILDER(Name("StridedSlice")
                            .Device(DEVICE_SYCL)
                            .TypeConstraint<int32>("T")
                            .HostMemory("input")
                            .HostMemory("begin")
                            .HostMemory("end")
                            .HostMemory("strides")
                            .HostMemory("output"),
                        StridedSliceOp<CPUDevice, int32>);
REGISTER_KERNEL_BUILDER(Name("StridedSliceGrad")
                            .Device(DEVICE_SYCL)
                            .TypeConstraint<int32>("T")
                            .HostMemory("shape")
                            .HostMemory("begin")
                            .HostMemory("end")
                            .HostMemory("strides")
                            .HostMemory("dy")
                            .HostMemory("output"),
                        StridedSliceGradOp<CPUDevice, int32>);
REGISTER_KERNEL_BUILDER(Name("StridedSliceAssign")
                            .Device(DEVICE_SYCL)
                            .TypeConstraint<int32>("T")
                            .HostMemory("ref")
                            .HostMemory("begin")
                            .HostMemory("end")
                            .HostMemory("strides"),
                        StridedSliceAssignOp<CPUDevice, int32>)
REGISTER_KERNEL_BUILDER(Name("ResourceStridedSliceAssign")
                            .Device(DEVICE_SYCL)
                            .TypeConstraint<int32>("T")
                            .HostMemory("ref")
                            .HostMemory("begin")
                            .HostMemory("end")
                            .HostMemory("strides"),
                        StridedSliceAssignOp<CPUDevice, int32>)
#undef REGISTER_SYCL
#endif  // TENSORFLOW_USE_SYCL
}  // namespace tensorflow
