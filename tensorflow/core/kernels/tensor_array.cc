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

#define EIGEN_USE_THREADS
#include "tensorflow/core/kernels/tensor_array.h"

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor_util.h"
#include "tensorflow/core/kernels/aggregate_ops_cpu.h"

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

namespace tensor_array {

#define TENSOR_ARRAY_WRITE_OR_ADD(Device, T)                                \
  template <>                                                               \
  Status AddToTensor<Device, T>(OpKernelContext * ctx, Tensor * sum,        \
                                const Tensor* current, const Tensor* add) { \
    functor::Add2Functor<Device, T> add_functor;                            \
    add_functor(ctx->template eigen_device<Device>(), sum->flat<T>(),       \
                current->flat<T>(), add->flat<T>());                        \
    return Status::OK();                                                    \
  }

#define TENSOR_ARRAY_WRITE_OR_ADD_CPU(T) TENSOR_ARRAY_WRITE_OR_ADD(CPUDevice, T)
TF_CALL_NUMBER_TYPES(TENSOR_ARRAY_WRITE_OR_ADD_CPU)
#undef TENSOR_ARRAY_WRITE_OR_ADD_CPU

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#define TENSOR_ARRAY_WRITE_OR_ADD_GPU(T) TENSOR_ARRAY_WRITE_OR_ADD(GPUDevice, T)
TF_CALL_GPU_NUMBER_TYPES(TENSOR_ARRAY_WRITE_OR_ADD_GPU);
TF_CALL_complex64(TENSOR_ARRAY_WRITE_OR_ADD_GPU);
TF_CALL_complex128(TENSOR_ARRAY_WRITE_OR_ADD_GPU);
#undef TENSOR_ARRAY_WRITE_OR_ADD_GPU

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#undef TENSOR_ARRAY_WRITE_OR_ADD

#define TENSOR_ARRAY_SET_ZERO(Device, T)                                      \
  template <>                                                                 \
  Status TensorSetZero<Device, T>(OpKernelContext * ctx, Tensor * value) {    \
    functor::SetZeroFunctor<Device, T> set_zero_functor;                      \
    set_zero_functor(ctx->template eigen_device<Device>(), value->flat<T>()); \
    return Status::OK();                                                      \
  }

#define TENSOR_ARRAY_SET_ZERO_CPU(T) TENSOR_ARRAY_SET_ZERO(CPUDevice, T)
TF_CALL_NUMBER_TYPES(TENSOR_ARRAY_SET_ZERO_CPU);
TF_CALL_bool(TENSOR_ARRAY_SET_ZERO_CPU);
#undef TENSOR_ARRAY_SET_ZERO_CPU

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#define TENSOR_ARRAY_SET_ZERO_GPU(T) TENSOR_ARRAY_SET_ZERO(GPUDevice, T)
TF_CALL_GPU_NUMBER_TYPES(TENSOR_ARRAY_SET_ZERO_GPU);
TF_CALL_complex64(TENSOR_ARRAY_SET_ZERO_GPU);
TF_CALL_complex128(TENSOR_ARRAY_SET_ZERO_GPU);
#undef TENSOR_ARRAY_SET_ZERO_GPU

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#undef TENSOR_ARRAY_SET_ZERO

}  // namespace tensor_array

std::atomic<int64> TensorArray::tensor_array_counter{0};

Status TensorArray::CopyShapesFrom(TensorArray* rhs,
                                   const TensorShape* shape_to_prepend) {
  mutex_lock l(mu_);
  mutex_lock l_rhs(rhs->mu_);
  TF_RETURN_IF_ERROR(LockedReturnIfClosed());
  TF_RETURN_IF_ERROR(rhs->LockedReturnIfClosed());
  if (tensors_.size() != rhs->tensors_.size()) {
    return errors::InvalidArgument(
        "TensorArray sizes do not match during CopyShapesFrom: ",
        handle_.vec<tstring>()(1), " has size ", tensors_.size(), " but rhs ",
        rhs->handle_.vec<tstring>()(1), " has size ", rhs->tensors_.size());
  }
  for (std::size_t i = 0; i < tensors_.size(); ++i) {
    // Skip "soft copy" of indices which have not been written.
    if (!rhs->tensors_[i].written) continue;

    // Copy the shape over.
    if (shape_to_prepend) {
      tensors_[i].shape = *shape_to_prepend;
      tensors_[i].shape.AppendShape(rhs->tensors_[i].shape);
    } else {
      tensors_[i].shape = rhs->tensors_[i].shape;
    }
    // Mark as written.  Reads will know that if written is true and
    // read is false, and cleared is false, to return zeros of the
    // appropriate shape.  Future aggregating writes will only use the shape
    // for validation.
    tensors_[i].written = true;
  }

  return Status::OK();
}

}  // namespace tensorflow
