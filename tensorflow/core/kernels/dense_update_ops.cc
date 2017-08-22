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

#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#endif

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/kernels/assign_op.h"
#include "tensorflow/core/kernels/dense_update_functor.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

template <typename Device, typename T>
class AssignOpT : public AssignOp {
 public:
  using AssignOp::AssignOp;

  void Copy(OpKernelContext* context, Tensor* lhs, const Tensor& rhs) override {
    functor::DenseUpdate<Device, T, ASSIGN> copy;
    copy(context->eigen_device<Device>(), lhs->flat<T>(), rhs.flat<T>());
  }
};

// TODO(jeff): Get rid of use_exclusive_lock_ option
template <typename Device, typename T, DenseUpdateType OP>
class DenseUpdateOp : public OpKernel {
 public:
  explicit DenseUpdateOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context,
                   context->GetAttr("use_locking", &use_exclusive_lock_));
    const DataType dt = DataTypeToEnum<T>::v();
    OP_REQUIRES_OK(context, context->MatchSignature({MakeRefType(dt), dt},
                                                    {MakeRefType(dt)}));
  }

  void Compute(OpKernelContext* context) override {
    // We always return the input ref.
    context->forward_ref_input_to_ref_output(0, 0);

    if (use_exclusive_lock_) {
      mutex_lock l(*context->input_ref_mutex(0));
      DoUpdate(context);
    } else {
      DoUpdate(context);
    }
  }

 private:
  void DoUpdate(OpKernelContext* context) {
    Tensor Tparams = context->mutable_input(0, use_exclusive_lock_);
    const Tensor& Tupdate = context->input(1);
    OP_REQUIRES(context, Tparams.IsInitialized(),
                errors::FailedPrecondition("Attempting to use uninitialized "
                                           "parameters: ",
                                           requested_input(0)));
    OP_REQUIRES(
        context, Tparams.IsSameSize(Tupdate),
        errors::InvalidArgument("Parameters and update must be the same size"));

    functor::DenseUpdate<Device, T, OP> update_functor;
    update_functor(context->template eigen_device<Device>(), Tparams.flat<T>(),
                   Tupdate.flat<T>());
  }

  bool use_exclusive_lock_;
};

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;
#ifdef TENSORFLOW_USE_SYCL
typedef Eigen::SyclDevice SYCLDevice;
#endif // TENSORFLOW_USE_SYCL

#define REGISTER_KERNELS(type)                                     \
  REGISTER_KERNEL_BUILDER(                                         \
      Name("Assign").Device(DEVICE_CPU).TypeConstraint<type>("T"), \
      AssignOpT<CPUDevice, type>);

TF_CALL_ALL_TYPES(REGISTER_KERNELS);
TF_CALL_QUANTIZED_TYPES(REGISTER_KERNELS);
#undef REGISTER_KERNELS

#if GOOGLE_CUDA
// Only register 'Assign' on GPU for the subset of types also supported by
// 'Variable' (see variable_ops.cc.)
#define REGISTER_GPU_KERNELS(type)                                 \
  REGISTER_KERNEL_BUILDER(                                         \
      Name("Assign").Device(DEVICE_GPU).TypeConstraint<type>("T"), \
      AssignOpT<GPUDevice, type>);

TF_CALL_GPU_ALL_TYPES(REGISTER_GPU_KERNELS);
#undef REGISTER_GPU_KERNELS
#endif  // GOOGLE_CUDA

#ifdef TENSORFLOW_USE_SYCL
#define REGISTER_SYCL_KERNELS(type)                                \
REGISTER_KERNEL_BUILDER(                                           \
    Name("Assign").Device(DEVICE_SYCL).TypeConstraint<type>("T"),  \
    AssignOpT<SYCLDevice, type>);

TF_CALL_GPU_NUMBER_TYPES_NO_HALF(REGISTER_SYCL_KERNELS);
#undef REGISTER_SYCL_KERNELS
#endif // TENSORFLOW_USE_SYCL

#define REGISTER_KERNELS(type)                                        \
  REGISTER_KERNEL_BUILDER(                                            \
      Name("AssignAdd").Device(DEVICE_CPU).TypeConstraint<type>("T"), \
      DenseUpdateOp<CPUDevice, type, DenseUpdateType::ADD>);          \
  REGISTER_KERNEL_BUILDER(                                            \
      Name("AssignSub").Device(DEVICE_CPU).TypeConstraint<type>("T"), \
      DenseUpdateOp<CPUDevice, type, DenseUpdateType::SUB>);

TF_CALL_NUMBER_TYPES(REGISTER_KERNELS);
#undef REGISTER_KERNELS

#if GOOGLE_CUDA
#define REGISTER_GPU_KERNELS(type)                                    \
  REGISTER_KERNEL_BUILDER(                                            \
      Name("AssignAdd").Device(DEVICE_GPU).TypeConstraint<type>("T"), \
      DenseUpdateOp<GPUDevice, type, DenseUpdateType::ADD>);          \
  REGISTER_KERNEL_BUILDER(                                            \
      Name("AssignSub").Device(DEVICE_GPU).TypeConstraint<type>("T"), \
      DenseUpdateOp<GPUDevice, type, DenseUpdateType::SUB>);
TF_CALL_GPU_NUMBER_TYPES(REGISTER_GPU_KERNELS);
#undef REGISTER_GPU_KERNELS
#endif  // end GOOGLE_CUDA

#ifdef TENSORFLOW_USE_SYCL
#define REGISTER_SYCL_KERNELS(type)                                         \
  REGISTER_KERNEL_BUILDER(                                             \
      Name("AssignAdd").Device(DEVICE_SYCL).TypeConstraint<type>("T"), \
      DenseUpdateOp<SYCLDevice, type, DenseUpdateType::ADD>);          \
  REGISTER_KERNEL_BUILDER(                                             \
      Name("AssignSub").Device(DEVICE_SYCL).TypeConstraint<type>("T"), \
      DenseUpdateOp<SYCLDevice, type, DenseUpdateType::SUB>);

TF_CALL_GPU_NUMBER_TYPES_NO_HALF(REGISTER_SYCL_KERNELS);
#undef REGISTER_SYCL_KERNELS
#endif // TENSORFLOW_USE_SYCL
}  // namespace tensorflow
