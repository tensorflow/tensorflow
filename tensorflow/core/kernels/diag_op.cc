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

// See docs in ../ops/array_ops.cc

#define EIGEN_USE_THREADS

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#define EIGEN_USE_GPU
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#include "tensorflow/core/kernels/diag_op.h"

#include <algorithm>
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/work_sharder.h"

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

// Generate the diagonal tensor with the diagonal set to the input tensor.
template <typename Device, typename T>
class DiagOp : public OpKernel {
 public:
  explicit DiagOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    const Tensor& diagonal = context->input(0);
    const int num_dims = diagonal.dims();
    OP_REQUIRES(
        context, 0 != num_dims,
        errors::InvalidArgument("Input must be at least rank 1, got 0"));
    TensorShape out_shape;
    for (int i = 0; i < num_dims; ++i) {
      out_shape.AddDim(diagonal.dim_size(i));
    }
    for (int i = 0; i < num_dims; ++i) {
      out_shape.AddDim(diagonal.dim_size(i));
    }
    Tensor* output_tensor = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, out_shape, &output_tensor));
    functor::DiagFunctor<Device, T> diagFunc;
    Status s =
        diagFunc(context, diagonal.NumElements(), diagonal.flat<T>().data(),
                 output_tensor->flat<T>().data());
    OP_REQUIRES_OK(context, s);
  }
};

// Extract the diagonal tensor with the diagonal set to the input tensor.
template <typename Device, typename T>
class DiagPartOp : public OpKernel {
 public:
  explicit DiagPartOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    const Tensor& tensor = context->input(0);
    const int num_dims = tensor.dims();
    const int out_dims = num_dims / 2;
    OP_REQUIRES(context, 0 == num_dims % 2,
                errors::InvalidArgument("The rank of the tensor should be \
                                         even and positive, got shape ",
                                        tensor.shape().DebugString()));
    for (int i = 0; i < out_dims; i++) {
      OP_REQUIRES(
          context, tensor.dim_size(i) == tensor.dim_size(i + out_dims),
          errors::InvalidArgument("Invalid shape ",
                                  tensor.shape().DebugString(), ": dimensions ",
                                  i, " and ", i + out_dims, " do not match."));
    }

    TensorShape out_shape;
    for (int i = 0; i < out_dims; ++i) {
      out_shape.AddDim(tensor.dim_size(i));
    }

    Tensor* output = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, out_shape, &output));
    functor::DiagPartFunctor<Device, T> diagPartFunc;
    Status s = diagPartFunc(context, out_shape.num_elements(),
                            tensor.flat<T>().data(), output->flat<T>().data());
    OP_REQUIRES_OK(context, s);
  }
};

// Implementation of the functor specialization for CPU.
//
// According to the diagonal definition,
// `output[i1,..., ik, i1,..., ik] = input[i1,..., ik]`,
//
// Let the rank of input is [s1,..., sk], then any offset of input's
// pointer can be represent by coordinate [i1,..., ik],
// where `index = i1*(s2*...*sk) + i2*(s3*...*sk) +... + ik`
//
// Let new_index is the offset of output's pointer with coordinate
// [i1,..., ik, i1,..., ik], then we have
// `new_index = i1*(s2*...sk*s1*...*sk) + i2*(s3*...*sk*s1*...*sk) +... + \
//              ik*(s1*...*sk) + i1*(s2*...*sk) + i2*(s3*...*sk) +... + ik
//            = (i1*(s2*...*sk) + i2*(s3*...*sk) +... + ik) * (1 + s1*...*sk)
//            = index * (1 + s1*...*sk)
//
// Let `size = s1*...*sk`, we finally have `new_index = index * (1 + size)`,
// which is the transfer function we use below.
// This trick make our implementations clear and easy to be parallel.
namespace functor {
template <typename T>
struct DiagFunctor<CPUDevice, T> {
  EIGEN_ALWAYS_INLINE Status operator()(OpKernelContext* context,
                                        const int64_t size, const T* in,
                                        T* out) {
    // This subprocess is responsible for writing values in index range
    // [start*size, limit*size)
    auto subDiag = [in, out, size](int64_t start, int64_t limit) {
      std::fill(out + size * start, out + size * limit, T());
      for (int64_t index = start; index < limit; ++index) {
        out[(1 + size) * index] = in[index];
      }
    };

    // Here, 5 is a empirical factor of cost_per_unit.
    auto worker_threads = *(context->device()->tensorflow_cpu_worker_threads());
    Shard(worker_threads.num_threads, worker_threads.workers, size, 5 * size,
          subDiag);
    return Status::OK();
  }
};

template <typename T>
struct DiagPartFunctor<CPUDevice, T> {
  EIGEN_ALWAYS_INLINE Status operator()(OpKernelContext* context,
                                        const int64_t size, const T* in,
                                        T* out) {
    // This subprocess is responsible for extracting values in index range
    // [start, limit)
    auto subDiagPart = [in, out, size](int64_t start, int64_t limit) {
      for (int64_t index = start; index < limit; ++index) {
        out[index] = in[(1 + size) * index];
      }
    };

    // Here, 5 is a empirical factor of cost_per_unit.
    auto worker_threads = *(context->device()->tensorflow_cpu_worker_threads());
    Shard(worker_threads.num_threads, worker_threads.workers, size, 5,
          subDiagPart);
    return Status::OK();
  }
};
}  // namespace functor

// Register the CPU kernels.
#define REGISTER_DIAGOP(T)                                    \
  REGISTER_KERNEL_BUILDER(                                    \
      Name("Diag").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      DiagOp<CPUDevice, T>)

TF_CALL_double(REGISTER_DIAGOP);
TF_CALL_float(REGISTER_DIAGOP);
TF_CALL_int32(REGISTER_DIAGOP);
TF_CALL_int64(REGISTER_DIAGOP);
TF_CALL_COMPLEX_TYPES(REGISTER_DIAGOP);
TF_CALL_half(REGISTER_DIAGOP);
#undef REGISTER_DIAGOP

#define REGISTER_DIAGPARTOP(T)                                    \
  REGISTER_KERNEL_BUILDER(                                        \
      Name("DiagPart").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      DiagPartOp<CPUDevice, T>)

TF_CALL_double(REGISTER_DIAGPARTOP);
TF_CALL_float(REGISTER_DIAGPARTOP);
TF_CALL_int32(REGISTER_DIAGPARTOP);
TF_CALL_int64(REGISTER_DIAGPARTOP);
TF_CALL_COMPLEX_TYPES(REGISTER_DIAGPARTOP);
TF_CALL_half(REGISTER_DIAGPARTOP);
#undef REGISTER_DIAGPARTOP

// Register the GPU kernels.
#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM

// Forward declarations of the functor specializations for GPU.
namespace functor {
extern template struct DiagFunctor<GPUDevice, double>;
extern template struct DiagFunctor<GPUDevice, float>;
extern template struct DiagFunctor<GPUDevice, int32>;
extern template struct DiagFunctor<GPUDevice, int64_t>;
extern template struct DiagFunctor<GPUDevice, complex64>;
extern template struct DiagFunctor<GPUDevice, complex128>;
}  // namespace functor

#define REGISTER_DIAGOP_GPU(T)                                \
  REGISTER_KERNEL_BUILDER(                                    \
      Name("Diag").Device(DEVICE_GPU).TypeConstraint<T>("T"), \
      DiagOp<GPUDevice, T>)

TF_CALL_double(REGISTER_DIAGOP_GPU);
TF_CALL_float(REGISTER_DIAGOP_GPU);
TF_CALL_int32(REGISTER_DIAGOP_GPU);
TF_CALL_int64(REGISTER_DIAGOP_GPU);
TF_CALL_COMPLEX_TYPES(REGISTER_DIAGOP_GPU);
TF_CALL_half(REGISTER_DIAGOP_GPU);
#undef REGISTER_DIAGOP_GPU

// Forward declarations of the functor specializations for GPU.
namespace functor {
extern template struct DiagPartFunctor<GPUDevice, double>;
extern template struct DiagPartFunctor<GPUDevice, float>;
extern template struct DiagPartFunctor<GPUDevice, int32>;
extern template struct DiagPartFunctor<GPUDevice, int64_t>;
extern template struct DiagPartFunctor<GPUDevice, complex64>;
extern template struct DiagPartFunctor<GPUDevice, complex128>;
extern template struct DiagPartFunctor<GPUDevice, Eigen::half>;
}  // namespace functor

#define REGISTER_DIAGPARTOP_GPU(T)                                \
  REGISTER_KERNEL_BUILDER(                                        \
      Name("DiagPart").Device(DEVICE_GPU).TypeConstraint<T>("T"), \
      DiagPartOp<GPUDevice, T>)

TF_CALL_double(REGISTER_DIAGPARTOP_GPU);
TF_CALL_float(REGISTER_DIAGPARTOP_GPU);
TF_CALL_int32(REGISTER_DIAGPARTOP_GPU);
TF_CALL_int64(REGISTER_DIAGPARTOP_GPU);
TF_CALL_COMPLEX_TYPES(REGISTER_DIAGPARTOP_GPU);
TF_CALL_half(REGISTER_DIAGPARTOP_GPU);
#undef REGISTER_DIAGPARTOP_GPU

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

}  // namespace tensorflow
