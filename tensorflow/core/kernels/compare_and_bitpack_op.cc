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

// See docs in ../ops/math_ops.cc

#define EIGEN_USE_THREADS

#include "tensorflow/core/kernels/compare_and_bitpack_op.h"

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/util/work_sharder.h"

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

template <typename Device, typename T>
class CompareAndBitpackOp : public OpKernel {
 public:
  explicit CompareAndBitpackOp(OpKernelConstruction* context)
      : OpKernel(context) {}

  void Compute(OpKernelContext* c) override {
    const Tensor& input_t = c->input(0);
    const Tensor& threshold_t = c->input(1);
    OP_REQUIRES(
        c, TensorShapeUtils::IsScalar(threshold_t.shape()),
        errors::InvalidArgument("Compare must be a scalar, but saw shape: ",
                                threshold_t.shape().DebugString()));
    const TensorShape& input_shape = input_t.shape();
    OP_REQUIRES(c, TensorShapeUtils::IsVectorOrHigher(input_shape),
                errors::InvalidArgument(
                    "Input should be at least a vector, but saw a scalar."));
    OP_REQUIRES(c, input_shape.dim_size(input_shape.dims() - 1) % 8 == 0,
                errors::InvalidArgument(
                    "Inner dimension of input should be "
                    "divisible by ",
                    8, ", but saw shape: ", input_shape.DebugString()));

    TensorShape output_shape = input_shape;
    int rank = input_shape.dims();
    output_shape.set_dim(rank - 1, input_shape.dim_size(rank - 1) / 8);

    Tensor* output_t;
    OP_REQUIRES_OK(c, c->allocate_output(0, output_shape, &output_t));

    auto input = input_t.flat_inner_dims<T>();
    auto threshold = threshold_t.scalar<T>();
    auto output = output_t->flat_inner_dims<uint8>();

    functor::CompareAndBitpack<Device, T> func;
    func(c, input, threshold, output);
  }
};

#define REGISTER_COMPARE_AND_BITPACK(type)                                    \
  REGISTER_KERNEL_BUILDER(                                                    \
      Name("CompareAndBitpack").Device(DEVICE_CPU).TypeConstraint<type>("T"), \
      CompareAndBitpackOp<CPUDevice, type>);

TF_CALL_REAL_NUMBER_TYPES(REGISTER_COMPARE_AND_BITPACK);
TF_CALL_bool(REGISTER_COMPARE_AND_BITPACK);

#undef REGISTER_COMPARE_AND_BITPACK

namespace functor {

template <typename T, class = void, class = void>
struct ComputeShard {
  static EIGEN_STRONG_INLINE void Compute(typename TTypes<T>::ConstMatrix input,
                                          typename TTypes<uint8>::Matrix output,
                                          const T& thresh, int64 start,
                                          int64 limit) {
    for (int64 i = start; i < limit; ++i) {
      uint8* out = output.data() + i;
      const T* block = input.data() + 8 * i;
      *out = ((((block[0] > thresh) << 7)) | (((block[1] > thresh) << 6)) |
              (((block[2] > thresh) << 5)) | (((block[3] > thresh) << 4)) |
              (((block[4] > thresh) << 3)) | (((block[5] > thresh) << 2)) |
              (((block[6] > thresh) << 1)) | (((block[7] > thresh))));
    }
  }
};

// Specialization for bool on systems where sizeof(bool) == 1.
template <typename T>
struct ComputeShard<T,
                    typename std::enable_if<std::is_same<T, bool>::value>::type,
                    typename std::enable_if<sizeof(T) == 1>::type> {
  static EIGEN_STRONG_INLINE void Compute(
      typename TTypes<bool>::ConstMatrix input,
      typename TTypes<uint8>::Matrix output, bool /*thresh*/, int64 start,
      int64 limit) {
#if __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
    for (int64 i = start; i < limit; ++i) {
      uint8* out = output.data() + i;
      const int64 block = *reinterpret_cast<const int64*>(input.data() + 8 * i);
      *out = ((((block & (1LL << (7 * 8))) >> (7 * 8 - 7))) |
              (((block & (1LL << (6 * 8))) >> (6 * 8 - 6))) |
              (((block & (1LL << (5 * 8))) >> (5 * 8 - 5))) |
              (((block & (1LL << (4 * 8))) >> (4 * 8 - 4))) |
              (((block & (1LL << (3 * 8))) >> (3 * 8 - 3))) |
              (((block & (1LL << (2 * 8))) >> (2 * 8 - 2))) |
              (((block & (1LL << 8)) >> (1 * 8 - 1))) | (((block & (1LL)))));
    }
#else
    for (int64 i = start; i < limit; ++i) {
      uint8* out = output.data() + i;
      const int64 block = *reinterpret_cast<const int64*>(input.data() + 8 * i);
      *out =
          ((((block & (1LL << (7 * 8))) >> (7 * 8 - 0))) |
           (((block & (1LL << (6 * 8))) >> (6 * 8 - 1))) |
           (((block & (1LL << (5 * 8))) >> (5 * 8 - 2))) |
           (((block & (1LL << (4 * 8))) >> (4 * 8 - 3))) |
           (((block & (1LL << (3 * 8))) >> (3 * 8 - 4))) |
           (((block & (1LL << (2 * 8))) >> (2 * 8 - 5))) |
           (((block & (1LL << 8)) >> (1 * 8 - 6))) | (((block & (1LL)) << 7)));
    }
#endif
  }
};

template <typename T>
struct CompareAndBitpack<CPUDevice, T> {
  void operator()(OpKernelContext* c, typename TTypes<T>::ConstMatrix input,
                  typename TTypes<T>::ConstScalar threshold,
                  TTypes<uint8>::Matrix output) {
    const T thresh = threshold();
    auto shard = [&, thresh](int64 start, int64 limit) {
      ComputeShard<T>::Compute(input, output, thresh, start, limit);
    };
    int64 total_shards = output.size();  // Approximate cmp as an add and
                                         // bitwise-or + shift as an add.
    const double total_cost = 8 * (Eigen::TensorOpCost::AddCost<T>() +
                                   Eigen::TensorOpCost::AddCost<uint8>());
    const int64 shard_cost = (total_cost >= static_cast<double>(kint64max))
                                 ? kint64max
                                 : static_cast<int64>(total_cost);

    auto worker_threads = *(c->device()->tensorflow_cpu_worker_threads());
    Shard(worker_threads.num_threads, worker_threads.workers, total_shards,
          shard_cost, shard);
  }
};

}  // namespace functor

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#define REGISTER_COMPARE_AND_BITPACK(type)                                    \
  REGISTER_KERNEL_BUILDER(                                                    \
      Name("CompareAndBitpack").Device(DEVICE_GPU).TypeConstraint<type>("T"), \
      CompareAndBitpackOp<GPUDevice, type>);

TF_CALL_GPU_NUMBER_TYPES(REGISTER_COMPARE_AND_BITPACK);
TF_CALL_bool(REGISTER_COMPARE_AND_BITPACK);

#undef REGISTER_COMPARE_AND_BITPACK

namespace functor {

#define DECLARE_GPU_SPEC(T)                                      \
  template <>                                                    \
  void CompareAndBitpack<GPUDevice, T>::operator()(              \
      OpKernelContext* c, typename TTypes<T>::ConstMatrix input, \
      typename TTypes<T>::ConstScalar threshold,                 \
      TTypes<uint8>::Matrix output);                             \
  extern template struct CompareAndBitpack<GPUDevice, T>;

TF_CALL_GPU_NUMBER_TYPES(DECLARE_GPU_SPEC)
TF_CALL_bool(DECLARE_GPU_SPEC)

#undef DECLARE_GPU_SPEC

}  // namespace functor

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

}  // namespace tensorflow
