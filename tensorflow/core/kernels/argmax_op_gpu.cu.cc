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

#if (defined(GOOGLE_CUDA) && GOOGLE_CUDA) || \
    (defined(TENSORFLOW_USE_ROCM) && TENSORFLOW_USE_ROCM)

#define EIGEN_USE_GPU

#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/kernels/argmax_op.h"
#include "tensorflow/core/kernels/reduction_gpu_kernels.cu.h"
#include "tensorflow/core/kernels/reduction_ops_common.h"

namespace tensorflow {

typedef Eigen::GpuDevice GPUDevice;

typedef tensorflow::TTypes<float>::Tensor::Index Index;

// To compute the argmax/argmin, we perform a reduction on KeyValuePairs, which
// are (flattened index, value) pairs.
template <typename T>
using KeyValuePair = cub::KeyValuePair<Index, T>;

namespace {

template <typename T, bool is_argmax>
struct MaxOrMinFunc;

// The reduction operator: Returns the KeyValuePair with the highest or lowest
// value.
template <typename T>
struct MaxOrMinFunc<T, true> {
  __host__ __device__ __forceinline__ KeyValuePair<T> operator()(
      const KeyValuePair<T>& lhs, const KeyValuePair<T>& rhs) {
    // If one value is NaN, we choose the other value. This behavior is not
    // guaranteed by the op and may change in the future.
    return (lhs.value > rhs.value || Eigen::numext::isnan(rhs.value)) ? lhs
                                                                      : rhs;
  }
};

template <typename T>
struct MaxOrMinFunc<T, false> {
  __host__ __device__ __forceinline__ KeyValuePair<T> operator()(
      const KeyValuePair<T>& lhs, const KeyValuePair<T>& rhs) {
    return (lhs.value < rhs.value || Eigen::numext::isnan(rhs.value)) ? lhs
                                                                      : rhs;
  }
};

// The output converter: Converts from a KeyValuePair to an index into a a
// specific dimension. dim1 is the size of the dimension being reduced. dim2 is
// the size of the dimension(s) after dim1.
template <typename T, typename Tout>
struct OutputConverter {
  OutputConverter(Index dim1, Index dim2) : dim1_(dim1), dim2_(dim2) {}

  __host__ __device__ __forceinline__ Tout
  operator()(const KeyValuePair<T>& key_value_pair) const {
    return static_cast<Tout>((key_value_pair.key / dim2_) % dim1_);
  }

  Index dim1_;
  Index dim2_;
};

}  // namespace

namespace functor {
namespace reduction_op_helper {

// Template specialization of IdentityValue, to return the identity value for
// the reduction. This is needed for ReduceImpl, a function we call. We return
// (0, -inf) for argmax and (0, inf) for argmin.
template <typename T>
struct IdentityValue<KeyValuePair<T>, MaxOrMinFunc<T, true>> {
  KeyValuePair<T> operator()() {
    return {0, -std::numeric_limits<T>::infinity()};
  }
};

template <typename T>
struct IdentityValue<KeyValuePair<T>, MaxOrMinFunc<T, false>> {
  KeyValuePair<T> operator()() {
    return {0, std::numeric_limits<T>::infinity()};
  }
};

}  // namespace reduction_op_helper
}  // namespace functor

template <typename T, typename Tout, bool is_argmax>
void DoGpuArgOp(OpKernelContext* context, const Tensor& input, int axis,
                Tensor* output) {
  // We collapse adjacent axes of the input tensor in order to view it as a
  // 3 dimensional tensor. The reduction axis is not collapsed, so the three new
  // axes will be the input axes to the left of the reduction axis, the
  // reduction axis, and the input axes to the right of the reduction axis.
  Index dim0 = 1;
  for (Index i = 0; i < axis; i++) {
    dim0 *= input.dim_size(i);
  }
  Index dim1 = input.dim_size(axis);
  Index dim2 = 1;
  for (Index i = axis + 1; i < input.dims(); i++) {
    dim2 *= input.dim_size(i);
  }
  DCHECK_EQ(dim0 * dim1 * dim2, input.NumElements());

  auto inp = input.shaped<T, 3>({dim0, dim1, dim2});
  auto out = output->shaped<Tout, 2>({dim0, dim2});

  // We call ReduceImpl to perform the reduction. The input iterator returns
  // KeyValuePairs. The reduction functor returns the KeyValuePair with the max
  // or min value. The output iterator converts the KeyValuePair into an index
  // into dim1.
  using InputIterType = cub::ArgIndexInputIterator<const T*>;
  using Functor = MaxOrMinFunc<T, is_argmax>;
  using OutputIterType =
      TransformOutputIterator<Tout, KeyValuePair<T>, OutputConverter<T, Tout>>;

  InputIterType inp_wrapper(inp.data());
  OutputIterType out_wrapper(out.data(), OutputConverter<T, Tout>(dim1, dim2));

  typedef const Eigen::array<TTypes<float>::Tensor::Index, 1>& ReductionAxes;
  Constants<GPUDevice> constants;

  // TODO(reedwm): We can probably improve performance by writing specialized
  // argmax kernels instead of relying on the generic ReduceImpl function
  functor::ReduceImpl<KeyValuePair<T>, Functor, OutputIterType, InputIterType,
                      ReductionAxes>(context, out_wrapper, inp_wrapper, 3, dim0,
                                     dim1, dim2, 2, constants.kOne, Functor());
}

#define DEFINE_GPU_ARG_OPS(T)                                              \
  template void DoGpuArgOp<T, int64, true>(OpKernelContext * context,      \
                                           const Tensor& input, int axis,  \
                                           Tensor* output);                \
  template void DoGpuArgOp<T, int64, false>(OpKernelContext * context,     \
                                            const Tensor& input, int axis, \
                                            Tensor* output);               \
  template void DoGpuArgOp<T, int32, true>(OpKernelContext * context,      \
                                           const Tensor& input, int axis,  \
                                           Tensor* output);                \
  template void DoGpuArgOp<T, int32, false>(OpKernelContext * context,     \
                                            const Tensor& input, int axis, \
                                            Tensor* output);

TF_CALL_GPU_NUMBER_TYPES(DEFINE_GPU_ARG_OPS);

#define DEFINE_GPU_SPEC(T)                              \
  template struct functor::ArgMax<GPUDevice, T, int64>; \
  template struct functor::ArgMin<GPUDevice, T, int64>; \
  template struct functor::ArgMax<GPUDevice, T, int32>; \
  template struct functor::ArgMin<GPUDevice, T, int32>;

TF_CALL_GPU_NUMBER_TYPES(DEFINE_GPU_SPEC);

}  // end namespace tensorflow

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
