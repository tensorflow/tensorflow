/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#define EIGEN_USE_GPU

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/kernels/searchsorted_op.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"

namespace tensorflow {
typedef Eigen::GpuDevice GPUDevice;

namespace {

template <typename T>
struct GpuNanAwareCompare {
  __device__ bool operator()(const T& a, const T& b) const {
    if constexpr (!std::is_integral<T>::value) {
      if (Eigen::numext::isnan(a)) return false;
      if (Eigen::numext::isnan(b)) return true;
    }
    return a < b;
  }
};

template <typename T, typename OutType, typename Compare>
__device__ OutType gpu_lower_bound(const T* first, OutType count, T val,
                                   Compare comp) {
  const T* orig_first = first;
  OutType step = 0;
  while (count > 0) {
    const T* it = first;
    step = count / 2;
    it += step;
    if (comp(*it, val)) {
      first = ++it;
      count -= step + 1;
    } else {
      count = step;
    }
  }
  return first - orig_first;
}

template <typename T, typename OutType, typename Compare>
__device__ OutType gpu_upper_bound(const T* first, OutType count, T val,
                                   Compare comp) {
  const T* orig_first = first;
  OutType step = 0;
  while (count > 0) {
    const T* it = first;
    step = count / 2;
    it += step;
    if (!comp(val, *it)) {
      first = ++it;
      count -= step + 1;
    } else {
      count = step;
    }
  }
  return first - orig_first;
}

template <typename T, typename OutType>
__global__ void UpperBoundKernel(const T* __restrict__ sorted_inputs,
                                 int64_t batch_size, int64_t sorted_inputs_size,
                                 int64_t values_size,
                                 const T* __restrict__ values,
                                 OutType* __restrict__ outputs) {
  for (int64_t work_unit_id : GpuGridRangeX(values_size * batch_size)) {
    int64_t bid = work_unit_id / values_size;
    T value = values[work_unit_id];
    outputs[work_unit_id] = gpu_upper_bound<T, OutType>(
        sorted_inputs + bid * sorted_inputs_size, sorted_inputs_size, value,
        GpuNanAwareCompare<T>());
  }
}

template <typename T, typename OutType>
__global__ void LowerBoundKernel(const T* __restrict__ sorted_inputs,
                                 int64_t batch_size, int64_t sorted_inputs_size,
                                 int64_t values_size,
                                 const T* __restrict__ values,
                                 OutType* __restrict__ outputs) {
  for (int64_t work_unit_id : GpuGridRangeX(values_size * batch_size)) {
    int64_t bid = work_unit_id / values_size;
    T value = values[work_unit_id];
    outputs[work_unit_id] = gpu_lower_bound<T, OutType>(
        sorted_inputs + bid * sorted_inputs_size, sorted_inputs_size, value,
        GpuNanAwareCompare<T>());
  }
}
}  // namespace

namespace functor {
template <typename T, typename OutType>
struct UpperBoundFunctor<GPUDevice, T, OutType> {
  static absl::Status Compute(
      OpKernelContext* context,
      const typename TTypes<T, 1>::ConstTensor& sorted_inputs,
      const typename TTypes<T, 1>::ConstTensor& values, int64_t batch_size,
      int64_t num_inputs, int64_t num_values,
      typename TTypes<OutType, 1>::Tensor* output) {
    const GPUDevice& device = context->eigen_device<GPUDevice>();
    if (values.size() == 0) {
      // GetGpuLaunchConfig requires work_element_count > 0
      return absl::OkStatus();
    }
    GpuLaunchConfig config = GetGpuLaunchConfig(values.size(), device);

    TF_CHECK_OK(GpuLaunchKernel(
        UpperBoundKernel<T, OutType>, config.block_count,
        config.thread_per_block, 0, device.stream(), sorted_inputs.data(),
        batch_size, num_inputs, num_values, values.data(), output->data()));

    return absl::OkStatus();
  }
};

template <typename T, typename OutType>
struct LowerBoundFunctor<GPUDevice, T, OutType> {
  static absl::Status Compute(
      OpKernelContext* context,
      const typename TTypes<T, 1>::ConstTensor& sorted_inputs,
      const typename TTypes<T, 1>::ConstTensor& values, int64_t batch_size,
      int64_t num_inputs, int64_t num_values,
      typename TTypes<OutType, 1>::Tensor* output) {
    const GPUDevice& device = context->eigen_device<GPUDevice>();
    if (values.size() == 0) {
      // GetGpuLaunchConfig requires work_element_count > 0
      return absl::OkStatus();
    }
    GpuLaunchConfig config = GetGpuLaunchConfig(values.size(), device);

    TF_CHECK_OK(GpuLaunchKernel(
        LowerBoundKernel<T, OutType>, config.block_count,
        config.thread_per_block, 0, device.stream(), sorted_inputs.data(),
        batch_size, num_inputs, num_values, values.data(), output->data()));

    return absl::OkStatus();
  }
};
}  // namespace functor

#define REGISTER_GPU_SPEC(type) \
  template struct functor::UpperBoundFunctor<GPUDevice, type, int32>;

TF_CALL_REAL_NUMBER_TYPES(REGISTER_GPU_SPEC);
#undef REGISTER_GPU_SPEC

#define REGISTER_GPU_SPEC(type) \
  template struct functor::UpperBoundFunctor<GPUDevice, type, int64>;

TF_CALL_REAL_NUMBER_TYPES(REGISTER_GPU_SPEC);
#undef REGISTER_GPU_SPEC

#define REGISTER_GPU_SPEC(type) \
  template struct functor::LowerBoundFunctor<GPUDevice, type, int32>;

TF_CALL_REAL_NUMBER_TYPES(REGISTER_GPU_SPEC);
#undef REGISTER_GPU_SPEC

#define REGISTER_GPU_SPEC(type) \
  template struct functor::LowerBoundFunctor<GPUDevice, type, int64>;

TF_CALL_REAL_NUMBER_TYPES(REGISTER_GPU_SPEC);
#undef REGISTER_GPU_SPEC
}  // namespace tensorflow

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
