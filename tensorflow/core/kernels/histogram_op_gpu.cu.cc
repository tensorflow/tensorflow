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

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#define EIGEN_USE_GPU

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/kernels/gpu_prim.h"
#include "tensorflow/core/kernels/histogram_op.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"

namespace tensorflow {

typedef Eigen::GpuDevice GPUDevice;

namespace functor {

// TODO(yongtang) int64 of atomicAdd is not supported yet.
template <typename T, typename Tout>
struct HistogramFixedWidthFunctor<GPUDevice, T, Tout> {
  static Status Compute(OpKernelContext* context,
                        const typename TTypes<T, 1>::ConstTensor& values,
                        const typename TTypes<T, 1>::ConstTensor& value_range,
                        int32 nbins, typename TTypes<Tout, 1>::Tensor& out) {
    tensorflow::AllocatorAttributes pinned_allocator;
    pinned_allocator.set_on_host(true);
    pinned_allocator.set_gpu_compatible(true);

    Tensor levels_tensor;
    TF_RETURN_IF_ERROR(context->allocate_temp(
        DataTypeToEnum<T>::value, TensorShape({nbins + 1}), &levels_tensor,
        pinned_allocator));
    auto levels = levels_tensor.flat<T>();

    const double step = static_cast<double>(value_range(1) - value_range(0)) /
                        static_cast<double>(nbins);
    levels(0) = std::numeric_limits<T>::lowest();
    for (int i = 1; i < nbins; i++) {
      levels(i) =
          static_cast<T>(static_cast<double>(value_range(0)) + step * i);
    }
    levels(nbins) = std::numeric_limits<T>::max();

    size_t temp_storage_bytes = 0;
    const T* d_samples = values.data();
    Tout* d_histogram = out.data();
    int num_levels = levels.size();
    T* d_levels = levels.data();
    int num_samples = values.size();
    const gpuStream_t& stream = GetGpuStream(context);

    // The first HistogramRange is to obtain the temp storage size required
    // with d_temp_storage = NULL passed to the call.
    auto err = gpuprim::DeviceHistogram::HistogramRange(
        /* d_temp_storage */ NULL,
        /* temp_storage_bytes */ temp_storage_bytes,
        /* d_samples */ d_samples,
        /* d_histogram */ d_histogram,
        /* num_levels */ num_levels,
        /* d_levels */ d_levels,
        /* num_samples */ num_samples,
        /* stream */ stream);
    if (err != gpuSuccess) {
      return errors::Internal(
          "Could not launch HistogramRange to get temp storage: ",
          GpuGetErrorString(err), ".");
    }

    Tensor temp_storage;
    TF_RETURN_IF_ERROR(context->allocate_temp(
        DataTypeToEnum<int8>::value,
        TensorShape({static_cast<int64_t>(temp_storage_bytes)}),
        &temp_storage));

    void* d_temp_storage = temp_storage.flat<int8>().data();

    // The second HistogramRange is to actual run with d_temp_storage
    // allocated with temp_storage_bytes.
    err = gpuprim::DeviceHistogram::HistogramRange(
        /* d_temp_storage */ d_temp_storage,
        /* temp_storage_bytes */ temp_storage_bytes,
        /* d_samples */ d_samples,
        /* d_histogram */ d_histogram,
        /* num_levels */ num_levels,
        /* d_levels */ d_levels,
        /* num_samples */ num_samples,
        /* stream */ stream);
    if (err != gpuSuccess) {
      return errors::Internal(
          "Could not launch HistogramRange: ", GpuGetErrorString(err), ".");
    }

    return Status::OK();
  }
};

}  // end namespace functor

#define REGISTER_GPU_SPEC(type) \
  template struct functor::HistogramFixedWidthFunctor<GPUDevice, type, int32>;

TF_CALL_GPU_NUMBER_TYPES(REGISTER_GPU_SPEC);
#undef REGISTER_GPU_SPEC

}  // namespace tensorflow

#endif  // GOOGLE_CUDA
