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

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/kernels/bincount_op.h"
#include "tensorflow/core/kernels/gpu_prim.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/determinism.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"

namespace tensorflow {

typedef Eigen::GpuDevice GPUDevice;

namespace functor {

template <typename Tidx, typename T>
struct BincountFunctor<GPUDevice, Tidx, T, false> {
  static Status Compute(OpKernelContext* context,
                        const typename TTypes<Tidx, 1>::ConstTensor& arr,
                        const typename TTypes<T, 1>::ConstTensor& weights,
                        typename TTypes<T, 1>::Tensor& output,
                        const Tidx num_bins) {
    if (weights.size() != 0) {
      return errors::InvalidArgument(
          "Weights should not be passed as it should be "
          "handled by unsorted_segment_sum");
    }
    if (output.size() == 0) {
      return Status::OK();
    }
    if (tensorflow::OpDeterminismRequired()) {
      // TODO(reedwm): Is this really nondeterministic?
      // DeviceHistogram::HistogramEven is called, and it is unclear
      // if it is deterministic on floating-point inputs.
      // See https://github.com/NVIDIA/cub/issues/471#issuecomment-1194682443.
      return errors::Unimplemented(
          "Determinism is not yet supported in GPU implementation of "
          "Bincount.");
    }
    // In case weight.size() == 0, use CUB
    size_t temp_storage_bytes = 0;
    const Tidx* d_samples = arr.data();
    T* d_histogram = output.data();
    int num_levels = output.size() + 1;
    Tidx lower_level = Tidx(0);
    Tidx upper_level = num_bins;
    int num_samples = arr.size();
    const gpuStream_t& stream = GetGpuStream(context);

    // The first HistogramEven is to obtain the temp storage size required
    // with d_temp_storage = NULL passed to the call.
    auto err = gpuprim::DeviceHistogram::HistogramEven(
        /* d_temp_storage */ NULL,
        /* temp_storage_bytes */ temp_storage_bytes,
        /* d_samples */ d_samples,
        /* d_histogram */ d_histogram,
        /* num_levels */ num_levels,
        /* lower_level */ lower_level,
        /* upper_level */ upper_level,
        /* num_samples */ num_samples,
        /* stream */ stream);
    if (err != gpuSuccess) {
      return errors::Internal(
          "Could not launch HistogramEven to get temp storage: ",
          GpuGetErrorString(err), ".");
    }
    Tensor temp_storage;
    TF_RETURN_IF_ERROR(context->allocate_temp(
        DataTypeToEnum<int8>::value,
        TensorShape({static_cast<int64_t>(temp_storage_bytes)}),
        &temp_storage));

    void* d_temp_storage = temp_storage.flat<int8>().data();
    // The second HistogramEven is to actual run with d_temp_storage
    // allocated with temp_storage_bytes.
    err = gpuprim::DeviceHistogram::HistogramEven(
        /* d_temp_storage */ d_temp_storage,
        /* temp_storage_bytes */ temp_storage_bytes,
        /* d_samples */ d_samples,
        /* d_histogram */ d_histogram,
        /* num_levels */ num_levels,
        /* lower_level */ lower_level,
        /* upper_level */ upper_level,
        /* num_samples */ num_samples,
        /* stream */ stream);
    if (err != gpuSuccess) {
      return errors::Internal(
          "Could not launch HistogramEven: ", GpuGetErrorString(err), ".");
    }
    return Status::OK();
  }
};

template <typename Tidx, typename T>
__global__ void BincountReduceKernel(const Tidx* in, T* out, const int nthreads,
                                     const Tidx num_bins) {
  GPU_1D_KERNEL_LOOP(index, nthreads) {
    Tidx bin = ldg(in + index);
    if (bin < num_bins) {
      out[bin] = T(1);
    }
  }
}

template <typename Tidx, typename T>
struct BincountFunctor<GPUDevice, Tidx, T, true> {
  static Status Compute(OpKernelContext* context,
                        const typename TTypes<Tidx, 1>::ConstTensor& arr,
                        const typename TTypes<T, 1>::ConstTensor& weights,
                        typename TTypes<T, 1>::Tensor& output,
                        const Tidx num_bins) {
    const int nthreads = arr.dimension(0);

    auto d = context->eigen_gpu_device();
    GpuLaunchConfig config = GetGpuLaunchConfig(nthreads, d);
    return GpuLaunchKernel(BincountReduceKernel<Tidx, T>, config.block_count,
                           config.thread_per_block, 0, d.stream(), arr.data(),
                           output.data(), nthreads, num_bins);
  }
};

template <typename Tidx, typename T, bool binary_count>
__global__ void BincountColReduceKernel(const Tidx* in, const T* weights,
                                        const int weights_size, T* out,
                                        const int num_rows, const int num_cols,
                                        const Tidx num_bins) {
  const int nthreads = num_rows * num_cols;
  GPU_1D_KERNEL_LOOP(index, nthreads) {
    Tidx bin = ldg(in + index);
    if (bin < num_bins) {
      int row = index / num_cols;
      int offset = row * num_bins + bin;
      if (binary_count) {
        out[offset] = T(1);
      } else {
        T value = (weights_size == 0) ? T(1) : ldg(weights + index);
        GpuAtomicAdd(out + offset, value);
      }
    }
  }
}

template <typename Tidx, typename T, bool binary_count>
__global__ void BincountColReduceSharedKernel(const Tidx* in, const T* weights,
                                              const int weights_size, T* out,
                                              const int num_rows,
                                              const int num_cols,
                                              const Tidx num_bins) {
  const int out_size = num_rows * num_bins;
  GPU_DYNAMIC_SHARED_MEM_DECL(sizeof(T), unsigned char, shared_col_mem);
  T* shared_col_bins = reinterpret_cast<T*>(shared_col_mem);
  for (unsigned int binIdx = threadIdx.x; binIdx < out_size;
       binIdx += blockDim.x) {
    shared_col_bins[binIdx] = T(0);
  }
  __syncthreads();
  const int nthreads = num_rows * num_cols;
  GPU_1D_KERNEL_LOOP(index, nthreads) {
    Tidx bin = ldg(in + index);
    if (bin < num_bins) {
      int row = index / num_cols;
      int offset = row * num_bins + bin;
      if (binary_count) {
        shared_col_bins[offset] = T(1);
      } else {
        T value = (weights_size == 0) ? T(1) : ldg(weights + index);
        GpuAtomicAdd(shared_col_bins + offset, value);
      }
    }
  }
  __syncthreads();
  for (unsigned int binIdx = threadIdx.x; binIdx < out_size;
       binIdx += blockDim.x) {
    if (binary_count) {
      // out[binIdx] = out[binIdx] & shared_col_bins[binIdx];
      if (shared_col_bins[binIdx]) {
        out[binIdx] = shared_col_bins[binIdx];
      }
    } else {
      GpuAtomicAdd(out + binIdx, shared_col_bins[binIdx]);
    }
  }
}

template <typename Tidx, typename T, bool binary_count>
struct BincountReduceFunctor<GPUDevice, Tidx, T, binary_count> {
  static Status Compute(OpKernelContext* context,
                        const typename TTypes<Tidx, 2>::ConstTensor& in,
                        const typename TTypes<T, 2>::ConstTensor& weights,
                        typename TTypes<T, 2>::Tensor& out,
                        const Tidx num_bins) {
    const int num_rows = in.dimension(0);
    const int num_cols = in.dimension(1);

    auto d = context->eigen_gpu_device();
    GpuLaunchConfig config = GetGpuLaunchConfig(num_rows * num_cols, d);

    // Use half of maximum shared memory, approximately 6 * 1024 inputs.
    int smem_max = d.sharedMemPerBlock() / 2;
    int smem_usage = out.size() * sizeof(T);
    if (smem_usage < smem_max) {
      return GpuLaunchKernel(
          BincountColReduceSharedKernel<Tidx, T, binary_count>,
          config.block_count, config.thread_per_block, smem_usage, d.stream(),
          in.data(), weights.data(), weights.size(), out.data(), num_rows,
          num_cols, num_bins);
    }
    return GpuLaunchKernel(
        BincountColReduceKernel<Tidx, T, binary_count>, config.block_count,
        config.thread_per_block, 0, d.stream(), in.data(), weights.data(),
        weights.size(), out.data(), num_rows, num_cols, num_bins);
  }
};

}  // end namespace functor

#define REGISTER_GPU_SPEC(T)                                                  \
  template struct functor::BincountFunctor<GPUDevice, int32, T, true>;        \
  template struct functor::BincountFunctor<GPUDevice, int64, T, true>;        \
  template struct functor::BincountFunctor<GPUDevice, int32, T, false>;       \
  template struct functor::BincountFunctor<GPUDevice, int64, T, false>;       \
  template struct functor::BincountReduceFunctor<GPUDevice, int32, T, true>;  \
  template struct functor::BincountReduceFunctor<GPUDevice, int64, T, true>;  \
  template struct functor::BincountReduceFunctor<GPUDevice, int32, T, false>; \
  template struct functor::BincountReduceFunctor<GPUDevice, int64, T, false>;

TF_CALL_int32(REGISTER_GPU_SPEC);
TF_CALL_float(REGISTER_GPU_SPEC);
#undef REGISTER_GPU_SPEC

}  // namespace tensorflow

#endif  // GOOGLE_CUDA
