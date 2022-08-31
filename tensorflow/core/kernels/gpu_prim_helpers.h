/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_CORE_KERNELS_GPU_PRIM_HELPERS_H_
#define TENSORFLOW_CORE_KERNELS_GPU_PRIM_HELPERS_H_

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#define EIGEN_USE_GPU

#include "tensorflow/compiler/xla/stream_executor/stream.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/kernels/gpu_prim.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"

namespace tensorflow {

namespace detail {

template <typename T>
__global__ void RangeInitKernel(const T start, const T delta, const T size,
                                T* out) {
  GPU_1D_KERNEL_LOOP(i, size) { out[i] = start + i * delta; }
}

// Initialize out with range start, start + delta, start + 2 * delta, ...
template <typename T>
Status RangeInit(const Eigen::GpuDevice& d, const T start, const T delta,
                 const T size, T* out) {
  if (size == 0) return OkStatus();
  GpuLaunchConfig config = GetGpuLaunchConfig(size, d);
  return GpuLaunchKernel(RangeInitKernel<T>, config.block_count,
                         config.thread_per_block, 0, d.stream(), start, delta,
                         size, out);
}

// Computes keys_out = sorted(keys_in), and indices_out = argsort(keys_in).
// If keys_out is not required, it can be set to nullptr.
// If indices_in is nullptr, the range of input indices [0, size) will be used.
template <bool Descending, typename Tkey, typename Tindex>
Status GpuRadixSortImpl(OpKernelContext* context, int size, const Tkey* keys_in,
                        Tkey* keys_out,            // Optional
                        const Tindex* indices_in,  // Optional
                        Tindex* indices_out, int num_bits = sizeof(Tkey) * 8) {
  if (size == 0) return OkStatus();
  if (num_bits == 0) {
    // Workaround for CUB failing when begin_bit = end_bit = 0 (e.g., when all
    // keys are 0, so no sorting is needed).
    se::Stream* stream = context->op_device_context()->stream();
    if (keys_out) {
      // Copy keys_in to keys_out.
      size_t num_bytes = size * sizeof(Tkey);
      se::DeviceMemoryBase src(const_cast<Tkey*>(keys_in), num_bytes);
      se::DeviceMemoryBase dst(keys_out, num_bytes);
      if (!stream->ThenMemcpy(&dst, src, num_bytes).ok()) {
        return errors::Internal("Failed to copy keys_in to keys_out");
      }
    }
    if (indices_in) {
      // Copy indices_in to indices_out.
      size_t num_bytes = size * sizeof(Tindex);
      se::DeviceMemoryBase src(const_cast<Tindex*>(indices_in), num_bytes);
      se::DeviceMemoryBase dst(indices_out, num_bytes);
      if (!stream->ThenMemcpy(&dst, src, num_bytes).ok()) {
        return errors::Internal("Failed to copy indices_in to indices_out");
      }
    } else {
      // Set output indices to range.
      const Eigen::GpuDevice& device =
          context->eigen_device<Eigen::GpuDevice>();
      TF_RETURN_IF_ERROR(detail::RangeInit(device, Tindex(0), Tindex(1),
                                           Tindex(size), indices_out));
    }
    return OkStatus();
  }
  // Allocate temporary inputs/outputs if necessary.
  Tensor tmp_indices_in;
  if (!indices_in) {
    TF_RETURN_IF_ERROR(context->allocate_temp(
        DataTypeToEnum<Tindex>::value, TensorShape({size}), &tmp_indices_in));
    Tindex* mutable_indices_in = tmp_indices_in.flat<Tindex>().data();
    indices_in = mutable_indices_in;
    const Eigen::GpuDevice& device = context->eigen_device<Eigen::GpuDevice>();
    // Initialize indices_in to the input index range.
    TF_RETURN_IF_ERROR(detail::RangeInit(device, Tindex(0), Tindex(1),
                                         Tindex(size), mutable_indices_in));
  }
  Tensor tmp_keys_out;
  if (!keys_out) {
    TF_RETURN_IF_ERROR(context->allocate_temp(
        DataTypeToEnum<Tkey>::value, TensorShape({size}), &tmp_keys_out));
    keys_out = tmp_keys_out.flat<Tkey>().data();
  }
  // Determine temporary device storage requirements.
  Tensor temp_storage;
  size_t temp_storage_bytes = 0;
  const auto& cu_stream = GetGpuStream(context);
  gpuError_t err;
  if constexpr (Descending) {
    err = gpuprim::DeviceRadixSort::SortPairsDescending(
        nullptr, temp_storage_bytes, keys_in, keys_out, indices_in, indices_out,
        size, /*begin_bit=*/0, /*end_bit=*/num_bits, cu_stream);
  } else {
    err = gpuprim::DeviceRadixSort::SortPairs(
        nullptr, temp_storage_bytes, keys_in, keys_out, indices_in, indices_out,
        size, /*begin_bit=*/0, /*end_bit=*/num_bits, cu_stream);
  }
  if (err != 0) {
    return errors::Internal(
        "Failed to launch gpuprim::DeviceRadixSort::SortPairs to calculate "
        "temp_storage_bytes, status: ",
        cudaGetErrorString(err));
  }
  // Allocate temporary storage.
  TF_RETURN_IF_ERROR(context->allocate_temp(
      DT_INT8, TensorShape({static_cast<int64_t>(temp_storage_bytes)}),
      &temp_storage));
  // Sort indices by keys.
  if constexpr (Descending) {
    err = gpuprim::DeviceRadixSort::SortPairsDescending(
        temp_storage.flat<int8>().data(), temp_storage_bytes, keys_in, keys_out,
        indices_in, indices_out, size, /*begin_bit=*/0, /*end_bit=*/num_bits,
        cu_stream);
  } else {
    err = gpuprim::DeviceRadixSort::SortPairs(
        temp_storage.flat<int8>().data(), temp_storage_bytes, keys_in, keys_out,
        indices_in, indices_out, size, /*begin_bit=*/0, /*end_bit=*/num_bits,
        cu_stream);
  }
  if (err != 0) {
    return errors::Internal(
        "Failed to launch gpuprim::DeviceRadixSort::SortPairs, "
        "temp_storage_bytes: ",
        temp_storage_bytes, "status: ", cudaGetErrorString(err));
  }
  return OkStatus();
}

}  // namespace detail

template <typename Tkey, typename Tindex>
Status GpuRadixSort(OpKernelContext* context, int size, const Tkey* keys_in,
                    Tkey* keys_out,            // Optional
                    const Tindex* indices_in,  // Optional
                    Tindex* indices_out, int num_bits = sizeof(Tkey) * 8) {
  return detail::GpuRadixSortImpl</*Descending=*/false>(
      context, size, keys_in, keys_out, indices_in, indices_out, num_bits);
}

template <typename Tkey, typename Tindex>
Status GpuRadixSortDescending(OpKernelContext* context, int size,
                              const Tkey* keys_in,
                              Tkey* keys_out,            // Optional
                              const Tindex* indices_in,  // Optional
                              Tindex* indices_out,
                              int num_bits = sizeof(Tkey) * 8) {
  return detail::GpuRadixSortImpl</*Descending=*/true>(
      context, size, keys_in, keys_out, indices_in, indices_out, num_bits);
}

template <typename InputIteratorT, typename OutputIteratorT>
Status GpuInclusivePrefixSum(OpKernelContext* context, int size,
                             InputIteratorT input, OutputIteratorT output) {
  static_assert(
      !std::is_same<typename std::remove_reference<decltype(*input)>::type,
                    bool>::value,
      "GpuInclusivePrefixSum does not work correct with booleans, please use "
      "TransformInputIterator to explicitly cast to an integer.");
  if (size == 0) return OkStatus();
  const auto& cu_stream = GetGpuStream(context);
  size_t temp_storage_bytes;
  auto err = gpuprim::DeviceScan::InclusiveSum(nullptr, temp_storage_bytes,
                                               input, output, size, cu_stream);
  if (err != 0) {
    return errors::Internal(
        "Failed to launch gpuprim::DeviceScan::InclusiveSum to calculate "
        "temp_storage_bytes, status: ",
        cudaGetErrorString(err));
  }
  Tensor temp_storage;
  TF_RETURN_IF_ERROR(context->allocate_temp(
      DT_INT8, TensorShape({static_cast<int64_t>(temp_storage_bytes)}),
      &temp_storage));
  err = gpuprim::DeviceScan::InclusiveSum(temp_storage.flat<int8>().data(),
                                          temp_storage_bytes, input, output,
                                          size, cu_stream);
  if (err != 0) {
    return errors::Internal(
        "Failed to launch gpuprim::DeviceScan::InclusiveSum, "
        "temp_storage_bytes: ",
        temp_storage_bytes, ", status: ", cudaGetErrorString(err));
  }
  return OkStatus();
}

// Note that this behaves deterministically for repeat calls on the same device.
template <typename InputIteratorT, typename OutputIteratorT,
          typename OffsetIteratorT, typename ReduceOp, typename T>
Status GpuSegmentedReduce(
    OpKernelContext* context, int num_segments, ReduceOp reduce_op,
    const T& initial_value,
    InputIteratorT input,             // [any]
    OffsetIteratorT segment_offsets,  // [num_segments + 1]
    OutputIteratorT output) {         // [num_segments]
  if (num_segments == 0) return OkStatus();
  const auto& cu_stream = GetGpuStream(context);
  size_t temp_storage_bytes;
  auto err = gpuprim::DeviceSegmentedReduce::Reduce(
      nullptr, temp_storage_bytes, input, output, num_segments, segment_offsets,
      segment_offsets + 1, reduce_op, initial_value, cu_stream);
  if (err != 0) {
    return errors::Internal(
        "Failed to launch gpuprim::DeviceSegmentedReduce::Reduce to calculate "
        "temp_storage_bytes, status: ",
        cudaGetErrorString(err));
  }
  Tensor temp_storage;
  TF_RETURN_IF_ERROR(context->allocate_temp(
      DT_INT8, TensorShape({static_cast<int64_t>(temp_storage_bytes)}),
      &temp_storage));
  err = gpuprim::DeviceSegmentedReduce::Reduce(
      temp_storage.flat<int8>().data(), temp_storage_bytes, input, output,
      num_segments, segment_offsets, segment_offsets + 1, reduce_op,
      initial_value, cu_stream);
  if (err != 0) {
    return errors::Internal(
        "Failed to launch gpuprim::DeviceSegmentedReduce::Reduce"
        ", temp_storage_bytes: ",
        temp_storage_bytes, ", status: ", cudaGetErrorString(err));
  }
  return OkStatus();
}

template <typename InputIteratorT, typename FlagIteratorT,
          typename OutputIteratorT, typename NumSelectedT = int>
Status GpuSelectFlagged(OpKernelContext* context, int size,
                        InputIteratorT input, FlagIteratorT flags,
                        OutputIteratorT output,
                        NumSelectedT* out_num_selected = nullptr) {
  const auto& cu_stream = GetGpuStream(context);
  Tensor out_num_selected_t;
  if (!out_num_selected) {
    TF_RETURN_IF_ERROR(
        context->allocate_temp(DataTypeToEnum<NumSelectedT>::value,
                               TensorShape({}), &out_num_selected_t));
    out_num_selected = out_num_selected_t.scalar<NumSelectedT>().data();
  }
  size_t temp_storage_bytes;
  auto err =
      gpuprim::DeviceSelect::Flagged(nullptr, temp_storage_bytes, input, flags,
                                     output, out_num_selected, size, cu_stream);
  if (err != 0) {
    return errors::Internal(
        "Failed to launch gpuprim::DeviceSelect::Flagged to calculate "
        "temp_storage_bytes, status: ",
        cudaGetErrorString(err));
  }
  Tensor temp_storage;
  TF_RETURN_IF_ERROR(context->allocate_temp(
      DT_INT8, TensorShape({static_cast<int64_t>(temp_storage_bytes)}),
      &temp_storage));
  err = gpuprim::DeviceSelect::Flagged(temp_storage.flat<int8>().data(),
                                       temp_storage_bytes, input, flags, output,
                                       out_num_selected, size, cu_stream);
  if (err != 0) {
    return errors::Internal(
        "Failed to launch gpuprim::DeviceSelect::Flagged, temp_storage_bytes: ",
        temp_storage_bytes, ", status: ", cudaGetErrorString(err));
  }
  return OkStatus();
}

}  // namespace tensorflow

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#endif  // TENSORFLOW_CORE_KERNELS_GPU_PRIM_HELPERS_H_
