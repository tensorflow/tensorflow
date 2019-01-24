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

#ifndef TENSORFLOW_CORE_KERNELS_WHERE_OP_GPU_CU_H_
#define TENSORFLOW_CORE_KERNELS_WHERE_OP_GPU_CU_H_

#if GOOGLE_CUDA

#define EIGEN_USE_GPU

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "third_party/cub/device/device_reduce.cuh"
#include "third_party/cub/device/device_select.cuh"
#include "third_party/cub/iterator/counting_input_iterator.cuh"
#include "third_party/cub/iterator/transform_input_iterator.cuh"
#include "tensorflow/core/framework/bounds_check.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/kernels/where_op.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/cuda_kernel_helper.h"

namespace tensorflow {

typedef Eigen::GpuDevice GPUDevice;

namespace functor {

template <int NDIM, typename TIndex>
__global__ void PropagateWhereIndicesKernel(
    const TIndex output_rows, const typename Eigen::array<TIndex, NDIM> strides,
    int64* output) {
  // TODO(ebrevdo): Use a multi-dimensional loop, increasing the
  // dimensions of individual indices manually, instead of relying on
  // a scalar loop variable and using integer division.
  CUDA_1D_KERNEL_LOOP(i, output_rows) {
    TIndex index_value = ldg(output + NDIM * i);
#pragma unroll
    for (int c = 0; c < NDIM; ++c) {
      *(output + NDIM * i + c) = index_value / strides[c];
      index_value %= strides[c];
    }
  }
}

namespace {

template <typename T>
struct IsNonzero {
  EIGEN_DEVICE_FUNC IsNonzero() : zero(T(0)) {}
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE bool operator()(const T& x) const {
    return (x != zero);
  }
  const T zero;
};

template <typename T, typename TIndex>
struct CubDeviceReduceCount {
  cudaError_t operator()(void* d_temp_storage, size_t& temp_storage_bytes,
                         const T* d_in, TIndex* d_out, int num_items,
                         cudaStream_t stream = 0,
                         bool debug_synchronous = false) {
    IsNonzero<T> is_nonzero;
    cub::TransformInputIterator<bool, IsNonzero<T>, const T*> is_nonzero_iter(
        d_in, is_nonzero);
    return cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes,
                                  is_nonzero_iter, d_out, num_items, stream,
                                  debug_synchronous);
  }
};

template <typename TIndex>
struct CubDeviceReduceCount<bool, TIndex> {
  cudaError_t operator()(void* d_temp_storage, size_t& temp_storage_bytes,
                         const bool* d_in, TIndex* d_out, int num_items,
                         cudaStream_t stream = 0,
                         bool debug_synchronous = false) {
    return cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_in,
                                  d_out, num_items, stream, debug_synchronous);
  }
};

template <typename T, typename TIndex, typename OutputIterator,
          bool IsConvertibleToBool>
struct CubDeviceSelectFlaggedCounter;

template <typename T, typename TIndex, typename OutputIterator>
struct CubDeviceSelectFlaggedCounter<T, TIndex, OutputIterator,
                                     false /*IsConvertibleToBool*/> {
  cudaError_t operator()(void* d_temp_storage, size_t& temp_storage_bytes,
                         const T* d_flags, OutputIterator d_out,
                         TIndex* d_num_selected_out, int num_items,
                         cudaStream_t stream = 0,
                         bool debug_synchronous = false) {
    cub::CountingInputIterator<TIndex> select_counter(0);
    IsNonzero<T> is_nonzero;
    cub::TransformInputIterator<bool, IsNonzero<T>, const T*> is_nonzero_iter(
        d_flags, is_nonzero);
    return cub::DeviceSelect::Flagged(
        d_temp_storage, temp_storage_bytes, select_counter /*d_in*/,
        is_nonzero_iter /*d_flags*/, d_out, d_num_selected_out, num_items,
        stream, debug_synchronous);
  }
};

template <typename T, typename TIndex, typename OutputIterator>
struct CubDeviceSelectFlaggedCounter<T, TIndex, OutputIterator,
                                     true /*IsConvertibleToBool*/> {
  cudaError_t operator()(void* d_temp_storage, size_t& temp_storage_bytes,
                         const T* d_flags, OutputIterator d_out,
                         TIndex* d_num_selected_out, int num_items,
                         cudaStream_t stream = 0,
                         bool debug_synchronous = false) {
    cub::CountingInputIterator<TIndex> select_counter(0);
    return cub::DeviceSelect::Flagged(
        d_temp_storage, temp_storage_bytes, select_counter /*d_in*/, d_flags,
        d_out, d_num_selected_out, num_items, stream, debug_synchronous);
  }
};

}  // namespace

template <typename T, typename TIndex>
struct NumTrue<GPUDevice, T, TIndex> {
  EIGEN_ALWAYS_INLINE static Status Compute(
      OpKernelContext* ctx, const GPUDevice& d,
      typename TTypes<T>::ConstFlat input,
      typename TTypes<TIndex>::Scalar num_true) {
    const cudaStream_t& cu_stream = GetCudaStream(ctx);

    std::size_t temp_storage_bytes = 0;
    const T* input_data = input.data();
    TIndex* num_true_data = num_true.data();

    // TODO(ebrevdo): sum doesn't work; perhaps need a different
    // iterator?
    auto reducer = CubDeviceReduceCount<T, TIndex>();
    auto first_success = reducer(/*temp_storage*/ nullptr, temp_storage_bytes,
                                 /*d_in*/ input_data,
                                 /*d_out*/ num_true_data,
                                 /*num_items*/ input.size(),
                                 /*stream*/ cu_stream);

    if (first_success != cudaSuccess) {
      return errors::Internal(
          "WhereOp: Could not launch cub::DeviceReduce::Sum to calculate "
          "temp_storage_bytes, status: ",
          cudaGetErrorString(first_success));
    }

    Tensor temp_storage;
    TF_RETURN_IF_ERROR(ctx->allocate_temp(
        DT_INT8, TensorShape({static_cast<int64>(temp_storage_bytes)}),
        &temp_storage));

    auto second_success = reducer(
        /*temp_storage*/ temp_storage.flat<int8>().data(), temp_storage_bytes,
        /*d_in*/ input_data,
        /*d_out*/ num_true_data,
        /*num_items*/ input.size(),
        /*stream*/ cu_stream);

    if (second_success != cudaSuccess) {
      return errors::Internal(
          "WhereOp: Could not launch cub::DeviceReduce::Sum to count "
          "number of true / nonzero indices.  temp_storage_bytes: ",
          temp_storage_bytes, ", status: ", cudaGetErrorString(second_success));
    }

    return Status::OK();
  }
};

#define NUMTRUE_GPU_FUNCTOR(T)                  \
  template struct NumTrue<GPUDevice, T, int32>; \
  template struct NumTrue<GPUDevice, T, int64>;

// We only need to declare the NumTrue functor once, but this file is
// included from where_op_gpu_impl_X.cu.cc for X=1,2,...
// Only declare for X = 1.
#if GPU_PROVIDED_DIM == 1

TF_CALL_WHERE_GPU_TYPES(NUMTRUE_GPU_FUNCTOR);

#endif  // GPU_PROVIDED_DIM == 1

#undef NUMTRUE_GPU_FUNCTOR

template <int NDIM>
class WhereOutputIterator {
 public:
  // Required iterator traits
  typedef WhereOutputIterator self_type;
  typedef std::ptrdiff_t difference_type;
  typedef void value_type;
  typedef void pointer;
  typedef int64& reference;

#if (THRUST_VERSION >= 100700)
  // Use Thrust's iterator categories so we can use these iterators in Thrust
  // 1.7 (or newer) methods
  typedef typename thrust::detail::iterator_facade_category<
      thrust::device_system_tag, thrust::random_access_traversal_tag,
      value_type,
      reference>::type iterator_category;  ///< The iterator category
#else
  typedef std::random_access_iterator_tag
      iterator_category;  ///< The iterator category
#endif  // THRUST_VERSION

  WhereOutputIterator(int64* ptr, const Eigen::DenseIndex max_row)
      : ptr_(ptr), max_row_(max_row) {}

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE int64& operator[](int n) const {
    // If the selection mechanism finds too many true values (because
    // the input tensor changed between allocation of output and now),
    // we may accidentally try to write past the allowable memory.  If
    // valid is false, then we don't do this.  Instead, we'll read off
    // the number of items found in Flagged()'s d_num_selected_out at
    // the end and confirm that it matches the number of rows of output.
    const bool valid = FastBoundsCheck(n, max_row_);
    return *(ptr_ + (valid ? (NDIM * n) : 0));
  }

 private:
  int64* ptr_;
  const Eigen::DenseIndex max_row_;
};

template <typename TIndex, typename T, int NDIM>
Eigen::array<TIndex, NDIM> CalculateStrides(
    typename TTypes<T, NDIM>::ConstTensor input) {
  const Eigen::DSizes<Eigen::DenseIndex, NDIM> dims = input.dimensions();
  Eigen::array<TIndex, NDIM> strides;
  EIGEN_STATIC_ASSERT((static_cast<int>(decltype(input)::Layout) ==
                       static_cast<int>(Eigen::RowMajor)),
                      INTERNAL_ERROR_INPUT_SHOULD_BE_ROWMAJOR);
  strides[NDIM - 1] = 1;
  for (int i = NDIM - 2; i >= 0; --i) {
    strides[i] = strides[i + 1] * dims[i + 1];
  }
  return strides;
}

template <int NDIM, typename T, typename TIndex>
struct Where<GPUDevice, NDIM, T, TIndex> {
  EIGEN_ALWAYS_INLINE static Status Compute(
      OpKernelContext* ctx, const GPUDevice& d,
      typename TTypes<T, NDIM>::ConstTensor input,
      typename TTypes<int64>::Matrix output, TIndex* found_true_host) {
    if (output.dimension(0) == 0) {
      // Nothing to do.
      return Status::OK();
    }

    const cudaStream_t& cu_stream = GetCudaStream(ctx);

    std::size_t temp_storage_bytes = 0;

    Tensor found_true_t;
    TF_RETURN_IF_ERROR(ctx->allocate_temp(DataTypeToEnum<TIndex>::v(),
                                          TensorShape({}), &found_true_t));
    TIndex* found_true_device = found_true_t.scalar<TIndex>().data();

    WhereOutputIterator<NDIM> output_iterator(
        output.data(),
        /* max_row */ output.dimension(0));

    typedef std::decay<T> DT;
    CubDeviceSelectFlaggedCounter<
        T, TIndex, decltype(output_iterator) /*OutputIterator*/,
        std::is_convertible<DT, bool>::value /*IsConvertibleToBool*/>
        counter;
    auto first_success = counter(/*temp_storage*/ nullptr, temp_storage_bytes,
                                 /*d_flags*/ input.data(),
                                 /*d_out*/ output_iterator,
                                 /*d_num_selected_out*/ found_true_device,
                                 /*num_items*/ input.size(),
                                 /*stream*/ cu_stream);
    if (first_success != cudaSuccess) {
      return errors::Internal(
          "WhereOp: Could not launch cub::DeviceSelect::Flagged to calculate "
          "temp_storage_bytes, status: ",
          cudaGetErrorString(first_success));
    }

    Tensor temp_storage;
    TF_RETURN_IF_ERROR(ctx->allocate_temp(
        DT_INT8, TensorShape({static_cast<int64>(temp_storage_bytes)}),
        &temp_storage));

    auto second_success = counter(
        /*temp_storage*/ temp_storage.flat<int8>().data(), temp_storage_bytes,
        /*d_flags*/ input.data(),
        /*d_out*/ output_iterator,
        /*d_num_selected_out*/ found_true_device,
        /*num_items*/ input.size(),
        /*stream*/ cu_stream);

    if (second_success != cudaSuccess) {
      return errors::Internal(
          "WhereOp: Could not launch cub::DeviceSelect::Flagged to copy "
          "indices out, status: ",
          cudaGetErrorString(second_success));
    }

    // TODO(ebrevdo): Find a way to synchronously copy back data from
    // found_true_device to *found_true_host.

    const Eigen::array<TIndex, NDIM> strides =
        CalculateStrides<TIndex, T, NDIM>(input);
    const TIndex output_rows = output.dimension(0);
    CudaLaunchConfig config = GetCudaLaunchConfig(output_rows, d);
    PropagateWhereIndicesKernel<NDIM, TIndex>
        <<<config.block_count, config.thread_per_block, 0, d.stream()>>>(
            output_rows, strides, output.data());

    return Status::OK();
  }
};

#define DECLARE_GPU_SPEC_INDEX(Dims, T, TIndex) \
  template struct Where<GPUDevice, Dims, T, TIndex>

#define DECLARE_GPU_SPEC(T)                           \
  DECLARE_GPU_SPEC_INDEX(GPU_PROVIDED_DIM, T, int32); \
  DECLARE_GPU_SPEC_INDEX(GPU_PROVIDED_DIM, T, int64)

TF_CALL_WHERE_GPU_TYPES(DECLARE_GPU_SPEC);

#undef DECLARE_GPU_SPEC
#undef DECLARE_GPU_SPEC_INDEX

}  // namespace functor

}  // namespace tensorflow

#endif  // GOOGLE_CUDA

#endif  // TENSORFLOW_CORE_KERNELS_WHERE_OP_GPU_CU_H_
