/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/kernels/gpu_device_array.h"
#include "tensorflow/core/kernels/gpu_device_array_gpu.h"
#include "tensorflow/core/kernels/gpu_prim.h"
#include "tensorflow/core/kernels/sparse/kernels.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/cuda_sparse.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"

namespace tensorflow {

typedef Eigen::GpuDevice GPUDevice;

namespace functor {

namespace {
struct StridedDataReader {
  StridedDataReader(const int64* begin, int stride)
      : begin_(begin), stride_(stride) {}

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE int operator()(int idx) const {
    return static_cast<int>(ldg(begin_ + idx * stride_));
  }

  const int64* begin_;
  const int stride_;
};
}  // namespace

template <>
Status CalculateNNZPerBatchMatrixFromIndices<GPUDevice>::operator()(
    OpKernelContext* c, TTypes<int64_t>::ConstMatrix indices,
    TTypes<int32>::Vec nnz_per_batch) {
  const auto& cu_stream = GetGpuStream(c);

  const int total_nnz = indices.dimension(0);
  const int size = nnz_per_batch.size();

  DCHECK_EQ(indices.rank(), 2);
  DCHECK_EQ(indices.dimension(1), 3);  // batch, row, col

  const int rank = indices.dimension(1);
  gpuprim::CountingInputIterator<int> row_counter(0);
  gpuprim::TransformInputIterator<int, StridedDataReader,
                                  gpuprim::CountingInputIterator<int>>
      indices_first_column(row_counter,
                           StridedDataReader(indices.data(), rank));

  std::size_t temp_storage_bytes = 0;

  DCHECK_NE(indices.data(), nullptr);
  DCHECK_NE(nnz_per_batch.data(), nullptr);

  auto first_success = gpuprim::DeviceHistogram::HistogramEven(
      /*d_temp_storage*/ nullptr,
      /*temp_storage_bytes&*/ temp_storage_bytes,
      /*d_samples*/ indices_first_column,
      /*d_histogram*/ nnz_per_batch.data(),
      /*num_levels*/ size + 1,
      /*lower_level*/ 0,
      /*upper_level*/ size,
      /*num_samples*/ total_nnz,
      /*stream*/ cu_stream);

  if (first_success != gpuSuccess) {
    return errors::Internal(
        "SparseTensorToCSRSparseMatrix: Could not launch "
        "gpuprim::DeviceHistogram::HistogramEven "
        "to calculate temp_storage_bytes, status: ",
        GpuGetErrorString(first_success));
  }

  Tensor temp_storage;
  TF_RETURN_IF_ERROR(c->allocate_temp(
      DT_INT8, TensorShape({static_cast<int64_t>(temp_storage_bytes)}),
      &temp_storage));
  DCHECK_NE(temp_storage.flat<int8>().data(), nullptr);
  auto second_success = gpuprim::DeviceHistogram::HistogramEven(
      /*d_temp_storage*/ temp_storage.flat<int8>().data(),
      /*temp_storage_bytes&*/ temp_storage_bytes,
      /*d_samples*/ indices_first_column,
      /*d_histogram*/ nnz_per_batch.data(),
      /*num_levels*/ size + 1,
      /*lower_level*/ 0,
      /*upper_level*/ size,
      /*num_samples*/ total_nnz,
      /*stream*/ cu_stream);

  if (second_success != gpuSuccess) {
    return errors::Internal(
        "SparseTensorToCSRSparseMatrix: Could not launch "
        "gpuprim::DeviceHistogram::HistogramEven "
        "to count nnz entries per batch.  temp_storage_bytes: ",
        temp_storage_bytes, ", status: ", GpuGetErrorString(second_success));
  }

  return Status::OK();
}

// TODO(ebrevdo): Write a custom batch-friendly impl of this to update
// the SparseTensor indices directly.
template <>
Status CSRSparseMatrixToCOOSparseMatrix<GPUDevice>::operator()(
    OpKernelContext* c, TTypes<const int>::UnalignedVec csr_row_ptr,
    TTypes<int>::UnalignedVec coo_row_ind) {
  GpuSparse gpu_sparse(c);
  const int nnz = coo_row_ind.size();
  TF_RETURN_IF_ERROR(gpu_sparse.Initialize());
  const int m = csr_row_ptr.size() - 1;  // rows
  return gpu_sparse.Csr2coo(csr_row_ptr.data(), nnz, m, coo_row_ind.data());
}

template <int stride>
__global__ void SparseTensorToCOOMatrixKernel(const int64* indices,
                                              int* coo_rows_out,
                                              int* coo_cols_out, int size) {
  const int offset = (stride == 3) ? 1 : 0;
  GPU_1D_KERNEL_LOOP(i, size) {
    coo_rows_out[i] = static_cast<int>(ldg(indices + i * stride + offset));
    coo_cols_out[i] = static_cast<int>(ldg(indices + i * stride + offset + 1));
  }
}

template <>
void SparseTensorToCOOSparseMatrix<GPUDevice>::operator()(
    const GPUDevice& d, TTypes<int64_t>::ConstVec host_dense_shape,
    TTypes<int64_t>::ConstMatrix indices, TTypes<int>::Vec coo_row_ind,
    TTypes<int>::Vec coo_col_ind) {
  const int stride = host_dense_shape.size();
  DCHECK(stride == 2 || stride == 3);
  DCHECK_EQ(stride, indices.dimension(1));
  const int size = coo_row_ind.dimension(0);
  GpuLaunchConfig config = GetGpuLaunchConfig(size, d);
  if (stride == 2) {
    TF_CHECK_OK(GpuLaunchKernel(SparseTensorToCOOMatrixKernel<2>,
                                config.block_count, config.thread_per_block, 0,
                                d.stream(), indices.data(), coo_row_ind.data(),
                                coo_col_ind.data(), size));
  } else {
    TF_CHECK_OK(GpuLaunchKernel(SparseTensorToCOOMatrixKernel<3>,
                                config.block_count, config.thread_per_block, 0,
                                d.stream(), indices.data(), coo_row_ind.data(),
                                coo_col_ind.data(), size));
  }
}

__global__ void COOMatrixToSparseTensorKernel2D(const int* coo_rows,
                                                const int* coo_cols,
                                                int64* indices_out, int size) {
  GPU_1D_KERNEL_LOOP(i, size) {
    indices_out[i * 2] = static_cast<int64_t>(ldg(coo_rows + i));
    indices_out[i * 2 + 1] = static_cast<int64_t>(ldg(coo_cols + i));
  }
}

__device__ inline int BinarySearchRange(int* range, int n, int x) {
  int left = 0;
  int right = n - 1;
  while (left < right) {
    int mid = left + (right - left) / 2;
    if (x < range[mid])
      right = mid - 1;
    else if (range[mid + 1] <= x)
      left = mid + 1;
    else
      return mid;  // range[mid] <= x < range[mid + 1].
  }
  return left;
}

__global__ void COOMatrixToSparseTensorKernel3D(
    const int* coo_rows, const int* coo_cols, int64* indices_out,
    GpuDeviceArrayStruct<int> batch_ptr_s, const int batch_size,
    const int size) {
  // Step 1: access the batch ptrs and copy to shared memory.
  const int* batch_ptr = GetGpuDeviceArrayOnDevice(&batch_ptr_s);
  extern __shared__ int local_batch_ptr[];
  for (int i = threadIdx.x; i < batch_size + 1; i += blockDim.x) {
    local_batch_ptr[i] = batch_ptr[i];
  }
  __syncthreads();

  GPU_1D_KERNEL_LOOP(i, size) {
    // TODO(ebrevdo): Consider special casing batch_size <= 3,
    // alternatively doing linear instead of binary search.  Requires
    // some benchmarks.
    const int b = BinarySearchRange(local_batch_ptr, batch_size, i);
    indices_out[i * 3] = static_cast<int64_t>(b);
    indices_out[i * 3 + 1] = static_cast<int64_t>(ldg(coo_rows + i));
    indices_out[i * 3 + 2] = static_cast<int64_t>(ldg(coo_cols + i));
  }
}

template <>
Status COOSparseMatrixToSparseTensor<GPUDevice>::operator()(
    OpKernelContext* c, TTypes<int64_t>::ConstVec host_dense_shape,
    TTypes<int>::ConstVec host_batch_ptr, TTypes<int>::Vec coo_row_ind,
    TTypes<int>::ConstVec coo_col_ind, TTypes<int64_t>::Matrix indices) {
  const int ndims = indices.dimension(1);
  DCHECK(ndims == 2 || ndims == 3);
  DCHECK_EQ(ndims, host_dense_shape.size());
  DCHECK_NE(coo_row_ind.data(), nullptr);
  DCHECK_NE(coo_col_ind.data(), nullptr);
  DCHECK_NE(indices.data(), nullptr);
  const GPUDevice& d = c->eigen_device<GPUDevice>();
  const int size = coo_row_ind.size();
  DCHECK_EQ(size, coo_col_ind.size());
  DCHECK_EQ(size, indices.dimension(0));
  if (ndims == 2) {
    GpuLaunchConfig config = GetGpuLaunchConfig(size, d);
    TF_CHECK_OK(GpuLaunchKernel(COOMatrixToSparseTensorKernel2D,
                                config.block_count, config.thread_per_block, 0,
                                d.stream(), coo_row_ind.data(),
                                coo_col_ind.data(), indices.data(), size));
    return Status::OK();
  } else {
    const int batch_size = host_dense_shape(0);
    GpuDeviceArrayOnHost<int> batch_ptr_copy(c, host_batch_ptr.size());
    TF_RETURN_IF_ERROR(batch_ptr_copy.Init());
    for (int i = 0; i < batch_size; ++i) {
      batch_ptr_copy.Set(i, host_batch_ptr(i));
    }
    TF_RETURN_IF_ERROR(batch_ptr_copy.Finalize());
    GpuLaunchConfig config = GetGpuLaunchConfig(size, d);
    // shared memory stores the batch pointers.
    const size_t shared_memory_size = sizeof(int) * (batch_size + 1);
    TF_CHECK_OK(
        GpuLaunchKernel(COOMatrixToSparseTensorKernel3D, config.block_count,
                        config.thread_per_block, shared_memory_size, d.stream(),
                        coo_row_ind.data(), coo_col_ind.data(), indices.data(),
                        batch_ptr_copy.data(), batch_size, size));
    return Status::OK();
  }
}

template <typename T>
__global__ void CSRSparseMatrixBatchMulVecKernel3D(
    const T* a_values, const T* b_batch_values, T* c_values,
    GpuDeviceArrayStruct<int> batch_ptr_s, const int batch_size,
    const int total_nnz) {
  // Step 1: Access the batch ptrs and copy to shared memory.
  //         Also copy the per-batch multipliers into shared memory.
  const int* batch_ptr = GetGpuDeviceArrayOnDevice(&batch_ptr_s);
  extern __shared__ int local_batch_ptr[];
  T* local_batch_values =
      reinterpret_cast<T*>(local_batch_ptr + batch_size + 1);
  for (int i = threadIdx.x; i < batch_size + 1; i += blockDim.x) {
    local_batch_ptr[i] = batch_ptr[i];
    if (i < batch_size) {
      local_batch_values[i] = b_batch_values[i];
    }
  }
  __syncthreads();

  GPU_1D_KERNEL_LOOP(i, total_nnz) {
    const int b = BinarySearchRange(local_batch_ptr, batch_size, i);
    c_values[i] = ldg(a_values + i) * local_batch_values[b];
  }
}

template <typename T>
Status CSRSparseMatrixBatchMulVecImpl(OpKernelContext* ctx,
                                      const CSRSparseMatrix& a,
                                      typename TTypes<T>::ConstFlat b,
                                      CSRSparseMatrix* c) {
  DCHECK_EQ(a.dims(), 3);
  const int total_nnz = a.total_nnz();
  Tensor c_values_t;
  TF_RETURN_IF_ERROR(ctx->allocate_temp(DataTypeToEnum<T>::value,
                                        TensorShape({total_nnz}), &c_values_t));
  TF_RETURN_IF_ERROR(CSRSparseMatrix::CreateCSRSparseMatrix(
      DataTypeToEnum<T>::value, a.dense_shape(), a.batch_pointers(),
      a.row_pointers(), a.col_indices(), c_values_t, c));

  auto a_values = a.values().flat<T>();
  auto c_values = c_values_t.flat<T>();

  auto host_dense_shape = a.dense_shape().vec<int64_t>();
  auto host_batch_ptr = a.batch_pointers().vec<int>();

  const GPUDevice& d = ctx->eigen_device<GPUDevice>();

  const int batch_size = host_dense_shape(0);
  DCHECK_EQ(b.size(), batch_size);

  GpuDeviceArrayOnHost<int> batch_ptr_copy(ctx, host_batch_ptr.size());
  TF_RETURN_IF_ERROR(batch_ptr_copy.Init());
  for (int i = 0; i < batch_size; ++i) {
    batch_ptr_copy.Set(i, host_batch_ptr(i));
  }
  TF_RETURN_IF_ERROR(batch_ptr_copy.Finalize());
  GpuLaunchConfig config = GetGpuLaunchConfig(total_nnz, d);
  // shared memory stores the batch pointers.
  const size_t shared_memory_size =
      (sizeof(int) * (batch_size + 1)  // local batch_pointers.
       + sizeof(T) * batch_size);      // local copy of b.
  TF_CHECK_OK(GpuLaunchKernel(
      CSRSparseMatrixBatchMulVecKernel3D<T>, config.block_count,
      config.thread_per_block, shared_memory_size, d.stream(), a_values.data(),
      b.data(), c_values.data(), batch_ptr_copy.data(), batch_size, total_nnz));

  return Status::OK();
}

#define DEFINE_SPARSE_MUL_VEC_GPU(T)                                        \
  template <>                                                               \
  CSRSparseMatrixBatchMulVec<GPUDevice, T>::CSRSparseMatrixBatchMulVec() {} \
  template <>                                                               \
  Status CSRSparseMatrixBatchMulVec<GPUDevice, T>::Compute(                 \
      OpKernelContext* ctx, const CSRSparseMatrix& a,                       \
      typename TTypes<T>::ConstFlat b, CSRSparseMatrix* c) {                \
    return CSRSparseMatrixBatchMulVecImpl<T>(ctx, a, b, c);                 \
  }

DEFINE_SPARSE_MUL_VEC_GPU(float);
DEFINE_SPARSE_MUL_VEC_GPU(double);
DEFINE_SPARSE_MUL_VEC_GPU(std::complex<float>);
DEFINE_SPARSE_MUL_VEC_GPU(std::complex<double>);

#undef DEFINE_SPARSE_MUL_VEC_GPU

template <typename T>
EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC void CalculateRowSoftmax(const int begin,
                                                               const int end,
                                                               const T* logits,
                                                               T* softmax) {
  // For each row, calculate the vector:
  //   softmax[row] = exp(shifted_logits[row]) / sum(exp(shifted_logits[row]))
  // where
  //   shifted_logits[row] = logits[row] - max(logits[row])
  // are the logits normalized for stability.
  T row_max = Eigen::NumTraits<T>::lowest();
  for (int r_i = begin; r_i < end; ++r_i) {
    row_max = Eigen::numext::maxi(row_max, ldg(logits + r_i));
  }
  T sum_exp = 0;
  for (int r_i = begin; r_i < end; ++r_i) {
    const T exp_i = Eigen::numext::exp(ldg(logits + r_i) - row_max);
    softmax[r_i] = exp_i;
    sum_exp += exp_i;
  }
  for (int r_i = begin; r_i < end; ++r_i) {
    softmax[r_i] = softmax[r_i] / sum_exp;
  }
}

template <typename T>
__global__ void CSRSparseMatrixSoftmaxKernel2D(const int rows,
                                               const int* row_ptr,
                                               const T* logits, T* softmax) {
  // TODO(ebrevdo): consider something like a merge-path based
  // algorithm to distribute the work in case the row sizes are
  // uneven:
  //   http://images.nvidia.com/events/sc15/pdfs/sc15-Merge-Based-Parallel-Sparse-Matrix-Vector-Multiplication-merrill.pdf
  GPU_1D_KERNEL_LOOP(row, rows) {
    CalculateRowSoftmax(ldg(row_ptr + row), ldg(row_ptr + row + 1), logits,
                        softmax);
  }
}

EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void CopyFromGpuDeviceArrayToLocal(
    GpuDeviceArrayStruct<int> cuda_ptr_s, int* local_ptr, int length) {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
  const int* cuda_ptr = GetGpuDeviceArrayOnDevice(&cuda_ptr_s);
  for (int i = threadIdx.x; i < length; i += blockDim.x) {
    local_ptr[i] = cuda_ptr[i];
  }
  __syncthreads();
#endif
}

template <typename T>
__global__ void CSRSparseMatrixSoftmaxKernel3D(
    const int size, const int rows, GpuDeviceArrayStruct<int> batch_ptr_s,
    const int* row_ptr, const T* logits, T* softmax) {
  // TODO(ebrevdo): consider something like a merge-path based
  // algorithm to distribute the work in case the row sizes are
  // uneven:
  //   http://images.nvidia.com/events/sc15/pdfs/sc15-Merge-Based-Parallel-Sparse-Matrix-Vector-Multiplication-merrill.pdf
  const int batch_size = size / rows;
  extern __shared__ int local_batch_ptr[];
  CopyFromGpuDeviceArrayToLocal(std::move(batch_ptr_s), local_batch_ptr,
                                batch_size + 1);

  GPU_1D_KERNEL_LOOP(i, size) {
    const int batch = i / rows;
    const int row = i % rows;
    const int batch_offset = local_batch_ptr[batch];
    const int row_offset = batch * (rows + 1) + row;
    CalculateRowSoftmax(batch_offset + ldg(row_ptr + row_offset),
                        batch_offset + ldg(row_ptr + row_offset + 1), logits,
                        softmax);
  }
}

template <typename T>
Status CSRSparseMatrixSoftmaxGPUImpl(OpKernelContext* ctx,
                                     const CSRSparseMatrix& logits,
                                     typename TTypes<T>::Vec softmax_values) {
  auto host_dense_shape = logits.dense_shape().vec<int64_t>();
  auto host_batch_ptr = logits.batch_pointers().vec<int32>();
  auto row_ptr = logits.row_pointers().vec<int32>();
  auto logits_values = logits.values().vec<T>();

  const int ndims = host_dense_shape.size();
  DCHECK(ndims == 2 || ndims == 3);
  const GPUDevice& d = ctx->eigen_device<GPUDevice>();
  if (ndims == 2) {
    const int rows = host_dense_shape(0);
    DCHECK_EQ(rows, row_ptr.size() - 1);
    GpuLaunchConfig config = GetGpuLaunchConfig(rows /*size*/, d);
    TF_CHECK_OK(GpuLaunchKernel(CSRSparseMatrixSoftmaxKernel2D<T>,
                                config.block_count, config.thread_per_block, 0,
                                d.stream(), rows /*size*/, row_ptr.data(),
                                logits_values.data(), softmax_values.data()));
  } else {
    const int batch_size = host_dense_shape(0);
    const int rows = host_dense_shape(1);
    DCHECK_EQ(batch_size, host_batch_ptr.size() - 1);
    DCHECK_EQ((rows + 1) * batch_size, row_ptr.size());
    const int size = rows * batch_size;

    GpuDeviceArrayOnHost<int> batch_ptr_copy(ctx, host_batch_ptr.size());
    TF_RETURN_IF_ERROR(batch_ptr_copy.Init());
    for (int i = 0; i < host_batch_ptr.size(); ++i) {
      batch_ptr_copy.Set(i, host_batch_ptr(i));
    }
    TF_RETURN_IF_ERROR(batch_ptr_copy.Finalize());

    GpuLaunchConfig config = GetGpuLaunchConfig(size, d);
    // shared memory stores the batch pointers.
    const size_t shared_memory_size = sizeof(int) * (batch_size + 1);
    TF_CHECK_OK(GpuLaunchKernel(CSRSparseMatrixSoftmaxKernel3D<T>,
                                config.block_count, config.thread_per_block,
                                shared_memory_size, d.stream(), size, rows,
                                batch_ptr_copy.data(), row_ptr.data(),
                                logits_values.data(), softmax_values.data()));
  }

  return Status::OK();
}

#define DEFINE_SOFTMAX_GPU(T)                                             \
  template <>                                                             \
  Status CSRSparseMatrixSoftmax<GPUDevice, T>::operator()(                \
      OpKernelContext* ctx, const CSRSparseMatrix& logits,                \
      typename TTypes<T>::Vec softmax_values) {                           \
    return CSRSparseMatrixSoftmaxGPUImpl<T>(ctx, logits, softmax_values); \
  }

DEFINE_SOFTMAX_GPU(float);
DEFINE_SOFTMAX_GPU(double);

#undef DEFINE_SOFTMAX_GPU

template <typename T>
EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC void CalculateRowSoftmaxGrad(
    const int softmax_begin, const int softmax_end, const int* softmax_col_ind,
    const T* softmax, const int grad_softmax_begin, const int grad_softmax_end,
    const int* grad_softmax_col_ind, const T* grad_softmax, T* gradient) {
  // Iterate from
  //   softmax_col_ind[softmax_begin] to
  //   softmax_col_ind[softmax_end]
  // and from
  //  grad_softmax_col_ind[grad_softmax_begin] to
  //  grad_softmax_col_ind[grad_softmax_end]
  //
  // looking for for matching indices.  In the softmax indices only, perform:
  //
  //   gradient = (grad_softmax - sum(grad_softmax * softmax)) * softmax
  //
  // where the sum is along the given row.
  T sum_prod = 0;
  for (int i = softmax_begin, j = grad_softmax_begin;
       i < softmax_end && j < grad_softmax_end;) {
    const int softmax_col = ldg(softmax_col_ind + i);
    const int grad_softmax_col = ldg(grad_softmax_col_ind + j);
    if (softmax_col == grad_softmax_col) {
      sum_prod += ldg(softmax + i) * ldg(grad_softmax + j);
      ++i;
      ++j;
    } else if (softmax_col > grad_softmax_col) {
      ++j;
    } else {
      ++i;
    }
  }

  // Find an upper bound on the column numbers in this row; for use in
  // the special case of a empty grad_softmax row and a non-empty
  // softmax row.
  const int softmax_col_upper_bound =
      (softmax_begin == softmax_end)
          ? -1
          : ldg(softmax_col_ind + softmax_end - 1) + 1;
  for (int i = softmax_begin, j = grad_softmax_begin; i < softmax_end;) {
    const int softmax_col = ldg(softmax_col_ind + i);
    // We need to keep a large grad_softmax_col value if we're at the
    // end of the grad_softmax row, so we can fill in the remainder of
    // the gradients row (the last if branch in this loop).
    const int grad_softmax_col = (j == grad_softmax_end)
                                     ? softmax_col_upper_bound
                                     : ldg(grad_softmax_col_ind + j);

    if (softmax_col == grad_softmax_col) {
      gradient[i] = (ldg(grad_softmax + j) - sum_prod) * ldg(softmax + i);
      ++i;
      ++j;
    } else if (softmax_col > grad_softmax_col) {
      // grad_softmax is nonzero here, but since softmax is zero, the
      // gradient is 0; so we skip it since the sparsity structure
      // already encodes this zero.
      ++j;
    } else {
      // grad_softmax is zero but softmax is not.
      gradient[i] = -sum_prod * ldg(softmax + i);
      ++i;
    }
  }
}

template <typename T>
__global__ void CSRSparseMatrixSoftmaxGradKernel2D(
    const int rows, const int* softmax_row_ptr, const int* softmax_col_ind,
    const T* softmax, const int* grad_softmax_row_ptr,
    const int* grad_softmax_col_ind, const T* grad_softmax, T* gradient) {
  // TODO(ebrevdo): consider something like a merge-path based
  // algorithm to distribute the work in case the row sizes are
  // uneven:
  //   http://images.nvidia.com/events/sc15/pdfs/sc15-Merge-Based-Parallel-Sparse-Matrix-Vector-Multiplication-merrill.pdf
  GPU_1D_KERNEL_LOOP(row, rows) {
    CalculateRowSoftmaxGrad(
        ldg(softmax_row_ptr + row) /*softmax_begin*/,
        ldg(softmax_row_ptr + row + 1) /*softmax_end*/, softmax_col_ind,
        softmax, ldg(grad_softmax_row_ptr + row) /*grad_softmax_begin*/,
        ldg(grad_softmax_row_ptr + row + 1) /*grad_softmax_end*/,
        grad_softmax_col_ind, grad_softmax, gradient);
  }
}

template <typename T>
__global__ void CSRSparseMatrixSoftmaxGradKernel3D(
    const int size, const int rows,
    GpuDeviceArrayStruct<int> softmax_and_grad_batch_ptr_s,
    const int* softmax_row_ptr, const int* softmax_col_ind, const T* softmax,
    const int* grad_softmax_row_ptr, const int* grad_softmax_col_ind,
    const T* grad_softmax, T* gradient) {
  // TODO(ebrevdo): consider something like a merge-path based
  // algorithm to distribute the work in case the row sizes are
  // uneven:
  //   http://images.nvidia.com/events/sc15/pdfs/sc15-Merge-Based-Parallel-Sparse-Matrix-Vector-Multiplication-merrill.pdf

  const int batch_size = size / rows;
  extern __shared__ int local_batch_ptr[];
  CopyFromGpuDeviceArrayToLocal(std::move(softmax_and_grad_batch_ptr_s),
                                local_batch_ptr, 2 * (batch_size + 1));

#define SOFTMAX_BATCH_PTR(i) local_batch_ptr[i];
#define GRAD_SOFTMAX_BATCH_PTR(i) local_batch_ptr[batch_size + 1 + i];

  GPU_1D_KERNEL_LOOP(i, size) {
    const int batch = i / rows;
    const int row = i % rows;
    const int softmax_batch_offset = SOFTMAX_BATCH_PTR(batch);
    const int grad_softmax_batch_offset = GRAD_SOFTMAX_BATCH_PTR(batch);
    const int row_offset = batch * (rows + 1) + row;
    CalculateRowSoftmaxGrad(
        softmax_batch_offset +
            ldg(softmax_row_ptr + row_offset) /*softmax_begin*/,
        softmax_batch_offset +
            ldg(softmax_row_ptr + row_offset + 1) /*softmax_end*/,
        softmax_col_ind, softmax,
        grad_softmax_batch_offset +
            ldg(grad_softmax_row_ptr + row_offset) /*grad_softmax_begin*/,
        grad_softmax_batch_offset +
            ldg(grad_softmax_row_ptr + row_offset + 1) /*grad_softmax_end*/,
        grad_softmax_col_ind, grad_softmax, gradient);
  }

#undef SOFTMAX_BATCH_PTR
#undef GRAD_SOFTMAX_BATCH_PTR
}

template <typename T>
Status CSRSparseMatrixSoftmaxGradGPUImpl(
    OpKernelContext* ctx, const CSRSparseMatrix& softmax,
    const CSRSparseMatrix& grad_softmax,
    typename TTypes<T>::Vec gradient_values) {
  auto host_dense_shape = softmax.dense_shape().vec<int64_t>();
  auto softmax_host_batch_ptr = softmax.batch_pointers().vec<int32>();
  auto softmax_row_ptr = softmax.row_pointers().vec<int32>();
  auto softmax_col_ind = softmax.col_indices().vec<int32>();
  auto softmax_values = softmax.values().vec<T>();
  auto grad_softmax_host_batch_ptr = grad_softmax.batch_pointers().vec<int32>();
  auto grad_softmax_row_ptr = grad_softmax.row_pointers().vec<int32>();
  auto grad_softmax_col_ind = grad_softmax.col_indices().vec<int32>();
  auto grad_softmax_values = grad_softmax.values().vec<T>();

  const int ndims = host_dense_shape.size();
  DCHECK(ndims == 2 || ndims == 3);
  const int rows = host_dense_shape(0);
  const GPUDevice& d = ctx->eigen_device<GPUDevice>();
  if (ndims == 2) {
    DCHECK_EQ(rows + 1, softmax_row_ptr.size());
    DCHECK_EQ(rows + 1, grad_softmax_row_ptr.size());
    GpuLaunchConfig config = GetGpuLaunchConfig(rows /*size*/, d);
    TF_CHECK_OK(GpuLaunchKernel(
        CSRSparseMatrixSoftmaxGradKernel2D<T>, config.block_count,
        config.thread_per_block, 0, d.stream(), rows /*size*/,
        softmax_row_ptr.data(), softmax_col_ind.data(), softmax_values.data(),
        grad_softmax_row_ptr.data(), grad_softmax_col_ind.data(),
        grad_softmax_values.data(), gradient_values.data()));
  } else {
    const int batch_size = host_dense_shape(0);
    const int rows = host_dense_shape(1);
    DCHECK_EQ(batch_size, softmax_host_batch_ptr.size() - 1);
    DCHECK_EQ(batch_size, grad_softmax_host_batch_ptr.size() - 1);
    DCHECK_EQ((rows + 1) * batch_size, softmax_row_ptr.size());
    DCHECK_EQ((rows + 1) * batch_size, grad_softmax_row_ptr.size());
    const int size = rows * batch_size;
    // The length of softmax_and_grad_batch_ptr_copy is 2 * (batch_size + 1)
    // The first (batch_size + 1) entries contain softmax_batch_ptr and
    // the second (batch_size + 1) entries contain grad_softmax_batch_ptr.
    GpuDeviceArrayOnHost<int> softmax_and_grad_batch_ptr_copy(
        ctx, 2 * softmax_host_batch_ptr.size());
    TF_RETURN_IF_ERROR(softmax_and_grad_batch_ptr_copy.Init());
    for (int i = 0; i < softmax_host_batch_ptr.size(); ++i) {
      softmax_and_grad_batch_ptr_copy.Set(i, softmax_host_batch_ptr(i));
      softmax_and_grad_batch_ptr_copy.Set(batch_size + 1 + i,
                                          grad_softmax_host_batch_ptr(i));
    }
    TF_RETURN_IF_ERROR(softmax_and_grad_batch_ptr_copy.Finalize());

    GpuLaunchConfig config = GetGpuLaunchConfig(size, d);
    // shared memory stores two copies of batch pointers: one for the
    // softmax CSR matrix, one for the grad_softmax CSR matrix.
    const size_t shared_memory_size = 2 * sizeof(int) * (batch_size + 1);
    TF_CHECK_OK(GpuLaunchKernel(
        CSRSparseMatrixSoftmaxGradKernel3D<T>, config.block_count,
        config.thread_per_block, shared_memory_size, d.stream(), size, rows,
        softmax_and_grad_batch_ptr_copy.data(), softmax_row_ptr.data(),
        softmax_col_ind.data(), softmax_values.data(),
        grad_softmax_row_ptr.data(), grad_softmax_col_ind.data(),
        grad_softmax_values.data(), gradient_values.data()));
  }

  return Status::OK();
}

#define DEFINE_SOFTMAX_GRAD_GPU(T)                                          \
  template <>                                                               \
  Status CSRSparseMatrixSoftmaxGrad<GPUDevice, T>::operator()(              \
      OpKernelContext* ctx, const CSRSparseMatrix& softmax,                 \
      const CSRSparseMatrix& grad_softmax,                                  \
      typename TTypes<T>::Vec gradient_values) {                            \
    return CSRSparseMatrixSoftmaxGradGPUImpl<T>(ctx, softmax, grad_softmax, \
                                                gradient_values);           \
  }

DEFINE_SOFTMAX_GRAD_GPU(float);
DEFINE_SOFTMAX_GRAD_GPU(double);

#undef DEFINE_SOFTMAX_GRAD_GPU

}  // namespace functor

}  // namespace tensorflow

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
