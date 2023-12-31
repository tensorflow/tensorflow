#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#define EIGEN_USE_GPU

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/kernels/sampled_addmm_op.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"

namespace tensorflow {

typedef Eigen::GpuDevice GPUDevice;

template <typename T>
__global__ void SampledADDMMCustomKernel(const int32_t* __restrict__ ind,
    const T* __restrict__ vals, const int32_t* __restrict__ ds,
    const T* __restrict__ mat1, const T* __restrict__ mat2,
    const int32_t batch_size, const T beta_, const T alpha_,
    const int32_t mat1_num_rows, const int32_t mat1_num_cols,
    const int32_t mat2_num_rows, const int32_t mat2_num_cols,
    const int32_t mat_num_batches, const int32_t sparse_rank,
    T* __restrict__ out) {
  const int32_t sparse_num_batches = sparse_rank == 3 ? ds[0] : 1;
  const int32_t sparse_num_rows = ds[sparse_rank == 2 ? 0 : 1];
  const int32_t sparse_num_cols = ds[sparse_rank == 2 ? 1 : 2];

  if (sparse_num_batches != mat_num_batches || sparse_num_rows != mat1_num_rows
      || sparse_num_cols != mat2_num_cols) {
    return;
  }

  GPU_1D_KERNEL_LOOP(batch_idx, mat_num_batches) {
    const int32_t sparse_batch_offset = batch_idx * batch_size;
    const int32_t indices_batch_offset = sparse_batch_offset * 2;
    const int32_t mat1_batch_offset = batch_idx * mat1_num_rows * mat1_num_cols;
    const int32_t mat2_batch_offset = batch_idx * mat2_num_rows * mat2_num_cols;
    
    for (int32_t i = 0; i < batch_size; ++i) {
      T val = vals[sparse_batch_offset + i];
      int32_t row_idx = ind[indices_batch_offset + 2 * i];
      int32_t col_idx = ind[indices_batch_offset + 2 * i + 1];
      T dot = 0;

      if (alpha_ != 0) {
        for (int32_t j = 0; j < mat1_num_cols; ++j) {
          auto mat1_idx = mat1_batch_offset + row_idx * mat1_num_cols + j;
          auto mat2_idx = mat2_batch_offset + j * mat2_num_cols + col_idx;
          dot += mat1[mat1_idx] * mat2[mat2_idx];
        }
      }

      out[sparse_batch_offset + i] = alpha_ * dot + beta_ * val;
    }
  }
}

namespace functor {

template <typename T>
struct SampledADDMMFunctor<GPUDevice, T> {
  static Status Compute(OpKernelContext* ctx, const Tensor& indices_t,
                     const Tensor& values_t, const Tensor& dense_shape_t,
                     const Tensor& mat1, const Tensor& mat2,
                     const int32_t batch_size, const T beta_, const T alpha_,
                     const int32_t mat1_num_rows, const int32_t mat1_num_cols,
                     const int32_t mat2_num_rows, const int32_t mat2_num_cols,
                     const int32_t mat_num_batches, const int32_t sparse_rank,
                     Tensor* out) {
    const GPUDevice& d = ctx->eigen_device<GPUDevice>();
    GpuLaunchConfig config = GetGpuLaunchConfig(mat_num_batches, d);
    
    TF_CHECK_OK(GpuLaunchKernel(SampledADDMMCustomKernel<T>,
                                config.block_count, config.thread_per_block,
                                0, d.stream(), indices_t.flat<int32_t>().data(),
                                values_t.flat<T>().data(), dense_shape_t.flat<int32_t>().data(),
                                mat1.flat<T>().data(), mat2.flat<T>().data(),
                                batch_size, beta_, alpha_, mat1_num_rows,
                                mat1_num_cols, mat2_num_rows, mat2_num_cols,
                                mat_num_batches, sparse_rank, 
                                out->flat<T>().data()));

    return OkStatus();
  }
};

} // namespace functor

template struct functor::SampledADDMMFunctor<GPUDevice, float>;

} // namespace tensorflow

#endif // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
