#define EIGEN_USE_THREADS

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/kernels/sampled_addmm_op.h"
#include "tensorflow/core/util/work_sharder.h"

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

namespace functor {

template <typename T>
struct SampledADDMMFunctor<CPUDevice, T> {
  static Status Compute(OpKernelContext* ctx, const Tensor& indices_t,
                     const Tensor& values_t, const Tensor& dense_shape_t,
                     const Tensor& mat1, const Tensor& mat2, 
                     const int32_t batch_size, const T beta_, const T alpha_,
                     const int32_t mat1_num_rows, const int32_t mat1_num_cols,
                     const int32_t mat2_num_rows, const int32_t mat2_num_cols,
                     const int32_t mat_num_batches, Tensor* out) {
    const T* mat1_ptr = mat1.flat<T>().data();
    const T* mat2_ptr = mat2.flat<T>().data();
    const T* values_ptr = values_t.flat<T>().data();
    const int32_t* indices_ptr = indices_t.flat<int32_t>().data();
    const int32_t* ds_ptr = dense_shape_t.flat<int32_t>().data();

    int32_t ds_num_rows = ds_ptr[0];
    int32_t ds_num_cols = ds_ptr[1];

    if (ds_num_rows != mat1_num_rows || ds_num_cols != mat2_num_cols) {
      return errors::InvalidArgument("Incorrect dense shape provided: dense_shape: (",
                                     ds_num_rows, ", ", ds_num_cols, "), expected: (",
                                     mat1_num_rows, ", ", mat2_num_cols, ")");
    } 

    auto output_flat = out->flat<T>();

    // Process the individual batches in parallel using a threadpool.
    auto shard = [&](int32_t batch_begin, int32_t batch_end) {
      for (int32_t batch_idx = batch_begin; batch_idx < batch_end; ++batch_idx) {
        const int32_t sparse_batch_offset = batch_idx * batch_size;
        const int32_t indices_batch_offset = sparse_batch_offset * 2;
        const int32_t mat1_batch_offset = batch_idx * mat1_num_rows * mat1_num_cols;
        const int32_t mat2_batch_offset = batch_idx * mat2_num_rows * mat2_num_cols;

        for (int32_t i = 0; i < batch_size; ++i) {
          T val = values_ptr[sparse_batch_offset + i];
          int32_t row_idx = indices_ptr[indices_batch_offset + 2 * i];
          int32_t col_idx = indices_ptr[indices_batch_offset + 2 * i + 1];
          T dot = 0;

          if (alpha_ != 0) {
            for (int32_t j = 0; j < mat1_num_cols; ++j) {
              int32_t mat1_idx = mat1_batch_offset + row_idx * mat1_num_cols + j;
              int32_t mat2_idx = mat2_batch_offset + j * mat2_num_cols + col_idx;
              dot += mat1_ptr[mat1_idx] * mat2_ptr[mat2_idx];
            }
          }

          output_flat(sparse_batch_offset + i) = alpha_ * dot + beta_ * val;
        }
      }
    };

    auto worker_threads = *(ctx->device()->tensorflow_cpu_worker_threads());
    Shard(worker_threads.num_threads, worker_threads.workers, mat_num_batches,
          batch_size, shard);

    return OkStatus();
  }
};

} // namespace functor

template <typename Device, typename T>
class SampledADDMMOp : public OpKernel {

  public:
    explicit SampledADDMMOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
      OP_REQUIRES_OK(ctx, ctx->GetAttr("beta", &beta_));
      OP_REQUIRES_OK(ctx, ctx->GetAttr("alpha", &alpha_));
    }

    void Compute(OpKernelContext* ctx) override {
      const Tensor& indices_t = ctx->input(0);
      const Tensor& values_t = ctx->input(1);
      const Tensor& dense_shape_t = ctx->input(2);
      const Tensor& mat1 = ctx->input(3);
      const Tensor& mat2 = ctx->input(4);

      const TensorShape& values_shape = values_t.shape();
      const int32_t batch_size = values_shape.dim_size(values_t.dims() - 1);

      const int32_t mat1_rank = mat1.dims();
      const int32_t mat2_rank = mat2.dims();

      const TensorShape& mat1_shape = mat1.shape();
      const TensorShape& mat2_shape = mat2.shape();
      
      const int32_t mat1_num_rows = mat1_shape.dim_size(mat1_rank - 2);
      const int32_t mat1_num_cols = mat1_shape.dim_size(mat1_rank - 1);
      const int32_t mat2_num_rows = mat2_shape.dim_size(mat2_rank - 2);
      const int32_t mat2_num_cols = mat2_shape.dim_size(mat2_rank - 1);
      const int32_t mat1_num_batches = mat1.NumElements() / (mat1_num_rows * mat1_num_cols); 

      Tensor* output = nullptr;
      OP_REQUIRES_OK(ctx, ctx->allocate_output(0, values_shape, &output));

      OP_REQUIRES_OK(ctx, functor::SampledADDMMFunctor<Device, T>::Compute(ctx,
                                     indices_t, values_t, dense_shape_t, 
                                     mat1, mat2, batch_size, 
                                     beta_, alpha_, 
                                     mat1_num_rows, mat1_num_cols, 
                                     mat2_num_rows, mat2_num_cols,
                                     mat1_num_batches, output));  
    }

  private:
    T beta_;
    T alpha_;
};

REGISTER_KERNEL_BUILDER(Name("SampledADDMM")            
                            .Device(DEVICE_CPU)         
                            .TypeConstraint<float>("T"),
                        SampledADDMMOp<CPUDevice, float>);

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
REGISTER_KERNEL_BUILDER(Name("SampledADDMM")
                            .Device(DEVICE_GPU)
                            .TypeConstraint<float>("T"),
                        SampledADDMMOp<GPUDevice, float>);
#endif // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

} // namespace tensorflow
