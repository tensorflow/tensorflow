#ifndef TENSORFLOW_CORE_KERNELS_SAMPLEDADDMM_OP_H_
#define TENSORFLOW_CORE_KERNELS_SAMPLEDADDMM_OP_H_

#include "tensorflow/core/framework/op_kernel.h"

namespace tensorflow {
namespace functor {

template <typename Device, typename T>
struct SampledADDMMFunctor {
  static Status Compute(OpKernelContext* ctx,
                     const Tensor& indices_t, const Tensor& values_t,
                     const Tensor& mat1, const Tensor& mat2,
                     const int32_t batch_size, const T beta_, const T alpha_,
                     const int32_t mat1_num_rows, const int32_t mat1_num_cols,
                     const int32_t mat2_num_rows, const int32_t mat2_num_cols,
                     const int32_t mat_num_batches, Tensor* out);
};

} // namespace functor
} // namespace tensorflow

#endif // TENSORFLOW_CORE_KERNELS_SAMPLEDADDMM_OP_H_
