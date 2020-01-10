#if GOOGLE_CUDA

#define EIGEN_USE_GPU

#include <stdio.h>

#include <memory>

#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor_types.h"

namespace tensorflow {

typedef Eigen::GpuDevice GPUDevice;

template <typename T>
void ConcatGPU(const GPUDevice& d,
               const std::vector<
                   std::unique_ptr<typename TTypes<T, 2>::ConstMatrix>>& inputs,
               typename TTypes<T, 2>::Matrix* output) {
  Eigen::array<ptrdiff_t, 2> offset(0, 0);
  for (int i = 0; i < inputs.size(); ++i) {
    Eigen::array<ptrdiff_t, 2> size = inputs[i]->dimensions();
    output->slice(offset, size).device(d) = *inputs[i];
    offset[1] += size[1];
  }
}

#define REGISTER_GPU(T)                                                       \
  template void ConcatGPU<T>(                                                 \
      const GPUDevice& d,                                                     \
      const std::vector<std::unique_ptr<typename TTypes<T, 2>::ConstMatrix>>& \
          inputs,                                                             \
      typename TTypes<T, 2>::Matrix* output);

TF_CALL_GPU_NUMBER_TYPES(REGISTER_GPU);
#undef REGISTER_GPU

}  // end namespace tensorflow

#endif  // GOOGLE_CUDA
