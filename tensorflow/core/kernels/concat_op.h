#ifndef TENSORFLOW_KERNELS_CONCAT_OP_H_
#define TENSORFLOW_KERNELS_CONCAT_OP_H_

#include <vector>

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/device_base.h"

namespace tensorflow {

// Assumes all inputs are nonempty
template <typename T>
void ConcatCPU(DeviceBase* d,
               const std::vector<
                   std::unique_ptr<typename TTypes<T, 2>::ConstMatrix>>& inputs,
               typename TTypes<T, 2>::Matrix* output);

// Assumes all inputs are nonempty
template <typename T>
void ConcatGPU(const Eigen::GpuDevice& d,
               const std::vector<
                   std::unique_ptr<typename TTypes<T, 2>::ConstMatrix>>& inputs,
               typename TTypes<T, 2>::Matrix* output);

}  // namespace tensorflow

#endif  // TENSORFLOW_KERNELS_CONCAT_OP_H_
