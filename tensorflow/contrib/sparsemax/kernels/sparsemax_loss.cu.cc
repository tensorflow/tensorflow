/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");

You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#if GOOGLE_CUDA

#define EIGEN_USE_GPU

#include "sparsemax_loss.h"
#include "tensorflow/core/framework/register_types.h"

namespace tensorflow {

typedef Eigen::GpuDevice GPUDevice;

// Compile the Eigen code for GPUDevice
template struct functor::SparsemaxLoss<GPUDevice, Eigen::half>;
template struct functor::SparsemaxLoss<GPUDevice, float>;
template struct functor::SparsemaxLoss<GPUDevice, double>;

}  // namespace tensorflow

#endif  // GOOGLE_CUDA
