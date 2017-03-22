/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
#ifndef _TENSORFLOW_CORE_KERNELS_ADJUST_HUE_OP_H
#define _TENSORFLOW_CORE_KERNELS_ADJUST_HUE_OP_H

#if GOOGLE_CUDA
#define EIGEN_USE_GPU

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace tensorflow {

typedef Eigen::GpuDevice GPUDevice;

namespace functor {

struct AdjustHueGPU {
  void operator()(
      GPUDevice* device,
      const int64 number_of_elements,
      const float* const input,
      const float* const delta,
      float* const output
  );
};

} // namespace functor
} // namespace tensorflow

#endif // GOOGLE_CUDA
#endif // _TENSORFLOW_CORE_KERNELS_ADJUST_HUE_OP_H
