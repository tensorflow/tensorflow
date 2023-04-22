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

#ifndef TENSORFLOW_CORE_KERNELS_SNAPSHOT_OP_H_
#define TENSORFLOW_CORE_KERNELS_SNAPSHOT_OP_H_

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#define EIGEN_USE_GPU
#endif

#define EIGEN_USE_THREADS

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op_kernel.h"

namespace tensorflow {
namespace functor {

// Functor used by SnapshotOp.
template <typename Device, typename Scalar>
struct Snapshot {
  void operator()(const Device& device,
                  typename TTypes<Scalar>::ConstTensor input,
                  typename TTypes<Scalar>::Tensor output) {
    device.memcpy(output.data(), input.data(), input.size() * sizeof(Scalar));
  }
};

}  // namespace functor
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_SNAPSHOT_OP_H_
