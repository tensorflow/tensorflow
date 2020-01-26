/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_KERNELS_CWISE_OP_FMA_H_
#define TENSORFLOW_CORE_KERNELS_CWISE_OP_FMA_H_

#define EIGEN_USE_THREADS

#include "tensorflow/core/platform/types.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace tensorflow {

enum FMAType { FMAType_Add, FMAType_Sub, FMAType_SubRev };

template <typename Device, typename T, FMAType Type>
class LaunchFusedMulAddOp {
 public:
  void operator()(const Device& device, T* out, const T* x1, const T* y1,
                  const T* x2, uint64 elements, bool broadcast_x1,
                  bool broadcast_y1, bool broadcast_x2);
};

template <typename Device, typename T, FMAType Type>
class LaunchFusedMulAdd2Op {
 public:
  void operator()(const Device& device, T* out, const T* x1, const T* y1,
                  const T* x2, const T* y2, uint64 elements, bool broadcast_x1,
                  bool broadcast_y1, bool broadcast_x2, bool broadcast_y2);
};

template <typename Device, typename T, FMAType Type>
class FallbackLaunchFusedMulAddOp {
 public:
  void operator()(const Device& device, T* out, const T* x1, const T* y1,
                  const T* x2, int64 dims[6], uint8 broadcast_masks[6]);
};

template <typename Device, typename T, FMAType Type>
class FallbackLaunchFusedMulAdd2Op {
 public:
  void operator()(const Device& device, T* out, const T* x1, const T* y1,
                  const T* x2, const T* y2, int64 dims[6],
                  uint8 broadcast_masks[6]);
};

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
typedef Eigen::GpuDevice GPUDevice;

template <typename T, FMAType Type>
class LaunchFusedMulAddOp<GPUDevice, T, Type> {
 public:
  typedef void exec_fun(const GPUDevice& device, T* out, const T* x1,
                        const T* y1, const T* x2, uint64 elements);

  template <int N>
  static void execute(const GPUDevice& device, T* out, const T* x1, const T* y1,
                      const T* x2, uint64 elements);

  void operator()(const GPUDevice& device, T* out, const T* x1, const T* y1,
                  const T* x2, uint64 elements, bool broadcast_x1,
                  bool broadcast_y1, bool broadcast_x2);
};

template <typename T, FMAType Type>
class LaunchFusedMulAdd2Op<GPUDevice, T, Type> {
 public:
  typedef void exec_fun(const GPUDevice& device, T* out, const T* x1,
                        const T* y1, const T* x2, const T* y2, uint64 elements);
  template <int N>
  static void execute(const GPUDevice& device, T* out, const T* x1, const T* y1,
                      const T* x2, const T* y2, uint64 elements);
  void operator()(const GPUDevice& device, T* out, const T* x1, const T* y1,
                  const T* x2, const T* y2, uint64 elements, bool broadcast_x1,
                  bool broadcast_y1, bool broadcast_x2, bool broadcast_y2);
};

template <typename T, FMAType Type>
class FallbackLaunchFusedMulAddOp<GPUDevice, T, Type> {
 public:
  void operator()(const GPUDevice& device, T* out, const T* x1, const T* y1,
                  const T* x2, int64 dims[6], uint8 broadcast_masks[6]);
};

template <typename T, FMAType Type>
class FallbackLaunchFusedMulAdd2Op<GPUDevice, T, Type> {
 public:
  void operator()(const GPUDevice& device, T* out, const T* x1, const T* y1,
                  const T* x2, const T* y2, int64 dims[6],
                  uint8 broadcast_masks[6]);
};
#endif

};  // namespace tensorflow

#endif
