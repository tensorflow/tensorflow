/* Copyright 2015 Google Inc. All Rights Reserved.

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

#if GOOGLE_CUDA

#define EIGEN_USE_GPU

#include "tensorflow/core/framework/bfloat16.h"
#include "tensorflow/core/kernels/cast_op.h"

namespace tensorflow {
namespace functor {

typedef Eigen::GpuDevice GPUDevice;

template <typename O, typename I>
struct CastFunctor<GPUDevice, O, I> {
  void operator()(const GPUDevice& d, typename TTypes<O>::Flat o,
                  typename TTypes<I>::ConstFlat i) {
    Cast<GPUDevice, O, I>(d, o, i);
  }
};

#define DEFINE(O, I) template struct CastFunctor<GPUDevice, O, I>;
DEFINE(float, double);
DEFINE(float, int32);
DEFINE(float, int64);
DEFINE(double, float);
DEFINE(double, int32);
DEFINE(double, int64);
DEFINE(int32, float);
DEFINE(int32, double);
DEFINE(int32, int64);
DEFINE(int64, float);
DEFINE(int64, double);
DEFINE(int64, int32);
DEFINE(int32, bool);
DEFINE(float, bool);
DEFINE(float, uint8);
DEFINE(uint8, float);
DEFINE(float, bfloat16);
DEFINE(bfloat16, float);
#undef DEFINE

}  // end namespace functor
}  // end namespace tensorflow

#endif  // GOOGLE_CUDA
