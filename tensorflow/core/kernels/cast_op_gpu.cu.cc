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

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM

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

#define DEFINE(O, I) template struct CastFunctor<GPUDevice, O, I>
#define DEFINE_ALL_FROM(in_type)        \
  DEFINE(in_type, bool);                \
  DEFINE(in_type, uint8);               \
  DEFINE(in_type, uint16);              \
  DEFINE(in_type, uint32);              \
  DEFINE(in_type, uint64);              \
  DEFINE(in_type, int8);                \
  DEFINE(in_type, int16);               \
  DEFINE(in_type, int32);               \
  DEFINE(in_type, int64);               \
  DEFINE(in_type, Eigen::half);         \
  DEFINE(in_type, float);               \
  DEFINE(in_type, double);              \
  DEFINE(in_type, std::complex<float>); \
  DEFINE(in_type, std::complex<double>)

DEFINE_ALL_FROM(bool);
DEFINE_ALL_FROM(uint8);
DEFINE_ALL_FROM(uint16);
DEFINE_ALL_FROM(uint32);
DEFINE_ALL_FROM(uint64);
DEFINE_ALL_FROM(int8);
DEFINE_ALL_FROM(int16);
DEFINE_ALL_FROM(int32);
DEFINE_ALL_FROM(int64);
DEFINE_ALL_FROM(Eigen::half);
DEFINE_ALL_FROM(float);
DEFINE_ALL_FROM(double);
DEFINE_ALL_FROM(std::complex<float>);
DEFINE_ALL_FROM(std::complex<double>);
DEFINE(bfloat16, float);
DEFINE(float, bfloat16);

#undef DEFINE_ALL_FROM
#undef DEFINE

}  // end namespace functor
}  // end namespace tensorflow

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
