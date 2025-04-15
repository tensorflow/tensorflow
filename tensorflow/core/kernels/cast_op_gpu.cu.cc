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

#if (defined(GOOGLE_CUDA) && GOOGLE_CUDA) || \
    (defined(TENSORFLOW_USE_ROCM) && TENSORFLOW_USE_ROCM)

#define EIGEN_USE_GPU

#include "tensorflow/core/framework/bfloat16.h"
#define SPECIALIZE_FOR_GPUS
#include "tensorflow/core/kernels/cast_op.h"
#undef SPECIALIZE_FOR_GPUS

namespace tensorflow {
namespace functor {

typedef Eigen::GpuDevice GPUDevice;

#if defined(MLIR_GENERATED_GPU_KERNELS_ENABLED)
CAST_FUNCTORS_SUBSET(GPUDevice);
#else
CAST_FUNCTORS(GPUDevice);
#endif

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

// Required functors not previously specialized for truncation.
DEFINE(double, float8_e5m2);
DEFINE(float, float8_e5m2);
DEFINE(bfloat16, float8_e5m2);
DEFINE(Eigen::half, float8_e5m2);
DEFINE(float8_e5m2, float8_e5m2);
DEFINE(float8_e4m3fn, float8_e5m2);

DEFINE(double, float8_e4m3fn);
DEFINE(float, float8_e4m3fn);
DEFINE(bfloat16, float8_e4m3fn);
DEFINE(Eigen::half, float8_e4m3fn);
DEFINE(float8_e4m3fn, float8_e4m3fn);

#if defined(MLIR_GENERATED_GPU_KERNELS_ENABLED)

// The cast from float to double is still needed for resize_bilinear_op.cc
DEFINE(double, float);

#else

DEFINE_ALL_FROM(bool);
DEFINE_ALL_FROM(uint8);
DEFINE_ALL_FROM(uint16);
DEFINE_ALL_FROM(uint32);
DEFINE_ALL_FROM(uint64);
DEFINE_ALL_FROM(int8);
DEFINE_ALL_FROM(int16);
DEFINE_ALL_FROM(int32);
DEFINE_ALL_FROM(int64);
DEFINE_ALL_FROM(double);
DEFINE_ALL_FROM(std::complex<double>);
#endif

#define DEFINE_ALL_TO_FLOAT(out_type) \
  DEFINE(out_type, bool);             \
  DEFINE(out_type, uint8);            \
  DEFINE(out_type, uint16);           \
  DEFINE(out_type, uint32);           \
  DEFINE(out_type, uint64);           \
  DEFINE(out_type, int8);             \
  DEFINE(out_type, int16);            \
  DEFINE(out_type, int32);            \
  DEFINE(out_type, int64);            \
  DEFINE(out_type, Eigen::half);      \
  DEFINE(out_type, bfloat16);         \
  DEFINE(out_type, float);            \
  DEFINE(out_type, std::complex<float>)

#define DEFINE_ALL_TO_HALF(out_type) \
  DEFINE(out_type, bool);            \
  DEFINE(out_type, uint8);           \
  DEFINE(out_type, uint16);          \
  DEFINE(out_type, uint32);          \
  DEFINE(out_type, uint64);          \
  DEFINE(out_type, int8);            \
  DEFINE(out_type, int16);           \
  DEFINE(out_type, int32);           \
  DEFINE(out_type, int64);           \
  DEFINE(out_type, Eigen::half);     \
  DEFINE(out_type, bfloat16)

DEFINE_ALL_TO_HALF(bfloat16);
DEFINE(bool, bfloat16);
DEFINE(uint8, bfloat16);
DEFINE(uint16, bfloat16);
DEFINE(uint32, bfloat16);
DEFINE(uint64, bfloat16);
DEFINE(int8, bfloat16);
DEFINE(int16, bfloat16);
DEFINE(int32, bfloat16);
DEFINE(int64, bfloat16);
DEFINE(std::complex<double>, bfloat16);
DEFINE(double, bfloat16);
DEFINE(bfloat16, std::complex<float>);
DEFINE(bfloat16, std::complex<double>);
DEFINE(bfloat16, double);

#if defined(MLIR_GENERATED_GPU_KERNELS_ENABLED)
// The cast from Eigen::half is still needed for depthwise_conv_grad_op.cc.
DEFINE(float, Eigen::half);
// The cast from float to float is still needed for resize_bilinear_op.cc.
DEFINE(float, float);
// The casts from complex to the complex element type is still needed for
// self_adjoint_eig_v2_op_gpu.cc
DEFINE(std::complex<float>, float);
DEFINE(std::complex<double>, double);

DEFINE(float, bfloat16);
DEFINE(Eigen::half, bfloat16);
DEFINE(std::complex<float>, bfloat16);
#else
DEFINE_ALL_TO_HALF(Eigen::half);
DEFINE_ALL_TO_FLOAT(float);
DEFINE_ALL_TO_FLOAT(std::complex<float>);
#endif

#define DEFINE_TO_INT(from_type) \
  DEFINE(from_type, int8);       \
  DEFINE(from_type, int16);      \
  DEFINE(from_type, int32);      \
  DEFINE(from_type, int64_t);    \
  DEFINE(from_type, uint8);      \
  DEFINE(from_type, uint16);     \
  DEFINE(from_type, uint32);     \
  DEFINE(from_type, uint64)

#define DEFINE_FROM_INT(out_type) \
  DEFINE(int8, out_type);         \
  DEFINE(int16, out_type);        \
  DEFINE(int32, out_type);        \
  DEFINE(int64_t, out_type);      \
  DEFINE(uint8, out_type);        \
  DEFINE(uint16, out_type);       \
  DEFINE(uint32, out_type);       \
  DEFINE(uint64, out_type)

DEFINE_TO_INT(int4);
DEFINE_TO_INT(uint4);
DEFINE_FROM_INT(int4);
DEFINE_FROM_INT(uint4);
DEFINE(int4, int4);
DEFINE(int4, uint4);
DEFINE(uint4, int4);
DEFINE(uint4, uint4);

#undef DEFINE_TO_INT
#undef DEFINE_FROM_INT
#undef DEFINE_ALL_TO_FLOAT
#undef DEFINE_ALL_TO_HALF
#undef DEFINE_ALL_FROM
#undef DEFINE

}  // end namespace functor
}  // end namespace tensorflow

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
