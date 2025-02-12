/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "Eigen/Core"  // from @eigen_archive
#include "tensorflow/core/kernels/cwise_ops_common.h"
#include "tensorflow/core/kernels/special_math/special_math_op_misc_impl.h"

namespace tensorflow {
REGISTER3(UnaryOp, CPU, "BesselI0", functor::bessel_i0, Eigen::half, float,
          double);
REGISTER3(UnaryOp, CPU, "BesselI1", functor::bessel_i1, Eigen::half, float,
          double);
REGISTER3(UnaryOp, CPU, "BesselI0e", functor::bessel_i0e, Eigen::half, float,
          double);
REGISTER3(UnaryOp, CPU, "BesselI1e", functor::bessel_i1e, Eigen::half, float,
          double);

REGISTER3(UnaryOp, CPU, "BesselK0", functor::bessel_k0, Eigen::half, float,
          double);
REGISTER3(UnaryOp, CPU, "BesselK1", functor::bessel_k1, Eigen::half, float,
          double);
REGISTER3(UnaryOp, CPU, "BesselK0e", functor::bessel_k0e, Eigen::half, float,
          double);
REGISTER3(UnaryOp, CPU, "BesselK1e", functor::bessel_k1e, Eigen::half, float,
          double);

REGISTER3(UnaryOp, CPU, "BesselJ0", functor::bessel_j0, Eigen::half, float,
          double);
REGISTER3(UnaryOp, CPU, "BesselJ1", functor::bessel_j1, Eigen::half, float,
          double);

REGISTER3(UnaryOp, CPU, "BesselY0", functor::bessel_y0, Eigen::half, float,
          double);
REGISTER3(UnaryOp, CPU, "BesselY1", functor::bessel_y1, Eigen::half, float,
          double);

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM

REGISTER3(UnaryOp, GPU, "BesselI0", functor::bessel_i0, Eigen::half, float,
          double);
REGISTER3(UnaryOp, GPU, "BesselI1", functor::bessel_i1, Eigen::half, float,
          double);
REGISTER3(UnaryOp, GPU, "BesselI0e", functor::bessel_i0e, Eigen::half, float,
          double);
REGISTER3(UnaryOp, GPU, "BesselI1e", functor::bessel_i1e, Eigen::half, float,
          double);

REGISTER3(UnaryOp, GPU, "BesselK0", functor::bessel_k0, Eigen::half, float,
          double);
REGISTER3(UnaryOp, GPU, "BesselK1", functor::bessel_k1, Eigen::half, float,
          double);
REGISTER3(UnaryOp, GPU, "BesselK0e", functor::bessel_k0e, Eigen::half, float,
          double);
REGISTER3(UnaryOp, GPU, "BesselK1e", functor::bessel_k1e, Eigen::half, float,
          double);

REGISTER3(UnaryOp, GPU, "BesselJ0", functor::bessel_j0, Eigen::half, float,
          double);
REGISTER3(UnaryOp, GPU, "BesselJ1", functor::bessel_j1, Eigen::half, float,
          double);

REGISTER3(UnaryOp, GPU, "BesselY0", functor::bessel_y0, Eigen::half, float,
          double);
REGISTER3(UnaryOp, GPU, "BesselY1", functor::bessel_y1, Eigen::half, float,
          double);
#endif
}  // namespace tensorflow
