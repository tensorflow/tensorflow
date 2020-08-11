/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0(the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifdef INTEL_MKL

// See docs in ../ops/math_ops.cc.

#define EIGEN_USE_THREADS
#include <iostream>
#include <vector>

#include "tensorflow/core/kernels/cwise_ops_common.h"

#include "tensorflow/core/util/mkl_util.h"

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;

template <typename Device, typename Functor>
class MklBinaryOp : public BinaryOp<Device, Functor> {
 public:
  explicit MklBinaryOp(OpKernelConstruction* context)
      : BinaryOp<Device, Functor>(context) {}

  void Compute(OpKernelContext* context) override {
    auto in0 = context->input(0);
    auto in1 = context->input(1);
    VLOG(1) << "Shapes (start mklbinaryop compute): "
            << in0.shape().DebugString() << " _and_ "
            << in1.shape().DebugString();

    // Call the TensorFlow BinaryOp Compute method
    BinaryOp<Device, Functor>::Compute(context);

    auto out = context->mutable_output(0);
    VLOG(1) << "Shapes (output): " << out->shape().DebugString();

    // Pass input shape through to output shape
    ForwardMklMetaDataInToOut(context, 0, 0);

    out = context->mutable_output(0);
    VLOG(1) << "Shapes (output): " << out->shape().DebugString();
  }
};

//---------- Registration macros for various element-wise ops -----------
// We will need to redefine "REGISTER" to include the mkl_op_registry flag
#pragma push_macro("REGISTER")
#undef REGISTER
#define REGISTER(OP, D, N, F, T)                               \
  REGISTER_KERNEL_BUILDER(                                     \
      Name(N)                                                  \
          .Device(DEVICE_##D)                                  \
          .TypeConstraint<T>("T")                              \
          .Label(mkl_op_registry::kMklLayoutDependentOpLabel), \
      OP<D##Device, F<T>>);

REGISTER6(MklBinaryOp, CPU, "_MklAdd", functor::add, float, Eigen::half, double,
          int32, int64, bfloat16);
REGISTER6(MklBinaryOp, CPU, "_MklAddV2", functor::add, float, Eigen::half,
          double, int32, int64, bfloat16);
REGISTER8(MklBinaryOp, CPU, "_MklSub", functor::sub, float, Eigen::half, double,
          int32, int64, complex64, complex128, bfloat16);
REGISTER6(MklBinaryOp, CPU, "_MklMul", functor::mul, float, Eigen::half, double,
          uint8, int32, bfloat16);
REGISTER6(MklBinaryOp, CPU, "_MklMaximum", functor::maximum, float, Eigen::half,
          double, int32, int64, bfloat16);
REGISTER6(MklBinaryOp, CPU, "_MklSquaredDifference",
          functor::squared_difference, float, Eigen::half, double, int32, int64,
          bfloat16);

#undef REGISTER
#pragma pop_macro("REGISTER")
//-----------------------------------------------------------------------

}  // end namespace tensorflow

#endif  // INTEL_MKL
