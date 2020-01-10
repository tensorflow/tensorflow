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

// See docs in ../ops/array_ops.cc.
#ifdef INTEL_MKL

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/logging.h"

#include "mkldnn.hpp"
#include "tensorflow/core/util/mkl_util.h"

namespace tensorflow {
typedef Eigen::ThreadPoolDevice CPUDevice;

template <typename Device, typename T>
class MklIdentityOp : public OpKernel {
 public:
  explicit MklIdentityOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    MklDnnShape dnn_shape_input;
    const int kInputIdx = 0, kOutputIdx = 0;
    GetMklShape(context, kInputIdx, &dnn_shape_input);

    if (dnn_shape_input.IsMklTensor()) {
      ForwardMklTensorInToOut(context, kInputIdx, kOutputIdx);
    } else {
      ForwardTfTensorInToOut(context, kInputIdx, kOutputIdx);
    }
  }

  // TensorFlow's IdentityOp has the following member function, so kept it
  // as it is.
  bool IsExpensive() override { return false; }
};

#define REGISTER_MKL_CPU(T)                                    \
  REGISTER_KERNEL_BUILDER(                                     \
      Name("_MklIdentity")                                     \
          .Device(DEVICE_CPU)                                  \
          .TypeConstraint<T>("T")                              \
          .Label(mkl_op_registry::kMklLayoutDependentOpLabel), \
      MklIdentityOp<CPUDevice, T>);

TF_CALL_float(REGISTER_MKL_CPU);
TF_CALL_bfloat16(REGISTER_MKL_CPU);
#undef REGISTER_MKL_CPU
}  // namespace tensorflow
#endif  // INTEL_MKL
