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

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/types.h"

namespace tensorflow {
namespace risc {
namespace experimental {

typedef Eigen::ThreadPoolDevice CPUDevice;

template <typename Device, typename T>
class RiscConvOp : public OpKernel {
 public:
  explicit RiscConvOp(OpKernelConstruction* context) : OpKernel(context) {
    // TODO(b/171294012): Implement RiscConv op.
  }

  void Compute(OpKernelContext* context) override {
    // TODO(b/171294012): Implement RiscConv op.
  }
};

#define REGISTER_CPU(T)                                           \
  REGISTER_KERNEL_BUILDER(                                        \
      Name("RiscConv").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      RiscConvOp<CPUDevice, T>);

REGISTER_CPU(float);
REGISTER_CPU(double);

}  // namespace experimental
}  // namespace risc
}  // namespace tensorflow
