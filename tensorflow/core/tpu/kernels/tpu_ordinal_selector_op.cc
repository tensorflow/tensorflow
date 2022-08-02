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

#include <memory>

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/tpu/kernels/tpu_ordinal_selector.h"

namespace tensorflow {
namespace {

// TPUOrdinalSelectorOp is a no-op for backward compatibility. The core
// selection algorithm happens inside TPUPartitionedCall.
class TPUOrdinalSelectorOp : public OpKernel {
 public:
  explicit TPUOrdinalSelectorOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  ~TPUOrdinalSelectorOp() override {}

  void Compute(OpKernelContext* ctx) override {
    Tensor output(DT_INT32, TensorShape({}));
    output.flat<int>().setValues({tpu::kDeferredCoreSelectionReserved});
    ctx->set_output(0, output);
    ctx->SetStatus(OkStatus());
  }

  bool IsExpensive() override { return false; }
};

}  // namespace

REGISTER_KERNEL_BUILDER(Name("TPUOrdinalSelector").Device(DEVICE_CPU),
                        TPUOrdinalSelectorOp);

REGISTER_KERNEL_BUILDER(Name("TPURoundRobin").Device(DEVICE_CPU),
                        TPUOrdinalSelectorOp);

}  // namespace tensorflow
