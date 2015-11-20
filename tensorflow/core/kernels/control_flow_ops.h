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

#ifndef TENSORFLOW_KERNELS_CONTROL_FLOW_OPS_H_
#define TENSORFLOW_KERNELS_CONTROL_FLOW_OPS_H_

#include "tensorflow/core/framework/op_kernel.h"

namespace tensorflow {

// A ControlTriggerOp is similar to a NoOp. However, it always treats the input
// control edges as Live edges. Its primary use so far is in the scheduling of
// recvs, where we add ControlTrigger nodes and use them to trigger recvs. We
// allow ControlTrigger nodes to be enabled by dead nodes.
class ControlTriggerOp : public OpKernel {
 public:
  explicit ControlTriggerOp(OpKernelConstruction* context)
      : OpKernel(context) {}
  void Compute(OpKernelContext* context) override {}
  bool IsExpensive() override { return false; }
};

}  // namespace tensorflow

#endif  // TENSORFLOW_KERNELS_CONTROL_FLOW_OPS_H_
