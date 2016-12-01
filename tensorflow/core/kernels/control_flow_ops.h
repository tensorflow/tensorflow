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

// A switch op has two inputs and two outputs. It forwards the value of
// Input:0 to the output specified by input:1. Input:1 is a boolean tensor.
// Input:0 is forwarded to output:0 if input:1 is false, otherwise to
// output:1.
class SwitchOp : public OpKernel {
 public:
  explicit SwitchOp(OpKernelConstruction* context) : OpKernel(context) {}
  void Compute(OpKernelContext* context) override;
  bool IsExpensive() override { return false; }
  ~SwitchOp() override {}

  TF_DISALLOW_COPY_AND_ASSIGN(SwitchOp);
};

// A merge op has n inputs and two outputs. It forwards the value of the
// first input that becomes available to its first output, and the
// index of the first input to its second output.
class MergeOp : public OpKernel {
 public:
  explicit MergeOp(OpKernelConstruction* context);
  void Compute(OpKernelContext* context) override;
  bool IsExpensive() override { return false; }
  ~MergeOp() override {}

  TF_DISALLOW_COPY_AND_ASSIGN(MergeOp);
};

// An enter op has one input and one output. It creates or finds
// the child frame that is uniquely identified by the frame_name,
// and makes its input available to the child frame.
class EnterOp : public OpKernel {
 public:
  explicit EnterOp(OpKernelConstruction* context) : OpKernel(context) {}
  void Compute(OpKernelContext* context) override;
  bool IsExpensive() override { return false; }
  ~EnterOp() override {}

  TF_DISALLOW_COPY_AND_ASSIGN(EnterOp);
};

// An exit op has one input and one output. It exits the current
// frame to its parent frame, and makes its input available to the
// parent frame.
class ExitOp : public OpKernel {
 public:
  explicit ExitOp(OpKernelConstruction* context) : OpKernel(context) {}
  void Compute(OpKernelContext* context) override;
  bool IsExpensive() override { return false; }
  ~ExitOp() override {}

  TF_DISALLOW_COPY_AND_ASSIGN(ExitOp);
};

// A next_iteration op has one input and one output. It makes its input
// available to the next iteration.
class NextIterationOp : public OpKernel {
 public:
  explicit NextIterationOp(OpKernelConstruction* context) : OpKernel(context) {}
  void Compute(OpKernelContext* context) override;
  bool IsExpensive() override { return false; }
  ~NextIterationOp() override {}

  TF_DISALLOW_COPY_AND_ASSIGN(NextIterationOp);
};

}  // namespace tensorflow

#endif  // TENSORFLOW_KERNELS_CONTROL_FLOW_OPS_H_
