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

#ifndef TENSORFLOW_CORE_KERNELS_CONTROL_FLOW_OPS_H_
#define TENSORFLOW_CORE_KERNELS_CONTROL_FLOW_OPS_H_

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

  SwitchOp(const SwitchOp&) = delete;
  void operator=(const SwitchOp&) = delete;
};

// An n-way switch op has two inputs and N outputs. It forwards the value of
// Input:0 to the output specified by Input:1. Input:1 is an integer tensor.
// Input:0 is forwarded to output:0 if Input:1 is 0, to output:1 if 1, and so
// forth. If Input:1 is <0 or >=num_outputs(), Input:0 is forwarded to
// output:num_outputs()-1.
class SwitchNOp : public OpKernel {
 public:
  explicit SwitchNOp(OpKernelConstruction* context) : OpKernel(context) {}
  void Compute(OpKernelContext* context) override;
  bool IsExpensive() override { return false; }
  ~SwitchNOp() override {}

  SwitchNOp(const SwitchNOp&) = delete;
  void operator=(const SwitchNOp&) = delete;
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

  MergeOp(const MergeOp&) = delete;
  void operator=(const MergeOp&) = delete;
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

  EnterOp(const EnterOp&) = delete;
  void operator=(const EnterOp&) = delete;
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

  ExitOp(const ExitOp&) = delete;
  void operator=(const ExitOp&) = delete;
};

// A next_iteration op has one input and one output. It makes its input
// available to the next iteration.
class NextIterationOp : public OpKernel {
 public:
  explicit NextIterationOp(OpKernelConstruction* context) : OpKernel(context) {}
  void Compute(OpKernelContext* context) override;
  bool IsExpensive() override { return false; }
  ~NextIterationOp() override {}

  NextIterationOp(const NextIterationOp&) = delete;
  void operator=(const NextIterationOp&) = delete;
};

// A LoopCond op has one input and one output. The input is a boolean
// scalar representing the taken branches of the "pivot" Switch that
// determines loop termination. As a contract, any high-level front-end
// should always use port '0' of the "pivot" switches for loop exit.
class LoopCondOp : public OpKernel {
 public:
  explicit LoopCondOp(OpKernelConstruction* context);
  ~LoopCondOp() override;

  void Compute(OpKernelContext* context) override;

  bool IsExpensive() override;

  LoopCondOp(const LoopCondOp&) = delete;
  void operator=(const LoopCondOp&) = delete;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_CONTROL_FLOW_OPS_H_
