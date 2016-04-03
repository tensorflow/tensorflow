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

#include "tensorflow/core/kernels/control_flow_ops.h"

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/platform/macros.h"

namespace tensorflow {

// A switch op has two inputs and two outputs. It forwards the value of
// Input:0 to the output specified by input:1. Input:1 is a boolean tensor.
// Input:0 is forwarded to output:0 if input:1 is false, otherwise to
// output:1.
class SwitchOp : public OpKernel {
 public:
  explicit SwitchOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    const Tensor& outputPorts = context->input(1);
    OP_REQUIRES(context, TensorShapeUtils::IsScalar(outputPorts.shape()),
                errors::InvalidArgument("The second input must be a scalar, "
                                        "but it has shape ",
                                        outputPorts.shape().DebugString()));

    bool pred = outputPorts.scalar<bool>()();
    int port = (pred) ? 1 : 0;
    if (IsRefType(context->input_dtype(0))) {
      context->forward_ref_input_to_ref_output(0, port);
    } else {
      context->set_output(port, context->input(0));
    }
  }

  bool IsExpensive() override { return false; }

  ~SwitchOp() override {}

  TF_DISALLOW_COPY_AND_ASSIGN(SwitchOp);
};

#define REGISTER_CPU_SWITCH(type)                         \
  REGISTER_KERNEL_BUILDER(Name("Switch")                  \
                              .Device(DEVICE_CPU)         \
                              .HostMemory("pred")         \
                              .TypeConstraint<type>("T"), \
                          SwitchOp)

#define REGISTER_CPU_REF_SWITCH(type)                     \
  REGISTER_KERNEL_BUILDER(Name("RefSwitch")               \
                              .Device(DEVICE_CPU)         \
                              .HostMemory("pred")         \
                              .TypeConstraint<type>("T"), \
                          SwitchOp)

#define REGISTER_GPU_SWITCH(type)                         \
  REGISTER_KERNEL_BUILDER(Name("Switch")                  \
                              .Device(DEVICE_GPU)         \
                              .HostMemory("pred")         \
                              .TypeConstraint<type>("T"), \
                          SwitchOp)

#define REGISTER_GPU_REF_SWITCH(type)                     \
  REGISTER_KERNEL_BUILDER(Name("RefSwitch")               \
                              .Device(DEVICE_GPU)         \
                              .HostMemory("pred")         \
                              .TypeConstraint<type>("T"), \
                          SwitchOp)

TF_CALL_ALL_TYPES(REGISTER_CPU_SWITCH);
TF_CALL_ALL_TYPES(REGISTER_CPU_REF_SWITCH);

TF_CALL_NUMBER_TYPES_NO_INT32(REGISTER_GPU_SWITCH);
TF_CALL_NUMBER_TYPES_NO_INT32(REGISTER_GPU_REF_SWITCH);

#undef REGISTER_CPU_SWITCH
#undef REGISTER_CPU_REF_SWITCH
#undef REGISTER_GPU_SWITCH
#undef REGISTER_GPU_REF_SWITCH

// Special GPU kernels for int32 and string.
// TODO(b/25387198): Also enable int32 in device memory. This kernel
// registration requires all int32 inputs and outputs to be in host memory.
#define REGISTER_GPU_HOST_KERNEL(type)                    \
  REGISTER_KERNEL_BUILDER(Name("Switch")                  \
                              .Device(DEVICE_GPU)         \
                              .HostMemory("data")         \
                              .HostMemory("pred")         \
                              .HostMemory("output_false") \
                              .HostMemory("output_true")  \
                              .TypeConstraint<type>("T"), \
                          SwitchOp)

#define REGISTER_GPU_HOST_REF_KERNEL(type)                \
  REGISTER_KERNEL_BUILDER(Name("RefSwitch")               \
                              .Device(DEVICE_GPU)         \
                              .HostMemory("data")         \
                              .HostMemory("pred")         \
                              .HostMemory("output_false") \
                              .HostMemory("output_true")  \
                              .TypeConstraint<type>("T"), \
                          SwitchOp)

REGISTER_GPU_HOST_KERNEL(int32);
REGISTER_GPU_HOST_REF_KERNEL(int32);
REGISTER_GPU_HOST_KERNEL(bool);
REGISTER_GPU_HOST_REF_KERNEL(bool);
REGISTER_GPU_HOST_KERNEL(string);
REGISTER_GPU_HOST_REF_KERNEL(string);

#undef REGISTER_GPU_HOST_KERNEL
#undef REGISTER_GPU_HOST_REF_KERNEL

class RefSelectOp : public OpKernel {
 public:
  explicit RefSelectOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("N", &num_ref_inputs_));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& index_tensor = context->input(0);
    OP_REQUIRES(context, TensorShapeUtils::IsScalar(index_tensor.shape()),
                errors::InvalidArgument("Index must be a scalar, "
                                        "but it has shape ",
                                        index_tensor.shape().DebugString()));

    int32 index = index_tensor.scalar<int32>()();

    OP_REQUIRES(context, index >= 0 && index < num_ref_inputs_,
                errors::InvalidArgument("Index must be in the range [0, ",
                                        num_ref_inputs_, ") but got ", index));
    context->forward_ref_input_to_ref_output(index + 1, 0);
  }

  bool IsExpensive() override { return false; }

  ~RefSelectOp() override {}

  TF_DISALLOW_COPY_AND_ASSIGN(RefSelectOp);

 private:
  int num_ref_inputs_;
};

#define REGISTER_CPU_REF_SELECT(type)                     \
  REGISTER_KERNEL_BUILDER(Name("RefSelect")               \
                              .Device(DEVICE_CPU)         \
                              .HostMemory("index")        \
                              .TypeConstraint<type>("T"), \
                          RefSelectOp)
TF_CALL_ALL_TYPES(REGISTER_CPU_REF_SELECT);

#undef REGISTER_CPU_REF_SWITCH

// A merge op has n inputs and two outputs. It forwards the value of the
// first input that becomes available to its first output, and the
// index of the first input to its second output.
class MergeOp : public OpKernel {
 public:
  explicit MergeOp(OpKernelConstruction* context) : OpKernel(context) {
    const DataType dt = context->input_type(0);
    const int num_in = context->num_inputs();
    OP_REQUIRES_OK(context, context->MatchSignature(DataTypeVector(num_in, dt),
                                                    {dt, DT_INT32}));
  }

  void Compute(OpKernelContext* context) override {
    bool input_seen = false;
    for (int i = 0; i < context->num_inputs(); ++i) {
      if (context->has_input(i)) {
        if (input_seen) {
          context->SetStatus(errors::Internal(
              "Merge can not have more than one valid input."));
          return;
        }
        input_seen = true;

        if (IsRefType(context->input_dtype(i))) {
          context->forward_ref_input_to_ref_output(i, 0);
        } else {
          context->set_output(0, context->input(i));
        }
        Tensor* value_index = nullptr;
        OP_REQUIRES_OK(context, context->allocate_output(1, TensorShape({}),
                                                         &value_index));
        value_index->scalar<int32>()() = i;
      }
    }
  }

  bool IsExpensive() override { return false; }

  ~MergeOp() override {}

  TF_DISALLOW_COPY_AND_ASSIGN(MergeOp);
};

REGISTER_KERNEL_BUILDER(Name("Merge").Device(DEVICE_CPU), MergeOp);
REGISTER_KERNEL_BUILDER(Name("RefMerge").Device(DEVICE_CPU), MergeOp);

#define REGISTER_GPU_KERNEL(type)                         \
  REGISTER_KERNEL_BUILDER(Name("Merge")                   \
                              .Device(DEVICE_GPU)         \
                              .TypeConstraint<type>("T")  \
                              .HostMemory("value_index"), \
                          MergeOp);

#define REGISTER_GPU_REF_KERNEL(type)                     \
  REGISTER_KERNEL_BUILDER(Name("RefMerge")                \
                              .Device(DEVICE_GPU)         \
                              .TypeConstraint<type>("T")  \
                              .HostMemory("value_index"), \
                          MergeOp);

TF_CALL_NUMBER_TYPES_NO_INT32(REGISTER_GPU_KERNEL);
TF_CALL_NUMBER_TYPES_NO_INT32(REGISTER_GPU_REF_KERNEL);
REGISTER_GPU_KERNEL(bool);
REGISTER_GPU_REF_KERNEL(bool);

#undef REGISTER_GPU_KERNEL
#undef REGISTER_GPU_REF_KERNEL

// Special GPU kernels for int32 and string.
// TODO(b/25387198): Also enable int32 in device memory. This kernel
// registration requires all int32 inputs and outputs to be in host memory.
#define REGISTER_GPU_HOST_KERNEL(type)                    \
  REGISTER_KERNEL_BUILDER(Name("Merge")                   \
                              .Device(DEVICE_GPU)         \
                              .HostMemory("inputs")       \
                              .HostMemory("output")       \
                              .HostMemory("value_index")  \
                              .TypeConstraint<type>("T"), \
                          MergeOp);                       \
  REGISTER_KERNEL_BUILDER(Name("RefMerge")                \
                              .Device(DEVICE_GPU)         \
                              .HostMemory("inputs")       \
                              .HostMemory("output")       \
                              .HostMemory("value_index")  \
                              .TypeConstraint<type>("T"), \
                          MergeOp)

REGISTER_GPU_HOST_KERNEL(int32);
REGISTER_GPU_HOST_KERNEL(string);

#undef REGISTER_GPU_HOST_KERNEL

// An enter op has one input and one output. It creates or finds
// the child frame that is uniquely identified by the frame_name,
// and makes its input available to the child frame.
class EnterOp : public OpKernel {
 public:
  explicit EnterOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    if (IsRefType(context->input_dtype(0))) {
      context->forward_ref_input_to_ref_output(0, 0);
    } else {
      context->set_output(0, context->input(0));
    }
  }

  bool IsExpensive() override { return false; }

  ~EnterOp() override {}

  TF_DISALLOW_COPY_AND_ASSIGN(EnterOp);
};

REGISTER_KERNEL_BUILDER(Name("Enter").Device(DEVICE_CPU), EnterOp);
REGISTER_KERNEL_BUILDER(Name("RefEnter").Device(DEVICE_CPU), EnterOp);

#define REGISTER_GPU_KERNEL(type) \
  REGISTER_KERNEL_BUILDER(        \
      Name("Enter").Device(DEVICE_GPU).TypeConstraint<type>("T"), EnterOp)
#define REGISTER_GPU_REF_KERNEL(type) \
  REGISTER_KERNEL_BUILDER(            \
      Name("RefEnter").Device(DEVICE_GPU).TypeConstraint<type>("T"), EnterOp)

TF_CALL_NUMBER_TYPES_NO_INT32(REGISTER_GPU_KERNEL);
TF_CALL_NUMBER_TYPES_NO_INT32(REGISTER_GPU_REF_KERNEL);
REGISTER_GPU_KERNEL(bool);
REGISTER_GPU_REF_KERNEL(bool);

#undef REGISTER_GPU_KERNEL
#undef REGISTER_GPU_REF_KERNEL

// Special GPU kernels for int32 and string.
// TODO(b/25387198): Also enable int32 in device memory. This kernel
// registration requires all int32 inputs and outputs to be in host memory.
#define REGISTER_GPU_HOST_KERNEL(type)                    \
  REGISTER_KERNEL_BUILDER(Name("Enter")                   \
                              .Device(DEVICE_GPU)         \
                              .HostMemory("data")         \
                              .HostMemory("output")       \
                              .TypeConstraint<type>("T"), \
                          EnterOp)

#define REGISTER_GPU_HOST_REF_KERNEL(type)                \
  REGISTER_KERNEL_BUILDER(Name("RefEnter")                \
                              .Device(DEVICE_GPU)         \
                              .HostMemory("data")         \
                              .HostMemory("output")       \
                              .TypeConstraint<type>("T"), \
                          EnterOp)

REGISTER_GPU_HOST_KERNEL(int32);
REGISTER_GPU_HOST_REF_KERNEL(int32);
REGISTER_GPU_HOST_KERNEL(string);
REGISTER_GPU_HOST_REF_KERNEL(string);

#undef REGISTER_GPU_HOST_KERNEL
#undef REGISTER_GPU_HOST_REF_KERNEL

// An exit op has one input and one output. It exits the current
// frame to its parent frame, and makes its input available to the
// parent frame.
class ExitOp : public OpKernel {
 public:
  explicit ExitOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    if (IsRefType(context->input_dtype(0))) {
      context->forward_ref_input_to_ref_output(0, 0);
    } else {
      context->set_output(0, context->input(0));
    }
  }

  bool IsExpensive() override { return false; }

  ~ExitOp() override {}

  TF_DISALLOW_COPY_AND_ASSIGN(ExitOp);
};

REGISTER_KERNEL_BUILDER(Name("Exit").Device(DEVICE_CPU), ExitOp);
REGISTER_KERNEL_BUILDER(Name("RefExit").Device(DEVICE_CPU), ExitOp);

#define REGISTER_GPU_KERNEL(type) \
  REGISTER_KERNEL_BUILDER(        \
      Name("Exit").Device(DEVICE_GPU).TypeConstraint<type>("T"), ExitOp);
#define REGISTER_GPU_REF_KERNEL(type) \
  REGISTER_KERNEL_BUILDER(            \
      Name("RefExit").Device(DEVICE_GPU).TypeConstraint<type>("T"), ExitOp);

TF_CALL_NUMBER_TYPES_NO_INT32(REGISTER_GPU_KERNEL);
REGISTER_GPU_KERNEL(bool);

#undef REGISTER_GPU_KERNEL
#undef REGISTER_GPU_REF_KERNEL

// Special GPU kernels for int32 and string.
// TODO(b/25387198): Also enable int32 in device memory. This kernel
// registration requires all int32 inputs and outputs to be in host memory.
#define REGISTER_GPU_HOST_KERNEL(type)                    \
  REGISTER_KERNEL_BUILDER(Name("Exit")                    \
                              .Device(DEVICE_GPU)         \
                              .HostMemory("data")         \
                              .HostMemory("output")       \
                              .TypeConstraint<type>("T"), \
                          ExitOp);                        \
  REGISTER_KERNEL_BUILDER(Name("RefExit")                 \
                              .Device(DEVICE_GPU)         \
                              .HostMemory("data")         \
                              .HostMemory("output")       \
                              .TypeConstraint<type>("T"), \
                          ExitOp)

REGISTER_GPU_HOST_KERNEL(int32);
REGISTER_GPU_HOST_KERNEL(string);

#undef REGISTER_GPU_HOST_KERNEL

// A next_iteration op has one input and one output. It makes its input
// available to the next iteration.
class NextIterationOp : public OpKernel {
 public:
  explicit NextIterationOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    if (IsRefType(context->input_dtype(0))) {
      context->forward_ref_input_to_ref_output(0, 0);
    } else {
      context->set_output(0, context->input(0));
    }
  }

  bool IsExpensive() override { return false; }

  ~NextIterationOp() override {}

  TF_DISALLOW_COPY_AND_ASSIGN(NextIterationOp);
};

REGISTER_KERNEL_BUILDER(Name("NextIteration").Device(DEVICE_CPU),
                        NextIterationOp);
REGISTER_KERNEL_BUILDER(Name("RefNextIteration").Device(DEVICE_CPU),
                        NextIterationOp);

#define REGISTER_GPU_KERNEL(type)                                            \
  REGISTER_KERNEL_BUILDER(                                                   \
      Name("NextIteration").Device(DEVICE_GPU).TypeConstraint<type>("T"),    \
      NextIterationOp);                                                      \
  REGISTER_KERNEL_BUILDER(                                                   \
      Name("RefNextIteration").Device(DEVICE_GPU).TypeConstraint<type>("T"), \
      NextIterationOp)

TF_CALL_NUMBER_TYPES_NO_INT32(REGISTER_GPU_KERNEL);
REGISTER_GPU_KERNEL(bool);

#undef REGISTER_GPU_KERNEL

// Special GPU kernels for int32 and string.
// TODO(b/25387198): Also enable int32 in device memory. This kernel
// registration requires all int32 inputs and outputs to be in host memory.
#define REGISTER_GPU_HOST_KERNEL(type)                    \
  REGISTER_KERNEL_BUILDER(Name("NextIteration")           \
                              .Device(DEVICE_GPU)         \
                              .HostMemory("data")         \
                              .HostMemory("output")       \
                              .TypeConstraint<type>("T"), \
                          NextIterationOp);               \
  REGISTER_KERNEL_BUILDER(Name("RefNextIteration")        \
                              .Device(DEVICE_GPU)         \
                              .HostMemory("data")         \
                              .HostMemory("output")       \
                              .TypeConstraint<type>("T"), \
                          NextIterationOp)

REGISTER_GPU_HOST_KERNEL(int32);
REGISTER_GPU_HOST_KERNEL(string);

#undef REGISTER_GPU_HOST_KERNEL

// A LoopCond op has one input and one output. The input is a boolean
// scalar representing the taken branches of the "pivot" Switch that
// determines loop termination. As a contract, any high-level front-end
// should always use port '0' of the "pivot" switches for loop exit.
class LoopCondOp : public OpKernel {
 public:
  explicit LoopCondOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    context->set_output(0, context->input(0));
  }

  bool IsExpensive() override { return false; }

  ~LoopCondOp() override {}

  TF_DISALLOW_COPY_AND_ASSIGN(LoopCondOp);
};

REGISTER_KERNEL_BUILDER(Name("LoopCond").Device(DEVICE_CPU), LoopCondOp);
REGISTER_KERNEL_BUILDER(Name("LoopCond")
                            .Device(DEVICE_GPU)
                            .HostMemory("input")
                            .HostMemory("output"),
                        LoopCondOp);

// ControlTrigger kernels
REGISTER_KERNEL_BUILDER(Name("ControlTrigger").Device(DEVICE_CPU),
                        ControlTriggerOp);

REGISTER_KERNEL_BUILDER(Name("ControlTrigger").Device(DEVICE_GPU),
                        ControlTriggerOp);

// When called, abort op will abort the current process. This can be used to
// abort remote PSs when needed.
class AbortOp : public OpKernel {
 public:
  explicit AbortOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("error_msg", &error_msg_));
  }

  void Compute(OpKernelContext* context) override {
    CHECK(false) << "Abort_op intentional failure; " << error_msg_;
  }

 private:
  string error_msg_;
};

REGISTER_KERNEL_BUILDER(Name("Abort").Device(DEVICE_CPU), AbortOp);
}  // namespace tensorflow
