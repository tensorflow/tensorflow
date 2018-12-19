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

#include "tensorflow/core/kernels/control_flow_ops.h"

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/platform/macros.h"

namespace tensorflow {

void SwitchOp::Compute(OpKernelContext* context) {
  const Tensor& outputPorts = context->input(1);
  OP_REQUIRES(context, TensorShapeUtils::IsScalar(outputPorts.shape()),
              errors::InvalidArgument("The second input must be a scalar, "
                                      "but it has shape ",
                                      outputPorts.shape().DebugString()));

  bool pred = outputPorts.scalar<bool>()();
  int port = (pred) ? 1 : 0;
  if (context->input_is_ref(0)) {
    context->forward_ref_input_to_ref_output(0, port);
  } else {
    context->set_output(port, context->input(0));
  }
}

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
TF_CALL_QUANTIZED_TYPES(REGISTER_CPU_SWITCH);
TF_CALL_QUANTIZED_TYPES(REGISTER_CPU_REF_SWITCH);
REGISTER_CPU_SWITCH(uint64);

TF_CALL_NUMBER_TYPES_NO_INT32(REGISTER_GPU_SWITCH);
TF_CALL_QUANTIZED_TYPES(REGISTER_GPU_SWITCH);
TF_CALL_NUMBER_TYPES_NO_INT32(REGISTER_GPU_REF_SWITCH);
TF_CALL_QUANTIZED_TYPES(REGISTER_GPU_REF_SWITCH);
REGISTER_GPU_SWITCH(uint64);
TF_CALL_variant(REGISTER_GPU_SWITCH);

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
REGISTER_GPU_HOST_KERNEL(ResourceHandle);

#undef REGISTER_GPU_HOST_KERNEL
#undef REGISTER_GPU_HOST_REF_KERNEL

#ifdef TENSORFLOW_USE_SYCL
#define REGISTER_SYCL_SWITCH(type)                        \
  REGISTER_KERNEL_BUILDER(Name("Switch")                  \
                              .Device(DEVICE_SYCL)        \
                              .HostMemory("pred")         \
                              .TypeConstraint<type>("T"), \
                          SwitchOp)
TF_CALL_REAL_NUMBER_TYPES_NO_INT32(REGISTER_SYCL_SWITCH);

#define REGISTER_SYCL_REF_SWITCH(type)                    \
  REGISTER_KERNEL_BUILDER(Name("RefSwitch")               \
                              .Device(DEVICE_SYCL)        \
                              .HostMemory("pred")         \
                              .TypeConstraint<type>("T"), \
                          SwitchOp)
TF_CALL_REAL_NUMBER_TYPES_NO_INT32(REGISTER_SYCL_REF_SWITCH);

#undef REGISTER_SYCL_SWITCH
#undef REGISTER_SYCL_REF_SWITCH

#define REGISTER_SYCL_HOST_KERNEL(type)                   \
  REGISTER_KERNEL_BUILDER(Name("Switch")                  \
                              .Device(DEVICE_SYCL)        \
                              .HostMemory("data")         \
                              .HostMemory("pred")         \
                              .HostMemory("output_false") \
                              .HostMemory("output_true")  \
                              .TypeConstraint<type>("T"), \
                          SwitchOp)

REGISTER_SYCL_HOST_KERNEL(bool);
REGISTER_SYCL_HOST_KERNEL(string);
REGISTER_SYCL_HOST_KERNEL(int32);

#define REGISTER_SYCL_HOST_REF_KERNEL(type)               \
  REGISTER_KERNEL_BUILDER(Name("RefSwitch")               \
                              .Device(DEVICE_SYCL)        \
                              .HostMemory("data")         \
                              .HostMemory("pred")         \
                              .HostMemory("output_false") \
                              .HostMemory("output_true")  \
                              .TypeConstraint<type>("T"), \
                          SwitchOp)

REGISTER_SYCL_HOST_REF_KERNEL(int32);
REGISTER_SYCL_HOST_REF_KERNEL(bool);
REGISTER_SYCL_HOST_REF_KERNEL(string);

#undef REGISTER_SYCL_HOST_KERNEL
#undef REGISTER_SYCL_HOST_REF_KERNEL
#endif  // TENSORFLOW_USE_SYCL

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

MergeOp::MergeOp(OpKernelConstruction* context) : OpKernel(context) {
  const DataType dt = context->input_type(0);
  const int num_in = context->num_inputs();
  OP_REQUIRES_OK(context, context->MatchSignature(DataTypeVector(num_in, dt),
                                                  {dt, DT_INT32}));
}

void MergeOp::Compute(OpKernelContext* context) {
  bool input_seen = false;
  for (int i = 0; i < context->num_inputs(); ++i) {
    if (context->has_input(i)) {
      if (input_seen) {
        context->SetStatus(
            errors::Internal("Merge can not have more than one valid input."));
        return;
      }
      input_seen = true;

      if (IsRefType(context->input_dtype(i))) {
        context->forward_ref_input_to_ref_output(i, 0);
      } else {
        context->set_output(0, context->input(i));
      }
      Tensor* value_index = nullptr;
      OP_REQUIRES_OK(
          context, context->allocate_output(1, TensorShape({}), &value_index));
      value_index->scalar<int32>()() = i;
    }
  }
}

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
TF_CALL_QUANTIZED_TYPES(REGISTER_GPU_KERNEL);
TF_CALL_QUANTIZED_TYPES(REGISTER_GPU_REF_KERNEL);
REGISTER_GPU_KERNEL(bool);
REGISTER_GPU_REF_KERNEL(bool);
REGISTER_GPU_KERNEL(uint64);

#undef REGISTER_GPU_KERNEL
#undef REGISTER_GPU_REF_KERNEL

#ifdef TENSORFLOW_USE_SYCL
#define REGISTER_SYCL_KERNEL(type)                        \
  REGISTER_KERNEL_BUILDER(Name("Merge")                   \
                              .Device(DEVICE_SYCL)        \
                              .TypeConstraint<type>("T")  \
                              .HostMemory("value_index"), \
                          MergeOp);
REGISTER_SYCL_KERNEL(bool);
TF_CALL_NUMBER_TYPES_NO_INT32(REGISTER_SYCL_KERNEL);

#define REGISTER_SYCL_REF_KERNEL(type)                    \
  REGISTER_KERNEL_BUILDER(Name("RefMerge")                \
                              .Device(DEVICE_SYCL)        \
                              .TypeConstraint<type>("T")  \
                              .HostMemory("value_index"), \
                          MergeOp);
REGISTER_SYCL_REF_KERNEL(bool);
TF_CALL_NUMBER_TYPES_NO_INT32(REGISTER_SYCL_REF_KERNEL);

#undef REGISTER_SYCL_KERNEL
#undef REGISTER_SYCL_REF_KERNEL
#endif  // TENSORFLOW_USE_SYCL

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
REGISTER_GPU_HOST_KERNEL(ResourceHandle);

#undef REGISTER_GPU_HOST_KERNEL

#ifdef TENSORFLOW_USE_SYCL
#define REGISTER_SYCL_HOST_KERNEL(type)                   \
  REGISTER_KERNEL_BUILDER(Name("Merge")                   \
                              .Device(DEVICE_SYCL)        \
                              .HostMemory("inputs")       \
                              .HostMemory("output")       \
                              .HostMemory("value_index")  \
                              .TypeConstraint<type>("T"), \
                          MergeOp);                       \
  REGISTER_KERNEL_BUILDER(Name("RefMerge")                \
                              .Device(DEVICE_SYCL)        \
                              .HostMemory("inputs")       \
                              .HostMemory("output")       \
                              .HostMemory("value_index")  \
                              .TypeConstraint<type>("T"), \
                          MergeOp)

REGISTER_SYCL_HOST_KERNEL(int32);
REGISTER_SYCL_HOST_KERNEL(string);
REGISTER_SYCL_HOST_KERNEL(ResourceHandle);

#undef REGISTER_SYCL_HOST_KERNEL
#endif  // TENSORFLOW_USE_SYCL

void EnterOp::Compute(OpKernelContext* context) {
  if (IsRefType(context->input_dtype(0))) {
    context->forward_ref_input_to_ref_output(0, 0);
  } else {
    context->set_output(0, context->input(0));
  }
}

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

#ifdef TENSORFLOW_USE_SYCL
#define REGISTER_SYCL_KERNEL(type) \
  REGISTER_KERNEL_BUILDER(         \
      Name("Enter").Device(DEVICE_SYCL).TypeConstraint<type>("T"), EnterOp)
REGISTER_SYCL_KERNEL(bool);
TF_CALL_NUMBER_TYPES_NO_INT32(REGISTER_SYCL_KERNEL);

#define REGISTER_SYCL_REF_KERNEL(type) \
  REGISTER_KERNEL_BUILDER(             \
      Name("RefEnter").Device(DEVICE_SYCL).TypeConstraint<type>("T"), EnterOp)
REGISTER_SYCL_REF_KERNEL(bool);
TF_CALL_NUMBER_TYPES_NO_INT32(REGISTER_SYCL_REF_KERNEL);

#undef REGISTER_SYCL_KERNEL
#undef REGISTER_SYCL_REF_KERNEL
#define REGISTER_SYCL_HOST_KERNEL(type)                   \
  REGISTER_KERNEL_BUILDER(Name("Enter")                   \
                              .Device(DEVICE_SYCL)        \
                              .HostMemory("data")         \
                              .HostMemory("output")       \
                              .TypeConstraint<type>("T"), \
                          EnterOp)

#define REGISTER_SYCL_HOST_REF_KERNEL(type)               \
  REGISTER_KERNEL_BUILDER(Name("RefEnter")                \
                              .Device(DEVICE_SYCL)        \
                              .HostMemory("data")         \
                              .HostMemory("output")       \
                              .TypeConstraint<type>("T"), \
                          EnterOp)

REGISTER_SYCL_HOST_KERNEL(int32);
REGISTER_SYCL_HOST_REF_KERNEL(int32);
REGISTER_SYCL_HOST_KERNEL(string);
REGISTER_SYCL_HOST_REF_KERNEL(string);
REGISTER_SYCL_HOST_KERNEL(ResourceHandle);

#undef REGISTER_SYCL_HOST_KERNEL
#undef REGISTER_SYCL_HOST_REF_KERNEL
#endif  // TENSORFLOW_USE_SYCL

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
REGISTER_GPU_HOST_KERNEL(ResourceHandle);

#undef REGISTER_GPU_HOST_KERNEL
#undef REGISTER_GPU_HOST_REF_KERNEL

void ExitOp::Compute(OpKernelContext* context) {
  if (IsRefType(context->input_dtype(0))) {
    context->forward_ref_input_to_ref_output(0, 0);
  } else {
    context->set_output(0, context->input(0));
  }
}

REGISTER_KERNEL_BUILDER(Name("Exit").Device(DEVICE_CPU), ExitOp);
REGISTER_KERNEL_BUILDER(Name("RefExit").Device(DEVICE_CPU), ExitOp);

#define REGISTER_GPU_KERNEL(type) \
  REGISTER_KERNEL_BUILDER(        \
      Name("Exit").Device(DEVICE_GPU).TypeConstraint<type>("T"), ExitOp);
#define REGISTER_GPU_REF_KERNEL(type) \
  REGISTER_KERNEL_BUILDER(            \
      Name("RefExit").Device(DEVICE_GPU).TypeConstraint<type>("T"), ExitOp);

TF_CALL_NUMBER_TYPES_NO_INT32(REGISTER_GPU_KERNEL);
TF_CALL_NUMBER_TYPES_NO_INT32(REGISTER_GPU_REF_KERNEL);
REGISTER_GPU_KERNEL(bool);
REGISTER_GPU_REF_KERNEL(bool);

#undef REGISTER_GPU_KERNEL
#undef REGISTER_GPU_REF_KERNEL

#ifdef TENSORFLOW_USE_SYCL
#define REGISTER_SYCL_KERNEL(type)                                         \
  REGISTER_KERNEL_BUILDER(                                                 \
      Name("Exit").Device(DEVICE_SYCL).TypeConstraint<type>("T"), ExitOp); \
  REGISTER_KERNEL_BUILDER(                                                 \
      Name("RefExit").Device(DEVICE_SYCL).TypeConstraint<type>("T"), ExitOp);
REGISTER_SYCL_KERNEL(bool);
TF_CALL_NUMBER_TYPES_NO_INT32(REGISTER_SYCL_KERNEL);

#undef REGISTER_SYCL_KERNEL
#undef REGISTER_SYCL_REF_KERNEL

#define REGISTER_SYCL_HOST_KERNEL(type)                   \
  REGISTER_KERNEL_BUILDER(Name("Exit")                    \
                              .Device(DEVICE_SYCL)        \
                              .HostMemory("data")         \
                              .HostMemory("output")       \
                              .TypeConstraint<type>("T"), \
                          ExitOp);                        \
  REGISTER_KERNEL_BUILDER(Name("RefExit")                 \
                              .Device(DEVICE_SYCL)        \
                              .HostMemory("data")         \
                              .HostMemory("output")       \
                              .TypeConstraint<type>("T"), \
                          ExitOp)

REGISTER_SYCL_HOST_KERNEL(int32);
REGISTER_SYCL_HOST_KERNEL(string);
#undef REGISTER_SYCL_HOST_KERNEL
#endif  // TENSORFLOW_USE_SYCL

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
REGISTER_GPU_HOST_KERNEL(ResourceHandle);

#undef REGISTER_GPU_HOST_KERNEL

void NextIterationOp::Compute(OpKernelContext* context) {
  if (IsRefType(context->input_dtype(0))) {
    context->forward_ref_input_to_ref_output(0, 0);
  } else {
    context->set_output(0, context->input(0));
  }
}

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
REGISTER_GPU_HOST_KERNEL(ResourceHandle);

#undef REGISTER_GPU_HOST_KERNEL

#ifdef TENSORFLOW_USE_SYCL
#define REGISTER_SYCL_KERNEL(type)                                            \
  REGISTER_KERNEL_BUILDER(                                                    \
      Name("NextIteration").Device(DEVICE_SYCL).TypeConstraint<type>("T"),    \
      NextIterationOp);                                                       \
  REGISTER_KERNEL_BUILDER(                                                    \
      Name("RefNextIteration").Device(DEVICE_SYCL).TypeConstraint<type>("T"), \
      NextIterationOp)
REGISTER_SYCL_KERNEL(bool);
TF_CALL_NUMBER_TYPES_NO_INT32(REGISTER_SYCL_KERNEL);

#undef REGISTER_SYCL_KERNEL

#define REGISTER_SYCL_HOST_KERNEL(type)                   \
  REGISTER_KERNEL_BUILDER(Name("NextIteration")           \
                              .Device(DEVICE_SYCL)        \
                              .HostMemory("data")         \
                              .HostMemory("output")       \
                              .TypeConstraint<type>("T"), \
                          NextIterationOp);               \
  REGISTER_KERNEL_BUILDER(Name("RefNextIteration")        \
                              .Device(DEVICE_SYCL)        \
                              .HostMemory("data")         \
                              .HostMemory("output")       \
                              .TypeConstraint<type>("T"), \
                          NextIterationOp)

REGISTER_SYCL_HOST_KERNEL(int32);
REGISTER_SYCL_HOST_KERNEL(string);
#undef REGISTER_SYCL_HOST_KERNEL
#endif  // TENSORFLOW_USE_SYCL

LoopCondOp::LoopCondOp(OpKernelConstruction* context) : OpKernel(context) {}
LoopCondOp::~LoopCondOp() = default;

void LoopCondOp::Compute(OpKernelContext* context) {
  CancellationManager* cm = context->cancellation_manager();
  if (cm != nullptr) {
    bool already_cancelled = cm->IsCancelled();
    OP_REQUIRES(context, !already_cancelled,
                errors::Cancelled("Loop execution was cancelled."));
  }

  context->set_output(0, context->input(0));
}

bool LoopCondOp::IsExpensive() { return false; }

REGISTER_KERNEL_BUILDER(Name("LoopCond").Device(DEVICE_CPU), LoopCondOp);
REGISTER_KERNEL_BUILDER(Name("LoopCond")
                            .Device(DEVICE_GPU)
                            .HostMemory("input")
                            .HostMemory("output"),
                        LoopCondOp);

#ifdef TENSORFLOW_USE_SYCL
REGISTER_KERNEL_BUILDER(Name("LoopCond")
                            .Device(DEVICE_SYCL)
                            .HostMemory("input")
                            .HostMemory("output"),
                        LoopCondOp);
#endif  // TENSORFLOW_USE_SYCL

// ControlTrigger kernels
REGISTER_KERNEL_BUILDER(Name("ControlTrigger").Device(DEVICE_CPU),
                        ControlTriggerOp);

REGISTER_KERNEL_BUILDER(Name("ControlTrigger").Device(DEVICE_GPU),
                        ControlTriggerOp);

#ifdef TENSORFLOW_USE_SYCL
REGISTER_KERNEL_BUILDER(Name("ControlTrigger").Device(DEVICE_SYCL),
                        ControlTriggerOp);
#endif  // TENSORFLOW_USE_SYCL

// When called, abort op will abort the current process. This can be used to
// abort remote PSs when needed.
class AbortOp : public OpKernel {
 public:
  explicit AbortOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("error_msg", &error_msg_));
    OP_REQUIRES_OK(
        context, context->GetAttr("exit_without_error", &exit_without_error_));
  }

  void Compute(OpKernelContext* context) override {
    if (!exit_without_error_) {
      LOG(FATAL) << "Abort_op intentional failure; " << error_msg_;
    } else {
      LOG(WARNING) << "Exiting the process: " << error_msg_;
      exit(0);
    }
  }

 private:
  string error_msg_;
  bool exit_without_error_;
};

REGISTER_KERNEL_BUILDER(Name("Abort").Device(DEVICE_CPU), AbortOp);

}  // namespace tensorflow
