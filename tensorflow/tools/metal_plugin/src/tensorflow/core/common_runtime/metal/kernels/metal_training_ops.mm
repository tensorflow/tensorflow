/* Copyright 2026 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/common_runtime/metal/kernels/metal_kernels.h"

#import <Metal/Metal.h>

#include <cstdint>
#include <string>
#include <vector>

#include "absl/log/log.h"
#include "tensorflow/c/kernels.h"
#include "tensorflow/c/kernels_experimental.h"
#include "tensorflow/c/tf_datatype.h"
#include "tensorflow/c/tf_status.h"
#include "tensorflow/c/tf_tensor.h"
#include "tensorflow/core/common_runtime/metal/kernels/metal_kernel_util.h"
#include "tensorflow/core/common_runtime/metal/kernels/metal_shader_library.h"
#include "tensorflow/core/common_runtime/metal/metal_platform.h"
#include "tensorflow/core/common_runtime/metal/metal_stream.h"

namespace tensorflow {
namespace metal {
namespace {

// ResourceApplyGradientDescent and ResourceApplyAdam, the weight update that
// closes a training step.
//
// The variable plumbing goes through the experimental kernel C API
// (TF_MaybeLockVariableInputMutexesInOrder, TF_GetInputTensorFromVariable),
// which exists so that a pluggable device can implement optimisers without
// reaching into core's resource manager. The arithmetic is a compute shader
// rather than MPSGraph's own optimiser ops: MPSGraph parameterises Adam
// differently, and matching TensorFlow's exact update, including how it folds
// the bias correction into a single step size, matters more here than reusing
// a library kernel. A mismatch would not fail, it would just train slightly
// differently from every other backend.

// Copies one device tensor onto another, for the variable machinery's
// copy-on-write path. Runs on the stream so it stays ordered with the update
// that follows.
void CopyTensorOnDevice(TF_OpKernelContext* ctx, TF_Tensor* source,
                        TF_Tensor* dest) {
  ScopedAutoreleasePool pool;
  TF_Status* status = TF_NewStatus();

  SP_Stream stream = StreamForContext(ctx, status);
  BufferSlice src_slice;
  BufferSlice dst_slice;
  if (TF_GetCode(status) == TF_OK &&
      SliceForTensor(source, &src_slice, status) &&
      SliceForTensor(dest, &dst_slice, status)) {
    const size_t bytes = TF_TensorByteSize(source);
    OrderedCommandBuffer command_buffer(stream);
    if (command_buffer.ok() && bytes > 0) {
      id<MTLBlitCommandEncoder> encoder =
          [command_buffer.get() blitCommandEncoder];
      [encoder copyFromBuffer:src_slice.buffer
                 sourceOffset:src_slice.offset
                     toBuffer:dst_slice.buffer
            destinationOffset:dst_slice.offset
                         size:bytes];
      [encoder endEncoding];
      command_buffer.Commit();
    }
  }
  if (TF_GetCode(status) != TF_OK) {
    LOG(ERROR) << "Metal: variable copy failed: " << TF_Message(status);
  }
  TF_DeleteStatus(status);
}

struct TrainingOp {
  TF_DataType dtype = TF_FLOAT;
  bool use_locking = false;
  bool use_nesterov = false;
};

template <bool kWantNesterov>
void* TrainingOp_Create(TF_OpKernelConstruction* ctx) {
  TF_Status* status = TF_NewStatus();
  auto* op = new TrainingOp();
  TF_OpKernelConstruction_GetAttrType(ctx, "T", &op->dtype, status);
  if (TF_GetCode(status) != TF_OK) {
    TF_OpKernelConstruction_Failure(ctx, status);
    TF_DeleteStatus(status);
    delete op;
    return nullptr;
  }

  TF_Bool use_locking = 0;
  TF_OpKernelConstruction_GetAttrBool(ctx, "use_locking", &use_locking, status);
  if (TF_GetCode(status) != TF_OK) TF_SetStatus(status, TF_OK, "");
  op->use_locking = use_locking != 0;

  if (kWantNesterov) {
    TF_Bool use_nesterov = 0;
    TF_OpKernelConstruction_GetAttrBool(ctx, "use_nesterov", &use_nesterov,
                                        status);
    if (TF_GetCode(status) != TF_OK) TF_SetStatus(status, TF_OK, "");
    op->use_nesterov = use_nesterov != 0;
  }

  TF_DeleteStatus(status);
  return op;
}

// float32 and float16, matching what CUDA registers. Half is storage only:
// every shader here widens to float before touching the arithmetic.
const char* ValueSuffix(TF_DataType t) {
  return t == TF_HALF ? "_half" : "_float";
}

void TrainingOp_Delete(void* kernel) {
  delete static_cast<TrainingOp*>(kernel);
}

// Holds the lock the variable machinery hands out, releasing it however the
// enclosing scope exits.
class ScopedVariableLock {
 public:
  ScopedVariableLock() = default;
  ~ScopedVariableLock() {
    if (holder_ != nullptr) TF_ReleaseVariableInputLockHolder(holder_);
  }
  ScopedVariableLock(const ScopedVariableLock&) = delete;
  ScopedVariableLock& operator=(const ScopedVariableLock&) = delete;

  TF_VariableInputLockHolder** address() { return &holder_; }

 private:
  TF_VariableInputLockHolder* holder_ = nullptr;
};

/*** GRADIENT DESCENT ***/

void ApplyGradientDescent_ComputeImpl(TrainingOp* op, TF_OpKernelContext* ctx,
                                      TF_Status* status) {
  const int locked_inputs[] = {0};
  ScopedVariableLock lock;
  TF_MaybeLockVariableInputMutexesInOrder(ctx, op->use_locking, /*sparse=*/false,
                                          locked_inputs, 1, &CopyTensorOnDevice,
                                          lock.address(), status);
  if (TF_GetCode(status) != TF_OK) return;

  ScopedTensor var;
  TF_GetInputTensorFromVariable(ctx, 0, op->use_locking,
                                /*isVariantType=*/false, /*sparse=*/false,
                                &CopyTensorOnDevice, var.address(), status);
  if (TF_GetCode(status) != TF_OK) return;

  ScopedTensor alpha;
  ScopedTensor delta;
  TF_GetInput(ctx, 1, alpha.address(), status);
  if (TF_GetCode(status) != TF_OK) return;
  TF_GetInput(ctx, 2, delta.address(), status);
  if (TF_GetCode(status) != TF_OK) return;

  const int64_t count = TF_TensorElementCount(var.get());
  if (count != TF_TensorElementCount(delta.get())) {
    TF_SetStatus(status, TF_INVALID_ARGUMENT,
                 "Metal: ResourceApplyGradientDescent variable and delta have "
                 "different sizes.");
    return;
  }
  if (count == 0) return;

  SP_Stream stream = StreamForContext(ctx, status);
  if (TF_GetCode(status) != TF_OK) return;

  id<MTLComputePipelineState> pipeline = PipelineFor(
      DeviceForStream(stream),
      (std::string("tf_apply_gradient_descent") + ValueSuffix(op->dtype))
          .c_str(),
      status);
  if (pipeline == nil) return;

  BufferSlice var_slice;
  BufferSlice alpha_slice;
  BufferSlice delta_slice;
  if (!SliceForTensor(var.get(), &var_slice, status)) return;
  if (!SliceForTensor(alpha.get(), &alpha_slice, status)) return;
  if (!SliceForTensor(delta.get(), &delta_slice, status)) return;

  OrderedCommandBuffer command_buffer(stream);
  if (!command_buffer.ok()) {
    TF_SetStatus(status, TF_RESOURCE_EXHAUSTED,
                 "Metal: could not create a command buffer for the optimizer.");
    return;
  }

  OptimizerParams params;
  params.count = static_cast<uint32_t>(count);
  params.use_nesterov = 0;
  params.padding0 = 0;
  params.padding1 = 0;

  id<MTLComputeCommandEncoder> encoder =
      [command_buffer.get() computeCommandEncoder];
  [encoder setComputePipelineState:pipeline];
  [encoder setBuffer:var_slice.buffer offset:var_slice.offset atIndex:0];
  [encoder setBuffer:alpha_slice.buffer offset:alpha_slice.offset atIndex:1];
  [encoder setBuffer:delta_slice.buffer offset:delta_slice.offset atIndex:2];
  [encoder setBytes:&params length:sizeof(params) atIndex:3];
  Dispatch1D(encoder, pipeline, params.count);
  [encoder endEncoding];
  command_buffer.Commit();
}

/*** ADAM ***/

void ApplyAdam_ComputeImpl(TrainingOp* op, TF_OpKernelContext* ctx,
                           TF_Status* status) {
  // var, m and v are all resources and are all written, so all three are
  // locked together and in index order, which is what prevents two optimiser
  // steps on the same variables from deadlocking against each other.
  const int locked_inputs[] = {0, 1, 2};
  ScopedVariableLock lock;
  TF_MaybeLockVariableInputMutexesInOrder(ctx, op->use_locking, /*sparse=*/false,
                                          locked_inputs, 3, &CopyTensorOnDevice,
                                          lock.address(), status);
  if (TF_GetCode(status) != TF_OK) return;

  ScopedTensor var;
  ScopedTensor m;
  ScopedTensor v;
  TF_GetInputTensorFromVariable(ctx, 0, op->use_locking, false, false,
                                &CopyTensorOnDevice, var.address(), status);
  if (TF_GetCode(status) != TF_OK) return;
  TF_GetInputTensorFromVariable(ctx, 1, op->use_locking, false, false,
                                &CopyTensorOnDevice, m.address(), status);
  if (TF_GetCode(status) != TF_OK) return;
  TF_GetInputTensorFromVariable(ctx, 2, op->use_locking, false, false,
                                &CopyTensorOnDevice, v.address(), status);
  if (TF_GetCode(status) != TF_OK) return;

  // beta1_power, beta2_power, lr, beta1, beta2, epsilon, grad.
  ScopedTensor scalars[6];
  for (int i = 0; i < 6; ++i) {
    TF_GetInput(ctx, 3 + i, scalars[i].address(), status);
    if (TF_GetCode(status) != TF_OK) return;
  }
  ScopedTensor grad;
  TF_GetInput(ctx, 9, grad.address(), status);
  if (TF_GetCode(status) != TF_OK) return;

  const int64_t count = TF_TensorElementCount(var.get());
  if (count != TF_TensorElementCount(grad.get()) ||
      count != TF_TensorElementCount(m.get()) ||
      count != TF_TensorElementCount(v.get())) {
    TF_SetStatus(status, TF_INVALID_ARGUMENT,
                 "Metal: ResourceApplyAdam variable, slots and gradient have "
                 "different sizes.");
    return;
  }
  if (count == 0) return;

  SP_Stream stream = StreamForContext(ctx, status);
  if (TF_GetCode(status) != TF_OK) return;

  id<MTLComputePipelineState> pipeline =
      PipelineFor(DeviceForStream(stream),
                  (std::string("tf_apply_adam") + ValueSuffix(op->dtype))
                      .c_str(),
                  status);
  if (pipeline == nil) return;

  BufferSlice slices[10];
  TF_Tensor* tensors[10] = {var.get(),        m.get(),
                            v.get(),          scalars[0].get(),
                            scalars[1].get(), scalars[2].get(),
                            scalars[3].get(), scalars[4].get(),
                            scalars[5].get(), grad.get()};
  for (int i = 0; i < 10; ++i) {
    if (!SliceForTensor(tensors[i], &slices[i], status)) return;
  }

  OrderedCommandBuffer command_buffer(stream);
  if (!command_buffer.ok()) {
    TF_SetStatus(status, TF_RESOURCE_EXHAUSTED,
                 "Metal: could not create a command buffer for Adam.");
    return;
  }

  OptimizerParams params;
  params.count = static_cast<uint32_t>(count);
  params.use_nesterov = op->use_nesterov ? 1 : 0;
  params.padding0 = 0;
  params.padding1 = 0;

  id<MTLComputeCommandEncoder> encoder =
      [command_buffer.get() computeCommandEncoder];
  [encoder setComputePipelineState:pipeline];
  for (int i = 0; i < 10; ++i) {
    [encoder setBuffer:slices[i].buffer offset:slices[i].offset atIndex:i];
  }
  [encoder setBytes:&params length:sizeof(params) atIndex:10];
  Dispatch1D(encoder, pipeline, params.count);
  [encoder endEncoding];
  command_buffer.Commit();
}

/*** MOMENTUM AND RMSPROP ***/

// These take one or more resource slots followed by scalars and the gradient.
// `slot_count` is how many leading inputs are resources; the rest are read as
// ordinary tensors in order.
//
// `grad_index` is passed rather than assumed to be last, because it is not.
// ResourceApplyRMSProp ends with the gradient, but ResourceApplyMomentum and
// ResourceApplyKerasMomentum are (var, accum, lr, grad, momentum), so taking
// the final input as the gradient compared the variable against a scalar and
// rejected every call.
void ApplySlotted_ComputeImpl(TrainingOp* op, TF_OpKernelContext* ctx,
                              const char* shader_base, int slot_count,
                              int scalar_count, int grad_index,
                              TF_Status* status) {
  std::vector<int> locked(slot_count);
  for (int i = 0; i < slot_count; ++i) locked[i] = i;
  ScopedVariableLock lock;
  TF_MaybeLockVariableInputMutexesInOrder(ctx, op->use_locking, /*sparse=*/false,
                                          locked.data(), slot_count,
                                          &CopyTensorOnDevice, lock.address(),
                                          status);
  if (TF_GetCode(status) != TF_OK) return;

  const int total = slot_count + scalar_count + 1;  // +1 for the gradient
  std::vector<ScopedTensor> tensors(total);
  for (int i = 0; i < slot_count; ++i) {
    TF_GetInputTensorFromVariable(ctx, i, op->use_locking, false, false,
                                  &CopyTensorOnDevice, tensors[i].address(),
                                  status);
    if (TF_GetCode(status) != TF_OK) return;
  }
  for (int i = slot_count; i < total; ++i) {
    TF_GetInput(ctx, i, tensors[i].address(), status);
    if (TF_GetCode(status) != TF_OK) return;
  }

  const int64_t count = TF_TensorElementCount(tensors[0].get());
  for (int i = 1; i < slot_count; ++i) {
    if (TF_TensorElementCount(tensors[i].get()) != count) {
      TF_SetStatus(status, TF_INVALID_ARGUMENT,
                   "Metal: optimizer slots differ in size from the variable.");
      return;
    }
  }
  if (TF_TensorElementCount(tensors[grad_index].get()) != count) {
    TF_SetStatus(status, TF_INVALID_ARGUMENT,
                 "Metal: optimizer gradient differs in size from the variable.");
    return;
  }
  if (count == 0) return;

  SP_Stream stream = StreamForContext(ctx, status);
  if (TF_GetCode(status) != TF_OK) return;
  const std::string shader = std::string(shader_base) + ValueSuffix(op->dtype);
  id<MTLComputePipelineState> pipeline =
      PipelineFor(DeviceForStream(stream), shader.c_str(), status);
  if (pipeline == nil) return;

  std::vector<BufferSlice> slices(total);
  for (int i = 0; i < total; ++i) {
    if (!SliceForTensor(tensors[i].get(), &slices[i], status)) return;
  }

  OrderedCommandBuffer command_buffer(stream);
  if (!command_buffer.ok()) {
    TF_SetStatus(status, TF_RESOURCE_EXHAUSTED,
                 "Metal: could not create a command buffer for the optimizer.");
    return;
  }

  OptimizerParams params;
  params.count = static_cast<uint32_t>(count);
  params.use_nesterov = op->use_nesterov ? 1 : 0;
  params.padding0 = 0;
  params.padding1 = 0;

  id<MTLComputeCommandEncoder> encoder =
      [command_buffer.get() computeCommandEncoder];
  [encoder setComputePipelineState:pipeline];
  for (int i = 0; i < total; ++i) {
    [encoder setBuffer:slices[i].buffer offset:slices[i].offset atIndex:i];
  }
  [encoder setBytes:&params length:sizeof(params) atIndex:total];
  Dispatch1D(encoder, pipeline, params.count);
  [encoder endEncoding];
  command_buffer.Commit();
}

// var, accum | lr, grad, momentum
void ApplyMomentum_ComputeImpl(TrainingOp* op, TF_OpKernelContext* ctx,
                               TF_Status* status) {
  ApplySlotted_ComputeImpl(op, ctx, "tf_apply_momentum",
                           /*slot_count=*/2, /*scalar_count=*/2,
                           /*grad_index=*/3, status);
}

// Keras folds the learning rate into the accumulator, so this is a different
// update from ResourceApplyMomentum and needs its own shader.
void ApplyKerasMomentum_ComputeImpl(TrainingOp* op, TF_OpKernelContext* ctx,
                                    TF_Status* status) {
  ApplySlotted_ComputeImpl(op, ctx, "tf_apply_keras_momentum",
                           /*slot_count=*/2, /*scalar_count=*/2,
                           /*grad_index=*/3, status);
}

// var, ms, mom | lr, rho, momentum, epsilon, grad
void ApplyRMSProp_ComputeImpl(TrainingOp* op, TF_OpKernelContext* ctx,
                              TF_Status* status) {
  ApplySlotted_ComputeImpl(op, ctx, "tf_apply_rms_prop",
                           /*slot_count=*/3, /*scalar_count=*/4,
                           /*grad_index=*/7, status);
}

/*** WRAPPERS AND REGISTRATION ***/

#define METAL_DEFINE_TRAINING_COMPUTE(NAME, IMPL)                             \
  void NAME(void* kernel, TF_OpKernelContext* ctx) {                          \
    ScopedAutoreleasePool pool;                                               \
    TF_Status* status = TF_NewStatus();                                       \
    auto* op = static_cast<TrainingOp*>(kernel);                              \
    if (op == nullptr) {                                                      \
      TF_SetStatus(status, TF_INTERNAL,                                       \
                   "Metal: optimizer kernel has no state.");                  \
    } else {                                                                  \
      IMPL(op, ctx, status);                                                  \
    }                                                                         \
    if (TF_GetCode(status) != TF_OK) TF_OpKernelContext_Failure(ctx, status); \
    TF_DeleteStatus(status);                                                  \
  }

METAL_DEFINE_TRAINING_COMPUTE(ApplyGradientDescent_Compute,
                              ApplyGradientDescent_ComputeImpl)
METAL_DEFINE_TRAINING_COMPUTE(ApplyAdam_Compute, ApplyAdam_ComputeImpl)
METAL_DEFINE_TRAINING_COMPUTE(ApplyMomentum_Compute, ApplyMomentum_ComputeImpl)
METAL_DEFINE_TRAINING_COMPUTE(ApplyKerasMomentum_Compute,
                              ApplyKerasMomentum_ComputeImpl)
METAL_DEFINE_TRAINING_COMPUTE(ApplyRMSProp_Compute, ApplyRMSProp_ComputeImpl)

#undef METAL_DEFINE_TRAINING_COMPUTE

void RegisterTraining(const char* op_name,
                      void* (*create)(TF_OpKernelConstruction*),
                      void (*compute)(void*, TF_OpKernelContext*),
                      const std::string& kernel_name, TF_DataType dtype) {
  TF_Status* status = TF_NewStatus();
  TF_KernelBuilder* builder = TF_NewKernelBuilder(
      op_name, kMetalDeviceType, create, compute, &TrainingOp_Delete);
  TF_KernelBuilder_TypeConstraint(builder, "T", dtype, status);
  // The variable handles are resource tensors and live on the host.
  TF_KernelBuilder_HostMemory(builder, "var");
  if (TF_GetCode(status) == TF_OK) {
    TF_RegisterKernelBuilder(kernel_name.c_str(), builder, status);
  } else {
    TF_DeleteKernelBuilder(builder);
  }
  if (TF_GetCode(status) != TF_OK) {
    LOG(ERROR) << "Metal: could not register kernel " << kernel_name << ": "
               << TF_Message(status);
  }
  TF_DeleteStatus(status);
}

}  // namespace

void RegisterMetalTrainingKernels() {
  static constexpr TF_DataType kDTypes[] = {TF_FLOAT, TF_HALF};
  static constexpr const char* kSuffixes[] = {"Float", "Half"};
  for (int i = 0; i < 2; ++i) {
    const TF_DataType t = kDTypes[i];
    const std::string s = kSuffixes[i];
    RegisterTraining("ResourceApplyGradientDescent", &TrainingOp_Create<false>,
                     &ApplyGradientDescent_Compute,
                     "MetalResourceApplyGradientDescent" + s, t);
    RegisterTraining("ResourceApplyAdam", &TrainingOp_Create<true>,
                     &ApplyAdam_Compute, "MetalResourceApplyAdam" + s, t);
    RegisterTraining("ResourceApplyMomentum", &TrainingOp_Create<true>,
                     &ApplyMomentum_Compute, "MetalResourceApplyMomentum" + s,
                     t);
    RegisterTraining("ResourceApplyKerasMomentum", &TrainingOp_Create<true>,
                     &ApplyKerasMomentum_Compute,
                     "MetalResourceApplyKerasMomentum" + s, t);
    RegisterTraining("ResourceApplyRMSProp", &TrainingOp_Create<false>,
                     &ApplyRMSProp_Compute, "MetalResourceApplyRMSProp" + s, t);
  }
}

}  // namespace metal
}  // namespace tensorflow
