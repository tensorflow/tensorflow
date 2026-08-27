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
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

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
#include "tensorflow/core/common_runtime/metal/kernels/metal_mps_graph.h"
#include "tensorflow/core/common_runtime/metal/metal_platform.h"
#include "tensorflow/core/common_runtime/metal/metal_stream.h"

namespace tensorflow {
namespace metal {
namespace {

// Assign, AssignAdd and AssignSub on reference variables.
//
// These are the TensorFlow 1 variables, the ones a graph holds by reference
// rather than through a resource handle. The kernel C API has an interface for
// exactly them, added so that a pluggable device could implement this trio:
// `TF_AssignRefVariable` performs the assignment, taking a copy callback for
// the part only the device knows how to do.
//
// The two updating forms have no such interface, because updating is
// arithmetic rather than copying. They read the variable's tensor, which for a
// reference input is the variable's own storage, add or subtract in place, and
// forward the reference to the output. That is what the reference contract
// means: the output is the same variable, not a new tensor.

int64_t ElementCount(const std::vector<int64_t>& s) {
  int64_t n = 1;
  for (int64_t d : s) n *= d;
  return n;
}

struct RefOp {
  TF_DataType dtype = TF_FLOAT;
  bool use_locking = false;
  bool validate_shape = true;
  // true for AssignAdd, false for AssignSub.
  bool add = true;
};

void* RefOp_Create(TF_OpKernelConstruction* ctx) {
  TF_Status* status = TF_NewStatus();
  auto* op = new RefOp();
  TF_OpKernelConstruction_GetAttrType(ctx, "T", &op->dtype, status);
  if (TF_GetCode(status) != TF_OK) {
    TF_SetStatus(status, TF_OK, "");
    op->dtype = TF_FLOAT;
  }
  TF_Bool flag = 0;
  TF_OpKernelConstruction_GetAttrBool(ctx, "use_locking", &flag, status);
  if (TF_GetCode(status) == TF_OK) op->use_locking = flag != 0;
  TF_SetStatus(status, TF_OK, "");
  flag = 1;
  TF_OpKernelConstruction_GetAttrBool(ctx, "validate_shape", &flag, status);
  if (TF_GetCode(status) == TF_OK) op->validate_shape = flag != 0;
  TF_SetStatus(status, TF_OK, "");
  TF_DeleteStatus(status);
  return op;
}

void RefOp_Delete(void* kernel) { delete static_cast<RefOp*>(kernel); }

// The copy the assignment interface asks the device to perform. It runs on the
// stream like everything else, so it is ordered against the work around it.
void CopyOnDevice(TF_OpKernelContext* ctx, TF_Tensor* source, TF_Tensor* dest) {
  ScopedAutoreleasePool pool;
  TF_Status* status = TF_NewStatus();
  SP_Stream stream = StreamForContext(ctx, status);
  BufferSlice src, dst;
  if (TF_GetCode(status) == TF_OK && SliceForTensor(source, &src, status) &&
      SliceForTensor(dest, &dst, status)) {
    const size_t bytes = TF_TensorByteSize(source);
    OrderedCommandBuffer command_buffer(stream);
    if (command_buffer.ok() && bytes > 0) {
      id<MTLBlitCommandEncoder> encoder =
          [command_buffer.get() blitCommandEncoder];
      [encoder copyFromBuffer:src.buffer
                 sourceOffset:src.offset
                     toBuffer:dst.buffer
            destinationOffset:dst.offset
                         size:bytes];
      [encoder endEncoding];
      command_buffer.Commit();
    }
  }
  if (TF_GetCode(status) != TF_OK) {
    LOG(ERROR) << "Metal: a reference variable copy failed: "
               << TF_Message(status);
  }
  TF_DeleteStatus(status);
}

void Assign_ComputeImpl(RefOp* op, TF_OpKernelContext* ctx,
                        TF_Status* status) {
  TF_AssignRefVariable(ctx, /*input_ref_index=*/0, /*output_ref_index=*/0,
                       /*value_index=*/1, op->use_locking, op->validate_shape,
                       &CopyOnDevice, status);
}

// Adds or subtracts into the variable's own storage.
void AssignUpdate_ComputeImpl(RefOp* op, TF_OpKernelContext* ctx,
                              TF_Status* status) {
  if (op->use_locking) {
    // The lock a reference variable uses is the kernel's own, and the C API
    // exposes no way to take it. Rather than race quietly, say so; the
    // attribute defaults to false and the graphs that set it are rare.
    TF_SetStatus(status, TF_UNIMPLEMENTED,
                 "Metal: AssignAdd and AssignSub on a reference variable "
                 "cannot take the variable's lock through this interface; "
                 "use_locking must be false.");
    return;
  }
  ScopedTensor variable, value;
  // For a reference input this is the variable's own tensor, so writing into
  // it updates the variable.
  TF_GetInput(ctx, 0, variable.address(), status);
  if (TF_GetCode(status) != TF_OK) return;
  TF_GetInput(ctx, 1, value.address(), status);
  if (TF_GetCode(status) != TF_OK) return;

  const std::vector<int64_t> var_shape = ShapeOf(variable.get());
  const std::vector<int64_t> value_shape = ShapeOf(value.get());
  if (var_shape != value_shape) {
    TF_SetStatus(status, TF_INVALID_ARGUMENT,
                 "Metal: an update must have the variable's shape.");
    return;
  }
  const int64_t count = ElementCount(var_shape);
  // The reference is forwarded whatever happens next, so that the output is
  // the variable rather than a copy of it.
  TF_OpKernelContext_ForwardRefInputToRefOutput(ctx, 0, 0);
  if (count == 0) return;

  SP_Stream stream = StreamForContext(ctx, status);
  if (TF_GetCode(status) != TF_OK) return;
  id<MTLDevice> device = DeviceForStream(stream);
  MPSDataType mps_dtype;
  if (!MPSTypeFor(op->dtype, &mps_dtype, status)) return;

  std::string key = op->add ? "RefAssignAdd" : "RefAssignSub";
  AppendShapeToKey(var_shape, &key);
  key.append("/t").append(std::to_string(static_cast<int>(op->dtype)));

  const bool add = op->add;
  const CachedGraph* cached = LookupOrBuildGraph(
      key,
      ^(CachedGraph* out) {
        MPSGraph* g = out->graph;
        MPSGraphTensor* a = [g placeholderWithShape:MPSShape(var_shape)
                                           dataType:mps_dtype
                                               name:nil];
        MPSGraphTensor* b = [g placeholderWithShape:MPSShape(var_shape)
                                           dataType:mps_dtype
                                               name:nil];
        [out->inputs addObject:a];
        [out->inputs addObject:b];
        [out->outputs
            addObject:add ? [g additionWithPrimaryTensor:a
                                         secondaryTensor:b
                                                    name:nil]
                          : [g subtractionWithPrimaryTensor:a
                                            secondaryTensor:b
                                                       name:nil]];
      },
      status);
  if (cached == nullptr) return;

  MPSGraphTensorData* var_data =
      TensorDataForTensor(variable.get(), op->dtype, device, status);
  if (var_data == nil) return;
  MPSGraphTensorData* value_data =
      TensorDataForTensor(value.get(), op->dtype, device, status);
  if (value_data == nil) return;
  // The result is written back over the variable, which is what makes this an
  // update rather than a new tensor. Reading and writing the same storage in
  // one graph is safe here because every element depends only on itself.
  MPSGraphTensorData* out_data =
      TensorDataForTensor(variable.get(), op->dtype, device, status);
  if (out_data == nil) return;
  RunGraph(stream, *cached, @[ var_data, value_data ], @[ out_data ], status);
}

#define METAL_REF_COMPUTE(NAME, IMPL, ADD)                                  \
  void NAME(void* kernel, TF_OpKernelContext* ctx) {                        \
    ScopedAutoreleasePool pool;                                             \
    TF_Status* status = TF_NewStatus();                                     \
    auto* op = static_cast<RefOp*>(kernel);                                 \
    if (op == nullptr) {                                                    \
      TF_SetStatus(status, TF_INTERNAL,                                     \
                   "Metal: a reference variable kernel has no state.");     \
    } else {                                                                \
      op->add = ADD;                                                        \
      IMPL(op, ctx, status);                                                \
    }                                                                       \
    if (TF_GetCode(status) != TF_OK) TF_OpKernelContext_Failure(ctx, status); \
    TF_DeleteStatus(status);                                                \
  }

METAL_REF_COMPUTE(Assign_Compute, Assign_ComputeImpl, true)
METAL_REF_COMPUTE(AssignAdd_Compute, AssignUpdate_ComputeImpl, true)
METAL_REF_COMPUTE(AssignSub_Compute, AssignUpdate_ComputeImpl, false)

#undef METAL_REF_COMPUTE

void Register(const char* op_name,
              void (*compute)(void*, TF_OpKernelContext*), TF_DataType dtype,
              const std::string& name) {
  TF_Status* status = TF_NewStatus();
  TF_KernelBuilder* builder = TF_NewKernelBuilder(
      op_name, kMetalDeviceType, &RefOp_Create, compute, &RefOp_Delete);
  TF_KernelBuilder_TypeConstraint(builder, "T", dtype, status);
  if (TF_GetCode(status) == TF_OK) {
    TF_RegisterKernelBuilder(name.c_str(), builder, status);
  } else {
    TF_DeleteKernelBuilder(builder);
  }
  if (TF_GetCode(status) != TF_OK) {
    LOG(ERROR) << "Metal: could not register kernel " << name << ": "
               << TF_Message(status);
  }
  TF_DeleteStatus(status);
}

}  // namespace

void RegisterMetalRefVariableKernels() {
  static constexpr TF_DataType kDTypes[] = {TF_FLOAT, TF_HALF, TF_INT32,
                                            TF_INT64};
  static constexpr const char* kSuffixes[] = {"Float", "Half", "Int32",
                                              "Int64"};
  for (int i = 0; i < 4; ++i) {
    Register("Assign", &Assign_Compute, kDTypes[i],
             std::string("MetalAssign") + kSuffixes[i]);
    Register("AssignAdd", &AssignAdd_Compute, kDTypes[i],
             std::string("MetalAssignAdd") + kSuffixes[i]);
    Register("AssignSub", &AssignSub_Compute, kDTypes[i],
             std::string("MetalAssignSub") + kSuffixes[i]);
  }
}

}  // namespace metal
}  // namespace tensorflow
