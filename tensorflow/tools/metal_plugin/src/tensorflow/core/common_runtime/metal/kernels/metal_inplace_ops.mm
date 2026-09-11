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

#include <algorithm>
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
#include "tensorflow/core/common_runtime/metal/metal_platform.h"
#include "tensorflow/core/common_runtime/metal/metal_stream.h"

namespace tensorflow {
namespace metal {
namespace {

// The two ops a parallel stack decomposes into.
//
// ParallelConcat itself never runs: every device, CUDA included, registers a
// kernel for it that fails on construction, because the graph rewrite always
// replaces it with an allocation followed by one update per stacked value.
// Registering a failing kernel here would add nothing, so what this file
// provides is the pair that does the work.

int64_t ElementCount(const std::vector<int64_t>& s) {
  int64_t n = 1;
  for (int64_t d : s) n *= d;
  return n;
}

struct InplaceOp {
  TF_DataType dtype = TF_FLOAT;
  std::vector<int64_t> shape;
  int64_t loc = 0;
};

void* InplaceOp_Create(TF_OpKernelConstruction* ctx) {
  TF_Status* status = TF_NewStatus();
  auto* op = new InplaceOp();
  TF_OpKernelConstruction_GetAttrType(ctx, "dtype", &op->dtype, status);
  if (TF_GetCode(status) != TF_OK) {
    TF_SetStatus(status, TF_OK, "");
    TF_OpKernelConstruction_GetAttrType(ctx, "T", &op->dtype, status);
    if (TF_GetCode(status) != TF_OK) {
      TF_SetStatus(status, TF_OK, "");
      op->dtype = TF_FLOAT;
    }
  }

  // The start op carries the output shape as a shape attribute, whose size
  // has to be asked for before it can be read.
  int32_t list_size = 0;
  int32_t rank = 0;
  TF_OpKernelConstruction_GetAttrSize(ctx, "shape", &list_size, &rank, status);
  if (TF_GetCode(status) == TF_OK && rank > 0) {
    std::vector<int64_t> dims(static_cast<size_t>(rank), 0);
    TF_OpKernelConstruction_GetAttrTensorShape(ctx, "shape", dims.data(),
                                               dims.size(), status);
    if (TF_GetCode(status) == TF_OK) op->shape = dims;
  }
  TF_SetStatus(status, TF_OK, "");

  int64_t loc = 0;
  TF_OpKernelConstruction_GetAttrInt64(ctx, "loc", &loc, status);
  if (TF_GetCode(status) == TF_OK) op->loc = loc;
  TF_SetStatus(status, TF_OK, "");

  TF_DeleteStatus(status);
  return op;
}

void InplaceOp_Delete(void* kernel) { delete static_cast<InplaceOp*>(kernel); }

/*** START ***/

// Allocates the destination the updates will fill in. Its contents are
// deliberately not initialised: every element is about to be written by an
// update, and zeroing first would double the traffic for nothing.
void Start_ComputeImpl(InplaceOp* op, TF_OpKernelContext* ctx,
                       TF_Status* status) {
  if (op->shape.empty()) {
    TF_SetStatus(status, TF_INVALID_ARGUMENT,
                 "Metal: the parallel stack's shape attribute is missing.");
    return;
  }
  const int64_t count = ElementCount(op->shape);
  ScopedTensor output;
  output.reset(TF_AllocateOutput(
      ctx, 0, op->dtype, op->shape.data(),
      static_cast<int>(op->shape.size()),
      static_cast<size_t>(count) * TF_DataTypeSize(op->dtype), status));
}

/*** UPDATE ***/

// Writes one stacked value into its row and passes the destination on. The
// input is forwarded to the output when the runtime allows it, so a stack of
// n values is one allocation and n row-sized copies rather than n copies of
// the whole thing.
void Update_ComputeImpl(InplaceOp* op, TF_OpKernelContext* ctx,
                        TF_Status* status) {
  ScopedTensor value, update;
  TF_GetInput(ctx, 0, value.address(), status);
  if (TF_GetCode(status) != TF_OK) return;
  TF_GetInput(ctx, 1, update.address(), status);
  if (TF_GetCode(status) != TF_OK) return;

  const std::vector<int64_t> shape = ShapeOf(value.get());
  if (shape.empty()) {
    TF_SetStatus(status, TF_INVALID_ARGUMENT,
                 "Metal: a parallel stack update needs a rank of at least 1.");
    return;
  }
  int64_t row_elements = 1;
  for (size_t i = 1; i < shape.size(); ++i) row_elements *= shape[i];
  // A negative location counts from the end, as it does in the CPU kernel.
  const int64_t loc = op->loc < 0 ? shape[0] + op->loc : op->loc;
  if (loc < 0 || loc >= shape[0]) {
    TF_SetStatus(status, TF_INVALID_ARGUMENT,
                 "Metal: the parallel stack location is out of range.");
    return;
  }
  if (NumElements(update.get()) != row_elements) {
    TF_SetStatus(status, TF_INVALID_ARGUMENT,
                 "Metal: a parallel stack update must be one row.");
    return;
  }

  // Forwarding the destination is what makes a stack of n values cost one
  // allocation rather than n.
  const int candidates[1] = {0};
  int forwarded = -1;
  ScopedTensor output;
  output.reset(TF_ForwardInputOrAllocateOutput(
      ctx, candidates, 1, 0, shape.data(), static_cast<int>(shape.size()),
      &forwarded, status));
  if (TF_GetCode(status) != TF_OK) return;

  SP_Stream stream = StreamForContext(ctx, status);
  if (TF_GetCode(status) != TF_OK) return;
  BufferSlice value_slice, update_slice, out_slice;
  if (!SliceForTensor(value.get(), &value_slice, status)) return;
  if (!SliceForTensor(update.get(), &update_slice, status)) return;
  if (!SliceForTensor(output.get(), &out_slice, status)) return;

  const size_t element = TF_DataTypeSize(op->dtype);
  OrderedCommandBuffer command_buffer(stream);
  if (!command_buffer.ok()) {
    TF_SetStatus(status, TF_RESOURCE_EXHAUSTED,
                 "Metal: could not create a command buffer for a parallel "
                 "stack update.");
    return;
  }
  id<MTLBlitCommandEncoder> encoder =
      [command_buffer.get() blitCommandEncoder];
  // When the input was not forwarded the output is a fresh allocation, so the
  // rest of the destination has to be carried over as well.
  if (out_slice.buffer != value_slice.buffer ||
      out_slice.offset != value_slice.offset) {
    [encoder copyFromBuffer:value_slice.buffer
               sourceOffset:value_slice.offset
                   toBuffer:out_slice.buffer
          destinationOffset:out_slice.offset
                       size:static_cast<NSUInteger>(ElementCount(shape)) *
                            element];
  }
  [encoder copyFromBuffer:update_slice.buffer
             sourceOffset:update_slice.offset
                 toBuffer:out_slice.buffer
        destinationOffset:out_slice.offset +
                          static_cast<size_t>(loc * row_elements) * element
                     size:static_cast<NSUInteger>(row_elements) * element];
  [encoder endEncoding];
  command_buffer.Commit();
}

#define METAL_INPLACE_COMPUTE(NAME, IMPL)                                   \
  void NAME(void* kernel, TF_OpKernelContext* ctx) {                        \
    ScopedAutoreleasePool pool;                                             \
    TF_Status* status = TF_NewStatus();                                     \
    auto* op = static_cast<InplaceOp*>(kernel);                             \
    if (op == nullptr) {                                                    \
      TF_SetStatus(status, TF_INTERNAL,                                     \
                   "Metal: a parallel stack kernel has no state.");         \
    } else {                                                                \
      IMPL(op, ctx, status);                                                \
    }                                                                       \
    if (TF_GetCode(status) != TF_OK) TF_OpKernelContext_Failure(ctx, status); \
    TF_DeleteStatus(status);                                                \
  }

METAL_INPLACE_COMPUTE(Start_Compute, Start_ComputeImpl)
METAL_INPLACE_COMPUTE(Update_Compute, Update_ComputeImpl)

#undef METAL_INPLACE_COMPUTE

// ParallelConcat itself, which is registered so that the op can be placed and
// then always fails if it is ever reached. That is not a stub: it is what
// every device does, CUDA included, because the graph rewrite replaces the op
// before execution and reaching it means the rewrite did not run. Failing here
// with that explanation is more useful than failing later somewhere else.
void* ParallelConcat_Create(TF_OpKernelConstruction* ctx) {
  TF_Status* status = TF_NewStatus();
  TF_SetStatus(status, TF_INTERNAL,
               "Found instance of parallel_stack which could not be properly "
               "replaced during graph construction.");
  TF_OpKernelConstruction_Failure(ctx, status);
  TF_DeleteStatus(status);
  return nullptr;
}

void ParallelConcat_Compute(void* kernel, TF_OpKernelContext* ctx) {
  ScopedAutoreleasePool pool;
  TF_Status* status = TF_NewStatus();
  TF_SetStatus(status, TF_INTERNAL,
               "Found instance of parallel_stack which could not be properly "
               "replaced during graph construction.");
  TF_OpKernelContext_Failure(ctx, status);
  TF_DeleteStatus(status);
}

void Register(const char* op_name,
              void (*compute)(void*, TF_OpKernelContext*), const char* attr,
              TF_DataType dtype, const std::string& name) {
  TF_Status* status = TF_NewStatus();
  TF_KernelBuilder* builder = TF_NewKernelBuilder(
      op_name, kMetalDeviceType, &InplaceOp_Create, compute,
      &InplaceOp_Delete);
  TF_KernelBuilder_TypeConstraint(builder, attr, dtype, status);
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

void RegisterMetalInplaceKernels() {
  static constexpr TF_DataType kDTypes[] = {TF_FLOAT, TF_HALF, TF_INT32,
                                            TF_INT64};
  static constexpr const char* kSuffixes[] = {"Float", "Half", "Int32",
                                              "Int64"};
  for (int i = 0; i < 4; ++i) {
    Register("_ParallelConcatStart", &Start_Compute, "dtype", kDTypes[i],
             std::string("Metal_ParallelConcatStart") + kSuffixes[i]);
    Register("_ParallelConcatUpdate", &Update_Compute, "T", kDTypes[i],
             std::string("Metal_ParallelConcatUpdate") + kSuffixes[i]);
  }

  // The op the rewrite replaces, registered so that the placement is possible
  // and the failure, if it ever happens, says why.
  for (int i = 0; i < 4; ++i) {
    TF_Status* status = TF_NewStatus();
    TF_KernelBuilder* builder = TF_NewKernelBuilder(
        "ParallelConcat", kMetalDeviceType, &ParallelConcat_Create,
        &ParallelConcat_Compute, &InplaceOp_Delete);
    TF_KernelBuilder_TypeConstraint(builder, "T", kDTypes[i], status);
    const std::string name =
        std::string("MetalParallelConcat") + kSuffixes[i];
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
}

}  // namespace metal
}  // namespace tensorflow
