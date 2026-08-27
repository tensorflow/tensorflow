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

// Deletes a TF_Tensor on scope exit. The C kernel API hands out owned tensors,
// and with several early-return paths per kernel, leaking one leaks device
// memory for the life of the process.
class ScopedTensor {
 public:
  ScopedTensor() = default;
  ~ScopedTensor() {
    if (tensor_ != nullptr) TF_DeleteTensor(tensor_);
  }
  ScopedTensor(const ScopedTensor&) = delete;
  ScopedTensor& operator=(const ScopedTensor&) = delete;

  TF_Tensor** address() { return &tensor_; }
  TF_Tensor* get() const { return tensor_; }
  void reset(TF_Tensor* tensor) {
    if (tensor_ != nullptr) TF_DeleteTensor(tensor_);
    tensor_ = tensor;
  }

 private:
  TF_Tensor* tensor_ = nullptr;
};

// One dispatch of a one-dimensional shader over `count` elements.
//
// Uses dispatchThreadgroups: rather than dispatchThreads:, so the grid rounds
// up to whole threadgroups and the call works regardless of GPU family. The
// shaders bounds-check against `count`, which is what makes the rounding safe.
void Dispatch1D(id<MTLComputeCommandEncoder> encoder,
                id<MTLComputePipelineState> pipeline, uint32_t count) {
  const NSUInteger threads_per_group =
      std::min<NSUInteger>(pipeline.maxTotalThreadsPerThreadgroup, count);
  const NSUInteger groups = (count + threads_per_group - 1) / threads_per_group;
  [encoder dispatchThreadgroups:MTLSizeMake(groups, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(threads_per_group, 1, 1)];
}

std::string DescribeShape(const std::vector<int64_t>& shape) {
  std::string text = "[";
  for (size_t i = 0; i < shape.size(); ++i) {
    if (i > 0) text += ",";
    text += std::to_string(shape[i]);
  }
  return text + "]";
}

/*** BINARY ELEMENTWISE ***/

enum class BinaryKind { kAdd, kSub, kMul };

// Per-instance kernel state, built once at construction from the node's "T"
// attribute so that Compute does no attribute lookups.
struct BinaryOp {
  std::string function_name;
  TF_DataType dtype = TF_FLOAT;
};

const char* ShaderForBinary(BinaryKind kind, TF_DataType dtype) {
  const bool is_half = dtype == TF_HALF;
  switch (kind) {
    case BinaryKind::kAdd:
      return is_half ? "tf_add_half" : "tf_add_float";
    case BinaryKind::kSub:
      return is_half ? "tf_sub_half" : "tf_sub_float";
    case BinaryKind::kMul:
      return is_half ? "tf_mul_half" : "tf_mul_float";
  }
  return nullptr;
}

// A distinct instantiation per op kind, which is how the kind reaches Compute:
// TF_NewKernelBuilder takes bare function pointers with no user data, so the
// only way to carry per-registration information is a distinct function.
template <BinaryKind kKind>
void* BinaryOp_Create(TF_OpKernelConstruction* ctx) {
  TF_Status* status = TF_NewStatus();
  auto* op = new BinaryOp();
  TF_OpKernelConstruction_GetAttrType(ctx, "T", &op->dtype, status);
  if (TF_GetCode(status) != TF_OK) {
    TF_OpKernelConstruction_Failure(ctx, status);
    TF_DeleteStatus(status);
    delete op;
    return nullptr;
  }
  op->function_name = ShaderForBinary(kKind, op->dtype);
  TF_DeleteStatus(status);
  return op;
}

void BinaryOp_Delete(void* kernel) { delete static_cast<BinaryOp*>(kernel); }

// Resolves the output shape for a binary op.
//
// Only two cases are accepted: identical shapes, and one operand being a
// scalar. Full NumPy broadcasting needs per-operand stride arithmetic in the
// shader and belongs with the wider op coverage. Anything else is rejected
// here, naming both shapes, so a graph fails loudly rather than quietly
// producing wrong numbers.
bool ResolveBinaryShape(TF_Tensor* lhs, TF_Tensor* rhs,
                        std::vector<int64_t>* out_shape, bool* lhs_is_scalar,
                        bool* rhs_is_scalar, TF_Status* status) {
  const std::vector<int64_t> lhs_shape = ShapeOf(lhs);
  const std::vector<int64_t> rhs_shape = ShapeOf(rhs);

  *lhs_is_scalar = false;
  *rhs_is_scalar = false;

  if (lhs_shape == rhs_shape) {
    *out_shape = lhs_shape;
    return true;
  }
  if (NumElements(lhs) == 1) {
    *lhs_is_scalar = true;
    *out_shape = rhs_shape;
    return true;
  }
  if (NumElements(rhs) == 1) {
    *rhs_is_scalar = true;
    *out_shape = lhs_shape;
    return true;
  }

  TF_SetStatus(
      status, TF_UNIMPLEMENTED,
      ("Metal: broadcasting " + DescribeShape(lhs_shape) + " against " +
       DescribeShape(rhs_shape) +
       " is not supported yet; only equal shapes or a scalar operand are.")
          .c_str());
  return false;
}

void BinaryOp_ComputeImpl(BinaryOp* op, TF_OpKernelContext* ctx,
                          TF_Status* status) {
  if (op == nullptr) {
    TF_SetStatus(status, TF_INTERNAL,
                 "Metal: binary kernel has no state; construction failed.");
    return;
  }

  ScopedTensor lhs;
  ScopedTensor rhs;
  TF_GetInput(ctx, 0, lhs.address(), status);
  if (TF_GetCode(status) != TF_OK) return;
  TF_GetInput(ctx, 1, rhs.address(), status);
  if (TF_GetCode(status) != TF_OK) return;

  std::vector<int64_t> out_shape;
  bool lhs_is_scalar = false;
  bool rhs_is_scalar = false;
  if (!ResolveBinaryShape(lhs.get(), rhs.get(), &out_shape, &lhs_is_scalar,
                          &rhs_is_scalar, status)) {
    return;
  }

  int64_t count = 1;
  for (int64_t dim : out_shape) count *= dim;

  ScopedTensor output;
  output.reset(TF_AllocateOutput(
      ctx, 0, op->dtype, out_shape.data(), static_cast<int>(out_shape.size()),
      static_cast<size_t>(count) * TF_DataTypeSize(op->dtype), status));
  if (TF_GetCode(status) != TF_OK) return;
  // Nothing to compute, and a zero-sized dispatch is a Metal error rather than
  // a no-op.
  if (count == 0) return;

  SP_Stream stream = StreamForContext(ctx, status);
  if (TF_GetCode(status) != TF_OK) return;

  id<MTLComputePipelineState> pipeline =
      PipelineFor(DeviceForStream(stream), op->function_name.c_str(), status);
  if (pipeline == nil) return;

  BufferSlice lhs_slice;
  BufferSlice rhs_slice;
  BufferSlice out_slice;
  if (!SliceForTensor(lhs.get(), &lhs_slice, status)) return;
  if (!SliceForTensor(rhs.get(), &rhs_slice, status)) return;
  if (!SliceForTensor(output.get(), &out_slice, status)) return;

  OrderedCommandBuffer command_buffer(stream);
  if (!command_buffer.ok()) {
    TF_SetStatus(
        status, TF_RESOURCE_EXHAUSTED,
        "Metal: could not create a command buffer for an elementwise op.");
    return;
  }

  ElementwiseParams params;
  params.count = static_cast<uint32_t>(count);
  params.lhs_is_scalar = lhs_is_scalar ? 1 : 0;
  params.rhs_is_scalar = rhs_is_scalar ? 1 : 0;
  params.padding = 0;

  id<MTLComputeCommandEncoder> encoder =
      [command_buffer.get() computeCommandEncoder];
  [encoder setComputePipelineState:pipeline];
  [encoder setBuffer:lhs_slice.buffer offset:lhs_slice.offset atIndex:0];
  [encoder setBuffer:rhs_slice.buffer offset:rhs_slice.offset atIndex:1];
  [encoder setBuffer:out_slice.buffer offset:out_slice.offset atIndex:2];
  [encoder setBytes:&params length:sizeof(params) atIndex:3];
  Dispatch1D(encoder, pipeline, params.count);
  [encoder endEncoding];
  command_buffer.Commit();
}

void BinaryOp_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* status = TF_NewStatus();
  BinaryOp_ComputeImpl(static_cast<BinaryOp*>(kernel), ctx, status);
  if (TF_GetCode(status) != TF_OK) TF_OpKernelContext_Failure(ctx, status);
  TF_DeleteStatus(status);
}

/*** CAST ***/

struct CastOp {
  std::string function_name;
  TF_DataType dst_dtype = TF_FLOAT;
};

void* CastOp_Create(TF_OpKernelConstruction* ctx) {
  TF_Status* status = TF_NewStatus();
  auto* op = new CastOp();
  TF_DataType src_dtype = TF_FLOAT;
  TF_OpKernelConstruction_GetAttrType(ctx, "SrcT", &src_dtype, status);
  if (TF_GetCode(status) == TF_OK) {
    TF_OpKernelConstruction_GetAttrType(ctx, "DstT", &op->dst_dtype, status);
  }
  if (TF_GetCode(status) != TF_OK) {
    TF_OpKernelConstruction_Failure(ctx, status);
    TF_DeleteStatus(status);
    delete op;
    return nullptr;
  }

  if (src_dtype == TF_FLOAT && op->dst_dtype == TF_HALF) {
    op->function_name = "tf_cast_float_to_half";
  } else if (src_dtype == TF_HALF && op->dst_dtype == TF_FLOAT) {
    op->function_name = "tf_cast_half_to_float";
  } else {
    // The type constraints below should prevent this, but a mismatch between
    // the registration and the shader table would otherwise show up as a
    // missing-function error deep in the pipeline cache.
    TF_SetStatus(status, TF_UNIMPLEMENTED,
                 "Metal: unsupported Cast type pair.");
    TF_OpKernelConstruction_Failure(ctx, status);
    TF_DeleteStatus(status);
    delete op;
    return nullptr;
  }
  TF_DeleteStatus(status);
  return op;
}

void CastOp_Delete(void* kernel) { delete static_cast<CastOp*>(kernel); }

void CastOp_ComputeImpl(CastOp* op, TF_OpKernelContext* ctx,
                        TF_Status* status) {
  if (op == nullptr) {
    TF_SetStatus(status, TF_INTERNAL,
                 "Metal: Cast kernel has no state; construction failed.");
    return;
  }

  ScopedTensor input;
  TF_GetInput(ctx, 0, input.address(), status);
  if (TF_GetCode(status) != TF_OK) return;

  const std::vector<int64_t> shape = ShapeOf(input.get());
  const int64_t count = NumElements(input.get());

  ScopedTensor output;
  output.reset(TF_AllocateOutput(
      ctx, 0, op->dst_dtype, shape.data(), static_cast<int>(shape.size()),
      static_cast<size_t>(count) * TF_DataTypeSize(op->dst_dtype), status));
  if (TF_GetCode(status) != TF_OK) return;
  if (count == 0) return;

  SP_Stream stream = StreamForContext(ctx, status);
  if (TF_GetCode(status) != TF_OK) return;

  id<MTLComputePipelineState> pipeline =
      PipelineFor(DeviceForStream(stream), op->function_name.c_str(), status);
  if (pipeline == nil) return;

  BufferSlice in_slice;
  BufferSlice out_slice;
  if (!SliceForTensor(input.get(), &in_slice, status)) return;
  if (!SliceForTensor(output.get(), &out_slice, status)) return;

  OrderedCommandBuffer command_buffer(stream);
  if (!command_buffer.ok()) {
    TF_SetStatus(status, TF_RESOURCE_EXHAUSTED,
                 "Metal: could not create a command buffer for Cast.");
    return;
  }

  ElementwiseParams params;
  params.count = static_cast<uint32_t>(count);
  params.lhs_is_scalar = 0;
  params.rhs_is_scalar = 0;
  params.padding = 0;

  id<MTLComputeCommandEncoder> encoder =
      [command_buffer.get() computeCommandEncoder];
  [encoder setComputePipelineState:pipeline];
  [encoder setBuffer:in_slice.buffer offset:in_slice.offset atIndex:0];
  [encoder setBuffer:out_slice.buffer offset:out_slice.offset atIndex:1];
  [encoder setBytes:&params length:sizeof(params) atIndex:2];
  Dispatch1D(encoder, pipeline, params.count);
  [encoder endEncoding];
  command_buffer.Commit();
}

void CastOp_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* status = TF_NewStatus();
  CastOp_ComputeImpl(static_cast<CastOp*>(kernel), ctx, status);
  if (TF_GetCode(status) != TF_OK) TF_OpKernelContext_Failure(ctx, status);
  TF_DeleteStatus(status);
}

/*** REGISTRATION ***/

void RegisterBinary(const char* op_name, void* (*create)(
                                            TF_OpKernelConstruction*),
                    TF_DataType dtype, const std::string& kernel_name) {
  TF_Status* status = TF_NewStatus();
  TF_KernelBuilder* builder = TF_NewKernelBuilder(
      op_name, kMetalDeviceType, create, &BinaryOp_Compute, &BinaryOp_Delete);
  TF_KernelBuilder_TypeConstraint(builder, "T", dtype, status);
  if (TF_GetCode(status) == TF_OK) {
    TF_RegisterKernelBuilder(kernel_name.c_str(), builder, status);
  } else {
    TF_DeleteKernelBuilder(builder);
  }
  if (TF_GetCode(status) != TF_OK) {
    // Logged, not fatal: a kernel that fails to register leaves the op
    // unplaceable on Metal, which core reports per graph, and that is a better
    // outcome than refusing to import TensorFlow at all.
    LOG(ERROR) << "Metal: could not register kernel " << kernel_name << ": "
               << TF_Message(status);
  }
  TF_DeleteStatus(status);
}

void RegisterCast(TF_DataType src, TF_DataType dst,
                  const std::string& kernel_name) {
  TF_Status* status = TF_NewStatus();
  TF_KernelBuilder* builder = TF_NewKernelBuilder(
      "Cast", kMetalDeviceType, &CastOp_Create, &CastOp_Compute,
      &CastOp_Delete);
  TF_KernelBuilder_TypeConstraint(builder, "SrcT", src, status);
  if (TF_GetCode(status) == TF_OK) {
    TF_KernelBuilder_TypeConstraint(builder, "DstT", dst, status);
  }
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

void RegisterMetalElementwiseKernels() {
  // AddV2 is what modern graphs emit; Add is kept for graphs still carrying
  // the v1 op.
  RegisterBinary("AddV2", &BinaryOp_Create<BinaryKind::kAdd>, TF_FLOAT,
                 "MetalAddV2Float");
  RegisterBinary("AddV2", &BinaryOp_Create<BinaryKind::kAdd>, TF_HALF,
                 "MetalAddV2Half");
  RegisterBinary("Add", &BinaryOp_Create<BinaryKind::kAdd>, TF_FLOAT,
                 "MetalAddFloat");
  RegisterBinary("Add", &BinaryOp_Create<BinaryKind::kAdd>, TF_HALF,
                 "MetalAddHalf");
  RegisterBinary("Sub", &BinaryOp_Create<BinaryKind::kSub>, TF_FLOAT,
                 "MetalSubFloat");
  RegisterBinary("Sub", &BinaryOp_Create<BinaryKind::kSub>, TF_HALF,
                 "MetalSubHalf");
  RegisterBinary("Mul", &BinaryOp_Create<BinaryKind::kMul>, TF_FLOAT,
                 "MetalMulFloat");
  RegisterBinary("Mul", &BinaryOp_Create<BinaryKind::kMul>, TF_HALF,
                 "MetalMulHalf");

  RegisterCast(TF_FLOAT, TF_HALF, "MetalCastFloatToHalf");
  RegisterCast(TF_HALF, TF_FLOAT, "MetalCastHalfToFloat");
}

}  // namespace metal
}  // namespace tensorflow
