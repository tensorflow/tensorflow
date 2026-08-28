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
#include <cstring>
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

// The max-pooling family that carries indices: MaxPoolWithArgmax, its
// gradient, and the second-order gradients.
//
// These do not go through MPSGraph. Its pooling-with-indices returns the
// winner's position inside the pooling window, whereas TensorFlow defines the
// index as a position in the flattened image, and there is no way to convert
// one into the other without knowing which window the value came from. They
// run as shaders that scan the window directly and emit TensorFlow's index.
//
// Everything here is float32 and NHWC. The gradient scatter accumulates with a
// Metal atomic, which exists for float and not for half, so half is left
// unregistered rather than silently accumulated in the wrong precision.

int64_t ElementCount(const std::vector<int64_t>& s) {
  int64_t n = 1;
  for (int64_t d : s) n *= d;
  return n;
}

bool ReadHostVector(TF_Tensor* t, std::vector<int64_t>* out,
                    TF_Status* status) {
  const int64_t count = TF_TensorElementCount(t);
  const TF_DataType dtype = TF_TensorType(t);
  const void* data = TF_TensorData(t);
  if (data == nullptr && count > 0) {
    TF_SetStatus(status, TF_INVALID_ARGUMENT,
                 "Metal: a host-memory argument has no data.");
    return false;
  }
  out->clear();
  for (int64_t i = 0; i < count; ++i) {
    if (dtype == TF_INT32) {
      out->push_back(static_cast<const int32_t*>(data)[i]);
    } else if (dtype == TF_INT64) {
      out->push_back(static_cast<const int64_t*>(data)[i]);
    } else {
      TF_SetStatus(status, TF_INVALID_ARGUMENT,
                   "Metal: expected an int32 or int64 argument.");
      return false;
    }
  }
  return true;
}

struct ArgmaxPoolOp {
  int window_h = 1, window_w = 1, stride_h = 1, stride_w = 1;
  bool same_padding = false;
  bool include_batch = false;
  TF_DataType index_dtype = TF_INT64;
  // The V2 form takes the window and strides as tensors instead.
  bool window_from_tensors = false;
};

void* ArgmaxPoolOp_Create(TF_OpKernelConstruction* ctx) {
  TF_Status* status = TF_NewStatus();
  auto* op = new ArgmaxPoolOp();

  char padding[16] = {0};
  TF_OpKernelConstruction_GetAttrString(ctx, "padding", padding,
                                        sizeof(padding) - 1, status);
  if (TF_GetCode(status) != TF_OK) {
    TF_OpKernelConstruction_Failure(ctx, status);
    TF_DeleteStatus(status);
    delete op;
    return nullptr;
  }
  if (std::strcmp(padding, "SAME") == 0) {
    op->same_padding = true;
  } else if (std::strcmp(padding, "VALID") != 0) {
    TF_SetStatus(status, TF_UNIMPLEMENTED,
                 "Metal: max pooling with indices supports SAME and VALID "
                 "padding only.");
    TF_OpKernelConstruction_Failure(ctx, status);
    TF_DeleteStatus(status);
    delete op;
    return nullptr;
  }

  // Only the second-order gradients have a data_format at all, and only NHWC
  // is handled: the shaders index NHWC directly.
  char format[8] = {0};
  TF_OpKernelConstruction_GetAttrString(ctx, "data_format", format,
                                        sizeof(format) - 1, status);
  if (TF_GetCode(status) == TF_OK && std::strcmp(format, "NHWC") != 0 &&
      format[0] != '\0') {
    TF_SetStatus(status, TF_UNIMPLEMENTED,
                 "Metal: max pooling with indices supports NHWC only.");
    TF_OpKernelConstruction_Failure(ctx, status);
    TF_DeleteStatus(status);
    delete op;
    return nullptr;
  }
  TF_SetStatus(status, TF_OK, "");

  int32_t ksize[4] = {1, 1, 1, 1};
  TF_OpKernelConstruction_GetAttrInt32List(ctx, "ksize", ksize, 4, status);
  if (TF_GetCode(status) == TF_OK) {
    int32_t strides[4] = {1, 1, 1, 1};
    TF_OpKernelConstruction_GetAttrInt32List(ctx, "strides", strides, 4,
                                             status);
    if (TF_GetCode(status) == TF_OK) {
      if (ksize[0] != 1 || ksize[3] != 1 || strides[0] != 1 ||
          strides[3] != 1) {
        TF_SetStatus(status, TF_UNIMPLEMENTED,
                     "Metal: pooling over the batch or channel dimension is "
                     "not supported.");
        TF_OpKernelConstruction_Failure(ctx, status);
        TF_DeleteStatus(status);
        delete op;
        return nullptr;
      }
      op->window_h = ksize[1];
      op->window_w = ksize[2];
      op->stride_h = strides[1];
      op->stride_w = strides[2];
    } else {
      op->window_from_tensors = true;
    }
  } else {
    op->window_from_tensors = true;
  }
  TF_SetStatus(status, TF_OK, "");

  TF_Bool include_batch = 0;
  TF_OpKernelConstruction_GetAttrBool(ctx, "include_batch_in_index",
                                      &include_batch, status);
  if (TF_GetCode(status) != TF_OK) TF_SetStatus(status, TF_OK, "");
  op->include_batch = include_batch != 0;

  TF_DataType targmax = TF_INT64;
  TF_OpKernelConstruction_GetAttrType(ctx, "Targmax", &targmax, status);
  if (TF_GetCode(status) != TF_OK) TF_SetStatus(status, TF_OK, "");
  op->index_dtype = targmax;

  TF_DeleteStatus(status);
  return op;
}

void ArgmaxPoolOp_Delete(void* kernel) {
  delete static_cast<ArgmaxPoolOp*>(kernel);
}

// Fills in the geometry shared by every shader here. `in_shape` is the
// unpooled NHWC shape; the pooled extent follows TensorFlow's padding rules.
bool Geometry(const ArgmaxPoolOp& op, int kh, int kw, int sh, int sw,
              const std::vector<int64_t>& in_shape, PoolIndexParams* params,
              std::vector<int64_t>* out_shape, TF_Status* status) {
  if (in_shape.size() != 4) {
    TF_SetStatus(status, TF_INVALID_ARGUMENT,
                 "Metal: max pooling with indices expects a rank-4 input.");
    return false;
  }
  const int64_t out_h =
      op.same_padding ? (in_shape[1] + sh - 1) / sh
                      : (in_shape[1] < kh ? 0 : (in_shape[1] - kh) / sh + 1);
  const int64_t out_w =
      op.same_padding ? (in_shape[2] + sw - 1) / sw
                      : (in_shape[2] < kw ? 0 : (in_shape[2] - kw) / sw + 1);
  *out_shape = {in_shape[0], out_h, out_w, in_shape[3]};

  const int64_t pad_h =
      std::max<int64_t>(0, (out_h - 1) * sh + kh - in_shape[1]);
  const int64_t pad_w =
      std::max<int64_t>(0, (out_w - 1) * sw + kw - in_shape[2]);
  params->batch = static_cast<uint32_t>(in_shape[0]);
  params->in_h = static_cast<uint32_t>(in_shape[1]);
  params->in_w = static_cast<uint32_t>(in_shape[2]);
  params->channels = static_cast<uint32_t>(in_shape[3]);
  params->out_h = static_cast<uint32_t>(out_h);
  params->out_w = static_cast<uint32_t>(out_w);
  params->kh = static_cast<uint32_t>(kh);
  params->kw = static_cast<uint32_t>(kw);
  params->stride_h = static_cast<uint32_t>(sh);
  params->stride_w = static_cast<uint32_t>(sw);
  params->pad_top = static_cast<int32_t>(op.same_padding ? pad_h / 2 : 0);
  params->pad_left = static_cast<int32_t>(op.same_padding ? pad_w / 2 : 0);
  params->count = static_cast<uint32_t>(ElementCount(*out_shape));
  params->include_batch = op.include_batch ? 1 : 0;
  params->padding0 = 0;
  params->padding1 = 0;
  return true;
}

const char* IndexSuffix(TF_DataType t) {
  return t == TF_INT32 ? "_i32" : "_i64";
}

/*** MAX POOL WITH ARGMAX ***/

void MaxPoolWithArgmax_ComputeImpl(ArgmaxPoolOp* op, TF_OpKernelContext* ctx,
                                   TF_Status* status) {
  ScopedTensor input;
  TF_GetInput(ctx, 0, input.address(), status);
  if (TF_GetCode(status) != TF_OK) return;
  const std::vector<int64_t> in_shape = ShapeOf(input.get());

  PoolIndexParams params;
  std::vector<int64_t> out_shape;
  if (!Geometry(*op, op->window_h, op->window_w, op->stride_h, op->stride_w,
                in_shape, &params, &out_shape, status)) {
    return;
  }

  const int64_t count = ElementCount(out_shape);
  ScopedTensor output, argmax;
  output.reset(TF_AllocateOutput(ctx, 0, TF_FLOAT, out_shape.data(), 4,
                                 static_cast<size_t>(count) * sizeof(float),
                                 status));
  if (TF_GetCode(status) != TF_OK) return;
  argmax.reset(TF_AllocateOutput(
      ctx, 1, op->index_dtype, out_shape.data(), 4,
      static_cast<size_t>(count) * TF_DataTypeSize(op->index_dtype), status));
  if (TF_GetCode(status) != TF_OK) return;
  if (count == 0) return;

  SP_Stream stream = StreamForContext(ctx, status);
  if (TF_GetCode(status) != TF_OK) return;
  const std::string fn =
      std::string("tf_maxpool_argmax_float") + IndexSuffix(op->index_dtype);
  id<MTLComputePipelineState> pipeline =
      PipelineFor(DeviceForStream(stream), fn.c_str(), status);
  if (pipeline == nil) return;

  BufferSlice in_slice, out_slice, idx_slice;
  if (!SliceForTensor(input.get(), &in_slice, status)) return;
  if (!SliceForTensor(output.get(), &out_slice, status)) return;
  if (!SliceForTensor(argmax.get(), &idx_slice, status)) return;

  OrderedCommandBuffer command_buffer(stream);
  if (!command_buffer.ok()) {
    TF_SetStatus(status, TF_RESOURCE_EXHAUSTED,
                 "Metal: could not create a command buffer for "
                 "MaxPoolWithArgmax.");
    return;
  }
  id<MTLComputeCommandEncoder> encoder =
      [command_buffer.get() computeCommandEncoder];
  [encoder setComputePipelineState:pipeline];
  [encoder setBuffer:in_slice.buffer offset:in_slice.offset atIndex:0];
  [encoder setBuffer:out_slice.buffer offset:out_slice.offset atIndex:1];
  [encoder setBuffer:idx_slice.buffer offset:idx_slice.offset atIndex:2];
  [encoder setBytes:&params length:sizeof(params) atIndex:3];
  Dispatch1D(encoder, pipeline, params.count);
  [encoder endEncoding];
  command_buffer.Commit();
}

/*** GRADIENTS FROM STORED INDICES ***/

// `scatter` selects MaxPoolGradWithArgmax, which writes an input-shaped result
// with atomic accumulation, from MaxPoolGradGradWithArgmax, which writes a
// pooled-shaped result by plain gather. Both take (input, grad, argmax) and
// both need the pooled geometry, so only the destination differs.
void ArgmaxGrad_ComputeImpl(ArgmaxPoolOp* op, TF_OpKernelContext* ctx,
                            bool scatter, TF_Status* status) {
  ScopedTensor input, grad, argmax;
  TF_GetInput(ctx, 0, input.address(), status);
  if (TF_GetCode(status) != TF_OK) return;
  TF_GetInput(ctx, 1, grad.address(), status);
  if (TF_GetCode(status) != TF_OK) return;
  TF_GetInput(ctx, 2, argmax.address(), status);
  if (TF_GetCode(status) != TF_OK) return;

  const std::vector<int64_t> in_shape = ShapeOf(input.get());
  PoolIndexParams params;
  std::vector<int64_t> pooled_shape;
  if (!Geometry(*op, op->window_h, op->window_w, op->stride_h, op->stride_w,
                in_shape, &params, &pooled_shape, status)) {
    return;
  }

  const std::vector<int64_t>& out_shape = scatter ? in_shape : pooled_shape;
  const int64_t out_count = ElementCount(out_shape);
  ScopedTensor output;
  output.reset(TF_AllocateOutput(ctx, 0, TF_FLOAT, out_shape.data(), 4,
                                 static_cast<size_t>(out_count) * sizeof(float),
                                 status));
  if (TF_GetCode(status) != TF_OK) return;
  if (out_count == 0) return;

  SP_Stream stream = StreamForContext(ctx, status);
  if (TF_GetCode(status) != TF_OK) return;
  const TF_DataType idx_dtype = TF_TensorType(argmax.get());
  const std::string fn =
      std::string(scatter ? "tf_maxpool_grad_with_argmax_float"
                          : "tf_maxpool_gradgrad_with_argmax_float") +
      IndexSuffix(idx_dtype);
  id<MTLComputePipelineState> pipeline =
      PipelineFor(DeviceForStream(stream), fn.c_str(), status);
  if (pipeline == nil) return;

  BufferSlice grad_slice, idx_slice, out_slice;
  if (!SliceForTensor(grad.get(), &grad_slice, status)) return;
  if (!SliceForTensor(argmax.get(), &idx_slice, status)) return;
  if (!SliceForTensor(output.get(), &out_slice, status)) return;

  OrderedCommandBuffer command_buffer(stream);
  if (!command_buffer.ok()) {
    TF_SetStatus(status, TF_RESOURCE_EXHAUSTED,
                 "Metal: could not create a command buffer for a max pooling "
                 "gradient.");
    return;
  }
  if (scatter) {
    // Overlapping windows share winners, so the destination accumulates and
    // has to start at zero. The fill goes in the same command buffer as the
    // dispatch, so the ordering belongs to the queue.
    id<MTLBlitCommandEncoder> zero = [command_buffer.get() blitCommandEncoder];
    [zero fillBuffer:out_slice.buffer
               range:NSMakeRange(out_slice.offset,
                                 static_cast<NSUInteger>(out_count) *
                                     sizeof(float))
               value:0];
    [zero endEncoding];
  }
  if (params.count > 0) {
    id<MTLComputeCommandEncoder> encoder =
        [command_buffer.get() computeCommandEncoder];
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:grad_slice.buffer offset:grad_slice.offset atIndex:0];
    [encoder setBuffer:idx_slice.buffer offset:idx_slice.offset atIndex:1];
    [encoder setBuffer:out_slice.buffer offset:out_slice.offset atIndex:2];
    [encoder setBytes:&params length:sizeof(params) atIndex:3];
    Dispatch1D(encoder, pipeline, params.count);
    [encoder endEncoding];
  }
  command_buffer.Commit();
}

/*** SECOND-ORDER GRADIENT WITHOUT STORED INDICES ***/

// MaxPoolGradGrad(orig_input, orig_output, grad) and its V2 form. The winner
// is recomputed from orig_input rather than read from an index tensor;
// orig_output is unused, exactly as in the CPU kernel, which keeps it only for
// shape inference.
void MaxPoolGradGrad_ComputeImpl(ArgmaxPoolOp* op, TF_OpKernelContext* ctx,
                                 TF_Status* status) {
  ScopedTensor orig_input, orig_output, grad;
  TF_GetInput(ctx, 0, orig_input.address(), status);
  if (TF_GetCode(status) != TF_OK) return;
  TF_GetInput(ctx, 1, orig_output.address(), status);
  if (TF_GetCode(status) != TF_OK) return;
  TF_GetInput(ctx, 2, grad.address(), status);
  if (TF_GetCode(status) != TF_OK) return;

  int kh = op->window_h, kw = op->window_w;
  int sh = op->stride_h, sw = op->stride_w;
  if (op->window_from_tensors) {
    ScopedTensor ksize_t, strides_t;
    TF_GetInput(ctx, 3, ksize_t.address(), status);
    if (TF_GetCode(status) != TF_OK) return;
    TF_GetInput(ctx, 4, strides_t.address(), status);
    if (TF_GetCode(status) != TF_OK) return;
    std::vector<int64_t> ks, st;
    if (!ReadHostVector(ksize_t.get(), &ks, status)) return;
    if (!ReadHostVector(strides_t.get(), &st, status)) return;
    if (ks.size() != 4 || st.size() != 4) {
      TF_SetStatus(status, TF_INVALID_ARGUMENT,
                   "Metal: pooling ksize and strides must have four entries.");
      return;
    }
    if (ks[0] != 1 || ks[3] != 1 || st[0] != 1 || st[3] != 1) {
      TF_SetStatus(status, TF_UNIMPLEMENTED,
                   "Metal: pooling over the batch or channel dimension is not "
                   "supported.");
      return;
    }
    kh = static_cast<int>(ks[1]);
    kw = static_cast<int>(ks[2]);
    sh = static_cast<int>(st[1]);
    sw = static_cast<int>(st[2]);
  }

  const std::vector<int64_t> in_shape = ShapeOf(orig_input.get());
  PoolIndexParams params;
  std::vector<int64_t> out_shape;
  if (!Geometry(*op, kh, kw, sh, sw, in_shape, &params, &out_shape, status)) {
    return;
  }

  const int64_t count = ElementCount(out_shape);
  ScopedTensor output;
  output.reset(TF_AllocateOutput(ctx, 0, TF_FLOAT, out_shape.data(), 4,
                                 static_cast<size_t>(count) * sizeof(float),
                                 status));
  if (TF_GetCode(status) != TF_OK) return;
  if (count == 0) return;

  SP_Stream stream = StreamForContext(ctx, status);
  if (TF_GetCode(status) != TF_OK) return;
  id<MTLComputePipelineState> pipeline = PipelineFor(
      DeviceForStream(stream), "tf_maxpool_gradgrad_float", status);
  if (pipeline == nil) return;

  BufferSlice in_slice, grad_slice, out_slice;
  if (!SliceForTensor(orig_input.get(), &in_slice, status)) return;
  if (!SliceForTensor(grad.get(), &grad_slice, status)) return;
  if (!SliceForTensor(output.get(), &out_slice, status)) return;

  OrderedCommandBuffer command_buffer(stream);
  if (!command_buffer.ok()) {
    TF_SetStatus(status, TF_RESOURCE_EXHAUSTED,
                 "Metal: could not create a command buffer for "
                 "MaxPoolGradGrad.");
    return;
  }
  id<MTLComputeCommandEncoder> encoder =
      [command_buffer.get() computeCommandEncoder];
  [encoder setComputePipelineState:pipeline];
  [encoder setBuffer:in_slice.buffer offset:in_slice.offset atIndex:0];
  [encoder setBuffer:grad_slice.buffer offset:grad_slice.offset atIndex:1];
  [encoder setBuffer:out_slice.buffer offset:out_slice.offset atIndex:2];
  [encoder setBytes:&params length:sizeof(params) atIndex:3];
  Dispatch1D(encoder, pipeline, params.count);
  [encoder endEncoding];
  command_buffer.Commit();
}

#define TF_METAL_ARGMAX_POOL_COMPUTE(NAME, BODY)                            \
  void NAME(void* kernel, TF_OpKernelContext* ctx) {                        \
    ScopedAutoreleasePool pool;                                             \
    TF_Status* status = TF_NewStatus();                                     \
    auto* op = static_cast<ArgmaxPoolOp*>(kernel);                          \
    if (op == nullptr) {                                                    \
      TF_SetStatus(status, TF_INTERNAL,                                     \
                   "Metal: a max pooling kernel has no state.");            \
    } else {                                                                \
      BODY;                                                                 \
    }                                                                       \
    if (TF_GetCode(status) != TF_OK) TF_OpKernelContext_Failure(ctx, status); \
    TF_DeleteStatus(status);                                                \
  }

TF_METAL_ARGMAX_POOL_COMPUTE(MaxPoolWithArgmax_Compute,
                             MaxPoolWithArgmax_ComputeImpl(op, ctx, status))
TF_METAL_ARGMAX_POOL_COMPUTE(MaxPoolGradWithArgmax_Compute,
                             ArgmaxGrad_ComputeImpl(op, ctx, /*scatter=*/true,
                                                    status))
TF_METAL_ARGMAX_POOL_COMPUTE(MaxPoolGradGradWithArgmax_Compute,
                             ArgmaxGrad_ComputeImpl(op, ctx, /*scatter=*/false,
                                                    status))
TF_METAL_ARGMAX_POOL_COMPUTE(MaxPoolGradGrad_Compute,
                             MaxPoolGradGrad_ComputeImpl(op, ctx, status))

void Register(const char* op_name,
              void (*compute)(void*, TF_OpKernelContext*),
              const std::string& name,
              const std::vector<const char*>& host_inputs) {
  TF_Status* status = TF_NewStatus();
  TF_KernelBuilder* builder =
      TF_NewKernelBuilder(op_name, kMetalDeviceType, &ArgmaxPoolOp_Create,
                          compute, &ArgmaxPoolOp_Delete);
  TF_KernelBuilder_TypeConstraint(builder, "T", TF_FLOAT, status);
  for (const char* input : host_inputs) {
    TF_KernelBuilder_HostMemory(builder, input);
  }
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

void RegisterMetalMaxPoolArgmaxKernels() {
  Register("MaxPoolWithArgmax", &MaxPoolWithArgmax_Compute,
           "MetalMaxPoolWithArgmaxFloat", {});
  Register("MaxPoolGradWithArgmax", &MaxPoolGradWithArgmax_Compute,
           "MetalMaxPoolGradWithArgmaxFloat", {});
  Register("MaxPoolGradGradWithArgmax", &MaxPoolGradGradWithArgmax_Compute,
           "MetalMaxPoolGradGradWithArgmaxFloat", {});
  Register("MaxPoolGradGrad", &MaxPoolGradGrad_Compute,
           "MetalMaxPoolGradGradFloat", {});
  // The V2 form reads its window and strides on the host to size the output.
  Register("MaxPoolGradGradV2", &MaxPoolGradGrad_Compute,
           "MetalMaxPoolGradGradV2Float", {"ksize", "strides"});
}

}  // namespace metal
}  // namespace tensorflow
