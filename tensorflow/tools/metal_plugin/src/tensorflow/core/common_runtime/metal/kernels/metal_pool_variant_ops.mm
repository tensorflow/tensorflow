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
#include <cstring>
#include <string>
#include <vector>

#include "absl/log/log.h"
#include "tensorflow/c/kernels.h"
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

// MaxPoolV2 and MaxPoolGradV2, whose window and strides arrive as tensors
// instead of attributes, and MaxPoolWithArgmax with its gradient.
//
// The V2 forms exist because a graph can compute its pooling window at run
// time. Everything downstream is the same as the attribute form, so the only
// real work is reading the two tensors on the host and keying the cached
// graph on their values rather than on a construction-time constant.

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
    if (dtype == TF_INT32) out->push_back(static_cast<const int32_t*>(data)[i]);
    else if (dtype == TF_INT64) out->push_back(static_cast<const int64_t*>(data)[i]);
    else {
      TF_SetStatus(status, TF_INVALID_ARGUMENT,
                   "Metal: expected an int32 or int64 argument.");
      return false;
    }
  }
  return true;
}

struct PoolVariantOp {
  TF_DataType dtype = TF_FLOAT;
  bool same_padding = false;
  bool nhwc = true;
  // Filled from attributes for the argmax forms, from tensors for the V2 ones.
  int window_h = 1, window_w = 1, stride_h = 1, stride_w = 1;
  bool from_attrs = false;
};

void* PoolVariantOp_Create(TF_OpKernelConstruction* ctx) {
  TF_Status* status = TF_NewStatus();
  auto* op = new PoolVariantOp();
  TF_OpKernelConstruction_GetAttrType(ctx, "T", &op->dtype, status);
  if (TF_GetCode(status) != TF_OK) {
    TF_OpKernelConstruction_Failure(ctx, status);
    TF_DeleteStatus(status);
    delete op;
    return nullptr;
  }
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
                 "Metal: only SAME and VALID padding are supported.");
    TF_OpKernelConstruction_Failure(ctx, status);
    TF_DeleteStatus(status);
    delete op;
    return nullptr;
  }
  char format[8] = {0};
  TF_OpKernelConstruction_GetAttrString(ctx, "data_format", format,
                                        sizeof(format) - 1, status);
  if (TF_GetCode(status) != TF_OK) {
    TF_SetStatus(status, TF_OK, "");
    op->nhwc = true;
  } else {
    op->nhwc = std::strcmp(format, "NCHW") != 0;
  }

  // MaxPoolWithArgmax keeps ksize and strides as attributes; the V2 forms do
  // not have them at all.
  int32_t ksize[4] = {1, 1, 1, 1};
  TF_OpKernelConstruction_GetAttrInt32List(ctx, "ksize", ksize, 4, status);
  if (TF_GetCode(status) == TF_OK) {
    int32_t strides[4] = {1, 1, 1, 1};
    TF_OpKernelConstruction_GetAttrInt32List(ctx, "strides", strides, 4,
                                             status);
    if (TF_GetCode(status) == TF_OK) {
      op->from_attrs = true;
      op->window_h = ksize[SpatialHeightIndex(op->nhwc)];
      op->window_w = ksize[SpatialWidthIndex(op->nhwc)];
      op->stride_h = strides[SpatialHeightIndex(op->nhwc)];
      op->stride_w = strides[SpatialWidthIndex(op->nhwc)];
    }
  }
  TF_SetStatus(status, TF_OK, "");
  TF_DeleteStatus(status);
  return op;
}

void PoolVariantOp_Delete(void* kernel) {
  delete static_cast<PoolVariantOp*>(kernel);
}

MPSGraphPooling2DOpDescriptor* Descriptor(const PoolVariantOp& op, int kh,
                                          int kw, int sh, int sw) {
  return [MPSGraphPooling2DOpDescriptor
      descriptorWithKernelWidth:static_cast<NSUInteger>(kw)
                   kernelHeight:static_cast<NSUInteger>(kh)
                      strideInX:static_cast<NSUInteger>(sw)
                      strideInY:static_cast<NSUInteger>(sh)
                   paddingStyle:op.same_padding ? MPSGraphPaddingStyleTF_SAME
                                                : MPSGraphPaddingStyleTF_VALID
                     dataLayout:op.nhwc ? MPSGraphTensorNamedDataLayoutNHWC
                                        : MPSGraphTensorNamedDataLayoutNCHW];
}

int64_t PoolExtent(int64_t input, int window, int stride, bool same) {
  if (same) return (input + stride - 1) / stride;
  return input < window ? 0 : (input - window) / stride + 1;
}

// Reads the window and stride for a V2 op, which passes both as tensors.
bool ReadWindow(TF_OpKernelContext* ctx, const PoolVariantOp& op,
                int ksize_index, int strides_index, int* kh, int* kw, int* sh,
                int* sw, TF_Status* status) {
  ScopedTensor ksize_t, strides_t;
  TF_GetInput(ctx, ksize_index, ksize_t.address(), status);
  if (TF_GetCode(status) != TF_OK) return false;
  TF_GetInput(ctx, strides_index, strides_t.address(), status);
  if (TF_GetCode(status) != TF_OK) return false;

  std::vector<int64_t> ks, st;
  if (!ReadHostVector(ksize_t.get(), &ks, status)) return false;
  if (!ReadHostVector(strides_t.get(), &st, status)) return false;
  if (ks.size() != 4 || st.size() != 4) {
    TF_SetStatus(status, TF_INVALID_ARGUMENT,
                 "Metal: pooling ksize and strides must have four entries.");
    return false;
  }
  const int batch = 0;
  const int channel = op.nhwc ? 3 : 1;
  if (ks[batch] != 1 || ks[channel] != 1 || st[batch] != 1 ||
      st[channel] != 1) {
    TF_SetStatus(status, TF_UNIMPLEMENTED,
                 "Metal: pooling over the batch or channel dimension is not "
                 "supported.");
    return false;
  }
  *kh = static_cast<int>(ks[SpatialHeightIndex(op.nhwc)]);
  *kw = static_cast<int>(ks[SpatialWidthIndex(op.nhwc)]);
  *sh = static_cast<int>(st[SpatialHeightIndex(op.nhwc)]);
  *sw = static_cast<int>(st[SpatialWidthIndex(op.nhwc)]);
  return true;
}

void AppendKey(const PoolVariantOp& op, int kh, int kw, int sh, int sw,
               std::string* key) {
  key->append("/k").append(std::to_string(kh)).push_back('x');
  key->append(std::to_string(kw));
  key->append("/s").append(std::to_string(sh)).push_back('x');
  key->append(std::to_string(sw));
  key->append(op.same_padding ? "/SAME" : "/VALID");
  key->append(op.nhwc ? "/NHWC" : "/NCHW");
  key->append("/t").append(std::to_string(static_cast<int>(op.dtype)));
}

/*** MAX POOL V2 ***/

void MaxPoolV2_ComputeImpl(PoolVariantOp* op, TF_OpKernelContext* ctx,
                           TF_Status* status) {
  ScopedTensor input;
  TF_GetInput(ctx, 0, input.address(), status);
  if (TF_GetCode(status) != TF_OK) return;
  int kh, kw, sh, sw;
  if (!ReadWindow(ctx, *op, 1, 2, &kh, &kw, &sh, &sw, status)) return;

  const std::vector<int64_t> in_shape = ShapeOf(input.get());
  if (in_shape.size() != 4) {
    TF_SetStatus(status, TF_INVALID_ARGUMENT,
                 "Metal: MaxPoolV2 expects a rank-4 input.");
    return;
  }
  const int h = SpatialHeightIndex(op->nhwc);
  const int w = SpatialWidthIndex(op->nhwc);
  std::vector<int64_t> out_shape = in_shape;
  out_shape[h] = PoolExtent(in_shape[h], kh, sh, op->same_padding);
  out_shape[w] = PoolExtent(in_shape[w], kw, sw, op->same_padding);

  const int64_t count = ElementCount(out_shape);
  ScopedTensor output;
  output.reset(TF_AllocateOutput(
      ctx, 0, op->dtype, out_shape.data(), 4,
      static_cast<size_t>(count) * TF_DataTypeSize(op->dtype), status));
  if (TF_GetCode(status) != TF_OK) return;
  if (count == 0) return;

  SP_Stream stream = StreamForContext(ctx, status);
  if (TF_GetCode(status) != TF_OK) return;
  id<MTLDevice> device = DeviceForStream(stream);
  MPSDataType mps_dtype;
  if (!MPSTypeFor(op->dtype, &mps_dtype, status)) return;

  std::string key = "MaxPoolV2";
  AppendShapeToKey(in_shape, &key);
  AppendKey(*op, kh, kw, sh, sw, &key);
  const PoolVariantOp captured = *op;

  const CachedGraph* cached = LookupOrBuildGraph(
      key,
      ^(CachedGraph* out) {
        MPSGraphTensor* x = [out->graph placeholderWithShape:MPSShape(in_shape)
                                                    dataType:mps_dtype
                                                        name:nil];
        [out->inputs addObject:x];
        [out->outputs addObject:[out->graph
                                    maxPooling2DWithSourceTensor:x
                                                      descriptor:Descriptor(
                                                                     captured,
                                                                     kh, kw, sh,
                                                                     sw)
                                                            name:nil]];
      },
      status);
  if (cached == nullptr) return;

  MPSGraphTensorData* in_data =
      TensorDataForTensor(input.get(), op->dtype, device, status);
  if (in_data == nil) return;
  MPSGraphTensorData* o_data =
      TensorDataForTensor(output.get(), op->dtype, device, status);
  if (o_data == nil) return;
  RunGraph(stream, *cached, @[ in_data ], @[ o_data ], status);
}

/*** MAX POOL GRAD V2 ***/

void MaxPoolGradV2_ComputeImpl(PoolVariantOp* op, TF_OpKernelContext* ctx,
                               TF_Status* status) {
  ScopedTensor orig_input, orig_output, grad;
  TF_GetInput(ctx, 0, orig_input.address(), status);
  if (TF_GetCode(status) != TF_OK) return;
  TF_GetInput(ctx, 1, orig_output.address(), status);
  if (TF_GetCode(status) != TF_OK) return;
  TF_GetInput(ctx, 2, grad.address(), status);
  if (TF_GetCode(status) != TF_OK) return;
  int kh, kw, sh, sw;
  if (!ReadWindow(ctx, *op, 3, 4, &kh, &kw, &sh, &sw, status)) return;

  const std::vector<int64_t> in_shape = ShapeOf(orig_input.get());
  const std::vector<int64_t> grad_shape = ShapeOf(grad.get());
  const int64_t count = ElementCount(in_shape);

  ScopedTensor output;
  output.reset(TF_AllocateOutput(
      ctx, 0, op->dtype, in_shape.data(), static_cast<int>(in_shape.size()),
      static_cast<size_t>(count) * TF_DataTypeSize(op->dtype), status));
  if (TF_GetCode(status) != TF_OK) return;
  if (count == 0) return;

  SP_Stream stream = StreamForContext(ctx, status);
  if (TF_GetCode(status) != TF_OK) return;
  id<MTLDevice> device = DeviceForStream(stream);
  MPSDataType mps_dtype;
  if (!MPSTypeFor(op->dtype, &mps_dtype, status)) return;

  std::string key = "MaxPoolGradV2";
  AppendShapeToKey(in_shape, &key);
  AppendShapeToKey(grad_shape, &key);
  AppendKey(*op, kh, kw, sh, sw, &key);
  const PoolVariantOp captured = *op;

  const CachedGraph* cached = LookupOrBuildGraph(
      key,
      ^(CachedGraph* out) {
        MPSGraphTensor* dy = [out->graph placeholderWithShape:MPSShape(grad_shape)
                                                     dataType:mps_dtype
                                                         name:nil];
        MPSGraphTensor* src =
            [out->graph placeholderWithShape:MPSShape(in_shape)
                                    dataType:mps_dtype
                                        name:nil];
        [out->inputs addObject:dy];
        [out->inputs addObject:src];
        [out->outputs addObject:[out->graph
                                    maxPooling2DGradientWithGradientTensor:dy
                                                              sourceTensor:src
                                                                descriptor:
                                                                    Descriptor(
                                                                        captured,
                                                                        kh, kw,
                                                                        sh, sw)
                                                                      name:nil]];
      },
      status);
  if (cached == nullptr) return;

  MPSGraphTensorData* g_data =
      TensorDataForTensor(grad.get(), op->dtype, device, status);
  if (g_data == nil) return;
  MPSGraphTensorData* s_data =
      TensorDataForTensor(orig_input.get(), op->dtype, device, status);
  if (s_data == nil) return;
  MPSGraphTensorData* o_data =
      TensorDataForTensor(output.get(), op->dtype, device, status);
  if (o_data == nil) return;
  RunGraph(stream, *cached, @[ g_data, s_data ], @[ o_data ], status);
}

// MaxPoolWithArgmax is deliberately absent.
//
// MPSGraph does return indices alongside the pooled values, but with
// returnIndicesMode set to MPSGraphPoolingReturnIndicesGlobalFlatten2D it
// yields the position *within the pooling window*, not the flattened position
// in the image. On a 4x4 input pooled 2x2 it returned 1, 3, 1, 2 where
// TensorFlow defines the answer as 1, 7, 13, 10.
//
// Converting one to the other is arithmetic on the window origin, which is
// straightforward for VALID padding and fiddly for SAME, where the origin can
// sit outside the image. Emitting indices in the wrong coordinate system would
// not fail loudly; it would quietly corrupt any model that unpools with them.
// Until the conversion is written and checked, TensorFlow places this op on
// the host, which is correct.

/*** WRAPPERS AND REGISTRATION ***/

#define METAL_COMPUTE(NAME, IMPL)                                             \
  void NAME(void* kernel, TF_OpKernelContext* ctx) {                          \
    ScopedAutoreleasePool pool;                                               \
    TF_Status* status = TF_NewStatus();                                       \
    auto* op = static_cast<PoolVariantOp*>(kernel);                           \
    if (op == nullptr) {                                                      \
      TF_SetStatus(status, TF_INTERNAL, "Metal: kernel has no state.");       \
    } else {                                                                  \
      IMPL(op, ctx, status);                                                  \
    }                                                                         \
    if (TF_GetCode(status) != TF_OK) TF_OpKernelContext_Failure(ctx, status); \
    TF_DeleteStatus(status);                                                  \
  }

METAL_COMPUTE(MaxPoolV2_Compute, MaxPoolV2_ComputeImpl)
METAL_COMPUTE(MaxPoolGradV2_Compute, MaxPoolGradV2_ComputeImpl)

#undef METAL_COMPUTE

void Register(const char* op_name,
              void (*compute)(void*, TF_OpKernelContext*), TF_DataType dtype,
              const std::string& name, std::vector<const char*> host_args) {
  TF_Status* status = TF_NewStatus();
  TF_KernelBuilder* builder =
      TF_NewKernelBuilder(op_name, kMetalDeviceType, &PoolVariantOp_Create,
                          compute, &PoolVariantOp_Delete);
  TF_KernelBuilder_TypeConstraint(builder, "T", dtype, status);
  for (const char* a : host_args) TF_KernelBuilder_HostMemory(builder, a);
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

void RegisterMetalPoolVariantKernels() {
  static constexpr TF_DataType kDTypes[] = {TF_FLOAT, TF_HALF};
  static constexpr const char* kSuffixes[] = {"Float", "Half"};
  for (int i = 0; i < 2; ++i) {
    const TF_DataType t = kDTypes[i];
    const std::string s = kSuffixes[i];
    // The window and strides are read on the host to size the output.
    Register("MaxPoolV2", &MaxPoolV2_Compute, t, "MetalMaxPoolV2" + s,
             {"ksize", "strides"});
    Register("MaxPoolGradV2", &MaxPoolGradV2_Compute, t,
             "MetalMaxPoolGradV2" + s, {"ksize", "strides"});
  }
}

}  // namespace metal
}  // namespace tensorflow
