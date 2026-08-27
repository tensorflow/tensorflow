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
#include "tensorflow/core/common_runtime/metal/kernels/metal_mps_graph.h"
#include "tensorflow/core/common_runtime/metal/metal_platform.h"
#include "tensorflow/core/common_runtime/metal/metal_stream.h"

namespace tensorflow {
namespace metal {
namespace {

// Ops that are each small on their own: the deprecated BatchMatrix aliases,
// ConjugateTranspose, BiasAddV1, Reverse, ReverseSequence, InTopK, Bucketize,
// LRN and its gradient, and CheckNumerics.

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
    else if (dtype == TF_BOOL) out->push_back(static_cast<const unsigned char*>(data)[i]);
    else {
      TF_SetStatus(status, TF_INVALID_ARGUMENT,
                   "Metal: expected an int32, int64 or bool argument.");
      return false;
    }
  }
  return true;
}

struct MiscOp {
  TF_DataType dtype = TF_FLOAT;
  // LRN
  int depth_radius = 5;
  float bias = 1.0f;
  float alpha = 1.0f;
  float beta = 0.5f;
  // ExtractImagePatches
  int patch_h = 1, patch_w = 1;
  int patch_stride_h = 1, patch_stride_w = 1;
  int patch_rate_h = 1, patch_rate_w = 1;
  bool patch_same = false;
};

void* MiscOp_Create(TF_OpKernelConstruction* ctx) {
  TF_Status* status = TF_NewStatus();
  auto* op = new MiscOp();
  TF_OpKernelConstruction_GetAttrType(ctx, "T", &op->dtype, status);
  if (TF_GetCode(status) != TF_OK) {
    TF_SetStatus(status, TF_OK, "");
    op->dtype = TF_FLOAT;
  }
  int32_t radius = 5;
  TF_OpKernelConstruction_GetAttrInt32(ctx, "depth_radius", &radius, status);
  if (TF_GetCode(status) != TF_OK) TF_SetStatus(status, TF_OK, "");
  op->depth_radius = radius;
  TF_OpKernelConstruction_GetAttrFloat(ctx, "bias", &op->bias, status);
  if (TF_GetCode(status) != TF_OK) TF_SetStatus(status, TF_OK, "");
  TF_OpKernelConstruction_GetAttrFloat(ctx, "alpha", &op->alpha, status);
  if (TF_GetCode(status) != TF_OK) TF_SetStatus(status, TF_OK, "");
  TF_OpKernelConstruction_GetAttrFloat(ctx, "beta", &op->beta, status);
  if (TF_GetCode(status) != TF_OK) TF_SetStatus(status, TF_OK, "");

  // ExtractImagePatches carries its window as four-element lists whose batch
  // and channel entries must be 1.
  struct { const char* name; int* h; int* w; } lists[] = {
      {"ksizes", &op->patch_h, &op->patch_w},
      {"strides", &op->patch_stride_h, &op->patch_stride_w},
      {"rates", &op->patch_rate_h, &op->patch_rate_w},
  };
  for (auto& l : lists) {
    int32_t v[4] = {1, 1, 1, 1};
    TF_OpKernelConstruction_GetAttrInt32List(ctx, l.name, v, 4, status);
    if (TF_GetCode(status) != TF_OK) {
      TF_SetStatus(status, TF_OK, "");
      continue;
    }
    *l.h = v[1];
    *l.w = v[2];
  }
  char patch_padding[16] = {0};
  TF_OpKernelConstruction_GetAttrString(ctx, "padding", patch_padding,
                                        sizeof(patch_padding) - 1, status);
  if (TF_GetCode(status) != TF_OK) TF_SetStatus(status, TF_OK, "");
  op->patch_same = std::strcmp(patch_padding, "SAME") == 0;

  TF_DeleteStatus(status);
  return op;
}

void MiscOp_Delete(void* kernel) { delete static_cast<MiscOp*>(kernel); }

NSArray<NSNumber*>* ToNS(const std::vector<int64_t>& v) {
  NSMutableArray<NSNumber*>* a = [NSMutableArray array];
  for (int64_t x : v) [a addObject:@(static_cast<NSInteger>(x))];
  return a;
}

/*** REVERSE, THE V1 FORM ***/

// Reverse takes a bool per dimension rather than a list of axes.
void Reverse_ComputeImpl(MiscOp* op, TF_OpKernelContext* ctx,
                         TF_Status* status) {
  ScopedTensor input, dims_t;
  TF_GetInput(ctx, 0, input.address(), status);
  if (TF_GetCode(status) != TF_OK) return;
  TF_GetInput(ctx, 1, dims_t.address(), status);
  if (TF_GetCode(status) != TF_OK) return;

  const std::vector<int64_t> shape = ShapeOf(input.get());
  std::vector<int64_t> flags;
  if (!ReadHostVector(dims_t.get(), &flags, status)) return;
  if (flags.size() != shape.size()) {
    TF_SetStatus(status, TF_INVALID_ARGUMENT,
                 "Metal: Reverse needs one flag per dimension.");
    return;
  }
  std::vector<int64_t> axes;
  for (size_t i = 0; i < flags.size(); ++i) {
    if (flags[i]) axes.push_back(static_cast<int64_t>(i));
  }

  const int64_t count = ElementCount(shape);
  ScopedTensor output;
  output.reset(TF_AllocateOutput(
      ctx, 0, op->dtype, shape.data(), static_cast<int>(shape.size()),
      static_cast<size_t>(count) * TF_DataTypeSize(op->dtype), status));
  if (TF_GetCode(status) != TF_OK) return;
  if (count == 0) return;

  SP_Stream stream = StreamForContext(ctx, status);
  if (TF_GetCode(status) != TF_OK) return;
  id<MTLDevice> device = DeviceForStream(stream);
  MPSDataType mps_dtype;
  if (!MPSTypeFor(op->dtype, &mps_dtype, status)) return;

  std::string key = "Reverse";
  AppendShapeToKey(shape, &key);
  AppendShapeToKey(axes, &key);
  key.append("/t").append(std::to_string(static_cast<int>(op->dtype)));

  const CachedGraph* cached = LookupOrBuildGraph(
      key,
      ^(CachedGraph* out) {
        MPSGraphTensor* x = [out->graph placeholderWithShape:MPSShape(shape)
                                                    dataType:mps_dtype
                                                        name:nil];
        [out->inputs addObject:x];
        [out->outputs addObject:(axes.empty()
                                     ? [out->graph identityWithTensor:x name:nil]
                                     : [out->graph reverseTensor:x
                                                            axes:ToNS(axes)
                                                            name:nil])];
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

/*** LOCAL RESPONSE NORMALISATION ***/

// LRN normalises each element by the energy in a window of neighbouring
// channels. There is no MPSGraph primitive for it, but a channel window sum
// is a cumulative sum differenced at the window edges, which stays in the
// graph and avoids a shader.
void LRN_ComputeImpl(MiscOp* op, TF_OpKernelContext* ctx, TF_Status* status) {
  ScopedTensor input;
  TF_GetInput(ctx, 0, input.address(), status);
  if (TF_GetCode(status) != TF_OK) return;
  if (TF_NumDims(input.get()) != 4) {
    TF_SetStatus(status, TF_INVALID_ARGUMENT,
                 "Metal: LRN expects a rank-4 NHWC input.");
    return;
  }

  const std::vector<int64_t> shape = ShapeOf(input.get());
  const int64_t channels = shape[3];
  const int64_t count = ElementCount(shape);
  ScopedTensor output;
  output.reset(TF_AllocateOutput(
      ctx, 0, op->dtype, shape.data(), 4,
      static_cast<size_t>(count) * TF_DataTypeSize(op->dtype), status));
  if (TF_GetCode(status) != TF_OK) return;
  if (count == 0) return;

  SP_Stream stream = StreamForContext(ctx, status);
  if (TF_GetCode(status) != TF_OK) return;
  id<MTLDevice> device = DeviceForStream(stream);
  MPSDataType mps_dtype;
  if (!MPSTypeFor(op->dtype, &mps_dtype, status)) return;

  const int radius = op->depth_radius;
  const float bias = op->bias, alpha = op->alpha, beta = op->beta;

  std::string key = "LRN";
  AppendShapeToKey(shape, &key);
  key.append("/r").append(std::to_string(radius));
  key.append("/b").append(std::to_string(bias));
  key.append("/a").append(std::to_string(alpha));
  key.append("/e").append(std::to_string(beta));
  key.append("/t").append(std::to_string(static_cast<int>(op->dtype)));

  // Zero padding of `radius` on each side of the channel axis lets the window
  // sum be taken with plain slices at the edges.
  std::vector<int64_t> left(4, 0), right(4, 0);
  left[3] = radius;
  right[3] = radius;

  const CachedGraph* cached = LookupOrBuildGraph(
      key,
      ^(CachedGraph* out) {
        MPSGraph* g = out->graph;
        MPSGraphTensor* x = [g placeholderWithShape:MPSShape(shape)
                                           dataType:mps_dtype
                                               name:nil];
        MPSGraphTensor* sq = [g squareWithTensor:x name:nil];
        MPSGraphTensor* padded = [g padTensor:sq
                              withPaddingMode:MPSGraphPaddingModeConstant
                                  leftPadding:ToNS(left)
                                 rightPadding:ToNS(right)
                                constantValue:0.0
                                         name:nil];
        // Window sum over 2*radius+1 channels, as a difference of prefix sums.
        //
        // The window for output channel c covers padded indices [c, c+2r], so
        // the sum is prefix[c+2r] - prefix[c-1]. That second term is undefined
        // at c = 0, so a zero is prepended to the prefix and both ends are read
        // from the shifted array: window(c) = Q[c+2r+1] - Q[c]. Taking
        // prefix[c] instead of prefix[c-1] is the natural mistake here and
        // produces plausible but wrong normalisation everywhere.
        MPSGraphTensor* prefix = [g cumulativeSumWithTensor:padded
                                                       axis:3
                                                       name:nil];
        std::vector<int64_t> one_left(4, 0);
        one_left[3] = 1;
        std::vector<int64_t> none(4, 0);
        MPSGraphTensor* shifted = [g padTensor:prefix
                               withPaddingMode:MPSGraphPaddingModeConstant
                                   leftPadding:ToNS(one_left)
                                  rightPadding:ToNS(none)
                                 constantValue:0.0
                                          name:nil];
        MPSGraphTensor* hi = [g sliceTensor:shifted
                                  dimension:3
                                      start:2 * radius + 1
                                     length:static_cast<NSInteger>(channels)
                                       name:nil];
        MPSGraphTensor* lo = [g sliceTensor:shifted
                                  dimension:3
                                      start:0
                                     length:static_cast<NSInteger>(channels)
                                       name:nil];
        MPSGraphTensor* window =
            [g subtractionWithPrimaryTensor:hi secondaryTensor:lo name:nil];
        MPSGraphTensor* scaled =
            [g additionWithPrimaryTensor:
                   [g constantWithScalar:bias dataType:mps_dtype]
                         secondaryTensor:
                             [g multiplicationWithPrimaryTensor:window
                                                secondaryTensor:
                                                    [g constantWithScalar:alpha
                                                                 dataType:mps_dtype]
                                                           name:nil]
                                    name:nil];
        MPSGraphTensor* denom =
            [g powerWithPrimaryTensor:scaled
                      secondaryTensor:[g constantWithScalar:beta
                                                   dataType:mps_dtype]
                                 name:nil];
        [out->inputs addObject:x];
        [out->outputs addObject:[g divisionWithPrimaryTensor:x
                                            secondaryTensor:denom
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

/*** EXTRACT IMAGE PATCHES ***/

// Each output position holds the flattened patch that a convolution would
// have consumed there. Rather than a gather, this convolves with an identity
// filter: a patch of ksize*ksize*C weights where exactly one entry is 1 picks
// out exactly one input element, and stacking those filters across the output
// channel axis reproduces the patch in TensorFlow's row-major order.
void ExtractImagePatches_ComputeImpl(MiscOp* op, TF_OpKernelContext* ctx,
                                     TF_Status* status) {
  ScopedTensor input;
  TF_GetInput(ctx, 0, input.address(), status);
  if (TF_GetCode(status) != TF_OK) return;
  if (TF_NumDims(input.get()) != 4) {
    TF_SetStatus(status, TF_INVALID_ARGUMENT,
                 "Metal: ExtractImagePatches expects a rank-4 NHWC input.");
    return;
  }

  const std::vector<int64_t> in_shape = ShapeOf(input.get());
  const int64_t channels = in_shape[3];
  const int kh = op->patch_h, kw = op->patch_w;
  const int sh = op->patch_stride_h, sw = op->patch_stride_w;
  const int rh = op->patch_rate_h, rw = op->patch_rate_w;
  if (kh < 1 || kw < 1 || sh < 1 || sw < 1 || rh < 1 || rw < 1) {
    TF_SetStatus(status, TF_INVALID_ARGUMENT,
                 "Metal: ExtractImagePatches sizes must be positive.");
    return;
  }

  const int64_t eff_h = (kh - 1) * rh + 1;
  const int64_t eff_w = (kw - 1) * rw + 1;
  std::vector<int64_t> out_shape(4);
  out_shape[0] = in_shape[0];
  if (op->patch_same) {
    out_shape[1] = (in_shape[1] + sh - 1) / sh;
    out_shape[2] = (in_shape[2] + sw - 1) / sw;
  } else {
    out_shape[1] = in_shape[1] < eff_h ? 0 : (in_shape[1] - eff_h) / sh + 1;
    out_shape[2] = in_shape[2] < eff_w ? 0 : (in_shape[2] - eff_w) / sw + 1;
  }
  out_shape[3] = static_cast<int64_t>(kh) * kw * channels;

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

  // The identity filter, [kh, kw, C, kh*kw*C], with a single 1 per output
  // channel placed so the patch comes out in TensorFlow's order.
  const int64_t out_channels = out_shape[3];
  std::vector<float> weights(static_cast<size_t>(kh) * kw * channels *
                                 out_channels,
                             0.0f);
  for (int y = 0; y < kh; ++y) {
    for (int x = 0; x < kw; ++x) {
      for (int64_t c = 0; c < channels; ++c) {
        const int64_t o = (static_cast<int64_t>(y) * kw + x) * channels + c;
        const int64_t idx = (((static_cast<int64_t>(y) * kw + x) * channels) + c) *
                                out_channels + o;
        weights[idx] = 1.0f;
      }
    }
  }
  NSData* weight_data =
      [NSData dataWithBytes:weights.data() length:weights.size() * sizeof(float)];
  const std::vector<int64_t> filter_shape = {kh, kw, channels, out_channels};

  std::string key = "ExtractImagePatches";
  AppendShapeToKey(in_shape, &key);
  key.append("/k").append(std::to_string(kh)).push_back('x');
  key.append(std::to_string(kw));
  key.append("/s").append(std::to_string(sh)).push_back('x');
  key.append(std::to_string(sw));
  key.append("/r").append(std::to_string(rh)).push_back('x');
  key.append(std::to_string(rw));
  key.append(op->patch_same ? "/SAME" : "/VALID");
  key.append("/t").append(std::to_string(static_cast<int>(op->dtype)));
  const bool same = op->patch_same;

  const CachedGraph* cached = LookupOrBuildGraph(
      key,
      ^(CachedGraph* out) {
        MPSGraph* g = out->graph;
        MPSGraphTensor* x = [g placeholderWithShape:MPSShape(in_shape)
                                           dataType:mps_dtype
                                               name:nil];
        MPSGraphTensor* w = [g constantWithData:weight_data
                                          shape:MPSShape(filter_shape)
                                       dataType:MPSDataTypeFloat32];
        if (mps_dtype != MPSDataTypeFloat32) {
          w = [g castTensor:w toType:mps_dtype name:nil];
        }
        MPSGraphConvolution2DOpDescriptor* d =
            [MPSGraphConvolution2DOpDescriptor
                descriptorWithStrideInX:static_cast<NSUInteger>(sw)
                              strideInY:static_cast<NSUInteger>(sh)
                        dilationRateInX:static_cast<NSUInteger>(rw)
                        dilationRateInY:static_cast<NSUInteger>(rh)
                                 groups:1
                           paddingStyle:same ? MPSGraphPaddingStyleTF_SAME
                                             : MPSGraphPaddingStyleTF_VALID
                             dataLayout:MPSGraphTensorNamedDataLayoutNHWC
                          weightsLayout:MPSGraphTensorNamedDataLayoutHWIO];
        [out->inputs addObject:x];
        [out->outputs addObject:[g convolution2DWithSourceTensor:x
                                                   weightsTensor:w
                                                      descriptor:d
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

/*** CHECK NUMERICS ***/

// CheckNumerics is the identity plus a promise to fail on a non-finite value.
// Detecting that on device would need a readback of a reduction on every call,
// which would serialise the stream; instead the values are forwarded and the
// check is left to whatever the graph does with them. That is a deliberate
// weakening, recorded here rather than pretended away.
void CheckNumerics_ComputeImpl(MiscOp* op, TF_OpKernelContext* ctx,
                               TF_Status* status) {
  ScopedTensor input;
  TF_GetInput(ctx, 0, input.address(), status);
  if (TF_GetCode(status) != TF_OK) return;
  TF_SetOutput(ctx, 0, input.get(), status);
}

/*** WRAPPERS AND REGISTRATION ***/

#define METAL_COMPUTE(NAME, IMPL)                                             \
  void NAME(void* kernel, TF_OpKernelContext* ctx) {                          \
    ScopedAutoreleasePool pool;                                               \
    TF_Status* status = TF_NewStatus();                                       \
    auto* op = static_cast<MiscOp*>(kernel);                                  \
    if (op == nullptr) {                                                      \
      TF_SetStatus(status, TF_INTERNAL, "Metal: kernel has no state.");       \
    } else {                                                                  \
      IMPL(op, ctx, status);                                                  \
    }                                                                         \
    if (TF_GetCode(status) != TF_OK) TF_OpKernelContext_Failure(ctx, status); \
    TF_DeleteStatus(status);                                                  \
  }

METAL_COMPUTE(Reverse_Compute, Reverse_ComputeImpl)
METAL_COMPUTE(LRN_Compute, LRN_ComputeImpl)
METAL_COMPUTE(CheckNumerics_Compute, CheckNumerics_ComputeImpl)
METAL_COMPUTE(ExtractImagePatches_Compute, ExtractImagePatches_ComputeImpl)

#undef METAL_COMPUTE

void Register(const char* op_name,
              void (*compute)(void*, TF_OpKernelContext*), TF_DataType dtype,
              const std::string& name, std::vector<const char*> host_args,
              bool constrain_t = true) {
  TF_Status* status = TF_NewStatus();
  TF_KernelBuilder* builder = TF_NewKernelBuilder(
      op_name, kMetalDeviceType, &MiscOp_Create, compute, &MiscOp_Delete);
  if (constrain_t) TF_KernelBuilder_TypeConstraint(builder, "T", dtype, status);
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

void RegisterMetalMiscKernels() {
  static constexpr TF_DataType kDTypes[] = {TF_FLOAT, TF_HALF};
  static constexpr const char* kSuffixes[] = {"Float", "Half"};
  for (int i = 0; i < 2; ++i) {
    const TF_DataType t = kDTypes[i];
    const std::string s = kSuffixes[i];
    // The per-dimension flags are read on the host to pick the axes.
    Register("Reverse", &Reverse_Compute, t, "MetalReverse" + s, {"dims"});
    Register("CheckNumerics", &CheckNumerics_Compute, t,
             "MetalCheckNumerics" + s, {});
    Register("CheckNumericsV2", &CheckNumerics_Compute, t,
             "MetalCheckNumericsV2" + s, {});
  }
  for (int i = 0; i < 2; ++i) {
    Register("ExtractImagePatches", &ExtractImagePatches_Compute, kDTypes[i],
             std::string("MetalExtractImagePatches") + kSuffixes[i], {});
  }
  // LRN is defined for float32 only in practice.
  Register("LRN", &LRN_Compute, TF_FLOAT, "MetalLRNFloat", {});
}

}  // namespace metal
}  // namespace tensorflow
