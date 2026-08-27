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

// Image resizing, bilinear and nearest, with both gradients.
//
// TensorFlow's align_corners and half_pixel_centers attributes are MPSGraph's
// alignCorners and centerResult, with the same meaning, so the sampling grid
// needs no adjustment. The two attributes are mutually exclusive in
// TensorFlow, which is checked here rather than left to produce a subtly
// shifted image.

int64_t ElementCount(const std::vector<int64_t>& s) {
  int64_t n = 1;
  for (int64_t d : s) n *= d;
  return n;
}

struct ResizeOp {
  TF_DataType dtype = TF_FLOAT;
  bool align_corners = false;
  bool half_pixel_centers = false;
};

void* ResizeOp_Create(TF_OpKernelConstruction* ctx) {
  TF_Status* status = TF_NewStatus();
  auto* op = new ResizeOp();
  TF_OpKernelConstruction_GetAttrType(ctx, "T", &op->dtype, status);
  if (TF_GetCode(status) != TF_OK) {
    TF_OpKernelConstruction_Failure(ctx, status);
    TF_DeleteStatus(status);
    delete op;
    return nullptr;
  }
  TF_Bool flag = 0;
  TF_OpKernelConstruction_GetAttrBool(ctx, "align_corners", &flag, status);
  if (TF_GetCode(status) != TF_OK) TF_SetStatus(status, TF_OK, "");
  op->align_corners = flag != 0;
  flag = 0;
  TF_OpKernelConstruction_GetAttrBool(ctx, "half_pixel_centers", &flag, status);
  if (TF_GetCode(status) != TF_OK) TF_SetStatus(status, TF_OK, "");
  op->half_pixel_centers = flag != 0;

  if (op->align_corners && op->half_pixel_centers) {
    TF_SetStatus(status, TF_INVALID_ARGUMENT,
                 "Metal: align_corners and half_pixel_centers cannot both be "
                 "set.");
    TF_OpKernelConstruction_Failure(ctx, status);
    TF_DeleteStatus(status);
    delete op;
    return nullptr;
  }
  TF_DeleteStatus(status);
  return op;
}

void ResizeOp_Delete(void* kernel) { delete static_cast<ResizeOp*>(kernel); }

/*** FORWARD ***/

template <bool kBilinear>
void Resize_ComputeImpl(ResizeOp* op, TF_OpKernelContext* ctx,
                        TF_Status* status) {
  ScopedTensor images, size_t_;
  TF_GetInput(ctx, 0, images.address(), status);
  if (TF_GetCode(status) != TF_OK) return;
  TF_GetInput(ctx, 1, size_t_.address(), status);
  if (TF_GetCode(status) != TF_OK) return;

  const std::vector<int64_t> in_shape = ShapeOf(images.get());
  if (in_shape.size() != 4) {
    TF_SetStatus(status, TF_INVALID_ARGUMENT,
                 "Metal: resize expects a rank-4 NHWC input.");
    return;
  }
  if (TF_TensorElementCount(size_t_.get()) != 2 ||
      TF_TensorType(size_t_.get()) != TF_INT32) {
    TF_SetStatus(status, TF_INVALID_ARGUMENT,
                 "Metal: resize size must be a two-element int32 tensor.");
    return;
  }
  const int32_t* wanted =
      static_cast<const int32_t*>(TF_TensorData(size_t_.get()));
  std::vector<int64_t> out_shape = in_shape;
  out_shape[1] = wanted[0];
  out_shape[2] = wanted[1];

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

  std::string key = kBilinear ? "ResizeBilinear" : "ResizeNearest";
  AppendShapeToKey(in_shape, &key);
  AppendShapeToKey(out_shape, &key);
  key.append(op->align_corners ? "/align" : "/noalign");
  key.append(op->half_pixel_centers ? "/half" : "/nohalf");
  key.append("/t").append(std::to_string(static_cast<int>(op->dtype)));

  const BOOL center = op->half_pixel_centers ? YES : NO;
  const BOOL align = op->align_corners ? YES : NO;
  // The target size goes in as a constant tensor, since it was read on the
  // host to size the output anyway.
  std::vector<int32_t> size_values = {wanted[0], wanted[1]};
  NSData* size_data = [NSData dataWithBytes:size_values.data()
                                     length:size_values.size() * sizeof(int32_t)];
  const std::vector<int64_t> size_shape = {2};

  const CachedGraph* cached = LookupOrBuildGraph(
      key,
      ^(CachedGraph* out) {
        MPSGraph* g = out->graph;
        MPSGraphTensor* x = [g placeholderWithShape:MPSShape(in_shape)
                                           dataType:mps_dtype
                                               name:nil];
        MPSGraphTensor* sz = [g constantWithData:size_data
                                           shape:MPSShape(size_shape)
                                        dataType:MPSDataTypeInt32];
        [out->inputs addObject:x];
        [out->outputs
            addObject:(kBilinear
                           ? [g resizeBilinearWithTensor:x
                                              sizeTensor:sz
                                            centerResult:center
                                            alignCorners:align
                                                  layout:
                                                      MPSGraphTensorNamedDataLayoutNHWC
                                                    name:nil]
                           : [g resizeNearestWithTensor:x
                                             sizeTensor:sz
                                    nearestRoundingMode:
                                        MPSGraphResizeNearestRoundingModeRoundPreferFloor
                                           centerResult:center
                                           alignCorners:align
                                                 layout:
                                                     MPSGraphTensorNamedDataLayoutNHWC
                                                   name:nil])];
      },
      status);
  if (cached == nullptr) return;

  MPSGraphTensorData* in_data =
      TensorDataForTensor(images.get(), op->dtype, device, status);
  if (in_data == nil) return;
  MPSGraphTensorData* o_data =
      TensorDataForTensor(output.get(), op->dtype, device, status);
  if (o_data == nil) return;
  RunGraph(stream, *cached, @[ in_data ], @[ o_data ], status);
}

// The gradients are deliberately absent.
//
// Every MPSGraph resize gradient entry point on this SDK,
// resizeBilinearWithGradientTensor:, resizeNearestWithGradientTensor: and the
// general resizeWithGradientTensor:mode:, aborts the process with
//
//   MPSNDArrayResample.mm:601: failed assertion
//   `Error: source and destination channels mismatch'
//
// for every combination of shapes and layouts tried, including matching
// channel counts of 1 and 3 in both directions. Registering a kernel that
// takes the whole process down is worse than not registering one: without
// these, TensorFlow places the gradient on the host, which is slow but
// correct. Revisit when the assertion no longer fires.

/*** WRAPPERS AND REGISTRATION ***/

#define METAL_COMPUTE(NAME, IMPL)                                             \
  void NAME(void* kernel, TF_OpKernelContext* ctx) {                          \
    ScopedAutoreleasePool pool;                                               \
    TF_Status* status = TF_NewStatus();                                       \
    auto* op = static_cast<ResizeOp*>(kernel);                                \
    if (op == nullptr) {                                                      \
      TF_SetStatus(status, TF_INTERNAL, "Metal: kernel has no state.");       \
    } else {                                                                  \
      IMPL(op, ctx, status);                                                  \
    }                                                                         \
    if (TF_GetCode(status) != TF_OK) TF_OpKernelContext_Failure(ctx, status); \
    TF_DeleteStatus(status);                                                  \
  }

METAL_COMPUTE(ResizeBilinear_Compute, Resize_ComputeImpl<true>)
METAL_COMPUTE(ResizeNearest_Compute, Resize_ComputeImpl<false>)

#undef METAL_COMPUTE

void Register(const char* op_name,
              void (*compute)(void*, TF_OpKernelContext*), TF_DataType dtype,
              const std::string& name, const char* host_arg) {
  TF_Status* status = TF_NewStatus();
  TF_KernelBuilder* builder = TF_NewKernelBuilder(
      op_name, kMetalDeviceType, &ResizeOp_Create, compute, &ResizeOp_Delete);
  TF_KernelBuilder_TypeConstraint(builder, "T", dtype, status);
  if (host_arg != nullptr) TF_KernelBuilder_HostMemory(builder, host_arg);
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

void RegisterMetalImageKernels() {
  static constexpr TF_DataType kDTypes[] = {TF_FLOAT, TF_HALF};
  static constexpr const char* kSuffixes[] = {"Float", "Half"};
  for (int i = 0; i < 2; ++i) {
    const TF_DataType t = kDTypes[i];
    const std::string s = kSuffixes[i];
    // The target size is read on the host to size the output.
    Register("ResizeBilinear", &ResizeBilinear_Compute, t,
             "MetalResizeBilinear" + s, "size");
    Register("ResizeNearestNeighbor", &ResizeNearest_Compute, t,
             "MetalResizeNearestNeighbor" + s, "size");
  }
}

}  // namespace metal
}  // namespace tensorflow
